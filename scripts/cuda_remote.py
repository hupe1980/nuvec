#!/usr/bin/env python3
"""Runs commands on a RunPod NVIDIA GPU against this working tree (the CUDA backend's
hardware gate: nuvec's maintainer machine is a Mac).

    scripts/cuda_remote.py up [--gpu ampere|hopper|"<type>,<type>"] [--secure]
    scripts/cuda_remote.py sync                 # rsync the working tree (no target/)
    scripts/cuda_remote.py run -- <command>     # in the synced tree, streamed
    scripts/cuda_remote.py ssh                  # print the ssh command
    scripts/cuda_remote.py down                 # terminate the pod
    scripts/cuda_remote.py session --gpu ... -- <command>   # up, sync, run, down

The API key comes from RUNPOD_API_KEY (or a `.env` file at the repository root) and is
never printed. The pod's id and address are kept in target/cuda-remote/pod.json; an
ephemeral SSH key in target/cuda-remote/. `session` always terminates the pod, also when
the command fails or is interrupted.
"""

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "target" / "cuda-remote"
API = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = "nvidia/cuda:13.2.0-devel-ubuntu24.04"
REMOTE = "/workspace/nuvec"
# Any Ampere card (sm_86), cheapest first.
AMPERE = ",".join([
    "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA RTX A4000", "NVIDIA A40",
    "NVIDIA RTX A6000", "NVIDIA GeForce RTX 3090 Ti",
])
HOPPER = "NVIDIA H100 PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 NVL"
# The tree to sync: this repository, or NUVEC_REMOTE_SRC.
SRC = Path(os.environ.get("NUVEC_REMOTE_SRC", ROOT)).resolve()
# The container installs sshd and the build tools, then waits; RunPod injects PUBLIC_KEY.
START = (
    "set -e; export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; "
    "apt-get install -y -qq openssh-server rsync curl git clang libclang-dev build-essential "
    "pkg-config python3 python3-venv >/dev/null; mkdir -p /root/.ssh /run/sshd; "
    'echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; '
    "chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; "
    "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.99.0 "
    ">/dev/null; "
    # A driver older than CUDA 13 (R580) runs the image's toolkit through the forward-
    # compatibility libraries (data-center GPUs).
    "drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); "
    'echo "export NUVEC_DRIVER=$drv" > /root/.nuvec_env; '
    'if [ "${drv%%.*}" -lt 580 ] && [ -d /usr/local/cuda/compat ]; then '
    'echo "export LD_LIBRARY_PATH=/usr/local/cuda/compat:\\$LD_LIBRARY_PATH" >> /root/.nuvec_env; fi; '
    "touch /root/.ready; sleep infinity"
)

# The deploy mutation (the REST create endpoint refuses this account's requests).
DEPLOY = """mutation($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName } }
}"""
STATUS = """query($id: String!) {
  pod(input: {podId: $id}) { desiredStatus runtime { ports { ip isIpPublic privatePort publicPort } } }
}"""


def key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        sys.exit("RUNPOD_API_KEY is not set (environment or .env)")
    return k


def api(method: str, path: str, body=None):
    req = urllib.request.Request(
        API + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key()}",
            "Content-Type": "application/json",
            # Python's default agent is refused by RunPod's edge (Cloudflare 1010).
            "User-Agent": "nuvec-cuda-remote/1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
            return json.loads(text) if text else None
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod {method} {path}: HTTP {e.code}: {e.read().decode()[:500]}")


def graphql(query: str, variables: dict) -> dict:
    req = urllib.request.Request(
        GRAPHQL,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Authorization": f"Bearer {key()}",
            "Content-Type": "application/json",
            "User-Agent": "nuvec-cuda-remote/1",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def state() -> dict:
    p = STATE / "pod.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save(s: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "pod.json").write_text(json.dumps(s, indent=2))


def ssh_key() -> Path:
    k = STATE / "id_ed25519"
    if not k.exists():
        STATE.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "nuvec-cuda-remote", "-f", str(k)],
            check=True,
        )
    return k


def ssh_args(s: dict) -> list[str]:
    return [
        "ssh", "-i", str(ssh_key()), "-p", str(s["port"]),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", f"root@{s['ip']}",
    ]


def up(gpu: str, secure: bool, attempts: int = 3) -> dict:
    """A ready pod; a host that does not start its container within 8 minutes (an image pull
    that stalls) is terminated and another one is tried."""
    for attempt in range(1, attempts + 1):
        s = up_once(gpu, secure)
        if s:
            return s
        print(f"attempt {attempt}: the container did not start; trying another host", flush=True)
    sys.exit(f"no pod became ready after {attempts} attempts")


def up_once(gpu: str, secure: bool) -> dict | None:
    gpu = {"ampere": AMPERE, "hopper": HOPPER}.get(gpu, gpu)
    if state().get("id"):
        sys.exit(f"a pod is recorded already ({state()['id']}); run `down` first")
    pub = (ssh_key().with_suffix(".pub")).read_text().strip()
    script = base64.b64encode(START.encode()).decode()
    pod, errors = None, []
    for g in [g.strip() for g in gpu.split(",")]:
        r = graphql(DEPLOY, {"input": {
            "name": "nuvec-cuda",
            "imageName": IMAGE,
            "gpuTypeId": g,
            "gpuCount": 1,
            "cloudType": "SECURE" if secure else "ALL",
            "containerDiskInGb": 60,
            "volumeInGb": 0,
            "minVcpuCount": 4,
            "minMemoryInGb": 16,
            "ports": "22/tcp",
            "supportPublicIp": True,
            "env": [{"key": "PUBLIC_KEY", "value": pub}, {"key": "NUVEC_START", "value": script}],
            "dockerArgs": "bash -c 'echo $NUVEC_START | base64 -d | bash'",
        }})
        pod = (r.get("data") or {}).get("podFindAndDeployOnDemand")
        if pod:
            break
        errors.append(f"{g}: {(r.get('errors') or [{}])[0].get('message', r)}")
    if not pod:
        sys.exit("no pod available:\n  " + "\n  ".join(errors))
    s = {"id": pod["id"], "gpu": pod["machine"]["gpuDisplayName"], "cost": pod["costPerHr"]}
    save(s)
    print(f"pod {s['id']} created ({s['gpu']}, ${s['cost']}/h); waiting for ssh", flush=True)
    started = time.time()
    deadline = started + 1200
    while time.time() < deadline:
        p = (graphql(STATUS, {"id": s["id"]}).get("data") or {}).get("pod") or {}
        if p.get("runtime") is None and time.time() - started > 480:
            down()
            return None
        ports = ((p.get("runtime") or {}).get("ports")) or []
        ssh = next((x for x in ports if x["privatePort"] == 22 and x["isIpPublic"]), None)
        if ssh:
            s.update(ip=ssh["ip"], port=ssh["publicPort"])
            save(s)
            ok = subprocess.run(
                ssh_args(s) + ["test -f /root/.ready"], capture_output=True
            ).returncode == 0
            if ok:
                drv = subprocess.run(ssh_args(s) + ["cat /root/.nuvec_env"], capture_output=True, text=True).stdout
                print(f"ready: {s['ip']}:{s['port']}\n{drv.strip()}", flush=True)
                return s
        time.sleep(10)
    down()
    return None


def sync() -> None:
    s = state()
    ssh = " ".join(shlex.quote(a) for a in ssh_args(s)[:-1])
    subprocess.run(ssh_args(s) + [f"mkdir -p {REMOTE}"], check=True)
    subprocess.run(
        [
            "rsync", "-az", "--delete", "-e", ssh,
            "--exclude", "target/", "--exclude", ".env", "--exclude", ".git/",
            "--exclude", "site/public/", "--exclude", "*.safetensors.part",
            f"{SRC}/", f"root@{s['ip']}:{REMOTE}/",
        ],
        check=True,
    )


def run(cmd: list[str]) -> int:
    s = state()
    line = (
        f"source /root/.cargo/env; source /root/.nuvec_env; cd {REMOTE}; "
        "export CUDA_HOME=/usr/local/cuda; "
    ) + " ".join(
        shlex.quote(c) for c in cmd
    )
    return subprocess.run(ssh_args(s) + [line]).returncode


def down() -> None:
    s = state()
    if s.get("id"):
        api("DELETE", f"/pods/{s['id']}")
        print(f"pod {s['id']} terminated", flush=True)
    (STATE / "pod.json").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("up", "session"):
        p = sub.add_parser(name)
        p.add_argument("--gpu", default=AMPERE, help="GPU types, comma-separated (any one)")
        p.add_argument("--secure", action="store_true", help="secure cloud (default community)")
        if name == "session":
            p.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("sync")
    r = sub.add_parser("run")
    r.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("ssh")
    sub.add_parser("down")
    a = ap.parse_args()
    strip = lambda c: c[1:] if c and c[0] == "--" else c
    if a.cmd == "up":
        up(a.gpu, a.secure)
    elif a.cmd == "sync":
        sync()
    elif a.cmd == "run":
        sys.exit(run(strip(a.command)))
    elif a.cmd == "ssh":
        print(" ".join(shlex.quote(x) for x in ssh_args(state())))
    elif a.cmd == "down":
        down()
    else:
        code = 1
        try:
            up(a.gpu, a.secure)
            sync()
            code = run(strip(a.command))
        finally:
            down()
        sys.exit(code)


if __name__ == "__main__":
    main()
