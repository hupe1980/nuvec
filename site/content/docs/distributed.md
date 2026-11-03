+++
title = "Distributed"
description = "nuvec::distributed runs one program across processes or Macs with MLX's collectives (all_sum, all_gather, sum_scatter, send, recv) over MLX's TCP ring or, experimentally, JACCL over Thunderbolt 5 RDMA, and splits linear layers for tensor parallelism."
weight = 115
[extra]
group = "Libraries"
+++

`nuvec::distributed` is MLX's `mx.distributed`: a group of processes, each with a rank, runs the same program, and collective operations combine their arrays. The collectives are ordinary lazy operations: they run when their result is evaluated, and they compose with `grad`, `vmap` and `compile`.

## Launching a group

Every process calls `init`, which reads MLX's launch environment and connects the first backend it describes: the TCP ring, then JACCL. `init_backend(strict, Backend::Ring)` (or `Backend::Jaccl`) names the backend, as MLX's `init(backend=...)`; a second call returns the same group, and `is_available(backend)` reports whether a backend can run on this machine.

The ring reads `MLX_HOSTFILE`, a JSON list with one entry per rank of `"host:port"` addresses, and `MLX_RANK`. The processes can share one Mac (over loopback) or run one per Mac (over Ethernet or Thunderbolt IP).

```json
[["10.0.0.1:5000"], ["10.0.0.2:5000"]]
```

Each address is one TCP connection to that rank's ring neighbour; list several per rank to spread large transfers over more connections. Without the environment, `init(false)` returns the group of this process alone, where every collective is the identity, so the same code runs on one machine unchanged.

```rust
use nuvec::prelude::*;
use nuvec::distributed;

# fn main() -> Result<()> {
let group = distributed::init(false)?;
let x = array![1.0, 2.0, 3.0];
let total = distributed::all_sum(&x, None); // the element-wise sum over every rank
let all = distributed::all_gather(&x, None); // every rank's x, in rank order
assert_eq!(total.shape(), [3]);
assert_eq!(all.shape(), [3 * group.size() as i32]);
# Ok(())
# }
```

## Collectives

| Operation | Result |
|---|---|
| `all_sum`, `all_max`, `all_min` | The element-wise reduction over the group; every rank gets the same bits, and reruns repeat them |
| `all_gather` | Every rank's array, concatenated along the first axis in rank order |
| `sum_scatter` | Rank `r`'s slice `r` of the first axis, summed over the group (the first axis must be divisible by the group size) |
| `send(x, dst)`, `recv(shape, dtype, src)`, `recv_like(x, src)` | Point to point (on the ring topologies, between neighbours); a receive of another dtype or shape is an error |
| `sum_gradients(x)` | `x`, with its gradient summed over the group |

Each takes the group as its last argument (`None`: the default group, which `init` sets). Collectives run on the CPU over the arrays' memory. On a GPU stream they run on the group's worker thread once their inputs are ready, and the GPU work that reads their results is committed by that thread once they are done (MLX's communication stream, without its GPU-side wait, which Metal fails after a few seconds). An input that nothing else holds is reduced or sent in place. `send` and `recv` frame each array, so every rank must run the same nuvec version.

## Gradients and compile

As in MLX, the gradient of `all_sum` passes the cotangent through (each rank's loss is its own), and the gradient of `all_gather` is this rank's rows. By the same rule, the gradient of `sum_scatter` is the cotangent in this rank's rows and zero elsewhere. A compiled function with collectives communicates on every call. For data-parallel training, average the gradients with `all_sum` before the optimizer step.

## Tensor-parallel layers

`nuvec_nn::sharded` splits a linear layer across a group as `mlx.nn` does: `AllToShardedLinear` keeps each rank's slice of the output features, and `ShardedToAllLinear` each rank's slice of the input features, summing the partial products with one `all_sum`. A pair of them splits an MLP (or attention's projections by heads) with one collective per pair. Both convert full-precision and quantized layers, and an uneven split is an error naming the layer. nuvec-lm shards whole models this way ([tensor parallelism](@/docs/llm.md#tensor-parallelism)).

```rust
use nuvec::distributed::Group;
use nuvec::{array, random};
use nuvec_nn::Linear;
use nuvec_nn::sharded::{AllToShardedLinear, ShardedToAllLinear};

# fn main() -> nuvec::Result<()> {
random::seed(0);
let group = Group::single();
let up = AllToShardedLinear::from_linear(&Linear::new(4, 16), &group)?;
let down = ShardedToAllLinear::from_linear(&Linear::new(16, 4), &group)?;
let y = down.forward(&up.forward(&array![[1.0, 2.0, 3.0, 4.0]]));
assert_eq!(y.shape(), [1, 4]);
# Ok(())
# }
```

## JACCL: RDMA over Thunderbolt 5 (experimental)

The `jaccl` feature of `nuvec` (`cargo add nuvec --features jaccl`) adds MLX's JACCL backend: the collectives over RDMA between Macs cabled with Thunderbolt 5 (macOS 26.2 or later). It is experimental: it is tested on an emulated fabric, not on Thunderbolt 5 hardware.

- **Enable RDMA** once per Mac: start in recoveryOS, open Terminal, run `rdma_ctl enable`, reboot. `nuvec::distributed::jaccl::devices()` then lists the Mac's RDMA devices with their port state and address; a device without an IPv4 address needs a link-local one on its Thunderbolt interface.
- **Cable** every pair of Macs (the mesh, up to 8 Macs) or every Mac to its two neighbours (the ring), with up to four cables per pair; large transfers use every cable at once.
- **Launch** with MLX's hostfile and `mlx.launch --backend jaccl` (or `--backend jaccl-ring` to prefer the ring), which starts any program: `mlx.launch --backend jaccl --hostfile hosts.json -- ./my-program`. It writes the device file (`MLX_IBV_DEVICES`: per rank, per peer `null`, a device name or a list of names), the rank and rank 0's coordinator address, and `init` picks them up. `jaccl::Config` and `jaccl::init_with` take the same configuration in code, and `jaccl::Options::side_channel` replaces the coordinator with the caller's own all-gather.

Reductions add the ranks' values in rank order (the mesh) or once per chunk owner (the ring), so every rank gets the same bits and reruns repeat them. Every message carries a sequence number, so a frame lost on the link fails the operation instead of corrupting a later one.

## Failures and re-forming

A peer that never starts, exits or loses its link fails every pending operation with an error naming its rank within the timeout (`NUVEC_DISTRIBUTED_TIMEOUT`, in seconds, default 30). The group then refuses further work with that first error. Nothing hangs or panics.

- **Finding a lost peer while idle:** `group.health()` checks every connection without running a collective or sending data. A peer whose process ended is reported at once, naming its rank. Every connection uses TCP keepalive, so a Mac that went to sleep or lost its cable is reported within about 15 s.
- **Re-forming in the same process:** `group.close()` releases a group's connections, after which its operations fail. A new group over the remaining Macs then forms with `init_with`, or with `init` from a new hostfile: `init` never returns a failed or closed group.

## Authentication

Set the same secret on every Mac to authenticate every connection: `NUVEC_DISTRIBUTED_KEY` for `init`, or `Options::key` (`jaccl::Options::key` for JACCL's side channel). Generate it with `openssl rand -hex 32`; shorter than 16 bytes is refused.

Both ends prove that they hold the key (HMAC-SHA256 over fresh challenges from each side), so a host without it cannot join a group, and a recorded handshake cannot be replayed. A stray connection is dropped while the group forms. A key on some ranks only, or different keys, fail with an error saying so, and the key never appears in an error.

The key authenticates who connects; it does not encrypt the traffic. On a network you do not trust, run the group over a VPN such as WireGuard, or on its own network segment.

## Performance

`just bench-distributed` times `all_sum` against MLX's ring backend on the same Mac. On an M4 Pro over loopback (2026-11-03) nuvec takes 0.24–0.87× MLX's time with 2 ranks and 0.32–0.83× with 4, from 4 KB to 256 MB. `uv run scripts/bench_distributed.py --hostfile hosts.json --backend jaccl` runs the same comparison on several Macs.
