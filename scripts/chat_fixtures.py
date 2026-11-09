# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers==5.22.1", "jinja2==3.1.6", "mlx==0.32.4", "mlx-lm==0.32.2", "huggingface_hub"]
# ///
"""Chat-template and chat-dataset fixtures for nuvec-lm, recorded from transformers and mlx-lm.

    uv run scripts/chat_fixtures.py corpus      # download the corpus' tokenizer files
    uv run scripts/chat_fixtures.py render      # crates/nuvec-lm/tests/fixtures/chat/<repo>.json
    uv run scripts/chat_fixtures.py datasets    # .../chat/datasets/<repo>.json
    uv run scripts/chat_fixtures.py generate    # .../chat/generate.json
    uv run scripts/chat_fixtures.py timing      # mlx-lm's load time of 1,000 chat rows

Templates and tokenizers are read from the Hugging Face cache (no weights are downloaded);
`strftime_now` is fixed to 2026-11-08 12:00.
"""

import argparse
import datetime as _dt
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/nuvec-lm/tests/fixtures/chat"
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "additional_chat_templates/*",
    "config.json",
]
SUPPORTED = {
    "llama", "mistral", "qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "gpt_oss",
    "gemma4", "gemma4_text",
}
CANDIDATES = [
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-30B-A3B-4bit",
    "mlx-community/Qwen3.8-27B-4bit",
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/Qwen3-4B-Thinking-2507-4bit",
    "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
    "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Llama-3.3-70B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
    "mlx-community/Ministral-8B-Instruct-2410-4bit",
    "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/gemma-4-26b-a4b-it-4bit",
    "mlx-community/gpt-oss-20b-MXFP4-Q8",
    "mlx-community/gpt-oss-120b-MXFP4-Q8",
    "lmstudio-community/gpt-oss-20b-MLX-8bit",
]
# Newer families are found by search (their repo names change with each release).
SEARCHES = ["Qwen3.5", "Qwen3.6"]
LOCAL = [
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/gpt-oss-20b-MXFP4-Q8",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen3.8-27B-4bit",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "A city, e.g. Paris"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7, "default": 1},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web <fast> & 'safe'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limits": {
                        "type": "object",
                        "properties": {"top_k": {"type": "number", "default": 1.0}},
                    },
                },
                "required": ["query"],
            },
        },
    },
]
NUMBER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tune",
            "description": "Grüße, 世界 🌍",
            "parameters": {
                "type": "object",
                "properties": {
                    "rate": {"type": "number", "default": 1e-05, "minimum": -0.0, "maximum": 1e16},
                    "scale": {"type": "number", "default": 0.0001, "multipleOf": 123456789012345.6},
                    "seed": {"type": "integer", "default": 9007199254740993},
                },
            },
        },
    }
]
CALL = {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
CONVERSATIONS = {
    "user": {"messages": [{"role": "user", "content": "Hello, who are you?"}]},
    "system": {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Name three primes."},
        ]
    },
    "turns": {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 = 4."},
            {"role": "user", "content": "  And times three?\n"},
        ]
    },
    "tools": {
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "id": "call_0", "function": CALL}],
            },
            {"role": "tool", "tool_call_id": "call_0", "name": "get_weather", "content": "{\"temperature\": 21.5}"},
            {"role": "assistant", "content": "It is 21.5 °C in Paris."},
        ],
    },
    "tools_string_arguments": {
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "sunny"},
        ],
    },
    # Mistral's templates need tool-call ids of 9 alphanumeric characters.
    "tools_short_ids": {
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "tool_calls": [{"type": "function", "id": "abc123DEF", "function": CALL}]},
            {"role": "tool", "tool_call_id": "abc123DEF", "content": "21.5"},
            {"role": "assistant", "content": "It is 21.5 °C."},
            {"role": "user", "content": "Thanks!"},
        ],
    },
    "reasoning": {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "reasoning_content": "Two plus two.", "content": "4"},
            {"role": "user", "content": "And 3 + 3?"},
            {"role": "assistant", "reasoning_content": "Three plus three.", "content": "6"},
        ]
    },
    "parts": {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Describe a cat."}]},
        ]
    },
    "unusual": {
        "tools": NUMBER_TOOLS,
        "messages": [
            {"role": "user", "content": "Grüße, 世界 🌍 <tag> & 'single' \"double\" \\ back\ttab"},
        ],
    },
    "empty_system": {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hi"},
        ]
    },
}


class FixedDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 8, 12, 0, 0)


def patch_clock():
    import transformers.utils.chat_template_utils as ctu

    ctu.datetime = FixedDatetime


def snapshot(repo, download=False):
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(repo, allow_patterns=TOKENIZER_FILES, local_files_only=not download)
        )
    except Exception as e:  # noqa: BLE001
        print(f"  {repo}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def corpus_repos():
    path = OUT / "corpus.json"
    return json.loads(path.read_text())["repos"]


def cmd_corpus(_args):
    from huggingface_hub import HfApi

    api = HfApi()
    candidates = list(CANDIDATES)
    for term in SEARCHES:
        for m in api.list_models(author="mlx-community", search=term, sort="downloads", limit=6):
            if m.id not in candidates and m.id.endswith("4bit"):
                candidates.append(m.id)
    repos = []
    for repo in candidates:
        path = snapshot(repo, download=True)
        if path is None or not (path / "tokenizer.json").exists():
            continue
        config = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
        model_type = config.get("model_type") or config.get("text_config", {}).get("model_type")
        if model_type not in SUPPORTED:
            print(f"  {repo}: model_type {model_type} not supported, skipped", file=sys.stderr)
            continue
        repos.append({"repo": repo, "model_type": model_type, "commit": path.name})
        print(f"{repo} ({model_type})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus.json").write_text(json.dumps({"repos": repos}, indent=1) + "\n")
    print(f"{len(repos)} repos")


def cases():
    for name, conv in CONVERSATIONS.items():
        for generation in (False, True):
            for thinking in (None, False):
                yield {
                    "conversation": name,
                    "add_generation_prompt": generation,
                    "enable_thinking": thinking,
                }
    yield {"conversation": "turns_prefill", "continue_final_message": True}
    yield {"conversation": "turns", "continue_final_message": True, "add_generation_prompt": True}


def conversation(name):
    if name == "turns_prefill":
        conv = json.loads(json.dumps(CONVERSATIONS["turns"]))
        conv["messages"].append({"role": "assistant", "content": "Twelve"})
        return conv
    return CONVERSATIONS[name]


def cmd_render(_args):
    from transformers import AutoTokenizer

    patch_clock()
    OUT.mkdir(parents=True, exist_ok=True)
    conversations = {n: conversation(n) for n in [*CONVERSATIONS, "turns_prefill"]}
    (OUT / "conversations.json").write_text(json.dumps(conversations, indent=1, ensure_ascii=False) + "\n")
    for entry in corpus_repos():
        repo = entry["repo"]
        path = snapshot(repo)
        if path is None:
            continue
        tok = AutoTokenizer.from_pretrained(path)
        records = []
        for case in cases():
            conv = conversation(case["conversation"])
            kwargs = {
                "tools": conv.get("tools"),
                "add_generation_prompt": case.get("add_generation_prompt", False),
                "continue_final_message": case.get("continue_final_message", False),
            }
            if case.get("enable_thinking") is not None:
                kwargs["enable_thinking"] = case["enable_thinking"]
            record = dict(case)
            try:
                record["text"] = tok.apply_chat_template(
                    json.loads(json.dumps(conv["messages"])), tokenize=False, **kwargs
                )
                if case.get("add_generation_prompt") and case.get("enable_thinking") is None:
                    record["ids"] = tok.apply_chat_template(
                        json.loads(json.dumps(conv["messages"])), tokenize=True, return_dict=False, **kwargs
                    )
            except Exception as e:  # noqa: BLE001
                record["error"] = str(e).splitlines()[0][:200]
            records.append(record)
        out = {
            "repo": repo,
            "special_tokens": tok.special_tokens_map,
            "templates": sorted(tok.chat_template) if isinstance(tok.chat_template, dict) else (
                ["default"] if tok.chat_template else []
            ),
            "cases": records,
        }
        name = repo.replace("/", "--")
        (OUT / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
        errors = sum("error" in r for r in records)
        print(f"{repo}: {len(records)} cases, {errors} errors")


CHAT_ROWS = [
    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"}]},
    {
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Times 3?"},
            {"role": "assistant", "content": "12"},
        ]
    },
    {"messages": CONVERSATIONS["tools"]["messages"], "tools": TOOLS},
    {"messages": [{"role": "user", "content": "Grüße 🌍"}, {"role": "assistant", "content": "Hallo, Welt!"}]},
    {"messages": [{"role": "user", "content": "Say nothing."}, {"role": "assistant", "content": ""}]},
]
COMPLETION_ROWS = [
    {"prompt": "Translate 'cat' to German.", "completion": "Katze"},
    {"prompt": "What is the boiling point of water?", "completion": "100 °C at sea level."},
    {"prompt": "List two colors.", "completion": "Red and blue."},
    {"prompt": "Weather in Paris?", "completion": "Sunny.", "tools": TOOLS},
    {"prompt": "Count to three.", "completion": "1, 2, 3"},
    {"prompt": "Hi", "completion": "Hello!"},
]
TEXT_ROWS = [
    {"text": "The quick brown fox jumps over the lazy dog."},
    {"text": "Hello, world!"},
    {"text": "Grüße, 世界 🌍"},
    {"text": "Line one\nLine two"},
    {"text": "A"},
    {"text": "Numbers: 1 2 3 4 5"},
]


def datasets_cases():
    yield "text", TEXT_ROWS, {}
    yield "chat", CHAT_ROWS, {}
    yield "completions", COMPLETION_ROWS, {}
    yield "chat_custom", [{"conversation": r["messages"], **({"tools": r["tools"]} if "tools" in r else {})} for r in CHAT_ROWS], {"chat_feature": "conversation"}
    yield "completions_custom", [{"question": r["prompt"], "answer": r["completion"]} for r in COMPLETION_ROWS], {"prompt_feature": "question", "completion_feature": "answer"}


def cmd_datasets(_args):
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.tuner.datasets import CacheDataset, create_dataset

    patch_clock()
    out_dir = OUT / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {name: data for name, data, _ in datasets_cases()}
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    for repo in LOCAL:
        path = snapshot(repo)
        if path is None:
            continue
        config = json.loads((path / "config.json").read_text())
        tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
        result = {"repo": repo, "eos_token_id": tokenizer.eos_token_id, "has_thinking": tokenizer.has_thinking, "sets": []}
        for name, data, keys in datasets_cases():
            for mask in (False, True):
                cfg = types.SimpleNamespace(mask_prompt=mask, **keys)
                entry = {"rows": name, "mask_prompt": mask, **keys}
                try:
                    ds = CacheDataset(create_dataset(json.loads(json.dumps(data)), tokenizer, cfg))
                    entry["out"] = [[list(map(int, ds[i][0])), int(ds[i][1])] for i in range(len(ds))]
                except Exception as e:  # noqa: BLE001
                    entry["error"] = str(e).splitlines()[0][:200]
                result["sets"].append(entry)
        (out_dir / f"{repo.replace('/', '--')}.json").write_text(json.dumps(result) + "\n")
        print(f"{repo}: eos {tokenizer.eos_token_id}, thinking {tokenizer.has_thinking}")


def cmd_generate(_args):
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    patch_clock()
    prompts = [
        {"prompt": "Write a haiku about the sea."},
        {"prompt": "Write a haiku about the sea.", "system_prompt": "You are a poet."},
        {"prompt": "Count to five.", "prefill_response": "One, two"},
        {"prompt": "Hello", "chat_template_config": {"enable_thinking": False}},
        {"prompt": "Hello", "ignore_chat_template": True},
    ]
    out = {"prompts": prompts, "repos": []}
    for repo in LOCAL:
        path = snapshot(repo)
        if path is None:
            continue
        config = json.loads((path / "config.json").read_text())
        tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
        ids = []
        for p in prompts:
            try:
                ids.append(generate_prompt(tokenizer, p))
            except Exception as e:  # noqa: BLE001
                ids.append({"error": str(e).splitlines()[0][:200]})
        out["repos"].append({"repo": repo, "ids": ids})
        print(repo)
    (OUT / "generate.json").write_text(json.dumps(out) + "\n")


def generate_prompt(tokenizer, p):
    """mlx_lm.generate's main(), as of 0.32.2."""
    prompt = p["prompt"]
    use_chat_template = not p.get("ignore_chat_template") and tokenizer.has_chat_template
    if use_chat_template:
        messages = [{"role": "system", "content": p["system_prompt"]}] if "system_prompt" in p else []
        messages.append({"role": "user", "content": prompt})
        has_prefill = "prefill_response" in p
        if has_prefill:
            messages.append({"role": "assistant", "content": p["prefill_response"]})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=has_prefill,
            add_generation_prompt=not has_prefill,
            **p.get("chat_template_config", {}),
        )
    return tokenizer.encode(prompt, add_special_tokens=not use_chat_template)


def cmd_timing(args):
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.tuner.datasets import CacheDataset, create_dataset

    path = snapshot(args.model)
    rows = [CHAT_ROWS[i % len(CHAT_ROWS)] for i in range(args.rows)]
    rows = [{**r, "messages": [*r["messages"][:-1], {**r["messages"][-1], "content": r["messages"][-1]["content"] + f" ({i})"}]} for i, r in enumerate(rows)]
    path_out = Path(args.write) if args.write else None
    if path_out:
        path_out.mkdir(parents=True, exist_ok=True)
        with open(path_out / "train.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    config = json.loads((path / "config.json").read_text())
    tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
    best = float("inf")
    for _ in range(args.rounds):
        start = time.perf_counter()
        ds = CacheDataset(create_dataset(json.loads(json.dumps(rows)), tokenizer, types.SimpleNamespace(mask_prompt=True)))
        for i in range(len(ds)):
            ds[i]
        best = min(best, time.perf_counter() - start)
    print(json.dumps({"engine": "mlx-lm", "rows": args.rows, "seconds": best}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("corpus")
    sub.add_parser("render")
    sub.add_parser("datasets")
    sub.add_parser("generate")
    timing = sub.add_parser("timing")
    timing.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    timing.add_argument("--rows", type=int, default=1000)
    timing.add_argument("--rounds", type=int, default=3)
    timing.add_argument("--write", help="also write the rows as DIR/train.jsonl")
    args = parser.parse_args()
    {"corpus": cmd_corpus, "render": cmd_render, "datasets": cmd_datasets, "generate": cmd_generate, "timing": cmd_timing}[args.command](args)


if __name__ == "__main__":
    main()
