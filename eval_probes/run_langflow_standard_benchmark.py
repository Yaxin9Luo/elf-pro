#!/usr/bin/env python3
"""Run generation-style standard benchmarks with a pretrained LangFlow model."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoTokenizer


CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(answer|final answer)\s*:\s*", "", text)
    return text.strip()


def trim_generation(text: str) -> str:
    stops = ["\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:", "### Instruction:"]
    end = len(text)
    for stop in stops:
        idx = text.find(stop)
        if idx >= 0:
            end = min(end, idx)
    return text[:end].strip()


def parse_choice(text: str, num_choices: int) -> str | None:
    valid = CHOICE_LETTERS[:num_choices]
    stripped = text.strip()
    patterns = [
        r"^\(?([A-Z])\)?(?:[\s\).:\-]|$)",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\)?\b",
        r"\(([A-Z])\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                return letter
    return None


def parse_yes_no(text: str) -> str | None:
    match = re.search(r"\b(yes|no)\b", text.strip(), flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def parse_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def numbers_equal(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return math.isclose(float(pred), float(str(gold).replace(",", "")), rel_tol=1e-4, abs_tol=1e-4)
    except ValueError:
        return normalize_text(pred) == normalize_text(str(gold))


def score_generation(row: dict[str, Any], answer: str) -> dict[str, Any]:
    scoring = row.get("scoring", "exact_text")
    gold = row.get("answer", row.get("output", ""))
    if scoring == "multiple_choice":
        choices = row.get("choices") or []
        pred = parse_choice(answer, len(choices))
        return {"metric": "accuracy", "correct": pred == gold, "prediction": pred, "gold": gold}
    if scoring == "yes_no":
        pred = parse_yes_no(answer)
        return {"metric": "accuracy", "correct": pred == gold, "prediction": pred, "gold": gold}
    if scoring == "numeric":
        pred = parse_number(answer)
        return {"metric": "accuracy", "correct": numbers_equal(pred, str(gold)), "prediction": pred, "gold": gold}
    if scoring == "ifeval":
        return {"metric": "official_ifeval", "correct": None, "prediction": None, "gold": None}
    pred_norm = normalize_text(answer)
    gold_norm = normalize_text(str(gold))
    return {
        "metric": "exact_match",
        "correct": pred_norm == gold_norm,
        "prediction": pred_norm,
        "gold": gold_norm,
    }


def strip_chat_prompt(prompt: str) -> str:
    text = prompt.strip()
    if text.startswith("User:"):
        text = text[len("User:"):].lstrip()
    if text.endswith("Assistant:"):
        text = text[: -len("Assistant:")].rstrip()
    return text


def format_prompt(prompt: str, prompt_format: str) -> str:
    if prompt_format == "raw":
        return prompt
    if prompt_format == "alpaca":
        instruction = strip_chat_prompt(prompt)
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )
    raise ValueError(f"Unknown prompt_format={prompt_format}")


def chunked(rows: list[dict[str, Any]], size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def resolve_checkpoint_file(path: str) -> str:
    checkpoint = Path(path)
    if checkpoint.is_file():
        return str(checkpoint)
    latest = checkpoint / "latest_checkpoint.txt"
    if latest.exists():
        latest_path = latest.read_text(encoding="utf-8").strip()
        if latest_path:
            return resolve_checkpoint_file(latest_path)
    candidates = [
        checkpoint / "model.safetensors",
        checkpoint / "pytorch_model.safetensors",
    ]
    for item in candidates:
        if item.exists():
            return str(item)
    nested = sorted(
        checkpoint.glob("checkpoint_*/model.safetensors"),
        key=lambda p: int(p.parent.name.split("_")[-1]) if p.parent.name.split("_")[-1].isdigit() else -1,
    )
    if nested:
        return str(nested[-1])
    matches = sorted(checkpoint.glob("*.safetensors"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"No safetensors checkpoint found under {checkpoint}")


def load_langflow_model(langflow_repo: str, checkpoint: str, device: torch.device):
    repo = Path(langflow_repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from langflow import LangFlow, LangFlowConfig

    checkpoint_file = Path(resolve_checkpoint_file(checkpoint))
    checkpoint_dir = checkpoint_file.parent
    config_source = checkpoint_dir if (checkpoint_dir / "config.json").exists() else repo / "langflow"
    config = LangFlowConfig.from_pretrained(config_source)
    model = LangFlow(config)
    state_dict = load_file(str(checkpoint_file), device=str(device))
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_trainer_state(checkpoint: str) -> dict[str, Any] | None:
    try:
        checkpoint_dir = Path(resolve_checkpoint_file(checkpoint)).parent
    except FileNotFoundError:
        return None
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def prepare_prompt_ids(
    tokenizer,
    prompts: list[str],
    *,
    max_input_length: int,
    add_bos: bool,
) -> list[list[int]]:
    prompt_ids = []
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if add_bos and bos_id is not None:
            ids = [int(bos_id)] + ids
        if len(ids) > max_input_length:
            ids = ids[-max_input_length:]
        if not ids and bos_id is not None:
            ids = [int(bos_id)]
        prompt_ids.append([int(x) for x in ids])
    return prompt_ids


def conditional_latent(clean: torch.Tensor, noise: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    alpha = torch.sigmoid(-gamma.float()).sqrt().to(clean.dtype)
    sigma = torch.sigmoid(gamma.float()).sqrt().to(clean.dtype)
    return clean * alpha + noise * sigma


@torch.no_grad()
def generate_prefix_conditioned_batch(
    model,
    tokenizer,
    prompt_ids_list: list[list[int]],
    *,
    max_new_tokens: int,
    num_steps: int,
    seed: int,
    device: torch.device,
) -> list[str]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")

    batch_size = len(prompt_ids_list)
    prefix_lens = torch.tensor([len(ids) for ids in prompt_ids_list], device=device, dtype=torch.long)
    max_prefix_len = int(prefix_lens.max().item()) if batch_size else 0
    seq_len = max_prefix_len + max_new_tokens
    if seq_len > model.config.model_length:
        raise ValueError(f"seq_len={seq_len} exceeds LangFlow model_length={model.config.model_length}")

    dtype = next(model.parameters()).dtype
    embed_dim = model.config.hidden_size
    prompt_ids = torch.zeros(batch_size, max_prefix_len, device=device, dtype=torch.long)
    prefix_mask = torch.zeros(batch_size, seq_len, 1, device=device, dtype=torch.bool)
    for i, ids in enumerate(prompt_ids_list):
        n = len(ids)
        if n:
            prompt_ids[i, :n] = torch.tensor(ids, device=device, dtype=torch.long)
            prefix_mask[i, :n] = True

    clean_full = torch.zeros(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    if max_prefix_len:
        clean_full[:, :max_prefix_len] = model._embed_tokens(prompt_ids)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype, generator=generator)
    prefix_noise = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype, generator=generator)

    eps = 1e-5
    t = torch.linspace(1.0 - eps, eps, num_steps, device=device)
    gamma = model.proposal(t)
    x_self_cond = None

    for i in range(len(gamma) - 1):
        gamma_t = gamma[i]
        gamma_s = gamma[i + 1]
        prefix_z_t = conditional_latent(clean_full, prefix_noise, gamma_t)
        z = torch.where(prefix_mask, prefix_z_t, z)

        gamma_expanded = gamma_t.unsqueeze(0).expand(batch_size)
        logits = model.forward(
            noisy_embeds=z,
            timesteps=gamma_expanded,
            x_self_cond=x_self_cond,
            return_dict=False,
        )
        probs = F.softmax(logits.float(), dim=-1)
        x_pred = model._embed_tokens(probs)
        x_pred = torch.where(prefix_mask, clean_full, x_pred)
        if model.config.self_conditioning:
            x_self_cond = x_pred

        z = model._euler_edm_step(z, x_pred, gamma_t, gamma_s)
        prefix_z_s = conditional_latent(clean_full, prefix_noise, gamma_s)
        z = torch.where(prefix_mask, prefix_z_s, z)

    gamma_final = gamma[-1]
    z = torch.where(prefix_mask, conditional_latent(clean_full, prefix_noise, gamma_final), z)
    logits = model.forward(
        noisy_embeds=z,
        timesteps=gamma_final.unsqueeze(0).expand(batch_size),
        x_self_cond=x_self_cond,
        return_dict=False,
    )
    full_ids = logits.argmax(dim=-1)

    suffix_ids = []
    eos_id = tokenizer.eos_token_id
    for i in range(batch_size):
        start = int(prefix_lens[i].item())
        ids = full_ids[i, start:start + max_new_tokens].detach().cpu().tolist()
        if eos_id is not None and eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        suffix_ids.append(ids)
    return tokenizer.batch_decode(suffix_ids, skip_special_tokens=True)


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_benchmark: dict[str, Counter] = defaultdict(Counter)
    by_task: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        score = row["score"]
        if score["correct"] is None:
            continue
        for key in (row["benchmark"], "overall"):
            by_benchmark[key]["n"] += 1
            by_benchmark[key]["correct"] += int(bool(score["correct"]))
        task_key = f"{row['benchmark']}:{row.get('task', '')}"
        by_task[task_key]["n"] += 1
        by_task[task_key]["correct"] += int(bool(score["correct"]))

    def as_scores(counter_map):
        return {
            key: {
                "n": int(val["n"]),
                "correct": int(val["correct"]),
                "accuracy": (float(val["correct"]) / val["n"]) if val["n"] else None,
            }
            for key, val in sorted(counter_map.items())
        }

    return {
        "schema_version": 1,
        "benchmark_file": args.benchmark_file,
        "model": args.checkpoint,
        "langflow_repo": args.langflow_repo,
        "prompt_format": args.prompt_format,
        "max_examples": args.max_examples,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_steps": args.num_steps,
        "max_new_tokens": args.max_new_tokens,
        "add_bos": args.add_bos,
        "trainer_state": load_trainer_state(args.checkpoint),
        "num_outputs": len(rows),
        "scores_by_benchmark": as_scores(by_benchmark),
        "scores_by_task": as_scores(by_task),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(Path(args.benchmark_file))
    if args.max_examples:
        rows = rows[: args.max_examples]
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.num_shards > 1:
        rows = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard_index]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
        use_fast=not args.use_slow_tokenizer,
    )
    model = load_langflow_model(args.langflow_repo, args.checkpoint, device)

    outputs = []
    for batch_index, batch in enumerate(tqdm(list(chunked(rows, args.batch_size)), desc="LangFlow standard benchmark")):
        prompts = [format_prompt(row["input"], args.prompt_format) for row in batch]
        prompt_ids = prepare_prompt_ids(
            tokenizer,
            prompts,
            max_input_length=args.max_input_length,
            add_bos=args.add_bos,
        )
        generations = generate_prefix_conditioned_batch(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            num_steps=args.num_steps,
            seed=args.seed + args.shard_index * 100000 + batch_index,
            device=device,
        )
        for row, prompt, generated in zip(batch, prompts, generations):
            answer = trim_generation(generated)
            score = score_generation(row, answer)
            outputs.append({
                **row,
                "prompt": prompt,
                "generated": generated,
                "answer_text": answer,
                "score": score,
                "sampling": {
                    "method": "langflow_prefix_conditioned_edm",
                    "steps": args.num_steps,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                    "add_bos": args.add_bos,
                },
            })

    write_jsonl(Path(args.output_jsonl), outputs)
    summary = summarize(outputs, args)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langflow_repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--benchmark_file", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--prompt_format", choices=["raw", "alpaca"], default="raw")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--add_bos", action="store_true", default=True)
    parser.add_argument("--no_add_bos", action="store_false", dest="add_bos")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
