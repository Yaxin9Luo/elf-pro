#!/usr/bin/env python3
"""Run generation-style standard benchmarks with an autoregressive HF model."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_input_length: int,
    max_new_tokens: int,
) -> list[str]:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_token_ids = generated[:, encoded["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)


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
        "model": args.model,
        "prompt_format": args.prompt_format,
        "max_examples": args.max_examples,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
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
    dtype = torch.float32
    if device.type != "cpu" and args.dtype == "bf16":
        dtype = torch.bfloat16
    elif device.type != "cpu" and args.dtype == "fp16":
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        use_fast=not args.use_slow_tokenizer,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    outputs = []
    for batch in tqdm(list(chunked(rows, args.batch_size)), desc="AR standard benchmark"):
        prompts = [format_prompt(row["input"], args.prompt_format) for row in batch]
        generations = generate_batch(
            model,
            tokenizer,
            prompts,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
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
            })

    write_jsonl(Path(args.output_jsonl), outputs)
    summary = summarize(outputs, args)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark_file", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--prompt_format", choices=["raw", "alpaca"], default="raw")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
