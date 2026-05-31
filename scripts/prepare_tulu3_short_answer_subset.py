#!/usr/bin/env python
"""Filter a Tulu3 SFT dataset into a short-answer ELF conditional dataset."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer


IMAGE_MARKERS = (
    "<image", "</image", "<img", "<|image", "[image", "![", "data:image",
    "<video", "<audio", "image token", "image_token",
)

TOOL_MARKERS = (
    "tool_call", "tool calls", "tool call", "json schema", "\"tools\"",
    "<tool", "</tool", "function call", "you have access to the following tools",
)

CODE_MARKERS = (
    "```", "<code", "</code", "def ", "class ", "import ", "#include",
)

REFUSAL_PREFIXES = (
    "i'm sorry", "i am sorry", "sorry,", "i can't", "i cannot",
    "i can’t", "i won’t", "i will not", "i'm unable", "i am unable",
)


def has_any(text, markers):
    lower = text.lower()
    return any(marker in lower for marker in markers)


def parse_csv(value):
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def parse_bucket_spec(value):
    buckets = []
    for item in parse_csv(value):
        if ":" not in item or "-" not in item:
            raise ValueError(f"Bad bucket spec item: {item!r}; expected MIN-MAX:COUNT")
        span, count = item.split(":", 1)
        lo, hi = span.split("-", 1)
        buckets.append((int(lo), int(hi), int(count)))
    return buckets


def ascii_ratio(text):
    if not text:
        return 1.0
    return sum(1 for ch in text if ord(ch) < 128) / len(text)


def first_reject_reason(row, args):
    prompt = str(row.get("input") or "")
    target = str(row.get("target") or "")
    source = str(row.get("source") or "")
    prompt_stripped = prompt.strip()
    target_stripped = target.strip()

    include_sources = parse_csv(args.include_source_substrings)
    exclude_sources = parse_csv(args.exclude_source_substrings)
    if include_sources and not any(part in source for part in include_sources):
        return "source_not_included"
    if exclude_sources and any(part in source for part in exclude_sources):
        return "source_excluded"
    if not prompt_stripped.endswith("Assistant:"):
        return "bad_prompt_tail"
    if args.single_turn_only:
        if prompt.count("User:") != 1 or prompt.count("Assistant:") != 1:
            return "not_single_turn"
    if not target_stripped:
        return "empty_target"
    if len(target_stripped) < args.min_target_chars:
        return "target_too_short"
    if "\n\n" in target_stripped or target_stripped.count("\n") > args.max_target_newlines:
        return "target_multiline"
    joined = prompt + "\n" + target
    if has_any(joined, IMAGE_MARKERS):
        return "image_marker"
    if has_any(prompt, TOOL_MARKERS):
        return "tool_marker"
    if args.drop_code and has_any(target, CODE_MARKERS):
        return "code_marker"
    normalized_target = target_stripped.lower().replace("’", "'").replace("‘", "'")
    if args.reject_refusals and normalized_target.startswith(REFUSAL_PREFIXES):
        return "refusal_target"
    if args.english_like:
        if ascii_ratio(prompt_stripped) < args.min_ascii_ratio or ascii_ratio(target_stripped) < args.min_ascii_ratio:
            return "non_english_like"
    if int(row.get("prompt_tokens") or len(row.get("condition_input_ids") or [])) > args.max_prompt_tokens:
        return "prompt_too_long"
    content_target_tokens = int(row.get("target_tokens") or len(row.get("input_ids") or []))
    if content_target_tokens > args.max_target_tokens:
        return "target_too_long"
    if int(row.get("total_tokens") or 0) + int(args.append_eos) > args.max_length:
        return "total_too_long"
    return None


def make_output_row(row, tokenizer, append_eos):
    cond_ids = [int(x) for x in row["condition_input_ids"]]
    target_ids = [int(x) for x in row["input_ids"]]
    if append_eos and tokenizer.eos_token_id is not None:
        eos = int(tokenizer.eos_token_id)
        if not target_ids or target_ids[-1] != eos:
            target_ids.append(eos)
    return {
        "index": int(row.get("index", -1)),
        "id": str(row.get("id", "")),
        "source": str(row.get("source", "")),
        "input": str(row["input"]),
        "target": str(row["target"]).strip(),
        "condition_input_ids": cond_ids,
        "input_ids": target_ids,
        "prompt_tokens": len(cond_ids),
        "target_tokens": len(target_ids),
        "target_tokens_no_eos": int(row.get("target_tokens") or max(0, len(target_ids) - int(append_eos))),
        "total_tokens": len(cond_ids) + len(target_ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probe_jsonl", required=True)
    parser.add_argument("--train_jsonl", default=None)
    parser.add_argument("--max_examples", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_tokens", type=int, default=512)
    parser.add_argument("--max_target_tokens", type=int, default=16)
    parser.add_argument("--min_target_chars", type=int, default=1)
    parser.add_argument("--max_target_newlines", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--append_eos", action="store_true")
    parser.add_argument("--single_turn_only", action="store_true")
    parser.add_argument("--drop_code", action="store_true")
    parser.add_argument("--reject_refusals", action="store_true")
    parser.add_argument("--english_like", action="store_true")
    parser.add_argument("--min_ascii_ratio", type=float, default=0.9)
    parser.add_argument("--include_source_substrings", default="")
    parser.add_argument("--exclude_source_substrings", default="")
    parser.add_argument(
        "--target_token_buckets",
        default="",
        help="Optional source target-token buckets, e.g. '1-16:3000,17-64:5000,65-128:2000'.",
    )
    args = parser.parse_args()

    ds = load_from_disk(args.input_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    reject_counts = Counter()
    candidates = []
    for idx, row in enumerate(ds):
        reason = first_reject_reason(row, args)
        if reason is not None:
            reject_counts[reason] += 1
            continue
        candidates.append(idx)

    prompt_to_target_idx = {}
    conflict_prompts = set()
    duplicate_same_target = 0
    for idx in candidates:
        row = ds[int(idx)]
        prompt = str(row["input"])
        target = str(row["target"]).strip()
        if prompt not in prompt_to_target_idx:
            prompt_to_target_idx[prompt] = (target, idx)
        elif prompt_to_target_idx[prompt][0] == target:
            duplicate_same_target += 1
        else:
            conflict_prompts.add(prompt)

    deduped_candidates = [
        int(idx)
        for prompt, (_, idx) in prompt_to_target_idx.items()
        if prompt not in conflict_prompts
    ]
    reject_counts["duplicate_prompt_same_target"] = duplicate_same_target
    reject_counts["conflicting_prompt_targets"] = len(conflict_prompts)

    rng = random.Random(args.seed)
    bucket_spec = parse_bucket_spec(args.target_token_buckets)
    if bucket_spec:
        selected = []
        remaining = set(deduped_candidates)
        bucket_candidate_counts = {}
        for lo, hi, count in bucket_spec:
            bucket = [
                idx for idx in deduped_candidates
                if idx in remaining and lo <= int(ds[int(idx)].get("target_tokens") or 0) <= hi
            ]
            bucket_candidate_counts[f"{lo}-{hi}"] = len(bucket)
            rng.shuffle(bucket)
            take = bucket[:count]
            selected.extend(take)
            remaining.difference_update(take)
        if args.max_examples > 0 and len(selected) < args.max_examples:
            filler = list(remaining)
            rng.shuffle(filler)
            selected.extend(filler[: args.max_examples - len(selected)])
        if args.max_examples > 0:
            selected = selected[: args.max_examples]
    else:
        bucket_candidate_counts = {}
        rng.shuffle(deduped_candidates)
        selected = deduped_candidates[: args.max_examples] if args.max_examples > 0 else deduped_candidates
    selected_rows = [make_output_row({**ds[int(idx)], "index": int(idx)}, tokenizer, args.append_eos) for idx in selected]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(selected_rows).save_to_disk(str(out_dir))

    probe_path = Path(args.probe_jsonl)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    with probe_path.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps({"input": row["input"], "output": row["target"]}, ensure_ascii=False) + "\n")

    if args.train_jsonl:
        train_path = Path(args.train_jsonl)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        with train_path.open("w", encoding="utf-8") as f:
            for row in selected_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_counts = Counter(row["source"] for row in selected_rows)
    target_len_counts = Counter(row["target_tokens"] for row in selected_rows)
    prompt_lens = [row["prompt_tokens"] for row in selected_rows] or [0]
    target_lens = [row["target_tokens"] for row in selected_rows] or [0]
    total_lens = [row["total_tokens"] for row in selected_rows] or [0]
    report = {
        "input_dataset": args.input_dataset,
        "output_dir": str(out_dir),
        "probe_jsonl": str(probe_path),
        "max_examples": args.max_examples,
        "selected_examples": len(selected_rows),
        "candidate_examples": len(deduped_candidates),
        "pre_dedupe_candidate_examples": len(candidates),
        "source_examples": len(ds),
        "reject_counts": dict(reject_counts.most_common()),
        "filters": {
            "single_turn_only": bool(args.single_turn_only),
            "append_eos": bool(args.append_eos),
            "drop_code": bool(args.drop_code),
            "reject_refusals": bool(args.reject_refusals),
            "english_like": bool(args.english_like),
            "min_ascii_ratio": args.min_ascii_ratio,
            "include_source_substrings": args.include_source_substrings,
            "exclude_source_substrings": args.exclude_source_substrings,
            "max_length": args.max_length,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_target_tokens_no_eos": args.max_target_tokens,
            "max_target_newlines": args.max_target_newlines,
            "target_token_buckets": args.target_token_buckets,
        },
        "bucket_candidate_counts": bucket_candidate_counts,
        "prompt_tokens": {
            "min": min(prompt_lens),
            "max": max(prompt_lens),
            "mean": sum(prompt_lens) / len(prompt_lens),
        },
        "target_tokens_with_eos": {
            "min": min(target_lens),
            "max": max(target_lens),
            "mean": sum(target_lens) / len(target_lens),
            "histogram": dict(sorted(target_len_counts.items())),
        },
        "total_tokens": {
            "min": min(total_lens),
            "max": max(total_lens),
            "mean": sum(total_lens) / len(total_lens),
        },
        "top_sources": source_counts.most_common(20),
        "samples": [
            {
                "id": row["id"],
                "source": row["source"],
                "input": row["input"],
                "target": row["target"],
                "prompt_tokens": row["prompt_tokens"],
                "target_tokens": row["target_tokens"],
            }
            for row in selected_rows[:20]
        ],
    }
    (out_dir / "tulu3_short_answer_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
