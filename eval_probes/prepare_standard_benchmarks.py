#!/usr/bin/env python3
"""Prepare generation-style benchmark JSONL files for CFM instruction eval.

The CFM model does not expose AR log-likelihoods, so these files use prompts
that ask for directly parseable generations where possible.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable


CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def chat_prompt(text: str) -> str:
    return "User: " + text.strip() + "\nAssistant:"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rows: list[dict[str, Any]], max_examples: int, seed: int) -> list[dict[str, Any]]:
    if max_examples <= 0 or len(rows) <= max_examples:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_examples))
    return [rows[i] for i in indices]


def normalize_answer_text(text: Any) -> str:
    if isinstance(text, (list, tuple)):
        return " ".join(str(x) for x in text)
    return str(text)


def make_choice_prompt(question: str, choices: list[str], *, instruction: str = "Answer with only the letter.") -> str:
    lines = [question.strip(), "", "Options:"]
    for i, choice in enumerate(choices):
        lines.append(f"{CHOICE_LETTERS[i]}. {str(choice).strip()}")
    lines.extend(["", instruction])
    return chat_prompt("\n".join(lines))


def choice_rows(
    *,
    benchmark: str,
    task: str,
    dataset_rows,
    question_fn: Callable[[dict[str, Any]], str],
    choices_fn: Callable[[dict[str, Any]], list[str]],
    answer_index_fn: Callable[[dict[str, Any]], int],
) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(dataset_rows):
        choices = [normalize_answer_text(x) for x in choices_fn(item)]
        if not choices:
            continue
        answer_idx = answer_index_fn(item)
        if answer_idx < 0 or answer_idx >= len(choices):
            continue
        answer = CHOICE_LETTERS[answer_idx]
        rows.append({
            "id": f"{benchmark}:{task}:{idx}",
            "benchmark": benchmark,
            "task": task,
            "input": make_choice_prompt(question_fn(item), choices),
            "output": answer,
            "scoring": "multiple_choice",
            "answer": answer,
            "choices": choices,
            "source_index": idx,
        })
    return rows


def prepare_ifeval(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("google/IFEval", split="train")
    rows = []
    for idx, item in enumerate(ds):
        prompt = item["prompt"]
        key = int(item.get("key", idx))
        rows.append({
            "id": f"ifeval:{key}",
            "benchmark": "ifeval",
            "task": "ifeval",
            "input": chat_prompt(prompt),
            "output": "",
            "scoring": "ifeval",
            "answer": None,
            "source_index": idx,
            "ifeval": {
                "key": key,
                "prompt": prompt,
                "instruction_id_list": item["instruction_id_list"],
                "kwargs": item["kwargs"],
            },
        })
    return rows


def extract_gsm8k_number(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if match:
        return match.group(1).replace(",", "")
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer)
    return matches[-1].replace(",", "") if matches else answer.strip()


def prepare_gsm8k(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for idx, item in enumerate(ds):
        answer = extract_gsm8k_number(item["answer"])
        prompt = (
            "Solve the following grade-school math problem. "
            "Give only the final numeric answer.\n\n" + item["question"]
        )
        rows.append({
            "id": f"gsm8k:test:{idx}",
            "benchmark": "gsm8k",
            "task": "main",
            "input": chat_prompt(prompt),
            "output": answer,
            "scoring": "numeric",
            "answer": answer,
            "source_index": idx,
        })
    return rows


def prepare_boolq(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("google/boolq", split="validation")
    rows = []
    for idx, item in enumerate(ds):
        answer = "Yes" if bool(item["answer"]) else "No"
        prompt = (
            "Read the passage and answer the question with only Yes or No.\n\n"
            f"Passage: {item['passage']}\n\nQuestion: {item['question']}"
        )
        rows.append({
            "id": f"boolq:validation:{idx}",
            "benchmark": "boolq",
            "task": "validation",
            "input": chat_prompt(prompt),
            "output": answer,
            "scoring": "yes_no",
            "answer": answer,
            "source_index": idx,
        })
    return rows


def prepare_arc_challenge(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    def choices(item):
        return item["choices"]["text"]

    def answer_idx(item):
        labels = [str(x) for x in item["choices"]["label"]]
        key = str(item["answerKey"])
        if key in labels:
            return labels.index(key)
        if key.isdigit():
            return int(key) - 1
        return CHOICE_LETTERS.index(key.upper())

    return choice_rows(
        benchmark="arc_challenge",
        task="test",
        dataset_rows=ds,
        question_fn=lambda x: x["question"],
        choices_fn=choices,
        answer_index_fn=answer_idx,
    )


def prepare_openbookqa(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("allenai/openbookqa", "main", split="test")

    def choices(item):
        return item["choices"]["text"]

    def answer_idx(item):
        labels = [str(x) for x in item["choices"]["label"]]
        key = str(item["answerKey"])
        if key in labels:
            return labels.index(key)
        return CHOICE_LETTERS.index(key.upper())

    return choice_rows(
        benchmark="openbookqa",
        task="test",
        dataset_rows=ds,
        question_fn=lambda x: x["question_stem"],
        choices_fn=choices,
        answer_index_fn=answer_idx,
    )


def prepare_hellaswag(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    return choice_rows(
        benchmark="hellaswag",
        task="validation",
        dataset_rows=ds,
        question_fn=lambda x: (
            "Choose the most plausible continuation.\n\n"
            f"Context: {x['ctx']}"
        ),
        choices_fn=lambda x: x["endings"],
        answer_index_fn=lambda x: int(x["label"]),
    )


def prepare_piqa(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("ybisk/piqa", split="validation")
    return choice_rows(
        benchmark="piqa",
        task="validation",
        dataset_rows=ds,
        question_fn=lambda x: "Choose the better solution.\n\nGoal: " + x["goal"],
        choices_fn=lambda x: [x["sol1"], x["sol2"]],
        answer_index_fn=lambda x: int(x["label"]),
    )


def prepare_winogrande(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    return choice_rows(
        benchmark="winogrande",
        task="validation",
        dataset_rows=ds,
        question_fn=lambda x: "Fill in the blank marked with _. \n\n" + x["sentence"],
        choices_fn=lambda x: [x["option1"], x["option2"]],
        answer_index_fn=lambda x: int(x["answer"]) - 1,
    )


def prepare_truthfulqa_mc(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rows = []
    for idx, item in enumerate(ds):
        targets = item.get("mc1_targets") or item.get("mc2_targets")
        if not targets:
            continue
        choices = [normalize_answer_text(x) for x in targets["choices"]]
        labels = [int(x) for x in targets["labels"]]
        if 1 not in labels:
            continue
        answer_idx = labels.index(1)
        rows.extend(choice_rows(
            benchmark="truthfulqa_mc1",
            task="validation",
            dataset_rows=[item],
            question_fn=lambda x: x["question"],
            choices_fn=lambda x, c=choices: c,
            answer_index_fn=lambda x, a=answer_idx: a,
        ))
        rows[-1]["id"] = f"truthfulqa_mc1:validation:{idx}"
        rows[-1]["source_index"] = idx
    return rows


def prepare_mmlu_pro(load_dataset) -> list[dict[str, Any]]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    def choices(item):
        value = item.get("options", item.get("choices"))
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [value]
        return list(value)

    def answer_idx(item):
        if "answer_index" in item:
            return int(item["answer_index"])
        answer = str(item.get("answer", item.get("answer_letter", ""))).strip()
        opts = choices(item)
        if answer in CHOICE_LETTERS:
            return CHOICE_LETTERS.index(answer)
        if answer in opts:
            return opts.index(answer)
        raise ValueError(f"Cannot parse MMLU-Pro answer: {answer!r}")

    return choice_rows(
        benchmark="mmlu_pro",
        task="test",
        dataset_rows=ds,
        question_fn=lambda x: x["question"],
        choices_fn=choices,
        answer_index_fn=answer_idx,
    )


PREPARERS: dict[str, Callable[[Any], list[dict[str, Any]]]] = {
    "ifeval": prepare_ifeval,
    "gsm8k": prepare_gsm8k,
    "boolq": prepare_boolq,
    "arc_challenge": prepare_arc_challenge,
    "openbookqa": prepare_openbookqa,
    "hellaswag": prepare_hellaswag,
    "piqa": prepare_piqa,
    "winogrande": prepare_winogrande,
    "truthfulqa_mc": prepare_truthfulqa_mc,
    "mmlu_pro": prepare_mmlu_pro,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="eval_probes/standard_benchmarks/data")
    parser.add_argument(
        "--benchmarks",
        default="ifeval,gsm8k,boolq,arc_challenge,openbookqa,hellaswag,piqa,winogrande,truthfulqa_mc,mmlu_pro",
    )
    parser.add_argument("--max_examples", type=int, default=0, help="Optional per-benchmark cap for smoke files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    output_dir = Path(args.output_dir)
    requested = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    manifest = {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "max_examples": args.max_examples,
        "benchmarks": {},
        "errors": {},
    }

    def loader(*loader_args, **kwargs):
        if args.cache_dir is not None:
            kwargs.setdefault("cache_dir", args.cache_dir)
        return load_dataset(*loader_args, **kwargs)

    for name in requested:
        if name not in PREPARERS:
            manifest["errors"][name] = "unknown benchmark"
            if args.fail_fast:
                raise KeyError(name)
            continue
        try:
            rows = PREPARERS[name](loader)
            rows = sample_rows(rows, args.max_examples, args.seed)
            path = output_dir / f"{name}.jsonl"
            write_jsonl(path, rows)
            manifest["benchmarks"][name] = {
                "path": str(path),
                "num_examples": len(rows),
                "scoring": sorted({row["scoring"] for row in rows}),
            }
            print(f"wrote {len(rows)} rows: {path}")
        except Exception as exc:
            message = repr(exc)
            if len(message) > 1000:
                message = message[:1000] + "...[truncated]"
            manifest["errors"][name] = message
            print(f"ERROR {name}: {exc}")
            if args.fail_fast:
                raise

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
