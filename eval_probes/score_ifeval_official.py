#!/usr/bin/env python3
"""Score CFM IFEval generations with the official Google evaluator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


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


def compact_ifeval_kwargs(kwargs_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert HF IFEval's wide kwargs dicts to official sparse kwargs."""
    compacted = []
    for kwargs in kwargs_list:
        compacted.append({key: value for key, value in kwargs.items() if value is not None})
    return compacted


def summarize(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    prompt_total = len(rows)
    prompt_correct = sum(1 for row in rows if row["follow_all_instructions"])
    instruction_total = sum(len(row["follow_instruction_list"]) for row in rows)
    instruction_correct = sum(sum(row["follow_instruction_list"]) for row in rows)
    return {
        "path": str(path),
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "prompt_accuracy": prompt_correct / prompt_total if prompt_total else None,
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
        "instruction_accuracy": instruction_correct / instruction_total if instruction_total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_jsonl", required=True)
    parser.add_argument("--generations_jsonl", required=True)
    parser.add_argument("--google_research_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()

    benchmark_rows = load_jsonl(Path(args.benchmark_jsonl))
    generation_rows = load_jsonl(Path(args.generations_jsonl))
    response_by_id = {row["id"]: row.get("answer_text", row.get("generated", "")) for row in generation_rows}

    input_rows = []
    response_rows = []
    missing = []
    for row in benchmark_rows:
        if row.get("scoring") != "ifeval":
            continue
        meta = row["ifeval"]
        row_id = row["id"]
        if row_id not in response_by_id:
            missing.append(row_id)
            continue
        input_rows.append({
            "key": meta["key"],
            "instruction_id_list": meta["instruction_id_list"],
            "prompt": meta["prompt"],
            "kwargs": compact_ifeval_kwargs(meta["kwargs"]),
        })
        response_rows.append({
            "prompt": meta["prompt"],
            "response": response_by_id[row_id],
        })

    if missing and not args.allow_partial:
        raise ValueError(f"Missing {len(missing)} generated IFEval rows, first={missing[:5]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_data = output_dir / "input_data.jsonl"
    input_response_data = output_dir / "input_response_data.jsonl"
    write_jsonl(input_data, input_rows)
    write_jsonl(input_response_data, response_rows)

    env = os.environ.copy()
    gr_dir_path = Path(args.google_research_dir).resolve()
    gr_dir = str(gr_dir_path)
    env["PYTHONPATH"] = gr_dir + os.pathsep + env.get("PYTHONPATH", "")
    if (gr_dir_path / "evaluation_main.py").is_file():
        cmd = [args.python, str(gr_dir_path / "evaluation_main.py")]
    else:
        cmd = [args.python, "-m", "instruction_following_eval.evaluation_main"]
    cmd.extend([
        "--input_data",
        str(input_data),
        "--input_response_data",
        str(input_response_data),
        "--output_dir",
        str(output_dir),
    ])
    subprocess.run(cmd, check=True, env=env)

    summary = {
        "strict": summarize(output_dir / "eval_results_strict.jsonl"),
        "loose": summarize(output_dir / "eval_results_loose.jsonl"),
        "missing_generations": len(missing),
        "allow_partial": bool(args.allow_partial),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
