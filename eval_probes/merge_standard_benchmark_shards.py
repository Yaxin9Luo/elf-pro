#!/usr/bin/env python3
"""Merge sharded CFM benchmark outputs and rebuild the summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def summarize(rows: list[dict[str, Any]], shard_paths: list[str]) -> dict[str, Any]:
    by_benchmark: dict[str, Counter] = defaultdict(Counter)
    by_task: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        score = row.get("score", {})
        if score.get("correct") is None:
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

    first = rows[0] if rows else {}
    return {
        "schema_version": 1,
        "shards": shard_paths,
        "checkpoint": first.get("checkpoint_meta"),
        "sampling": first.get("sampling"),
        "num_outputs": len(rows),
        "scores_by_benchmark": as_scores(by_benchmark),
        "scores_by_task": as_scores(by_task),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# CFM Standard Benchmark Report",
        "",
        f"Outputs: `{summary['num_outputs']}`",
        "",
        "## Scores",
        "",
        "| benchmark | n | correct | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for key, val in summary["scores_by_benchmark"].items():
        acc = "-" if val["accuracy"] is None else f"{val['accuracy']:.4f}"
        lines.append(f"| {key} | {val['n']} | {val['correct']} | {acc} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_jsonl", nargs="+", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    rows = []
    seen_ids = set()
    for item in args.shard_jsonl:
        for row in load_jsonl(Path(item)):
            row_id = row["id"]
            if row_id in seen_ids:
                raise ValueError(f"Duplicate benchmark id while merging shards: {row_id}")
            seen_ids.add(row_id)
            rows.append(row)

    rows.sort(key=lambda row: (row.get("benchmark", ""), row.get("source_index", 0), row.get("id", "")))
    output_jsonl = Path(args.output_jsonl)
    write_jsonl(output_jsonl, rows)
    summary = summarize(rows, args.shard_jsonl)
    summary_json = Path(args.summary_json)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, summary_json.with_suffix(".md"))
    print(f"merged {len(rows)} rows -> {output_jsonl}")
    print(f"wrote summary -> {summary_json}")


if __name__ == "__main__":
    main()
