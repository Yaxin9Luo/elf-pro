#!/usr/bin/env python3
"""Summarize CFM standard benchmark result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_scores(result_dir: Path) -> dict[str, Any]:
    scores = {}
    for path in sorted(result_dir.glob("*.summary.json")):
        if path.name == "summary.json":
            continue
        summary = load_json(path)
        benchmark = path.name.removesuffix(".summary.json")
        overall = summary.get("scores_by_benchmark", {}).get("overall")
        if overall is None:
            continue
        scores[benchmark] = {
            "metric": "accuracy",
            "n": overall.get("n"),
            "correct": overall.get("correct"),
            "score": overall.get("accuracy"),
            "summary": str(path),
        }

    ifeval_summary = result_dir / "ifeval_official" / "summary.json"
    if ifeval_summary.is_file():
        official = load_json(ifeval_summary)
        strict = official.get("strict", {})
        loose = official.get("loose", {})
        scores["ifeval_strict_prompt"] = {
            "metric": "prompt_accuracy",
            "n": strict.get("prompt_total"),
            "correct": strict.get("prompt_correct"),
            "score": strict.get("prompt_accuracy"),
            "summary": str(ifeval_summary),
        }
        scores["ifeval_strict_instruction"] = {
            "metric": "instruction_accuracy",
            "n": strict.get("instruction_total"),
            "correct": strict.get("instruction_correct"),
            "score": strict.get("instruction_accuracy"),
            "summary": str(ifeval_summary),
        }
        scores["ifeval_loose_prompt"] = {
            "metric": "prompt_accuracy",
            "n": loose.get("prompt_total"),
            "correct": loose.get("prompt_correct"),
            "score": loose.get("prompt_accuracy"),
            "summary": str(ifeval_summary),
        }
        scores["ifeval_loose_instruction"] = {
            "metric": "instruction_accuracy",
            "n": loose.get("instruction_total"),
            "correct": loose.get("instruction_correct"),
            "score": loose.get("instruction_accuracy"),
            "summary": str(ifeval_summary),
        }
    return scores


def write_markdown(scores: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# CFM Standard Benchmark Summary",
        "",
        "| benchmark | metric | n | correct | score |",
        "|---|---|---:|---:|---:|",
    ]
    for name, score in sorted(scores.items()):
        value = score.get("score")
        score_text = "-" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {name} | {score.get('metric', '-')} | "
            f"{score.get('n', '-')} | {score.get('correct', '-')} | {score_text} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    scores = collect_scores(result_dir)
    output_json = Path(args.output_json) if args.output_json else result_dir / "aggregate_summary.json"
    output_md = Path(args.output_md) if args.output_md else result_dir / "aggregate_summary.md"
    output_json.write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(scores, output_md)
    print(json.dumps(scores, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
