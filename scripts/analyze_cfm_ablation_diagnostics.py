#!/usr/bin/env python3
"""Summarize CFM SFT ablation probes by data and failure slices."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path("eval_probes")
DEFAULT_CONFIG_PATH = ROOT / "sft_eval_harness_config.json"


DEFAULT_EXPERIMENTS = {
    "synthetic_128": {
        "label": "Synthetic 128",
        "cfm": ROOT / "short_qa_overfit_128/cfm_sanity_short_qa_overfit_params.json",
        "trajectory": ROOT / "short_qa_overfit_128/sampling_high_noise_all128_params.json",
        "metadata": None,
    },
    "synthetic_10k": {
        "label": "Synthetic 10K",
        "cfm": ROOT / "short_qa_10k/cfm_sanity_random256_seed12345_params.json",
        "trajectory": ROOT / "short_qa_10k/sampling_trajectory_random16_seed12345_params.json",
        "metadata": None,
    },
    "tulu_short_10k": {
        "label": "Tulu3 Short QA 10K",
        "cfm": ROOT / "tulu3_short_answer_clean_en_10k/cfm_sanity_random256_seed12345_params.json",
        "trajectory": ROOT / "tulu3_short_answer_clean_en_10k/sampling_trajectory_random16_seed12345_params.json",
        "metadata": ROOT / "dataset_metadata/tulu3_short_answer_clean_en_10k_train_metadata.jsonl",
    },
    "tulu_mixed_10k": {
        "label": "Tulu3 Mixed Length 10K",
        "cfm": ROOT / "tulu3_mixed_length_clean_en_10k/cfm_sanity_random256_seed12345_params.json",
        "trajectory": ROOT / "tulu3_mixed_length_clean_en_10k/sampling_trajectory_random16_seed12345_params.json",
        "metadata": ROOT / "dataset_metadata/tulu3_mixed_length_clean_en_10k_train_metadata.jsonl",
    },
}


DEFAULT_GATES = {
    "A_discrete_short": {
        "label": "A: discrete short answer",
        "rules": [
            {
                "answer_type": ["yes_no", "number", "single_word_or_symbol", "structured_json_or_list"],
                "target_len_bucket": ["<=2", "3-8", "9-16"],
            }
        ],
    },
    "B_natural_short": {
        "label": "B: natural short answer",
        "rules": [
            {
                "answer_type": ["short_phrase", "sentence", "refusal"],
                "target_len_bucket": ["3-8", "9-16", "17-64"],
            }
        ],
    },
    "C_long_answer": {
        "label": "C: long answer",
        "rules": [{"answer_type": ["multi_line", "long_form"]}, {"target_len_bucket": ["17-64", ">64"]}],
    },
}


DEFAULT_GATE_THRESHOLDS = {
    "clean_decode_min": 0.98,
    "t01_correct_min": 0.8,
    "t01_condition_gap_min": 0.5,
    "trajectory_t0_uniform_min": 0.8,
    "per_gate": {
        "C_long_answer": {
            "_comment": (
                "Long answers should not be judged by token-exact accuracy. "
                "token_acc / trajectory thresholds are intentionally relaxed; "
                "promote to semantic / structure metrics in a future revision."
            ),
            "clean_decode_min": 0.95,
            "t01_correct_min": 0.4,
            "t01_condition_gap_min": 0.3,
            "trajectory_t0_uniform_min": 0.4,
            "skip_metrics": ["t01_correct_min", "trajectory_t0_uniform_min"],
        }
    },
}


@dataclass
class MeanStat:
    n: int = 0
    total: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def add(self, value: float | None) -> None:
        if value is None or math.isnan(value):
            return
        self.n += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    @property
    def mean(self) -> float | None:
        if self.n == 0:
            return None
        return self.total / self.n

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "mean": self.mean, "min": self.min_value, "max": self.max_value}


@dataclass
class BucketStat:
    n: int = 0
    clean: MeanStat = field(default_factory=MeanStat)
    denoise: dict[str, MeanStat] = field(default_factory=lambda: defaultdict(MeanStat))
    denoise_mse: dict[str, MeanStat] = field(default_factory=lambda: defaultdict(MeanStat))
    trajectory: dict[str, MeanStat] = field(default_factory=lambda: defaultdict(MeanStat))
    trajectory_exact: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "clean_decode": self.clean.as_dict(),
            "denoise_token_acc": {k: v.as_dict() for k, v in sorted(self.denoise.items())},
            "denoise_x_mse": {k: v.as_dict() for k, v in sorted(self.denoise_mse.items())},
            "trajectory_token_acc": {k: v.as_dict() for k, v in sorted(self.trajectory.items())},
            "trajectory_exact": dict(sorted(self.trajectory_exact.items())),
        }


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate
    config_relative = base_dir / path
    if config_relative.exists():
        return config_relative
    return candidate


def load_harness_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "experiments": DEFAULT_EXPERIMENTS,
            "gates": DEFAULT_GATES,
            "gate_thresholds": DEFAULT_GATE_THRESHOLDS,
        }
    config = load_json(path)
    base_dir = path.parent
    experiments: dict[str, dict[str, Any]] = {}
    for name, exp in config.get("experiments", {}).items():
        experiments[name] = {
            "label": exp["label"],
            "cfm": resolve_path(exp.get("cfm"), base_dir),
            "trajectory": resolve_path(exp.get("trajectory"), base_dir),
            "metadata": resolve_path(exp.get("metadata"), base_dir),
        }
    return {
        "experiments": experiments,
        "gates": config.get("gates", DEFAULT_GATES),
        "gate_thresholds": config.get("gate_thresholds", DEFAULT_GATE_THRESHOLDS),
    }


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def text_key(input_text: str, target: str) -> tuple[str, str]:
    return (input_text.strip(), target.strip())


def load_metadata_lookup(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(path):
        lookup.setdefault(text_key(row.get("input", ""), row.get("target", "")), row)
    return lookup


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text))


def source_group(source: str | None) -> str:
    source = source or "unknown"
    mapping = [
        ("flan_v2", "flan_v2"),
        ("table_gpt", "table_gpt"),
        ("open_math", "open_math"),
        ("personahub", "personahub"),
        ("no_robots", "no_robots"),
        ("sciriff", "sciriff"),
        ("wildchat", "wildchat"),
        ("synthetic_finalresp", "synthetic_finalresp"),
        ("oasst", "oasst"),
        ("coconot", "coconot"),
        ("codealpaca", "codealpaca"),
    ]
    for needle, group in mapping:
        if needle in source:
            return group
    if source.startswith("synthetic"):
        return "synthetic"
    if source == "unknown":
        return "unknown"
    return "other_tulu"


def len_bucket(value: int | None, breaks: list[int], labels: list[str]) -> str:
    if value is None:
        return "unknown"
    for upper, label in zip(breaks, labels):
        if value <= upper:
            return label
    return labels[-1]


def answer_type(target: str) -> str:
    stripped = target.strip()
    lower = stripped.lower()
    words = word_count(stripped)
    if lower in {"yes", "no"}:
        return "yes_no"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?%?", stripped):
        return "number"
    if stripped.startswith("{") or stripped.startswith("["):
        return "structured_json_or_list"
    if "can't assist" in lower or "cannot assist" in lower or "i'm sorry" in lower:
        return "refusal"
    if "\n" in stripped:
        return "multi_line"
    if words <= 2:
        return "single_word_or_symbol"
    if words <= 8:
        return "short_phrase"
    if words <= 32:
        return "sentence"
    return "long_form"


def prompt_type(prompt: str) -> str:
    lower = prompt.lower()
    q_count = len(re.findall(r"\bq\s*:", lower))
    a_count = len(re.findall(r"\ba\s*:", lower))
    if "detailed instructions" in lower or "given the task definition" in lower or "instructions:" in lower:
        return "flan_instruction"
    if "|" in prompt and ("|---|" in prompt or prompt.count("|") >= 8):
        return "table_or_structured"
    if "options:" in lower or "choose one" in lower or "options" in lower:
        return "multiple_choice"
    if "json" in lower or "return the output as" in lower or "extract all unique entities" in lower:
        return "extraction_or_json"
    if "step-by-step" in lower or "reasoning" in lower:
        return "reasoning"
    if q_count >= 2 and a_count >= 1:
        return "few_shot_qa"
    if re.search(r"\b(add|minus|double|half|prime|number|calculate|sum|multiply)\b", lower):
        return "math_or_numeric"
    if re.search(r"\bpoison|weapon|harm|kill|malware|explosive\b", lower):
        return "safety"
    return "open_instruction"


def example_features(example: dict[str, Any], meta: dict[str, Any] | None, experiment: str) -> dict[str, str]:
    prompt = example.get("input", "")
    target = example.get("expected", "")
    prompt_tokens = meta.get("prompt_tokens") if meta else None
    target_tokens = meta.get("target_tokens") if meta else None
    total_tokens = meta.get("total_tokens") if meta else None
    if prompt_tokens is None:
        prompt_tokens = word_count(prompt)
    if target_tokens is None:
        target_tokens = word_count(target)
    if total_tokens is None:
        total_tokens = prompt_tokens + target_tokens
    source = meta.get("source") if meta else ("synthetic" if experiment.startswith("synthetic") else "unknown")
    return {
        "source_group": source_group(source),
        "source": source or "unknown",
        "prompt_type": prompt_type(prompt),
        "answer_type": answer_type(target),
        "prompt_len_bucket": len_bucket(prompt_tokens, [32, 128, 256, 512], ["<=32", "33-128", "129-256", "257-512", ">512"]),
        "target_len_bucket": len_bucket(target_tokens, [2, 8, 16, 64], ["<=2", "3-8", "9-16", "17-64", ">64"]),
        "total_len_bucket": len_bucket(total_tokens, [64, 160, 320, 640], ["<=64", "65-160", "161-320", "321-640", ">640"]),
    }


def rule_matches(features: dict[str, str], rule: dict[str, list[str]]) -> bool:
    for key, allowed in rule.items():
        if features.get(key) not in set(allowed):
            return False
    return True


def assign_gate(features: dict[str, str], gates: dict[str, Any]) -> str:
    for gate_name, gate in gates.items():
        for rule in gate.get("rules", []):
            if rule_matches(features, rule):
                return gate_name
    return "ungated"


def make_empty_group_tree() -> dict[str, dict[str, BucketStat]]:
    return {
        "curriculum_gate": defaultdict(BucketStat),
        "source_group": defaultdict(BucketStat),
        "prompt_type": defaultdict(BucketStat),
        "answer_type": defaultdict(BucketStat),
        "prompt_len_bucket": defaultdict(BucketStat),
        "target_len_bucket": defaultdict(BucketStat),
        "total_len_bucket": defaultdict(BucketStat),
    }


def _safe_float(value: Any) -> float | None:
    """Coerce numeric JSON fields to float, returning None for missing/non-numeric.

    The aggregate code previously coerced missing fields to 0.0, which then
    entered the running mean as a real zero observation and silently pulled
    metrics down. Returning None lets MeanStat.add skip the sample instead.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_experiment(name: str, cfg: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    cfm = load_json(cfg["cfm"])
    trajectory = load_json(cfg["trajectory"]) if cfg.get("trajectory") and cfg["trajectory"].exists() else None
    metadata = load_metadata_lookup(cfg.get("metadata"))
    groups = make_empty_group_tree()
    overview = BucketStat()
    missing_metadata = 0
    cfm_examples_total = 0
    missing_field_counts: dict[str, int] = defaultdict(int)
    worst_denoise: list[dict[str, Any]] = []
    worst_trajectory: list[dict[str, Any]] = []

    feature_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for ex in cfm.get("examples", []):
        cfm_examples_total += 1
        key = text_key(ex.get("input", ""), ex.get("expected", ""))
        meta = metadata.get(key)
        if metadata and not meta:
            missing_metadata += 1
        features = example_features(ex, meta, name)
        features["curriculum_gate"] = assign_gate(features, gates)
        feature_by_key[key] = features
        overview.n += 1
        clean_token_acc = _safe_float(ex.get("clean_decode", {}).get("token_acc"))
        if clean_token_acc is None:
            missing_field_counts["clean_decode.token_acc"] += 1
        overview.clean.add(clean_token_acc)
        for tree_name, group_key in features.items():
            if tree_name not in groups:
                continue
            bucket = groups[tree_name][group_key]
            bucket.n += 1
            bucket.clean.add(clean_token_acc)
        for row in ex.get("denoise", []):
            t = _safe_float(row.get("t"))
            if t is None:
                missing_field_counts["denoise.t"] += 1
                continue
            variant = row.get("variant", "unknown")
            metric_key = f"t={t:.1f}:{variant}"
            acc = _safe_float(row.get("token_acc_batch"))
            x_mse = _safe_float(row.get("x_mse"))
            if acc is None:
                missing_field_counts["denoise.token_acc_batch"] += 1
            if x_mse is None:
                missing_field_counts["denoise.x_mse"] += 1
            overview.denoise[metric_key].add(acc)
            overview.denoise_mse[metric_key].add(x_mse)
            for tree_name, group_key in features.items():
                if tree_name not in groups:
                    continue
                groups[tree_name][group_key].denoise[metric_key].add(acc)
                groups[tree_name][group_key].denoise_mse[metric_key].add(x_mse)
            if variant == "correct" and any(abs(t - target) < 1e-6 for target in (0.1, 0.3)):
                worst_denoise.append({
                    "metric": metric_key,
                    "token_acc": acc if acc is not None else float("nan"),
                    "x_mse": x_mse if x_mse is not None else float("nan"),
                    "input": ex.get("input", "")[:240],
                    "expected": ex.get("expected", ""),
                    "clean_generated": ex.get("clean_decode", {}).get("generated", ""),
                    **features,
                })

    if trajectory:
        for ex in trajectory.get("examples", []):
            key = text_key(ex.get("input", ""), ex.get("expected", ""))
            meta = metadata.get(key)
            features = feature_by_key.get(key) or example_features(ex, meta, name)
            features.setdefault("curriculum_gate", assign_gate(features, gates))
            for run in ex.get("runs", []):
                t_start = _safe_float(run.get("t_start"))
                if t_start is None:
                    missing_field_counts["trajectory.t_start"] += 1
                    continue
                schedule = run.get("schedule", "unknown")
                metric_key = f"t_start={t_start:.2f}:{schedule}"
                acc = _safe_float(run.get("final_token_acc"))
                if acc is None:
                    missing_field_counts["trajectory.final_token_acc"] += 1
                overview.trajectory[metric_key].add(acc)
                if acc is not None and acc >= 0.999:
                    overview.trajectory_exact[metric_key] += 1
                for tree_name, group_key in features.items():
                    if tree_name not in groups:
                        continue
                    groups[tree_name][group_key].trajectory[metric_key].add(acc)
                    if acc is not None and acc >= 0.999:
                        groups[tree_name][group_key].trajectory_exact[metric_key] += 1
                if abs(t_start) < 1e-6:
                    worst_trajectory.append({
                        "metric": metric_key,
                        "token_acc": acc if acc is not None else float("nan"),
                        "generated": run.get("final_generated", ""),
                        "input": ex.get("input", "")[:240],
                        "expected": ex.get("expected", ""),
                        **features,
                    })

    metadata_match_rate: float | None = None
    if metadata and cfm_examples_total > 0:
        matched = cfm_examples_total - missing_metadata
        metadata_match_rate = matched / cfm_examples_total

    return {
        "label": cfg["label"],
        "paths": {k: str(v) for k, v in cfg.items() if isinstance(v, Path)},
        "overview": overview.as_dict(),
        "metadata_matches": {
            "metadata_rows": len(metadata),
            "cfm_examples": cfm_examples_total,
            "missing_cfm_examples": missing_metadata,
            "match_rate": metadata_match_rate,
            "expected_metadata": metadata is not None and len(metadata) > 0,
        },
        "missing_fields": dict(missing_field_counts),
        "groups": {
            tree_name: {k: stat.as_dict() for k, stat in sorted(tree.items())}
            for tree_name, tree in groups.items()
        },
        "worst_denoise": sorted(
            worst_denoise,
            key=lambda x: (
                x["token_acc"] if not math.isnan(x["token_acc"]) else float("inf"),
                x["x_mse"] if not math.isnan(x["x_mse"]) else float("inf"),
            ),
        )[:12],
        "worst_trajectory": sorted(
            worst_trajectory,
            key=lambda x: x["token_acc"] if not math.isnan(x["token_acc"]) else float("inf"),
        )[:12],
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def mean_at(stat: dict[str, Any], section: str, key: str) -> float | None:
    return stat.get(section, {}).get(key, {}).get("mean")


def exact_at(stat: dict[str, Any], key: str) -> str:
    n = stat.get("trajectory_token_acc", {}).get(key, {}).get("n")
    if not n:
        return "-"
    exact = stat.get("trajectory_exact", {}).get(key, 0)
    return f"{exact}/{n}"


def condition_gap(stat: dict[str, Any], t: str = "0.1") -> float | None:
    correct = mean_at(stat, "denoise_token_acc", f"t={t}:correct")
    zero = mean_at(stat, "denoise_token_acc", f"t={t}:zero")
    shuffled = mean_at(stat, "denoise_token_acc", f"t={t}:shuffled")
    if correct is None or zero is None or shuffled is None:
        return None
    return correct - max(zero, shuffled)


def resolve_gate_thresholds(
    thresholds: dict[str, Any], gate_name: str | None
) -> tuple[dict[str, float], set[str]]:
    """Merge global thresholds with optional per-gate overrides.

    Returns the active threshold dict plus the set of metric keys that should
    be skipped entirely for this gate (so they neither fail nor pass it).
    Per-gate overrides live under ``thresholds["per_gate"][gate_name]`` to
    let C-long-answer use looser bars without polluting A/B and to let us
    explicitly skip metrics that are not meaningful at long horizons.
    """
    base_keys = (
        "clean_decode_min",
        "t01_correct_min",
        "t01_condition_gap_min",
        "trajectory_t0_uniform_min",
    )
    active = {key: thresholds.get(key) for key in base_keys}
    skip: set[str] = set()
    per_gate = thresholds.get("per_gate", {}) if isinstance(thresholds, dict) else {}
    override = per_gate.get(gate_name) if isinstance(per_gate, dict) else None
    if isinstance(override, dict):
        for key in base_keys:
            if key in override:
                active[key] = override[key]
        for key in override.get("skip_metrics", []) or []:
            skip.add(key)
    return active, skip


def gate_status(
    stat: dict[str, Any], thresholds: dict[str, Any], gate_name: str | None = None
) -> str:
    """Decide pass/fail/incomplete for a single bucket.

    Each metric votes independently. A metric that is missing for this bucket
    counts as "missing" rather than silently passing — previously a None
    metric was treated as a non-failure, so a bucket with only a non-None
    trajectory could be reported as "pass" even with no clean / t0.1 / gap.
    Per-gate ``skip_metrics`` opt a metric out: it neither fails nor blocks
    the pass status (used for long answers where token-exact is not the
    right signal).
    """
    active, skip = resolve_gate_thresholds(thresholds, gate_name)
    metrics = {
        "clean_decode_min": stat.get("clean_decode", {}).get("mean"),
        "t01_correct_min": mean_at(stat, "denoise_token_acc", "t=0.1:correct"),
        "t01_condition_gap_min": condition_gap(stat, "0.1"),
        "trajectory_t0_uniform_min": mean_at(
            stat, "trajectory_token_acc", "t_start=0.00:uniform"
        ),
    }
    fail_any = False
    missing_any = False
    for key, value in metrics.items():
        if key in skip:
            continue
        threshold = active.get(key)
        if threshold is None:
            continue
        if value is None:
            missing_any = True
            continue
        if value < threshold:
            fail_any = True
    if fail_any:
        return "fail"
    if missing_any:
        return "incomplete"
    return "pass"


def group_rows(exp: dict[str, Any], group_name: str, metric: str, min_n: int) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for key, stat in exp["groups"].get(group_name, {}).items():
        if stat["n"] < min_n:
            continue
        mean = mean_at(stat, "denoise_token_acc", metric)
        if mean is None:
            continue
        rows.append((key, stat))
    rows.sort(key=lambda kv: (mean_at(kv[1], "denoise_token_acc", metric) or -1, -kv[1]["n"]))
    return rows


def render_group_table(exp: dict[str, Any], group_name: str, min_n: int) -> list[str]:
    lines = [
        f"#### {exp['label']} by `{group_name}`",
        "",
        "| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = group_rows(exp, group_name, "t=0.1:correct", min_n)
    if not rows:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")
        return lines
    for key, stat in rows:
        lines.append(
            "| "
            + " | ".join([
                key,
                str(stat["n"]),
                fmt(stat["clean_decode"]["mean"]),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.1:correct")),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.1:zero")),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.1:shuffled")),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.3:correct")),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.5:correct")),
                fmt(mean_at(stat, "trajectory_token_acc", "t_start=0.00:uniform")),
                exact_at(stat, "t_start=0.00:uniform"),
            ])
            + " |"
        )
    return lines


def render_gate_summary(results: dict[str, Any], thresholds: dict[str, Any], min_n: int) -> list[str]:
    per_gate = thresholds.get("per_gate", {}) if isinstance(thresholds, dict) else {}
    note_lines = []
    if per_gate:
        note_lines.append("Per-gate threshold overrides:")
        for name, override in per_gate.items():
            if not isinstance(override, dict):
                continue
            comment = override.get("_comment")
            skip = override.get("skip_metrics") or []
            note = f"- `{name}`"
            if skip:
                note += f" skips metrics {sorted(skip)}"
            if comment:
                note += f" — {comment}"
            note_lines.append(note)
    lines = [
        "## Curriculum Gate Summary",
        "",
        "Gate status uses configurable thresholds from `eval_probes/sft_eval_harness_config.json`. Current thresholds are intentionally aspirational and are meant to flag bottlenecks, not to claim model quality.",
        "Status `incomplete` means the bucket has at least one required metric missing (no longer silently treated as pass).",
        "",
    ]
    if note_lines:
        lines.extend(note_lines)
        lines.append("")
    lines.extend([
        "| experiment | gate | n | status | clean | t0.1 correct | t0.1 gap | t0.3 correct | t0.5 correct | t_start=0 uniform | exact |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for exp in results["experiments"].values():
        gates = exp["groups"].get("curriculum_gate", {})
        for gate_name, stat in sorted(gates.items()):
            if stat["n"] < min_n:
                continue
            lines.append(
                "| "
                + " | ".join([
                    exp["label"],
                    gate_name,
                    str(stat["n"]),
                    gate_status(stat, thresholds, gate_name),
                    fmt(stat["clean_decode"]["mean"]),
                    fmt(mean_at(stat, "denoise_token_acc", "t=0.1:correct")),
                    fmt(condition_gap(stat, "0.1")),
                    fmt(mean_at(stat, "denoise_token_acc", "t=0.3:correct")),
                    fmt(mean_at(stat, "denoise_token_acc", "t=0.5:correct")),
                    fmt(mean_at(stat, "trajectory_token_acc", "t_start=0.00:uniform")),
                    exact_at(stat, "t_start=0.00:uniform"),
                ])
                + " |"
            )
    return lines


def render_metadata_coverage(results: dict[str, Any]) -> list[str]:
    rows = []
    for exp in results["experiments"].values():
        info = exp.get("metadata_matches", {}) or {}
        if not info.get("expected_metadata"):
            continue
        n = info.get("cfm_examples", 0)
        missing = info.get("missing_cfm_examples", 0)
        match_rate = info.get("match_rate")
        missing_fields = exp.get("missing_fields", {}) or {}
        flagged_fields = ", ".join(
            f"{k}={v}" for k, v in sorted(missing_fields.items()) if v
        ) or "-"
        rows.append((exp["label"], n, missing, match_rate, flagged_fields))
    if not rows:
        return []
    lines = [
        "## Metadata & Field Coverage",
        "",
        "Metadata join uses `(input.strip(), target.strip())`. Rows that fail to match fall back to `source=unknown` and are not included in source-group / source slices, so a low match rate silently biases the breakdown. Missing-field counts cover JSON keys read by the harness; non-zero values mean the underlying probe artifact is incomplete and the corresponding metric was skipped (not coerced to 0).",
        "",
        "| experiment | cfm rows | missing meta | match rate | missing fields |",
        "|---|---:|---:|---:|---|",
    ]
    for label, n, missing, match_rate, flagged in rows:
        lines.append(
            "| "
            + " | ".join([
                label,
                str(n),
                str(missing),
                fmt(match_rate),
                flagged,
            ])
            + " |"
        )
    lines.append("")
    return lines


def render_markdown(results: dict[str, Any], min_n: int, thresholds: dict[str, Any]) -> str:
    lines = [
        "# CFM SFT Eval Harness Report",
        "",
        "This report is generated from fixed probe artifacts only. It is the evaluation harness for text-only Continuous Flow Matching SFT instruction experiments.",
        "Slice metrics are sample-level macro means over probe examples, so they can differ slightly from token-weighted summaries inside the raw JSON files.",
        "",
        "## Overall Metrics",
        "",
        "| experiment | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t_start=0 uniform | exact | t_start=0 logit-tail | exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, exp in results["experiments"].items():
        stat = exp["overview"]
        lines.append(
            "| "
            + " | ".join([
                exp["label"],
                str(stat["n"]),
                fmt(stat["clean_decode"]["mean"]),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.1:correct")),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.1:zero")),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.1:shuffled")),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.3:correct")),
                fmt(mean_at(stat, "denoise_token_acc", "t=0.5:correct")),
                fmt(mean_at(stat, "trajectory_token_acc", "t_start=0.00:uniform")),
                exact_at(stat, "t_start=0.00:uniform"),
                fmt(mean_at(stat, "trajectory_token_acc", "t_start=0.00:logit_tail")),
                exact_at(stat, "t_start=0.00:logit_tail"),
            ])
            + " |"
        )

    coverage_lines = render_metadata_coverage(results)
    if coverage_lines:
        lines += [""] + coverage_lines

    lines += [""] + render_gate_summary(results, thresholds=thresholds, min_n=min_n)
    lines += [
        "",
        "## Fine-Grained Slices",
        "",
        "Tables are sorted by low-noise-to-high-noise bottleneck metric `t=0.1:correct` ascending. Small groups below the `min_n` threshold are omitted.",
        "",
    ]
    detail_keys = [
        k for k in ("synthetic_10k", "tulu_short_10k", "tulu_mixed_10k")
        if k in results["experiments"]
    ]
    for exp_key in detail_keys:
        exp = results["experiments"][exp_key]
        for group in ("curriculum_gate", "source_group", "prompt_type", "answer_type", "prompt_len_bucket", "target_len_bucket"):
            lines.extend(render_group_table(exp, group, min_n=min_n))
            lines.append("")

    lines += [
        "## Worst t0.1 / t0.3 Controlled Denoise Examples",
        "",
    ]
    for exp_key in detail_keys:
        exp = results["experiments"][exp_key]
        lines.append(f"### {exp['label']}")
        lines.append("")
        lines.append("| metric | acc | source_group | prompt_type | answer_type | expected | clean | prompt prefix |")
        lines.append("|---|---:|---|---|---|---|---|---|")
        for row in exp["worst_denoise"][:8]:
            lines.append(
                "| "
                + " | ".join([
                    row["metric"],
                    fmt(row["token_acc"]),
                    row["source_group"],
                    row["prompt_type"],
                    row["answer_type"],
                    row["expected"].replace("|", "\\|").replace("\n", " ")[:80],
                    row["clean_generated"].replace("|", "\\|").replace("\n", " ")[:80],
                    row["input"].replace("|", "\\|").replace("\n", " ")[:120],
                ])
                + " |"
            )
        lines.append("")

    lines += [
        "## Worst Pure-Noise Trajectory Examples",
        "",
    ]
    for exp_key in detail_keys:
        exp = results["experiments"][exp_key]
        lines.append(f"### {exp['label']}")
        lines.append("")
        lines.append("| metric | acc | source_group | prompt_type | answer_type | expected | generated | prompt prefix |")
        lines.append("|---|---:|---|---|---|---|---|---|")
        for row in exp["worst_trajectory"][:8]:
            lines.append(
                "| "
                + " | ".join([
                    row["metric"],
                    fmt(row["token_acc"]),
                    row["source_group"],
                    row["prompt_type"],
                    row["answer_type"],
                    row["expected"].replace("|", "\\|").replace("\n", " ")[:80],
                    row["generated"].replace("|", "\\|").replace("\n", " ")[:80],
                    row["input"].replace("|", "\\|").replace("\n", " ")[:120],
                ])
                + " |"
            )
        lines.append("")

    lines += [
        "## Readout",
        "",
        "- Clean decode is near-saturated across all current ablations, so the immediate bottleneck is not latent-to-token decoding.",
        "- Synthetic 10K is already harder than Synthetic 128: pure-noise trajectory is no longer perfect, and the failures are concentrated in short discrete answers such as numbers and yes/no labels.",
        "- Tulu Short 10K has weak controlled denoise at `t=0.1` even though outputs are short. This points to data/task distribution rather than answer length alone.",
        "- Tulu Mixed 10K has better single-step denoise than Tulu Short but worse exact pure-noise trajectory for long answers, which separates local repair ability from long-horizon sampling stability.",
        "- The next decision should be based on which slices dominate the weak `t=0.1:correct` groups: source mix, prompt templates, answer type, or length.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagnostics")
    parser.add_argument("--min-n", type=int, default=5)
    args = parser.parse_args()

    config = load_harness_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": str(args.config),
        "gates": config["gates"],
        "gate_thresholds": config["gate_thresholds"],
        "experiments": {},
    }
    for name, cfg in config["experiments"].items():
        results["experiments"][name] = aggregate_experiment(name, cfg, gates=config["gates"])

    json_text = json.dumps(results, ensure_ascii=False, indent=2)
    md_text = render_markdown(results, min_n=args.min_n, thresholds=config["gate_thresholds"])
    output_paths = [
        args.output_dir / "sft_eval_harness_report.json",
        args.output_dir / "cfm_ablation_breakdown.json",
    ]
    for path in output_paths:
        path.write_text(json_text)
        print(f"wrote {path}")
    output_paths = [
        args.output_dir / "sft_eval_harness_report.md",
        args.output_dir / "cfm_ablation_breakdown.md",
    ]
    for path in output_paths:
        path.write_text(md_text)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
