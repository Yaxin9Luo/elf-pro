#!/usr/bin/env python3
"""Evaluate autoregressive pretrained LMs on the fixed instruction probes.

This is a task-level baseline for the same prompt/target JSONL files used by
the CFM SFT probes. It does not evaluate ELF-specific denoising metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = Path("eval_probes/probe_inputs/tulu3_short_answer_clean_en_10k_probe_valid_random256_seed12345.jsonl")
DEFAULT_HARNESS_CONFIG = Path("eval_probes/sft_eval_harness_config.json")
DEFAULT_OUTPUT_DIR = Path("eval_probes/ar_baselines")


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
class ArStats:
    n: int = 0
    exact: MeanStat = field(default_factory=MeanStat)
    prefix: MeanStat = field(default_factory=MeanStat)
    contains: MeanStat = field(default_factory=MeanStat)
    token_f1: MeanStat = field(default_factory=MeanStat)
    target_nll: MeanStat = field(default_factory=MeanStat)

    def add(self, row: dict[str, Any]) -> None:
        self.n += 1
        self.exact.add(float(row["exact_match"]))
        self.prefix.add(float(row["prefix_match"]))
        self.contains.add(float(row["contains_target"]))
        self.token_f1.add(row["token_f1"])
        self.target_nll.add(row["target_nll"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "exact_match": self.exact.as_dict(),
            "prefix_match": self.prefix.as_dict(),
            "contains_target": self.contains.as_dict(),
            "token_f1": self.token_f1.as_dict(),
            "target_nll": self.target_nll.as_dict(),
            "target_ppl": safe_exp(self.target_nll.mean),
        }


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def safe_exp(value: float | None) -> float | None:
    if value is None:
        return None
    return math.exp(min(value, 80.0))


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def infer_metadata_path(prompts_path: Path) -> Path | None:
    name = prompts_path.name
    if name.startswith("tulu3_short_answer_clean_en_10k"):
        return Path("eval_probes/dataset_metadata/tulu3_short_answer_clean_en_10k_train_metadata.jsonl")
    if name.startswith("tulu3_mixed_length_clean_en_10k"):
        return Path("eval_probes/dataset_metadata/tulu3_mixed_length_clean_en_10k_train_metadata.jsonl")
    return None


def load_gates(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path).get("gates", {})


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


def example_features(example: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, str]:
    prompt = example.get("input", "")
    target = get_target(example)
    prompt_tokens = meta.get("prompt_tokens") if meta else None
    target_tokens = meta.get("target_tokens") if meta else None
    total_tokens = meta.get("total_tokens") if meta else None
    if prompt_tokens is None:
        prompt_tokens = word_count(prompt)
    if target_tokens is None:
        target_tokens = word_count(target)
    if total_tokens is None:
        total_tokens = prompt_tokens + target_tokens
    source = meta.get("source") if meta else "unknown"
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


def get_target(row: dict[str, Any]) -> str:
    if "target" in row:
        return row["target"]
    if "output" in row:
        return row["output"]
    if "expected" in row:
        return row["expected"]
    raise KeyError("Prompt row must contain one of: target, output, expected")


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_boundary(text: str, idx: int) -> bool:
    if idx >= len(text):
        return True
    return text[idx].isspace() or text[idx] in ".,;:!?)]}'\""


def answer_prefix_match(prediction: str, target: str) -> bool:
    if not target or not prediction.startswith(target):
        return False
    return _is_boundary(prediction, len(target))


def answer_contains(prediction: str, target: str) -> bool:
    if not target:
        return False
    start = prediction.find(target)
    while start >= 0:
        end = start + len(target)
        if (start == 0 or _is_boundary(prediction, start - 1)) and _is_boundary(prediction, end):
            return True
        start = prediction.find(target, start + 1)
    return False


def trim_generation(text: str) -> str:
    stops = ["\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"]
    end = len(text)
    for stop in stops:
        idx = text.find(stop)
        if idx >= 0:
            end = min(end, idx)
    return text[:end].strip()


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize_answer(prediction).lower().split()
    target_tokens = normalize_answer(target).lower().split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for token in target_tokens:
        counts[token] += 1
    overlap = 0
    for token in pred_tokens:
        if counts[token] > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def target_with_prefix(prompt: str, target: str, mode: str) -> str:
    if mode == "none":
        return target
    if mode == "space":
        return " " + target
    if mode == "auto_space":
        return target if prompt.endswith((" ", "\n", "\t")) else " " + target
    raise ValueError(f"Unknown target prefix mode: {mode}")


def chunked(rows: list[dict[str, Any]], size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
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


def score_targets_batch(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    max_context_tokens: int,
    target_prefix: str,
) -> list[dict[str, float | int]]:
    input_ids_list: list[list[int]] = []
    labels_list: list[list[int]] = []
    attention_list: list[list[int]] = []

    for row in rows:
        prompt = row["input"]
        target = get_target(row)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(
            target_with_prefix(prompt, target, target_prefix),
            add_special_tokens=False,
        )
        if len(target_ids) == 0:
            target_ids = [tokenizer.eos_token_id]
        max_prompt_len = max(1, max_context_tokens - len(target_ids))
        prompt_ids = prompt_ids[-max_prompt_len:]
        full_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attention_list.append([1] * len(full_ids))

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id
    for ids, labels, attention in zip(input_ids_list, labels_list, attention_list):
        pad = max_len - len(ids)
        ids.extend([pad_id] * pad)
        labels.extend([-100] * pad)
        attention.extend([0] * pad)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=model.device)
    labels = torch.tensor(labels_list, dtype=torch.long, device=model.device)
    attention_mask = torch.tensor(attention_list, dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    target_mask = shift_labels.ne(-100)
    nll_sum = (losses * target_mask).sum(dim=1)
    token_count = target_mask.sum(dim=1).clamp(min=1)
    nll = nll_sum / token_count
    return [
        {"target_nll": float(nll[i].item()), "target_tokens": int(token_count[i].item())}
        for i in range(len(rows))
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    prompts_path = Path(args.prompts)
    metadata_path = Path(args.metadata) if args.metadata else infer_metadata_path(prompts_path)
    gates = load_gates(Path(args.harness_config))
    rows = load_jsonl(prompts_path)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    metadata = load_metadata_lookup(metadata_path)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = args.dtype
    if device.type == "cpu" and dtype_name != "fp32":
        dtype_name = "fp32"
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16 if dtype_name == "fp16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model = model.to(device).eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    examples: list[dict[str, Any]] = []
    overview = ArStats()
    groups: dict[str, dict[str, ArStats]] = {
        "curriculum_gate": defaultdict(ArStats),
        "source_group": defaultdict(ArStats),
        "prompt_type": defaultdict(ArStats),
        "answer_type": defaultdict(ArStats),
        "prompt_len_bucket": defaultdict(ArStats),
        "target_len_bucket": defaultdict(ArStats),
        "total_len_bucket": defaultdict(ArStats),
    }

    for batch in tqdm(list(chunked(rows, args.batch_size)), desc="AR eval"):
        prompts = [row["input"] for row in batch]
        generations = generate_batch(
            model, tokenizer, prompts,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        scores = score_targets_batch(
            model, tokenizer, batch,
            max_context_tokens=args.max_context_tokens,
            target_prefix=args.target_prefix,
        )
        for row, generation, score in zip(batch, generations, scores):
            target = get_target(row)
            answer = trim_generation(generation)
            pred_norm = normalize_answer(answer)
            target_norm = normalize_answer(target)
            meta = metadata.get(text_key(row.get("input", ""), target))
            features = example_features(row, meta)
            features["curriculum_gate"] = assign_gate(features, gates)
            example = {
                "input": row.get("input", ""),
                "expected": target,
                "generated": generation,
                "answer": answer,
                "exact_match": pred_norm == target_norm,
                "prefix_match": answer_prefix_match(pred_norm, target_norm),
                "contains_target": answer_contains(pred_norm, target_norm),
                "token_f1": token_f1(answer, target),
                "target_nll": score["target_nll"],
                "target_ppl": safe_exp(score["target_nll"]),
                "target_tokens": score["target_tokens"],
                **features,
            }
            examples.append(example)
            overview.add(example)
            for tree_name, tree in groups.items():
                tree[example[tree_name]].add(example)

    return {
        "schema_version": 1,
        "model": args.model,
        "prompts": str(prompts_path),
        "metadata": str(metadata_path) if metadata_path else None,
        "generation": {
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "target_prefix": args.target_prefix,
            "dtype": dtype_name,
            "device": str(device),
        },
        "overview": overview.as_dict(),
        "groups": {
            tree_name: {k: stat.as_dict() for k, stat in sorted(tree.items())}
            for tree_name, tree in groups.items()
        },
        "examples": examples,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# AR Instruction Baseline Report",
        "",
        f"Model: `{report['model']}`",
        f"Prompts: `{report['prompts']}`",
        "",
        "## Overall",
        "",
        "| n | exact | prefix | contains | token_f1 | target_nll | target_ppl |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ov = report["overview"]
    lines.append(
        "| {n} | {exact} | {prefix} | {contains} | {f1} | {nll} | {ppl} |".format(
            n=ov["n"],
            exact=fmt(ov["exact_match"]["mean"]),
            prefix=fmt(ov["prefix_match"]["mean"]),
            contains=fmt(ov["contains_target"]["mean"]),
            f1=fmt(ov["token_f1"]["mean"]),
            nll=fmt(ov["target_nll"]["mean"]),
            ppl=fmt(ov["target_ppl"]),
        )
    )
    lines.extend([
        "",
        "## Curriculum Gates",
        "",
        "| gate | n | exact | prefix | contains | token_f1 | target_nll |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for gate, stat in report["groups"]["curriculum_gate"].items():
        lines.append(
            "| {gate} | {n} | {exact} | {prefix} | {contains} | {f1} | {nll} |".format(
                gate=gate,
                n=stat["n"],
                exact=fmt(stat["exact_match"]["mean"]),
                prefix=fmt(stat["prefix_match"]["mean"]),
                contains=fmt(stat["contains_target"]["mean"]),
                f1=fmt(stat["token_f1"]["mean"]),
                nll=fmt(stat["target_nll"]["mean"]),
            )
        )
    lines.extend([
        "",
        "## Worst Examples By Token F1",
        "",
        "| token_f1 | target | generated |",
        "|---:|---|---|",
    ])
    worst = sorted(report["examples"], key=lambda ex: (ex["token_f1"], ex["exact_match"]))[:12]
    for ex in worst:
        target = normalize_answer(ex["expected"]).replace("|", "\\|")[:120]
        answer = normalize_answer(ex["answer"]).replace("|", "\\|")[:160]
        lines.append(f"| {ex['token_f1']:.3f} | {target} | {answer} |")
    path.write_text("\n".join(lines) + "\n")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AR LM baseline on fixed instruction probes.")
    parser.add_argument("--model", default="gpt2-large", help="HF causal LM id or local model path.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="Probe JSONL with input + target/output fields.")
    parser.add_argument("--metadata", default=None, help="Optional metadata JSONL. Auto-inferred for built-in Tulu probes.")
    parser.add_argument("--harness_config", default=str(DEFAULT_HARNESS_CONFIG), help="Harness config providing A/B/C gates.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=960)
    parser.add_argument("--max_context_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--target_prefix", choices=["auto_space", "space", "none"], default="auto_space")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_markdown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_DIR / f"{safe_name(args.model)}_{Path(args.prompts).stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate(args)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not args.no_markdown:
        write_markdown(report, output.with_suffix(".md"))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
