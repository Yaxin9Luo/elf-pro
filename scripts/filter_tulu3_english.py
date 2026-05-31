#!/usr/bin/env python3
"""Filter an already-tokenized Tulu3 SFT dataset to English-like rows.

The input is expected to be the ELF conditional text dataset format:

  - input: rendered prompt/history text, for inspection and filtering
  - target: final assistant text, for inspection and filtering
  - condition_input_ids: tokenized prompt/history
  - input_ids: tokenized assistant target

The script preserves all columns and saves a filtered HuggingFace dataset that
can be used directly as `data_path` by the existing dataloader.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


EN_STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "answer", "are", "as",
    "at", "be", "because", "by", "can", "context", "do", "does", "for",
    "from", "given", "has", "have", "how", "if", "in", "instructions", "is",
    "it", "its", "of", "on", "or", "please", "question", "reply", "should",
    "task", "that", "the", "their", "there", "this", "to", "use", "user",
    "was", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "write", "you", "your",
}

NON_EN_STOPWORDS = {
    # Spanish / Portuguese / Catalan / Italian.
    "al", "alguna", "alguno", "anche", "año", "aos", "aquesta", "aquest",
    "aqui", "aquí", "as", "até", "cada", "che", "como", "com", "con",
    "cosa", "cuál", "da", "das", "de", "del", "della", "des", "di", "do",
    "dos", "du", "el", "els", "em", "en", "es", "esta", "está", "este",
    "esto", "eu", "fa", "gli", "ha", "il", "isso", "la", "las", "le",
    "les", "lo", "los", "mais", "mas", "más", "me", "mi", "moi", "muito",
    "na", "não", "no", "nos", "o", "os", "ou", "para", "parce", "per",
    "pero", "por", "porque", "porqué", "que", "qué", "se", "si", "sí",
    "son", "sont", "su", "sur", "te", "una", "une", "uno", "vai", "vos",
    # French.
    "au", "aux", "avec", "cette", "dans", "elle", "elles", "est", "être",
    "faire", "ils", "je", "mes", "ne", "nous", "pas", "plus", "pour",
    "quand", "qui", "suis", "sur", "une", "vous",
    # German / Dutch.
    "aber", "auf", "aus", "das", "dem", "den", "der", "des", "die", "ein",
    "eine", "einen", "einer", "er", "für", "het", "ich", "ist", "mit",
    "nicht", "sie", "und", "van", "voor", "was", "wie", "zijn",
}

NON_LATIN_RANGES = (
    ("cjk", 0x4E00, 0x9FFF),
    ("cjk_ext", 0x3400, 0x4DBF),
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("hangul", 0xAC00, 0xD7AF),
    ("cyrillic", 0x0400, 0x04FF),
    ("arabic", 0x0600, 0x06FF),
    ("hebrew", 0x0590, 0x05FF),
    ("thai", 0x0E00, 0x0E7F),
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("greek", 0x0370, 0x03FF),
)

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


def ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for ch in text if ord(ch) < 128) / len(text)


def letter_counts(text: str) -> tuple[int, int, Counter]:
    total_letters = 0
    non_ascii_letters = 0
    scripts = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        total_letters += 1
        if ord(ch) >= 128:
            non_ascii_letters += 1
        cp = ord(ch)
        for name, lo, hi in NON_LATIN_RANGES:
            if lo <= cp <= hi:
                scripts[name] += 1
                break
    return total_letters, non_ascii_letters, scripts


def token_stats(text: str) -> tuple[int, int, int]:
    words = [w.strip("'").lower() for w in WORD_RE.findall(text)]
    words = [w for w in words if w]
    en_hits = sum(1 for w in words if w in EN_STOPWORDS)
    non_en_hits = sum(1 for w in words if w in NON_EN_STOPWORDS)
    return len(words), en_hits, non_en_hits


def first_language_reject(text: str, *, ascii_min: float, non_ascii_letter_max: float) -> str | None:
    stripped = (text or "").strip()
    if not stripped:
        return None

    if "¿" in stripped or "¡" in stripped:
        return "inverted_spanish_punctuation"

    total_letters, non_ascii_letters, scripts = letter_counts(stripped)
    if total_letters:
        for script, count in scripts.items():
            if count / total_letters >= 0.005 or count >= 3:
                return f"non_latin_script:{script}"
        if non_ascii_letters / total_letters > non_ascii_letter_max:
            return "too_many_non_ascii_letters"

    if ascii_ratio(stripped) < ascii_min:
        return "low_ascii_ratio"

    word_count, en_hits, non_en_hits = token_stats(stripped)
    if word_count >= 6 and non_en_hits >= 3 and non_en_hits >= en_hits:
        return "non_english_stopwords"
    if word_count >= 12 and non_en_hits >= 5 and non_en_hits >= 0.7 * max(1, en_hits):
        return "non_english_stopwords"
    if word_count < 6 and non_en_hits >= 2 and non_en_hits > en_hits:
        return "short_non_english_stopwords"
    if non_ascii_letters > 0 and non_en_hits >= 1 and en_hits == 0:
        return "accented_non_english_short_text"

    return None


def first_reject_reason(row: dict, args: argparse.Namespace) -> str | None:
    prompt = str(row.get("input") or "")
    target = str(row.get("target") or "")
    joined = prompt + "\n" + target

    reason = first_language_reject(
        prompt,
        ascii_min=args.prompt_ascii_min,
        non_ascii_letter_max=args.prompt_non_ascii_letter_max,
    )
    if reason is not None:
        return "prompt_" + reason

    # Short entity-like targets can be language-neutral, so judge them mostly by
    # prompt language. Longer targets should be English too.
    target_words, _, _ = token_stats(target)
    if target_words >= args.min_target_words_for_language_check:
        reason = first_language_reject(
            target,
            ascii_min=args.target_ascii_min,
            non_ascii_letter_max=args.target_non_ascii_letter_max,
        )
        if reason is not None:
            return "target_" + reason

    if args.joined_language_check:
        reason = first_language_reject(
            joined,
            ascii_min=args.joined_ascii_min,
            non_ascii_letter_max=args.joined_non_ascii_letter_max,
        )
        if reason is not None:
            return "joined_" + reason

    return None


def row_lengths(row: dict) -> tuple[int, int, int]:
    prompt_len = int(row.get("prompt_tokens") or len(row.get("condition_input_ids") or []))
    target_len = int(row.get("target_tokens") or len(row.get("input_ids") or []))
    total_len = int(row.get("total_tokens") or (prompt_len + target_len))
    return prompt_len, target_len, total_len


def update_length_stats(stats: dict, row: dict) -> None:
    prompt_len, target_len, total_len = row_lengths(row)
    stats["count"] += 1
    for name, value in (
        ("prompt_tokens", prompt_len),
        ("target_tokens", target_len),
        ("total_tokens", total_len),
    ):
        item = stats[name]
        item["sum"] += value
        item["min"] = value if item["min"] is None else min(item["min"], value)
        item["max"] = value if item["max"] is None else max(item["max"], value)


def finalize_length_stats(stats: dict) -> dict:
    count = stats["count"]
    if count == 0:
        return {}
    result = {}
    for name in ("prompt_tokens", "target_tokens", "total_tokens"):
        item = stats[name]
        result[name] = {
            "min": item["min"],
            "max": item["max"],
            "mean": item["sum"] / count,
        }
    result["total_token_sum"] = stats["total_tokens"]["sum"]
    result["target_token_sum"] = stats["target_tokens"]["sum"]
    return result


def compact_row(row: dict, idx: int, reason: str | None = None) -> dict:
    prompt_len, target_len, total_len = row_lengths(row)
    item = {
        "index": idx,
        "id": str(row.get("id", "")),
        "source": str(row.get("source", "")),
        "input": str(row.get("input", ""))[:800],
        "target": str(row.get("target", ""))[:800],
        "prompt_tokens": prompt_len,
        "target_tokens": target_len,
        "total_tokens": total_len,
    }
    if reason is not None:
        item["reject_reason"] = reason
    return item


def main() -> None:
    from datasets import load_from_disk

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_json", default=None)
    parser.add_argument("--kept_sample_jsonl", default=None)
    parser.add_argument("--rejected_sample_jsonl", default=None)
    parser.add_argument("--max_rows", type=int, default=0, help="Debug only; 0 means all rows.")
    parser.add_argument("--prompt_ascii_min", type=float, default=0.92)
    parser.add_argument("--target_ascii_min", type=float, default=0.90)
    parser.add_argument("--joined_ascii_min", type=float, default=0.92)
    parser.add_argument("--prompt_non_ascii_letter_max", type=float, default=0.015)
    parser.add_argument("--target_non_ascii_letter_max", type=float, default=0.025)
    parser.add_argument("--joined_non_ascii_letter_max", type=float, default=0.02)
    parser.add_argument("--min_target_words_for_language_check", type=int, default=4)
    parser.add_argument("--joined_language_check", action="store_true")
    args = parser.parse_args()

    ds = load_from_disk(args.input_dataset)
    limit = min(len(ds), args.max_rows) if args.max_rows and args.max_rows > 0 else len(ds)

    kept_indices: list[int] = []
    reject_counts = Counter()
    reject_examples: dict[str, list[dict]] = defaultdict(list)
    kept_examples: list[dict] = []
    source_counts = Counter()
    length_stats = {
        "count": 0,
        "prompt_tokens": {"sum": 0, "min": None, "max": None},
        "target_tokens": {"sum": 0, "min": None, "max": None},
        "total_tokens": {"sum": 0, "min": None, "max": None},
    }

    for idx in range(limit):
        row = ds[int(idx)]
        reason = first_reject_reason(row, args)
        if reason is None:
            kept_indices.append(idx)
            source_counts[str(row.get("source", ""))] += 1
            update_length_stats(length_stats, row)
            if len(kept_examples) < 50:
                kept_examples.append(compact_row(row, idx))
            continue
        reject_counts[reason] += 1
        if len(reject_examples[reason]) < 5:
            reject_examples[reason].append(compact_row(row, idx, reason))

    out_dir = Path(args.output_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    filtered = ds.select(kept_indices)
    filtered.save_to_disk(str(out_dir))

    report = {
        "input_dataset": args.input_dataset,
        "output_dir": str(out_dir),
        "source_rows": len(ds),
        "processed_rows": limit,
        "kept_rows": len(kept_indices),
        "rejected_rows": limit - len(kept_indices),
        "kept_fraction": len(kept_indices) / max(1, limit),
        "reject_counts": dict(reject_counts.most_common()),
        "top_kept_sources": source_counts.most_common(30),
        "length_stats": finalize_length_stats(length_stats),
        "filters": {
            "prompt_ascii_min": args.prompt_ascii_min,
            "target_ascii_min": args.target_ascii_min,
            "joined_ascii_min": args.joined_ascii_min,
            "prompt_non_ascii_letter_max": args.prompt_non_ascii_letter_max,
            "target_non_ascii_letter_max": args.target_non_ascii_letter_max,
            "joined_non_ascii_letter_max": args.joined_non_ascii_letter_max,
            "min_target_words_for_language_check": args.min_target_words_for_language_check,
            "joined_language_check": bool(args.joined_language_check),
        },
        "kept_samples": kept_examples[:20],
        "rejected_samples_by_reason": dict(reject_examples),
    }

    report_path = Path(args.report_json) if args.report_json else out_dir / "english_filter_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.kept_sample_jsonl:
        path = Path(args.kept_sample_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in kept_examples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if args.rejected_sample_jsonl:
        path = Path(args.rejected_sample_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for reason, items in reject_examples.items():
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input_dataset": args.input_dataset,
        "output_dir": str(out_dir),
        "processed_rows": limit,
        "kept_rows": len(kept_indices),
        "rejected_rows": limit - len(kept_indices),
        "kept_fraction": len(kept_indices) / max(1, limit),
        "report_json": str(report_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
