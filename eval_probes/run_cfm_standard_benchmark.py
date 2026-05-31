#!/usr/bin/env python3
"""Run generation-style standard benchmarks with an ELF CFM checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from configs.config import load_config_from_yaml, load_sampling_configs
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch, mask_after_eos, shift_left
from utils.sampling_utils import get_sampling_steps


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
    stops = ["\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"]
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


def load_checkpoint_weights(path: str, use_params: bool = False):
    kwargs = {"map_location": "cpu"}
    try:
        kwargs["mmap"] = True
        ckpt = torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        ckpt = torch.load(path, **kwargs)
    state_dict = ckpt["params"] if use_params else ckpt.get("ema_params1", ckpt["params"])
    meta = {"step": int(ckpt.get("step", 0)), "epoch": int(ckpt.get("epoch", 0))}
    return state_dict, meta


def make_dataset(rows: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    examples = []
    for i, row in enumerate(rows):
        target = str(row.get("output", ""))
        examples.append({
            "index": i,
            "input": row["input"],
            "target": target,
            "condition_input_ids": tokenizer(row["input"], add_special_tokens=False)["input_ids"],
            "input_ids": tokenizer(target, add_special_tokens=False)["input_ids"],
        })
    return examples


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, ckpt_meta: dict[str, int]) -> dict[str, Any]:
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
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_meta": ckpt_meta,
        "weights": "params" if args.use_params else "ema_params1",
        "sampling_config": args.sampling_config,
        "max_examples": args.max_examples,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_outputs": len(rows),
        "scores_by_benchmark": as_scores(by_benchmark),
        "scores_by_task": as_scores(by_task),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# CFM Standard Benchmark Report",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Benchmark file: `{summary['benchmark_file']}`",
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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark_file", required=True)
    parser.add_argument("--sampling_config", default="eval_probes/sft_probe_sampling_64.yml")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_params", action="store_true")
    args = parser.parse_args()

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    all_rows = load_jsonl(Path(args.benchmark_file))
    if args.max_examples and args.max_examples > 0:
        all_rows = all_rows[:args.max_examples]
    rows = [row for i, row in enumerate(all_rows) if i % args.num_shards == args.shard_index]

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = load_config_from_yaml(args.config)
    config.max_input_length = args.max_input_length
    config.batch_size = args.batch_size
    config.global_batch_size = args.batch_size
    config.num_samples = len(rows)
    config.sampling_configs = load_sampling_configs(args.sampling_config)
    config.online_eval = False
    config.use_wandb = False

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    dataset = make_dataset(rows, tokenizer)

    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=len(tokenizer),
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    state_dict, ckpt_meta = load_checkpoint_weights(args.checkpoint, use_params=args.use_params)
    model.load_state_dict(state_dict)
    del state_dict
    model = model.to(device).eval()

    dataloader = get_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=False,
        tokenizer=tokenizer,
    )

    sampling_config = config.sampling_configs[0]
    param_dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(args.seed + args.shard_index)
    torch.manual_seed(args.seed + args.shard_index)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + args.shard_index)

    outputs = []
    row_cursor = 0
    for batch in tqdm(dataloader, desc="CFM benchmark generation"):
        input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
        cond_seq_mask = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()
        encoder_attention_mask = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
        cond_seq = encode_text(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            encoder=encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
        ).to(param_dtype)
        z = torch.randn((input_ids.shape[0], config.max_length, model.text_encoder_dim), dtype=param_dtype)
        z = (z * config.denoiser_noise_scale).to(device)
        t_steps = get_sampling_steps(
            n_steps=sampling_config.num_sampling_steps[0],
            time_schedule=sampling_config.time_schedule,
            P_mean=config.denoiser_p_mean,
            P_std=config.denoiser_p_std,
            device=device,
            dtype=param_dtype,
        )
        latent = _generate_samples_single_batch(
            model=model,
            generator=generator,
            z=z,
            t_steps=t_steps,
            cond_seq=cond_seq,
            cond_seq_mask=cond_seq_mask,
            config=config,
            sampling_config=sampling_config,
            cfg_scale=sampling_config.cfgs[0],
            self_cond_cfg_scale=sampling_config.self_cond_cfg_scales[0],
        )
        predicted_ids = _dlm_decode_batch(
            z=latent,
            model=model,
            t_final_val=t_steps[-1].item(),
            config=config,
            self_cond_cfg_scale=sampling_config.self_cond_cfg_scales[0],
        )
        cond_len = cond_seq_mask.to(torch.int32).sum(dim=1)
        gen_length = config.max_length - config.max_input_length
        predicted_ids = shift_left(predicted_ids, cond_len, pad_token_id)[:, :gen_length]
        predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

        for i in range(predicted_ids.shape[0]):
            source = rows[row_cursor]
            generated = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
            answer = trim_generation(generated)
            score = score_generation(source, answer)
            outputs.append({
                **source,
                "generated": generated,
                "answer_text": answer,
                "score": score,
                "checkpoint_meta": ckpt_meta,
                "sampling": {
                    "method": sampling_config.sampling_method,
                    "steps": sampling_config.num_sampling_steps[0],
                    "cfg": sampling_config.cfgs[0],
                    "self_cond_cfg": sampling_config.self_cond_cfg_scales[0],
                    "seed": args.seed,
                },
            })
            row_cursor += 1

    output_jsonl = Path(args.output_jsonl)
    write_jsonl(output_jsonl, outputs)
    summary = summarize(outputs, args, ckpt_meta)
    summary_json = Path(args.summary_json) if args.summary_json else output_jsonl.with_suffix(".summary.json")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, summary_json.with_suffix(".md"))
    print(f"wrote {len(outputs)} outputs to {output_jsonl}")
    print(f"wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
