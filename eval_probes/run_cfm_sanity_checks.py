#!/usr/bin/env python
"""Sanity checks for ELF CFM text SFT checkpoints.

Checks:
1. Clean latent decode: encode gold target with T5, then decode with DLM head.
2. Controlled denoise: add noise to target latents and compare correct / zero /
   shuffled conditions at fixed timesteps.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from configs.config import load_config_from_yaml
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.data_utils import get_dataloader, get_pad_token_id, load_jsonl_dataset
from utils.encoder_utils import encode_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", default="0.9,0.7,0.5,0.3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_params", action="store_true", help="Use raw params instead of EMA params.")
    return parser.parse_args()


def load_checkpoint_weights(path, use_params=False):
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


def masked_mean(values, mask):
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def decode_latents(model, z, config, self_cond_cfg_scale=1.0):
    batch_size = z.shape[0]
    t = torch.ones((batch_size,), dtype=z.dtype, device=z.device)
    sc_batch = None
    if config.num_self_cond_cfg_tokens > 0:
        sc_batch = torch.full((batch_size,), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(z.is_cuda and bool(config.use_bf16))):
        _, logits = model(
            z_input,
            t,
            deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
        )
    return logits.argmax(dim=-1), logits


def slice_target(ids, cond_len, valid_len):
    return ids[cond_len:valid_len]


def token_accuracy(pred_ids, gold_ids):
    if gold_ids.numel() == 0:
        return 0.0
    n = min(pred_ids.numel(), gold_ids.numel())
    return float((pred_ids[:n] == gold_ids[:n]).float().mean().item())


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = load_config_from_yaml(args.config)
    config.max_input_length = args.max_input_length
    config.batch_size = args.batch_size
    config.global_batch_size = args.batch_size
    config.use_wandb = False
    config.online_eval = False

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    dataset = load_jsonl_dataset(args.prompts, tokenizer, input_key="input", output_key="output")

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

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    timesteps = [float(x) for x in args.timesteps.split(",") if x.strip()]
    rows = []
    clean_accs = []
    denoise_summary = {}
    sample_id = 0

    for batch in dataloader:
        input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
        attention_mask = torch.from_numpy(np.array(batch["attention_mask"])).to(device).float()
        cond_mask = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()
        encoder_attention_mask = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
        cond_mask3 = cond_mask.unsqueeze(-1)
        target_mask = attention_mask * (1.0 - cond_mask)

        x0 = encode_text(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            encoder=encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            use_bf16=bool(config.use_bf16),
        ).to(next(model.parameters()).dtype)

        clean_pred_ids, _ = decode_latents(model, x0, config)
        batch_rows = []
        for i in range(input_ids.shape[0]):
            cond_len = int(cond_mask[i].sum().item())
            valid_len = int(attention_mask[i].sum().item())
            gold = slice_target(input_ids[i], cond_len, valid_len)
            pred = slice_target(clean_pred_ids[i], cond_len, valid_len)
            acc = token_accuracy(pred.detach().cpu(), gold.detach().cpu())
            clean_accs.append(acc)
            batch_rows.append({
                "id": sample_id + i,
                "input": batch["input"][i],
                "expected": batch["target"][i],
                "clean_decode": {
                    "token_acc": acc,
                    "generated": tokenizer.decode(pred.detach().cpu().numpy(), skip_special_tokens=True).strip(),
                },
                "denoise": [],
            })

        for t_val in timesteps:
            t = torch.full((input_ids.shape[0],), t_val, dtype=x0.dtype, device=device)
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=device) * config.denoiser_noise_scale
            z_base = t.reshape(-1, 1, 1) * x0 + (1 - t.reshape(-1, 1, 1)) * noise
            noise_x_mse = float(masked_mean((z_base - x0).pow(2).mean(dim=-1), target_mask).item())

            variants = {
                "correct": x0,
                "zero": torch.zeros_like(x0),
                "shuffled": x0.roll(shifts=1, dims=0),
            }
            for variant_name, cond_latents in variants.items():
                z = torch.where(cond_mask3 > 0, cond_latents, z_base)
                x_pred_prev = torch.where(cond_mask3 > 0, cond_latents, torch.zeros_like(x0))
                sc_batch = torch.ones((input_ids.shape[0],), dtype=x0.dtype, device=device)
                model_input = torch.cat([z, x_pred_prev], dim=-1) if config.self_cond_prob > 0 else z
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and bool(config.use_bf16))):
                    x_pred, _ = model(
                        model_input,
                        t,
                        deterministic=True,
                        self_cond_cfg_scale=sc_batch if config.num_self_cond_cfg_tokens > 0 else None,
                    )
                x_mse = float(masked_mean((x_pred - x0).pow(2).mean(dim=-1), target_mask).item())
                v_mse = float(masked_mean((((x_pred - z) - (x0 - z)) / max(1e-6, 1.0 - t_val)).pow(2).mean(dim=-1), target_mask).item())
                cos = float(masked_mean(F.cosine_similarity(x_pred.float(), x0.float(), dim=-1), target_mask).item())
                decoded_ids, _ = decode_latents(model, torch.where(cond_mask3 > 0, cond_latents, x_pred), config)

                accs = []
                for i in range(input_ids.shape[0]):
                    cond_len = int(cond_mask[i].sum().item())
                    valid_len = int(attention_mask[i].sum().item())
                    gold = slice_target(input_ids[i], cond_len, valid_len)
                    pred = slice_target(decoded_ids[i], cond_len, valid_len)
                    accs.append(token_accuracy(pred.detach().cpu(), gold.detach().cpu()))
                    batch_rows[i]["denoise"].append({
                        "t": t_val,
                        "variant": variant_name,
                        "x_mse": x_mse,
                        "v_mse": v_mse,
                        "cosine": cos,
                        "token_acc_batch": float(np.mean(accs)),
                    })
                key = f"t={t_val}:{variant_name}"
                denoise_summary.setdefault(key, []).append({
                    "noise_x_mse": noise_x_mse,
                    "x_mse": x_mse,
                    "v_mse": v_mse,
                    "cosine": cos,
                    "token_acc": float(np.mean(accs)),
                })

        rows.extend(batch_rows)
        sample_id += len(batch_rows)

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_meta": ckpt_meta,
        "weights": "params" if args.use_params else "ema_params1",
        "num_examples": len(rows),
        "clean_decode": {
            "mean_token_acc": float(np.mean(clean_accs)) if clean_accs else 0.0,
            "min_token_acc": float(np.min(clean_accs)) if clean_accs else 0.0,
            "max_token_acc": float(np.max(clean_accs)) if clean_accs else 0.0,
        },
        "controlled_denoise": {},
    }
    for key, vals in denoise_summary.items():
        summary["controlled_denoise"][key] = {
            metric: float(np.mean([v[metric] for v in vals]))
            for metric in ("noise_x_mse", "x_mse", "v_mse", "cosine", "token_acc")
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "examples": rows}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
