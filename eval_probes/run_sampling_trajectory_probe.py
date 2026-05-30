#!/usr/bin/env python
"""Trace ELF CFM sampling trajectories from noised gold target latents."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
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
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_examples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_starts", default="0.95,0.9,0.7,0.5,0.3,0.1,0.0")
    parser.add_argument("--uniform_steps", type=int, default=32)
    parser.add_argument("--logit_steps", type=int, default=64)
    parser.add_argument("--trace_every", type=int, default=1)
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


def token_accuracy(pred_ids, gold_ids):
    if gold_ids.numel() == 0:
        return 0.0
    n = min(pred_ids.numel(), gold_ids.numel())
    return float((pred_ids[:n] == gold_ids[:n]).float().mean().item())


@torch.no_grad()
def decode_latents(model, z, config):
    batch_size = z.shape[0]
    t = torch.ones((batch_size,), dtype=z.dtype, device=z.device)
    sc_batch = None
    if config.num_self_cond_cfg_tokens > 0:
        sc_batch = torch.ones((batch_size,), dtype=z.dtype, device=z.device)
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(z.is_cuda and bool(config.use_bf16))):
        _, logits = model(
            z_input,
            t,
            deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
        )
    return logits.argmax(dim=-1)


def target_slice(ids, cond_len, valid_len):
    return ids[cond_len:valid_len]


def build_uniform_steps(t_start, steps, device, dtype):
    return torch.linspace(float(t_start), 1.0, int(steps) + 1, device=device, dtype=dtype)


def build_logit_tail_steps(t_start, total_steps, config, device, dtype):
    full = get_sampling_steps(
        n_steps=int(total_steps),
        time_schedule="logit_normal",
        P_mean=config.denoiser_p_mean,
        P_std=config.denoiser_p_std,
        device=device,
        dtype=dtype,
    )
    keep = full[full > float(t_start)]
    start = torch.tensor([float(t_start)], device=device, dtype=dtype)
    if keep.numel() == 0:
        return torch.cat([start, torch.ones((1,), device=device, dtype=dtype)])
    if keep[-1].item() < 1.0:
        keep = torch.cat([keep, torch.ones((1,), device=device, dtype=dtype)])
    return torch.cat([start, keep])


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = load_config_from_yaml(args.config)
    config.max_input_length = args.max_input_length
    config.batch_size = 1
    config.global_batch_size = 1
    config.use_wandb = False
    config.online_eval = False

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    dataset = load_jsonl_dataset(args.prompts, tokenizer, input_key="input", output_key="output")
    dataset = dataset[: args.max_examples]

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
        batch_size=1,
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
    dtype = next(model.parameters()).dtype
    t_starts = [float(x) for x in args.t_starts.split(",") if x.strip()]
    results = {
        "checkpoint": args.checkpoint,
        "checkpoint_meta": ckpt_meta,
        "weights": "params" if args.use_params else "ema_params1",
        "seed": args.seed,
        "schedules": {
            "uniform": {"steps": args.uniform_steps},
            "logit_tail": {"source_total_steps": args.logit_steps, "p_mean": config.denoiser_p_mean, "p_std": config.denoiser_p_std},
        },
        "examples": [],
    }

    for ex_idx, batch in enumerate(dataloader):
        input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
        attention_mask = torch.from_numpy(np.array(batch["attention_mask"])).to(device).float()
        cond_mask = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()
        encoder_attention_mask = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
        target_mask = attention_mask * (1.0 - cond_mask)
        cond_mask3 = cond_mask.unsqueeze(-1)
        cond_len = int(cond_mask[0].sum().item())
        valid_len = int(attention_mask[0].sum().item())
        gold_ids = target_slice(input_ids[0], cond_len, valid_len).detach().cpu()

        x0 = encode_text(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            encoder=encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            use_bf16=bool(config.use_bf16),
        ).to(dtype)

        clean_ids = decode_latents(model, x0, config)[0].detach().cpu()
        clean_target_ids = target_slice(clean_ids, cond_len, valid_len)
        example = {
            "id": ex_idx,
            "input": batch["input"][0],
            "expected": batch["target"][0],
            "clean_decode": {
                "token_acc": token_accuracy(clean_target_ids, gold_ids),
                "generated": tokenizer.decode(clean_target_ids.numpy(), skip_special_tokens=True).strip(),
            },
            "runs": [],
        }

        for t_start in t_starts:
            base_noise = torch.randn_like(x0) * config.denoiser_noise_scale
            z_start = float(t_start) * x0 + (1.0 - float(t_start)) * base_noise
            z_start = restore_cond(z_start, x0, cond_mask)
            schedule_defs = [
                ("uniform", build_uniform_steps(t_start, args.uniform_steps, device, dtype)),
                ("logit_tail", build_logit_tail_steps(t_start, args.logit_steps, config, device, dtype)),
            ]

            for schedule_name, t_steps in schedule_defs:
                z = z_start.clone()
                x_pred_prev = restore_cond(torch.zeros_like(z), x0, cond_mask)
                trace = []
                step_count = int(t_steps.numel() - 1)

                for step_idx in range(step_count):
                    t = float(t_steps[step_idx].item())
                    t_next = float(t_steps[step_idx + 1].item())
                    z_before = z
                    z, x_pred = _ode_step(
                        model=model,
                        z=z,
                        t=t,
                        t_next=t_next,
                        x_pred_prev=x_pred_prev,
                        config=config,
                        cfg_scale=1.0,
                        self_cond_cfg_scale=1.0,
                        cond_seq=x0,
                        cond_seq_mask=cond_mask,
                    )
                    x_pred_prev = x_pred

                    should_trace = (
                        step_idx == 0
                        or step_idx == step_count - 1
                        or ((step_idx + 1) % max(1, args.trace_every) == 0)
                    )
                    if should_trace:
                        z_decoded = decode_latents(model, restore_cond(z, x0, cond_mask), config)[0].detach().cpu()
                        pred_decoded = decode_latents(model, restore_cond(x_pred, x0, cond_mask), config)[0].detach().cpu()
                        z_target_ids = target_slice(z_decoded, cond_len, valid_len)
                        pred_target_ids = target_slice(pred_decoded, cond_len, valid_len)
                        trace.append({
                            "step": step_idx + 1,
                            "t": t,
                            "t_next": t_next,
                            "dt": t_next - t,
                            "z_before_mse": float(masked_mean((z_before - x0).pow(2).mean(dim=-1), target_mask).item()),
                            "z_after_mse": float(masked_mean((z - x0).pow(2).mean(dim=-1), target_mask).item()),
                            "x_pred_mse": float(masked_mean((x_pred - x0).pow(2).mean(dim=-1), target_mask).item()),
                            "z_after_token_acc": token_accuracy(z_target_ids, gold_ids),
                            "x_pred_token_acc": token_accuracy(pred_target_ids, gold_ids),
                        })

                final_ids = decode_latents(model, restore_cond(z, x0, cond_mask), config)[0].detach().cpu()
                final_target_ids = target_slice(final_ids, cond_len, valid_len)
                example["runs"].append({
                    "t_start": t_start,
                    "schedule": schedule_name,
                    "num_steps": step_count,
                    "initial_mse": float(masked_mean((z_start - x0).pow(2).mean(dim=-1), target_mask).item()),
                    "final_mse": float(masked_mean((z - x0).pow(2).mean(dim=-1), target_mask).item()),
                    "final_token_acc": token_accuracy(final_target_ids, gold_ids),
                    "final_generated": tokenizer.decode(final_target_ids.numpy(), skip_special_tokens=True).strip(),
                    "trace": trace,
                })

        results["examples"].append(example)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    for ex in results["examples"]:
        print(f"\nExample {ex['id']}: {ex['input'][:120].replace(chr(10), ' ')}")
        print(f"clean_acc={ex['clean_decode']['token_acc']:.3f}")
        for run in ex["runs"]:
            print(
                f"t_start={run['t_start']:.2f} schedule={run['schedule']:<10} "
                f"steps={run['num_steps']:<2d} init_mse={run['initial_mse']:.4f} "
                f"final_mse={run['final_mse']:.4f} final_acc={run['final_token_acc']:.3f} "
                f"final={run['final_generated'][:100].replace(chr(10), ' ')}"
            )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
