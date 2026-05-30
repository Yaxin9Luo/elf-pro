#!/usr/bin/env python
"""Lightweight instruction-following probe for an ELF text SFT checkpoint."""

import argparse
import json
import os
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

from configs.config import load_config_from_yaml, load_sampling_configs
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.data_utils import get_dataloader, get_pad_token_id, load_jsonl_dataset
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch, mask_after_eos, shift_left
from utils.sampling_utils import get_sampling_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--sampling_config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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
    return state_dict, int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = load_config_from_yaml(args.config)
    config.eval_data_path = args.prompts
    config.max_input_length = args.max_input_length
    config.batch_size = args.batch_size
    config.global_batch_size = args.batch_size
    config.num_samples = 10**9
    config.sampling_configs = load_sampling_configs(args.sampling_config)
    config.online_eval = False
    config.use_wandb = False

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    dataset = load_jsonl_dataset(args.prompts, tokenizer, input_key="input", output_key="output")

    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    vocab_size = len(tokenizer)
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    state_dict, ckpt_step, ckpt_epoch = load_checkpoint_weights(args.checkpoint, use_params=args.use_params)
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
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows = []
    sample_id = 0
    for batch in dataloader:
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
            rows.append({
                "id": sample_id,
                "checkpoint_step": ckpt_step,
                "checkpoint_epoch": ckpt_epoch,
                "sampling": {
                    "method": sampling_config.sampling_method,
                    "steps": sampling_config.num_sampling_steps[0],
                    "cfg": sampling_config.cfgs[0],
                    "self_cond_cfg": sampling_config.self_cond_cfg_scales[0],
                    "seed": args.seed,
                },
                "input": batch["input"][i],
                "expected": batch["target"][i],
                "generated": tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True).strip(),
            })
            sample_id += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
