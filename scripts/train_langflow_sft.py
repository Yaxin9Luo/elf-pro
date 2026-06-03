#!/usr/bin/env python3
"""Instruction-tune LangFlow on prompt/target SFT text pairs.

This is intentionally separate from the ELF trainer. LangFlow uses GPT-2 token
embeddings, while the existing ELF Tulu3 dataset also contains T5 token columns.
For LangFlow SFT we read the preserved text columns and tokenize them again with
the GPT-2 tokenizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


DEFAULT_TULU3_ENGLISH = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/mmdata/"
    "text_sft/tulu3_sft_mixture_t5_1024_english"
)


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[torch.device, int, int, int]:
    if is_dist():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return torch.device("cuda", local_rank), rank, local_rank, world_size

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), 0, local_rank, 1
    return torch.device("cpu"), 0, 0, 1


def log_rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def set_seed(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_checkpoint_file(path: str) -> str:
    checkpoint = Path(path)
    if checkpoint.is_file():
        return str(checkpoint)
    latest = checkpoint / "latest_checkpoint.txt"
    if latest.exists():
        latest_path = latest.read_text(encoding="utf-8").strip()
        if latest_path:
            return resolve_checkpoint_file(latest_path)
    for name in ("model.safetensors", "pytorch_model.safetensors"):
        candidate = checkpoint / name
        if candidate.exists():
            return str(candidate)
    nested = sorted(
        checkpoint.glob("checkpoint_*/model.safetensors"),
        key=lambda p: int(p.parent.name.split("_")[-1]) if p.parent.name.split("_")[-1].isdigit() else -1,
    )
    if nested:
        return str(nested[-1])
    matches = sorted(checkpoint.glob("*.safetensors"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"No safetensors checkpoint found under {checkpoint}")


def load_langflow_model(langflow_repo: str, checkpoint: str, device: torch.device):
    repo = Path(langflow_repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from langflow import LangFlow, LangFlowConfig

    checkpoint_file = Path(resolve_checkpoint_file(checkpoint))
    checkpoint_dir = checkpoint_file.parent
    config_source = checkpoint_dir if (checkpoint_dir / "config.json").exists() else repo / "langflow"
    config = LangFlowConfig.from_pretrained(config_source)
    model = LangFlow(config)
    state_dict = load_file(str(checkpoint_file), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def resolve_resume_dir(output_dir: str, resume: str) -> Path | None:
    if resume in ("", "none", "None", "false", "False"):
        return None
    resume_path = Path(output_dir) if resume == "auto" else Path(resume)
    try:
        checkpoint_file = Path(resolve_checkpoint_file(str(resume_path)))
    except FileNotFoundError:
        if resume == "auto":
            return None
        raise
    return checkpoint_file.parent


def load_resume_state(checkpoint_dir: Path) -> dict[str, Any]:
    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        return torch.load(state_path, map_location="cpu")
    trainer_path = checkpoint_dir / "trainer_state.json"
    if trainer_path.exists():
        trainer_state = json.loads(trainer_path.read_text(encoding="utf-8"))
        return {"step": int(trainer_state.get("step", 0)), "epoch": int(trainer_state.get("epoch", 0))}
    return {"step": 0, "epoch": 0}


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_sft_dataset(path: str, split: str):
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        if split not in ds:
            raise KeyError(f"Split {split!r} not found in {path}; available={list(ds.keys())}")
        ds = ds[split]
    required = {"input", "target"}
    missing = sorted(required.difference(ds.column_names))
    if missing:
        raise KeyError(f"SFT dataset must contain text columns {sorted(required)}; missing={missing}")
    return ds


class PromptTargetCollator:
    def __init__(
        self,
        tokenizer,
        *,
        max_length: int,
        max_target_length: int,
        add_bos: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.add_bos = add_bos
        self.pad_id = int(tokenizer.eos_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.bos_id = int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id)

    def _build_example(self, prompt: str, target: str) -> dict[str, Any]:
        prompt_ids = self.tokenizer(prompt or "", add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target or "", add_special_tokens=False)["input_ids"]

        prefix = [self.bos_id] + prompt_ids if self.add_bos else list(prompt_ids)
        if not prefix:
            prefix = [self.bos_id]

        target_with_eos = list(target_ids) + [self.eos_id]
        target_truncated = 0
        if len(target_with_eos) > self.max_target_length:
            target_truncated = len(target_with_eos) - self.max_target_length
            if self.max_target_length <= 1:
                target_with_eos = [self.eos_id]
            else:
                target_with_eos = target_with_eos[: self.max_target_length - 1] + [self.eos_id]

        min_prefix = 1
        if len(target_with_eos) > self.max_length - min_prefix:
            target_truncated += len(target_with_eos) - (self.max_length - min_prefix)
            keep = self.max_length - min_prefix
            if keep <= 1:
                target_with_eos = [self.eos_id]
            else:
                target_with_eos = target_with_eos[: keep - 1] + [self.eos_id]

        available_prefix = self.max_length - len(target_with_eos)
        prompt_truncated = max(0, len(prefix) - available_prefix)
        if len(prefix) > available_prefix:
            if self.add_bos and available_prefix >= 1:
                prefix = [self.bos_id] + prefix[-(available_prefix - 1):]
            else:
                prefix = prefix[-available_prefix:]

        input_ids = prefix + target_with_eos
        target_mask = [0] * len(prefix) + [1] * len(target_with_eos)
        return {
            "prefix_ids": prefix,
            "target_ids": target_with_eos,
            "input_ids": input_ids,
            "target_mask": target_mask,
            "prompt_truncated": prompt_truncated,
            "target_truncated": target_truncated,
        }

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        built = [self._build_example(str(ex["input"]), str(ex["target"])) for ex in examples]
        max_prefix_len = max(len(item["prefix_ids"]) for item in built)
        max_len = min(self.max_length, max_prefix_len + self.max_target_length)
        batch_ids = []
        batch_attn = []
        batch_target = []
        batch_condition = []
        batch_noise_tail = []
        for item in built:
            prefix_ids = item["prefix_ids"]
            target_ids = item["target_ids"]
            ids = prefix_ids + target_ids
            tail = max_len - len(ids)
            if tail < 0:
                raise ValueError(f"Internal truncation error: example length {len(ids)} exceeds batch length {max_len}")
            batch_ids.append(ids + [self.pad_id] * tail)
            batch_attn.append([1] * max_len)
            batch_target.append([0] * len(prefix_ids) + [1] * len(target_ids) + [0] * tail)
            batch_condition.append([1] * len(prefix_ids) + [0] * (len(target_ids) + tail))
            batch_noise_tail.append([0] * len(ids) + [1] * tail)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.bool),
            "target_mask": torch.tensor(batch_target, dtype=torch.bool),
            "condition_mask": torch.tensor(batch_condition, dtype=torch.bool),
            "noise_tail_mask": torch.tensor(batch_noise_tail, dtype=torch.bool),
            "prompt_truncated": torch.tensor([x["prompt_truncated"] for x in built], dtype=torch.long),
            "target_truncated": torch.tensor([x["target_truncated"] for x in built], dtype=torch.long),
        }


def conditional_latent(clean: torch.Tensor, noise: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    alpha = torch.sigmoid(-gamma.float()).sqrt()[:, None, None].to(clean.dtype)
    sigma = torch.sigmoid(gamma.float()).sqrt()[:, None, None].to(clean.dtype)
    return clean * alpha + noise * sigma


def embed_softmax(model, logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    return model._embed_tokens(probs)


def compute_loss(
    model,
    raw_model,
    batch: dict[str, torch.Tensor],
    *,
    self_cond_prob: float,
    t_eps: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    target_mask = batch["target_mask"] & attention_mask
    condition_mask = batch["condition_mask"] & attention_mask
    known_mask = condition_mask | target_mask

    clean = raw_model._embed_tokens(input_ids)
    q = torch.rand(input_ids.shape[0], device=input_ids.device, generator=generator)
    q = q.mul(1.0 - 2.0 * t_eps).add(t_eps)
    gamma = raw_model.proposal(q)
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    z_known = conditional_latent(clean, noise, gamma)
    z = torch.where(known_mask[..., None], z_known, noise)

    x_self_cond = None
    if getattr(raw_model.config, "self_conditioning", False) and self_cond_prob > 0.0:
        with torch.no_grad():
            logits_sc = raw_model(noisy_embeds=z, timesteps=gamma, return_dict=False)
            x_pred_sc = embed_softmax(raw_model, logits_sc)
            x_pred_sc = torch.where(condition_mask[..., None], clean, x_pred_sc)
        keep = torch.rand(input_ids.shape[0], device=input_ids.device, generator=generator) < self_cond_prob
        keep = keep[:, None, None]
        x_self_cond = torch.where(keep, x_pred_sc, torch.zeros_like(x_pred_sc))

    logits = model(noisy_embeds=z, timesteps=gamma, x_self_cond=x_self_cond, return_dict=False)
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        input_ids.reshape(-1),
        reduction="none",
    ).view_as(input_ids)

    denom = target_mask.sum().clamp_min(1)
    loss = (per_token * target_mask.float()).sum() / denom
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        target_correct = ((pred == input_ids) & target_mask).sum()
        metrics = {
            "loss_sum": float((per_token * target_mask.float()).sum().detach().float().item()),
            "target_tokens": float(denom.detach().float().item()),
            "target_correct": float(target_correct.detach().float().item()),
            "examples": float(input_ids.shape[0]),
            "prompt_truncated": float(batch["prompt_truncated"].sum().item()),
            "target_truncated": float(batch["target_truncated"].sum().item()),
            "prompt_truncated_examples": float((batch["prompt_truncated"] > 0).sum().item()),
            "target_truncated_examples": float((batch["target_truncated"] > 0).sum().item()),
            "noise_tail_tokens": float(batch["noise_tail_mask"].sum().item()),
            "gamma_sum": float(gamma.detach().float().sum().item()),
        }
    return loss, metrics


def reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = sorted(metrics)
    values = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
    if is_dist():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {k: float(v.item()) for k, v in zip(keys, values)}


def lr_for_step(step: int, *, lr: float, min_lr_ratio: float, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return lr
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(total_steps - warmup_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def save_checkpoint(
    model,
    tokenizer,
    args: argparse.Namespace,
    *,
    step: int,
    epoch: int,
    rank: int,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    if rank != 0:
        return
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt_dir = Path(args.output_dir) / f"checkpoint_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
    save_file(state, ckpt_dir / "model.safetensors")
    raw_model.config.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")

    repo = Path(args.langflow_repo).resolve() / "langflow"
    for name in ("model.py", "config.py", "__init__.py"):
        src = repo / name
        if src.exists():
            shutil.copy2(src, ckpt_dir / name)

    state_json = {
        "step": step,
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics or {},
    }
    (ckpt_dir / "trainer_state.json").write_text(json.dumps(state_json, indent=2) + "\n", encoding="utf-8")
    if optimizer is not None and args.save_optimizer_state:
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            ckpt_dir / "training_state.pt",
        )
    latest = Path(args.output_dir) / "latest_checkpoint.txt"
    latest.write_text(str(ckpt_dir) + "\n", encoding="utf-8")
    print(f"saved checkpoint: {ckpt_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langflow_repo", default="../LangFlow")
    parser.add_argument("--init_checkpoint", default="../checkpoints/Continuous-Rivals-Discrete/langflow-owt")
    parser.add_argument("--data_path", default=DEFAULT_TULU3_ENGLISH)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="outputs/langflow-owt-tulu3-english-sft")
    parser.add_argument("--resume", default="none", help="Use 'auto' for output_dir/latest_checkpoint.txt, 'none', or a checkpoint path.")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=768)
    parser.add_argument("--self_cond_prob", type=float, default=0.5)
    parser.add_argument("--t_eps", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--drop_last", action="store_true", default=True)
    parser.add_argument("--no_drop_last", action="store_false", dest="drop_last")
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--save_optimizer_state", action="store_true")
    parser.add_argument("--disable_saves", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--add_bos", action="store_true", default=True)
    parser.add_argument("--no_add_bos", action="store_false", dest="add_bos")
    parser.add_argument("--freeze_proposal", action="store_true", default=True)
    parser.add_argument("--train_proposal", action="store_false", dest="freeze_proposal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, rank, _local_rank, world_size = setup_distributed()
    if args.grad_accum_steps != 1:
        raise ValueError("grad_accum_steps > 1 is not supported until DDP no_sync and final flush are implemented")
    set_seed(args.seed, rank)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    log_rank0(rank, f"loading tokenizer={args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
        use_fast=not args.use_slow_tokenizer,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("LangFlow SFT requires a tokenizer with an EOS token")
    tokenizer.pad_token = tokenizer.eos_token

    log_rank0(rank, f"loading dataset={args.data_path}")
    dataset = load_sft_dataset(args.data_path, args.split)
    if args.max_train_samples:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    log_rank0(rank, f"dataset_rows={len(dataset)} columns={dataset.column_names}")

    resume_dir = resolve_resume_dir(args.output_dir, args.resume)
    model_checkpoint = str(resume_dir) if resume_dir is not None else args.init_checkpoint
    if resume_dir is not None:
        log_rank0(rank, f"resuming from checkpoint={resume_dir}")
    log_rank0(rank, f"loading LangFlow checkpoint={model_checkpoint}")
    model = load_langflow_model(args.langflow_repo, model_checkpoint, device)
    if args.max_length > model.config.model_length:
        raise ValueError(f"max_length={args.max_length} exceeds model_length={model.config.model_length}")
    if not args.freeze_proposal:
        raise ValueError("--train_proposal is not supported in this standalone SFT script")
    if args.freeze_proposal:
        for param in model.proposal.parameters():
            param.requires_grad_(False)
    model.train()
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=args.drop_last,
        )
        if is_dist()
        else None
    )
    collator = PromptTargetCollator(
        tokenizer,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        add_bos=args.add_bos,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        drop_last=args.drop_last,
        persistent_workers=args.num_workers > 0,
    )

    updates_per_epoch = math.ceil(len(dataloader) / args.grad_accum_steps)
    planned_steps = args.max_steps if args.max_steps > 0 else max(1, updates_per_epoch * args.epochs)
    log_rank0(
        rank,
        json.dumps(
            {
                "world_size": world_size,
                "per_gpu_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * world_size * args.grad_accum_steps,
                "epochs": args.epochs,
                "planned_optimizer_steps": planned_steps,
                "max_length": args.max_length,
                "max_target_length": args.max_target_length,
            },
            indent=2,
        ),
    )

    if is_dist():
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    raw_model = model.module if isinstance(model, DDP) else model
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    global_step = 0
    start_epoch = 0
    if resume_dir is not None:
        resume_state = load_resume_state(resume_dir)
        global_step = int(resume_state.get("step", 0))
        start_epoch = int(resume_state.get("epoch", 0))
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            optimizer_to(optimizer, device)
        log_rank0(rank, f"resume_state step={global_step} start_epoch={start_epoch} optimizer={'optimizer' in resume_state}")

    generator = torch.Generator(device=device) if device.type == "cuda" else None
    if generator is not None:
        generator.manual_seed(args.seed + rank * 100003)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    micro_step = 0
    running: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)

    if start_epoch >= args.epochs:
        log_rank0(rank, f"resume checkpoint already reached epoch {start_epoch}; nothing to train for epochs={args.epochs}")
        if is_dist():
            dist.barrier()
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, metrics = compute_loss(
                model,
                raw_model,
                batch,
                self_cond_prob=args.self_cond_prob,
                t_eps=args.t_eps,
                generator=generator,
            )
            (loss / args.grad_accum_steps).backward()
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value

            micro_step += 1
            if micro_step % args.grad_accum_steps != 0:
                continue

            lr = lr_for_step(
                global_step,
                lr=args.lr,
                min_lr_ratio=args.min_lr_ratio,
                warmup_steps=args.warmup_steps,
                total_steps=planned_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every == 0:
                reduced = reduce_metrics(running, device)
                tokens = max(1.0, reduced["target_tokens"])
                log_rank0(
                    rank,
                    (
                        f"step={global_step} epoch={epoch + 1} lr={lr:.3e} "
                        f"loss={reduced['loss_sum'] / tokens:.4f} "
                        f"target_acc={reduced['target_correct'] / tokens:.4f} "
                        f"target_tokens={int(tokens)} "
                        f"prompt_trunc={int(reduced['prompt_truncated'])} "
                        f"prompt_trunc_ex={int(reduced['prompt_truncated_examples'])} "
                        f"target_trunc={int(reduced['target_truncated'])} "
                        f"target_trunc_ex={int(reduced['target_truncated_examples'])} "
                        f"noise_tail={int(reduced['noise_tail_tokens'])} "
                        f"gamma_mean={reduced['gamma_sum'] / max(1.0, reduced['examples']):.4f}"
                    ),
                )
                running = {}

            if not args.disable_saves and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_checkpoint(model, tokenizer, args, step=global_step, epoch=epoch + 1, rank=rank, optimizer=optimizer)
                if is_dist():
                    dist.barrier()

            if args.max_steps > 0 and global_step >= args.max_steps:
                if not args.disable_saves:
                    save_checkpoint(model, tokenizer, args, step=global_step, epoch=epoch + 1, rank=rank, optimizer=optimizer)
                if is_dist():
                    dist.barrier()
                    dist.destroy_process_group()
                return

        if not args.disable_saves and args.save_every_epoch:
            save_checkpoint(model, tokenizer, args, step=global_step, epoch=epoch + 1, rank=rank, optimizer=optimizer)
            if is_dist():
                dist.barrier()

    if not args.disable_saves:
        save_checkpoint(model, tokenizer, args, step=global_step, epoch=args.epochs, rank=rank, optimizer=optimizer)
        if is_dist():
            dist.barrier()
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
