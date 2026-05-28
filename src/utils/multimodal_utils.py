"""Helpers for image-text ELF training and generation."""

from typing import Dict

import torch

from utils.encoder_utils import encode_text
from utils.train_utils import unwrap_model


_VISION_EXTRA_KEYS = ("pixel_attention_mask", "spatial_shapes")


def _batch_tensor(batch: Dict, key: str, device, dtype=None):
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.to(device, non_blocking=True)
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value


@torch.no_grad()
def encode_vision_tokens(vision_encoder, batch: Dict, device, use_bf16: bool) -> torch.Tensor:
    pixel_values = _batch_tensor(batch, "pixel_values", device)
    extra = {}
    for key in _VISION_EXTRA_KEYS:
        if key in batch:
            extra[key] = _batch_tensor(batch, key, device)
    autocast_enabled = bool(use_bf16) and pixel_values.is_cuda
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        return vision_encoder(pixel_values=pixel_values, **extra)


def build_image_text_latents(
    model,
    text_encoder,
    vision_encoder,
    batch: Dict,
    config,
    device,
    dtype,
    use_bf16: bool,
    include_target: bool = True,
) -> torch.Tensor:
    """Build `[visual_cond, prompt_cond, target]` latents padded to config.max_length."""
    if vision_encoder is None:
        raise ValueError("image_text batches require a vision_encoder")

    prompt_input_ids = _batch_tensor(batch, "prompt_input_ids", device).long()
    prompt_attention_mask = _batch_tensor(batch, "prompt_attention_mask", device, torch.float32)
    prompt_lens = _batch_tensor(batch, "prompt_lens", device).long()

    prompt_latents = encode_text(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        encoder=text_encoder,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
        use_bf16=use_bf16,
    ).to(dtype)

    target_latents = None
    target_lens = None
    if include_target:
        target_input_ids = _batch_tensor(batch, "target_input_ids", device).long()
        target_attention_mask = _batch_tensor(batch, "target_attention_mask", device, torch.float32)
        target_lens = _batch_tensor(batch, "target_lens", device).long()
        target_latents = encode_text(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            encoder=text_encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            use_bf16=use_bf16,
        ).to(dtype)

    vision_tokens = encode_vision_tokens(vision_encoder, batch, device=device, use_bf16=use_bf16)
    inner_model = unwrap_model(model)
    visual_latents = inner_model.project_vision(vision_tokens).to(dtype)

    rows = []
    batch_size = visual_latents.shape[0]
    max_length = int(config.max_length)
    for i in range(batch_size):
        p_len = int(prompt_lens[i].item())
        pieces = [visual_latents[i], prompt_latents[i, :p_len]]
        if include_target:
            t_len = int(target_lens[i].item())
            pieces.append(target_latents[i, :t_len])
        seq = torch.cat(pieces, dim=0)
        if seq.shape[0] < max_length:
            pad = torch.zeros(
                (max_length - seq.shape[0], seq.shape[-1]),
                dtype=seq.dtype,
                device=seq.device,
            )
            seq = torch.cat([seq, pad], dim=0)
        rows.append(seq[:max_length])
    return torch.stack(rows, dim=0)
