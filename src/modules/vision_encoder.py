"""Frozen vision encoder helpers for image-text ELF experiments."""

from typing import Any, Dict

import torch
import torch.nn as nn

from utils.logging_utils import log_for_0


class FrozenVisionEncoder(nn.Module):
    """Thin wrapper around a HuggingFace vision-language model's vision tower."""

    def __init__(self, model_name: str, dtype: Any = torch.float32):
        super().__init__()
        from transformers import AutoModel

        log_for_0(f"Loading vision encoder: {model_name}...")
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.hidden_size = self._infer_hidden_size()

    def _infer_hidden_size(self) -> int:
        cfg = getattr(self.model, "config", None)
        vision_cfg = getattr(cfg, "vision_config", cfg)
        for name in ("hidden_size", "embed_dim", "projection_dim"):
            value = getattr(vision_cfg, name, None)
            if value is not None:
                return int(value)
        raise ValueError(f"Could not infer vision hidden size for {self.model_name}")

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        image_kwargs: Dict[str, torch.Tensor] = {
            k: v for k, v in kwargs.items() if v is not None
        }
        tower = getattr(self.model, "vision_model", None)
        if tower is not None:
            out = tower(pixel_values=pixel_values, **image_kwargs, return_dict=True)
        else:
            out = self.model(pixel_values=pixel_values, **image_kwargs, return_dict=True)

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        vision_out = getattr(out, "vision_model_output", None)
        if vision_out is not None and getattr(vision_out, "last_hidden_state", None) is not None:
            return vision_out.last_hidden_state
        if isinstance(out, (tuple, list)) and out:
            return out[0]
        raise ValueError(f"Vision encoder {self.model_name} did not return patch tokens")


def get_vision_encoder(model_name: str, dtype: Any = torch.float32):
    """Return `(image_processor, frozen_vision_encoder)` for image-text experiments.

    We deliberately load only the image side. SigLIP2 ships a Gemma tokenizer
    that older transformers (<4.45) can't parse, and we never need it here —
    text encoding goes through the T5 encoder.
    """
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_name)
    encoder = FrozenVisionEncoder(model_name, dtype=dtype)
    return processor, encoder
