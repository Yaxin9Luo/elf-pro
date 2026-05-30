#!/usr/bin/env python
"""Frozen T5 text embedder, wrapping `transformers.T5EncoderModel`."""

from typing import Any, Optional

import torch
import torch.nn as nn

from utils.logging_utils import log_for_0


class T5EncoderConfig:
    """Configuration class for T5Encoder."""

    def __init__(self, model_name: str, dtype: Any):
        self.model_name = model_name
        self.dtype = dtype
        self.vocab_size: int = 0
        self.d_model: int = 0
        self.d_kv: int = 0
        self.d_ff: int = 0
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.is_gated_act: bool = False

    @classmethod
    def from_pretrained(cls, model_name: str, dtype: Any = torch.float32) -> "T5EncoderConfig":
        cfg = cls(model_name, dtype)
        defaults = {
            "t5-small": dict(vocab_size=32128, d_model=512, d_kv=64, d_ff=2048,
                             num_layers=6, num_heads=8, is_gated_act=False),
            "t5-base":  dict(vocab_size=32128, d_model=768, d_kv=64, d_ff=3072,
                             num_layers=12, num_heads=12, is_gated_act=False),
            "t5-large": dict(vocab_size=32128, d_model=1024, d_kv=64, d_ff=4096,
                             num_layers=24, num_heads=16, is_gated_act=False),
        }
        if model_name in defaults:
            for k, v in defaults[model_name].items():
                setattr(cfg, k, v)
        return cfg


class T5Encoder(nn.Module):
    """T5 encoder used as a frozen text embedder."""

    def __init__(self, config: T5EncoderConfig, *, pretrained: bool = True):
        super().__init__()
        from transformers import T5EncoderModel, T5Config

        if pretrained:
            self.model = T5EncoderModel.from_pretrained(config.model_name)
        else:
            hf_config = T5Config.from_pretrained(config.model_name)
            self.model = T5EncoderModel(hf_config)

        hf = self.model.config
        config.vocab_size = hf.vocab_size
        config.d_model = hf.d_model
        config.d_kv = hf.d_kv
        config.d_ff = hf.d_ff
        config.num_layers = hf.num_layers
        config.num_heads = hf.num_heads
        config.is_gated_act = bool(getattr(hf, "is_gated_act", False))
        self.config = config

    def _forward_with_3d_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the HF T5 encoder with an asymmetric self-attention mask."""
        encoder = self.model.encoder
        input_shape = input_ids.size()
        if attention_mask.shape != (input_shape[0], input_shape[1], input_shape[1]):
            raise ValueError(
                "3D T5 encoder attention_mask must have shape "
                f"(batch, seq, seq), got {tuple(attention_mask.shape)} for input_ids {tuple(input_shape)}"
            )

        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = encoder.embed_tokens(input_ids)
        seq_length = input_shape[1]
        cache_position = torch.arange(seq_length, device=inputs_embeds.device)

        additive_mask = attention_mask[:, None, :, :].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        additive_mask = (1.0 - additive_mask) * torch.finfo(inputs_embeds.dtype).min

        head_mask = encoder.get_head_mask(None, encoder.config.num_layers)
        hidden_states = encoder.dropout(inputs_embeds)
        position_bias = None

        for i, layer_module in enumerate(encoder.block):
            layer_outputs = layer_module(
                hidden_states,
                additive_mask,
                position_bias,
                None,
                None,
                None,
                layer_head_mask=head_mask[i],
                cross_attn_layer_head_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

        hidden_states = encoder.final_layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        use_3d_mask = False
        if attention_mask is not None and attention_mask.ndim == 3:
            first_row = attention_mask[:, :1, :]
            if torch.all(attention_mask == first_row):
                attention_mask = first_row.squeeze(1)
            else:
                use_3d_mask = True
        was_training = self.model.training
        if deterministic:
            self.model.eval()
        try:
            if use_3d_mask:
                return self._forward_with_3d_attention_mask(input_ids=input_ids, attention_mask=attention_mask)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            if not deterministic and was_training:
                self.model.train()
        return out.last_hidden_state


def get_encoder(model_name: str, dtype: Any):
    """Return `(config, model)`. Weights are downloaded on first use."""
    log_for_0(f"Loading T5 Encoder: {model_name}...")
    config = T5EncoderConfig.from_pretrained(model_name, dtype=dtype)
    model = T5Encoder(config, pretrained=True)
    if dtype is not None:
        model = model.to(dtype)
    return config, model
