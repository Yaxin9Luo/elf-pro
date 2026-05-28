import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.encoder_utils import build_self_attn_cond_masks
from utils.logging_utils import log_for_0


def is_image_text_config(config) -> bool:
    return getattr(config, "data_modality", "text") == "image_text"


def _process_count() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _process_index() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def get_pad_token_id(tokenizer, pad_token: str = "pad") -> int:
    """Resolve the token id used for padding, optionally using EOS as pad."""
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
    return token_id


def prepare_batch(batch: Dict, config, generator: torch.Generator) -> Dict:
    """Convert numpy batch to torch tensors and sample label-drop decisions."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            result[k] = v
        else:
            result[k] = v

    batch_size = result["input_ids"].shape[0]
    label_drop_mask = torch.zeros((batch_size,), dtype=torch.bool)
    if config.label_drop_prob > 0:
        u = torch.rand((batch_size,), generator=generator)
        label_drop_mask = u < config.label_drop_prob
    result["label_drop_mask"] = label_drop_mask
    return result


def pad_and_truncate(ids_list, target_len, pad_token_id):
    """Pad or truncate sequences to target_len, return stacked array and lengths."""
    padded, lengths = [], []
    for ids in ids_list:
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate([ids, np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)])
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


def _clean_text(text: str) -> str:
    return str(text or "").replace("<image>", "").strip()


def _extract_turn_text(turn) -> str:
    value = turn.get("value", turn.get("content", ""))
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text"):
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        value = "\n".join(parts)
    return _clean_text(value)


def _turn_role(turn) -> str:
    return str(turn.get("from", turn.get("role", ""))).lower()


def _parse_image_text_record(record, index: int, train_stage: str, image_root: Optional[str], base_dir: str):
    image = record.get("image", record.get("image_path", record.get("image_file")))
    if isinstance(image, (list, tuple)):
        return None
    if not image:
        return None
    image_path = str(image)
    if not os.path.isabs(image_path):
        root = image_root or base_dir
        image_path = os.path.join(root, image_path)

    prompt, target = None, None
    conversations = record.get("conversations", record.get("messages"))
    if conversations:
        assistant_turns = [
            (i, turn) for i, turn in enumerate(conversations)
            if _turn_role(turn) in ("assistant", "gpt")
        ]
        if not assistant_turns:
            return None
        if train_stage == "vision_warmup":
            target = _extract_turn_text(assistant_turns[0][1])
            prompt = "Describe the image."
        else:
            target_idx, target_turn = assistant_turns[-1]
            target = _extract_turn_text(target_turn)
            user_parts = [
                _extract_turn_text(turn)
                for turn in conversations[:target_idx]
                if _turn_role(turn) in ("human", "user")
            ]
            prompt = "\n".join(part for part in user_parts if part).strip()
            if not prompt:
                prompt = "Describe the image."
    else:
        prompt = _clean_text(record.get("input", record.get("prompt", "Describe the image.")))
        target = _clean_text(
            record.get("target", record.get("output", record.get("caption", record.get("text", ""))))
        )
        if train_stage == "vision_warmup":
            prompt = "Describe the image."

    if not target:
        return None
    return {
        "index": index,
        "image": image,
        "image_path": image_path,
        "input": prompt,
        "target": target,
    }


def _load_json_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "annotations", "instances"):
            if isinstance(data.get(key), list):
                return data[key]
    return data


def load_image_text_dataset(path: str, train_stage: str = "mm_instruct", image_root: Optional[str] = None):
    """Load LLaVA-style image-text records from JSON/JSONL or a HF dataset path."""
    base_dir = os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else os.getcwd()
    if os.path.isfile(path) and path.endswith((".json", ".jsonl")):
        records = _load_json_records(path)
    else:
        ds = load_dataset_split(path)
        records = [ds[i] for i in range(len(ds))]
    if not isinstance(records, list):
        raise ValueError(f"Expected image-text dataset {path!r} to contain a list of records")

    examples = []
    for i, record in enumerate(records):
        parsed = _parse_image_text_record(record, i, train_stage, image_root, base_dir)
        if parsed is not None:
            examples.append(parsed)
    log_for_0(f"Loaded {len(examples)} image-text examples from {path}")
    return examples


def _pad_1d(ids, target_len: int, pad_token_id: int):
    ids = np.array(ids[:target_len], dtype=np.int64)
    length = int(min(len(ids), target_len))
    if length < target_len:
        ids = np.concatenate([ids, np.full(target_len - length, pad_token_id, dtype=np.int64)])
    return ids, length


def _collate_image_text(
    batch_list,
    tokenizer,
    vision_processor,
    max_seq_length: int,
    pad_token_id: int,
    max_prompt_length: int,
    num_visual_tokens: int,
):
    if tokenizer is None:
        raise ValueError("image_text dataloader requires a tokenizer")
    if vision_processor is None:
        raise ValueError("image_text dataloader requires a vision_processor")
    if num_visual_tokens <= 0:
        raise ValueError("num_visual_tokens must be positive for image_text data")
    if max_seq_length <= num_visual_tokens:
        raise ValueError("max_length must be larger than num_visual_tokens")

    from PIL import Image

    max_prompt_budget = max(0, max_seq_length - num_visual_tokens - 1)
    prompt_width = max(1, min(max_prompt_length or max_prompt_budget, max_prompt_budget))
    target_width = max(1, max_seq_length - num_visual_tokens)

    prompt_ids, prompt_lens = [], []
    target_ids, target_lens = [], []
    final_ids = np.full((len(batch_list), max_seq_length), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((len(batch_list), max_seq_length), dtype=np.float32)
    cond_seq_mask = np.zeros((len(batch_list), max_seq_length), dtype=np.float32)

    images = []
    for i, item in enumerate(batch_list):
        prompt = _clean_text(item.get("input", ""))
        target = _clean_text(item.get("target", ""))
        p_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        t_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
        p_tokens = p_tokens[:prompt_width]
        available_target = max(1, max_seq_length - num_visual_tokens - len(p_tokens))
        t_tokens = t_tokens[:available_target]

        p_padded, p_len = _pad_1d(p_tokens, prompt_width, pad_token_id)
        t_padded, t_len = _pad_1d(t_tokens, target_width, pad_token_id)
        prompt_ids.append(p_padded)
        target_ids.append(t_padded)
        prompt_lens.append(p_len)
        target_lens.append(t_len)

        cond_len = num_visual_tokens + p_len
        total_len = min(max_seq_length, cond_len + t_len)
        attention_mask[i, :total_len] = 1.0
        cond_seq_mask[i, :cond_len] = 1.0
        if t_len > 0 and cond_len < max_seq_length:
            final_ids[i, cond_len:total_len] = np.array(t_tokens[:total_len - cond_len], dtype=np.int64)

        image_path = item["image_path"]
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))

    image_inputs = vision_processor(images=images, return_tensors="pt")
    prompt_arr = np.stack(prompt_ids)
    target_arr = np.stack(target_ids)
    prompt_mask = np.zeros_like(prompt_arr, dtype=np.float32)
    target_mask = np.zeros_like(target_arr, dtype=np.float32)
    for i, (p_len, t_len) in enumerate(zip(prompt_lens, target_lens)):
        prompt_mask[i, :p_len] = 1.0
        target_mask[i, :t_len] = 1.0

    result = {
        "input_ids": final_ids,
        "prompt_input_ids": prompt_arr,
        "prompt_attention_mask": prompt_mask,
        "target_input_ids": target_arr,
        "target_attention_mask": target_mask,
        "prompt_lens": np.array(prompt_lens, dtype=np.int64),
        "target_lens": np.array(target_lens, dtype=np.int64),
        "attention_mask": attention_mask,
        "cond_seq_mask": cond_seq_mask,
        "encoder_attention_mask": attention_mask,
    }
    for key, value in image_inputs.items():
        result[key] = value
    for key in ("index", "input", "target", "image", "image_path"):
        if key in batch_list[0]:
            result[key] = [item[key] for item in batch_list]
    return result


def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    max_seq_length: int = 512,
    pad_token_id: int = 0,
    max_input_seq_length: Optional[int] = None,
    distributed: bool = True,
    tokenizer=None,
    vision_processor=None,
    data_modality: str = "text",
    train_stage: str = "text",
    max_prompt_length: int = 128,
    num_visual_tokens: int = 0,
):
    """Create a DataLoader."""

    def collate_fn(batch_list):
        if data_modality == "image_text" or "image_path" in batch_list[0]:
            return _collate_image_text(
                batch_list=batch_list,
                tokenizer=tokenizer,
                vision_processor=vision_processor,
                max_seq_length=max_seq_length,
                pad_token_id=pad_token_id,
                max_prompt_length=max_prompt_length,
                num_visual_tokens=num_visual_tokens,
            )

        input_ids_list = [np.array(item["input_ids"]) for item in batch_list]

        has_condition = "condition_input_ids" in batch_list[0]
        if has_condition:
            seq_list, cond_lens = [], []
            for item in batch_list:
                cond = np.array(item["condition_input_ids"])[:max_input_seq_length]
                inp = np.array(item["input_ids"])
                seq_list.append(np.concatenate([cond, inp]))
                cond_lens.append(len(cond))
            cond_lens = np.array(cond_lens)
        else:
            seq_list = input_ids_list
            cond_lens = np.zeros(len(input_ids_list), dtype=np.int32)

        ids, total_lens = pad_and_truncate(seq_list, max_seq_length, pad_token_id)
        pos = np.arange(max_seq_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        encoder_attn, attn, pred = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
        if not has_condition:
            encoder_attn = attn
        result = {
            "input_ids": ids,
            "encoder_attention_mask": encoder_attn,
            "attention_mask": attn,
            "cond_seq_mask": pred,
        }
        for key in ("index", "input", "target"):
            if key in batch_list[0]:
                result[key] = [item[key] for item in batch_list]
        return result

    common = dict(
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
        drop_last=drop_last, persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=_process_count(), rank=_process_index(),
            shuffle=shuffle, drop_last=drop_last,
        )
        return DataLoader(dataset, sampler=sampler, **common)
    return DataLoader(dataset, shuffle=shuffle, **common)


def load_jsonl_dataset(path, tokenizer, input_key="input", output_key="output"):
    """Load a JSONL eval set (one `{input, output}` example per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "index": i,
                "input": data[input_key],
                "target": data[output_key],
                "condition_input_ids": tokenizer(data[input_key], add_special_tokens=False)["input_ids"],
                "input_ids": tokenizer(data[output_key], add_special_tokens=False)["input_ids"],
            })
    return examples


# ============================================
# Dataset loading
# ============================================

def _looks_like_save_to_disk_arrow(ds) -> bool:
    """Detect HF datasets uploaded via `save_to_disk` (returns 1-row of metadata)."""
    return (
        len(ds) == 1
        and any(c.startswith("_") for c in ds.column_names)
        and not any(not c.startswith("_") for c in ds.column_names)
    )


def load_dataset_split(path: str, dataset_cache_dir=None):
    """Load a dataset.

    Order of attempts:
      1. If `path` is a local dir of `.arrow` shards, use our minimal
         pyarrow-backed reader (works around `datasets<3` not understanding the
         newer `_type: "List"` features metadata).
      2. Try HuggingFace Hub.
      3. Fall back to `load_from_disk`.
    """
    import os
    import glob
    from datasets import DatasetDict, load_dataset as hf_load_dataset, load_from_disk

    if os.path.isdir(path) and glob.glob(os.path.join(path, "*.arrow")):
        from utils.local_arrow_dataset import LocalArrowDataset
        log_for_0(f"Loading local arrow shards from {path} (LocalArrowDataset)")
        return LocalArrowDataset(path)

    ds = None
    try:
        ds = hf_load_dataset(path, cache_dir=dataset_cache_dir)
    except Exception:
        ds = load_from_disk(path)

    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        if len(splits) != 1:
            raise ValueError(f"Expected dataset at {path!r} to have a single split, got {splits}.")
        ds = ds[splits[0]]

    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download
        log_for_0(
            f"Dataset at {path!r} looks like a save_to_disk-format HF repo; "
            f"re-downloading via snapshot_download + load_from_disk."
        )
        local_dir = snapshot_download(repo_id=path, repo_type="dataset", cache_dir=dataset_cache_dir)
        ds = load_from_disk(local_dir)
        if isinstance(ds, DatasetDict):
            splits = list(ds.keys())
            if len(splits) != 1:
                raise ValueError(f"Expected dataset at {path!r} to have a single split, got {splits}.")
            ds = ds[splits[0]]

    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_dataset(config, dataset_cache_dir=None):
    """Resolve config.data_path / config.eval_data_path into train/eval datasets."""
    if is_image_text_config(config):
        log_for_0(f"Loading image-text dataset from {config.data_path}...")
        train_dataset = load_image_text_dataset(
            config.data_path,
            train_stage=getattr(config, "train_stage", "mm_instruct"),
            image_root=getattr(config, "image_root", None),
        )
        log_for_0(f"Train size: {len(train_dataset)}")
        eval_dataset = None
        if config.eval_data_path:
            eval_dataset = load_image_text_dataset(
                config.eval_data_path,
                train_stage=getattr(config, "train_stage", "mm_instruct"),
                image_root=getattr(config, "image_root", None),
            )
            log_for_0(f"Eval size: {len(eval_dataset)}")
        else:
            log_for_0("No eval dataset")
        return train_dataset, eval_dataset

    log_for_0(f"Loading dataset from {config.data_path}...")
    train_dataset = load_dataset_split(config.data_path, dataset_cache_dir)
    log_for_0(f"Train size: {len(train_dataset)}")

    eval_dataset = None
    if config.eval_data_path:
        eval_dataset = load_dataset_split(config.eval_data_path, dataset_cache_dir)
        log_for_0(f"Eval size: {len(eval_dataset)}")
    else:
        log_for_0("No eval dataset")
    return train_dataset, eval_dataset
