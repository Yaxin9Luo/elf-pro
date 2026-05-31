# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Repository

PyTorch port of ELF (Embedded Language Flows) — a flow-matching diffusion language model
that generates text in a frozen T5 encoder's latent space, then decodes latents to tokens
with a built-in DLM head. Targets parity with the JAX paper numbers.

## Common commands

Set up env (Python 3.10+, <3.13):

```bash
conda create -n elf python=3.10 -y && conda activate elf
pip install -r requirements.txt          # or: pip install -e .  (uses pyproject.toml)
```

Single-GPU training / eval:

```bash
bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
bash scripts/launch.sh eval  src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-owt-torch
```

Multi-GPU single-host (uses `torchrun`):

```bash
NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

Multi-host: also set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`.

Override config fields without editing YAML (repeatable):

```bash
... --config_override use_bf16=true --config_override use_compile=true
```

Force CPU (debugging): pass `--use_cpu` to `train.py` / `eval.py`.

Standalone PPL on already-generated samples (single-GPU friendly when the eval
GPU can't fit ELF + GPT-2 Large at once):

```bash
python scripts/eval_ppl.py --input outputs/<run>/<sampling_dir>/all_generated_*.jsonl \
    --batch_size 16
```

Checkpoints / datasets are auto-downloaded from HuggingFace when `--checkpoint_path`
or `data_path` is a repo id (e.g. `embedded-language-flows/ELF-B-owt-torch`,
`embedded-language-flows/openwebtext-t5`).

There are no tests, lint config, or formatter in this repo.

## Architecture

### Latent-space diffusion LM

1. **Frozen T5 encoder** (`src/modules/t5_encoder.py`) maps token ids → latents
   `x0 ∈ R^(B,S,d_model)`. Latents are normalized by `(latent_mean, latent_std)`
   from config (per-dataset, e.g. `0.0 / 0.2` for OWT-T5).
2. **ELF Transformer** (`src/modules/model.py`) is a denoising backbone that
   maps a noisy latent `z` and timestep `t` back to clean `x_pred`. It also
   carries a **decoder head** (factored unembedding `hidden → text_encoder_dim → vocab`)
   that maps latents back to token logits — same backbone, two heads.
3. Sampling: ODE/SDE rollout (`src/utils/sampling_utils.py`,
   `src/utils/generation_utils.py`) integrates `z` from t≈0 noise to t≈1, then
   the decoder head argmaxes the final latent into token ids.

ELF-B / ELF-M / ELF-L sizes are defined in `ELF_models` factory at the bottom
of `src/modules/model.py`.

### Conditioning prefix tokens

`ELF.forward` prepends three groups of learnable prefix tokens before the
transformer blocks (and strips them off afterward):

- `mode_tokens` — gated by `decoder_step_active`; signals "decoder mode" vs
  "denoiser mode" so a single backbone serves both heads.
- `t_emb_tokens` — added to a `TimestepEmbedder` projection of `t`.
- `self_cond_cfg_tokens` — added to an embedding of the **self-cond CFG scale**
  (a per-example float, sampled log-uniformly during training, fixed during eval).

Counts come from `num_model_mode_tokens / num_time_tokens / num_self_cond_cfg_tokens`.
The RoPE module is told about the prefix length via `num_empty_token` so prefix
positions don't get rotated. Conditional generation (XSum / WMT) uses an
in-sequence cond prefix instead — distinct from these prefix *tokens*.

### Training loss: per-example CE/L2 mixture

`src/train_step.py` is the single mini-batch step. The key non-obvious bit:

- Every example independently flips a coin at `decoder_prob` to pick **decoder
  (CE)** vs **denoiser (L2)** branch (per-example, not per-step).
- One forward consumes a **mixed** input: `decoder_z` rows interpolated between
  `x0` and noise at logit-normal-sampled `λ_t` for the CE branch, `denoiser_z`
  rows from `add_noise(x0, ε, t, ...)` for the L2 branch.
- Both heads run; CE/L2 per-token losses are then masked to their rows and
  combined with a single denominator. Comments in `train_step.py` explain why
  `find_unused_parameters=False` is safe under DDP — both heads stay live every
  step via the mixed input.
- Self-conditioning + SC-CFG guidance only apply on the denoiser branch; the
  self-cond half is zeroed out for decoder-mode rows.

### Self-conditioning + SC-CFG

When `self_cond_prob > 0`, the network sees its own previous prediction
concatenated to `z` (input has `2 * text_encoder_dim` channels) — see
`self_cond_proj` in `model.py`. During training, an unconditional forward is
run under `no_grad` to produce `x_pred_init`, optionally guided by an SC-CFG
scale that biases the v-target toward (cond − uncond). At sampling time the
same SC-CFG mechanic is applied (`_forward_sample_self_cond`). This is **separate
from input/condition CFG (`cfg_scale`)** which is only meaningful for conditional
tasks.

### Conditioning masks (cond seq vs. attention)

`src/utils/encoder_utils.py::build_self_attn_cond_masks` splits each sequence
into condition tokens (e.g. translation source) and target tokens. It produces
three masks used everywhere: `encoder_attention_mask` (cond can only attend to
cond, target can attend to all), `attention_mask` (validity), `cond_seq_mask`
(per-position 1=cond, 0=target). `restore_cond` / `restore_vx` snap clean cond
positions back after every denoising step so cond tokens don't drift.

### Distributed / numerics

- Init in `train.py::run_training` uses one shared seed for parameter init,
  then re-seeds **per-rank** (`config.seed + rank`) so dropout and the
  Bernoulli decoder/denoiser branch coin diverge across ranks. Comments explain
  that a shared seed makes ranks pick the same branch in lockstep and produces
  spiky decoder gradients.
- BF16 autocast is used for forward passes (`use_bf16`); output heads (final
  layer + decoder unembedding) drop back to fp32 inside `with autocast(...,
  enabled=False)` blocks in `model.py`.
- `torch.compile` is applied to the **inner** module before DDP wrapping so
  checkpoint I/O (which calls `unwrap_model` to strip `_orig_mod` and DDP
  `.module`) keeps working. `unwrap_model` is in `src/utils/train_utils.py`.
- DDP all-reduce on metrics happens only every `log_freq` steps (not every
  step) to avoid extra syncs.

### Optimizer

Default optimizer is **Muon** (`src/utils/muon_utils.py`, third-party
`muon-optimizer` package) with auxiliary AdamW for non-matmul params. Switch
with `optimizer: adamw` in YAML. Effective LR is `blr * (global_batch *
grad_accum) / 256` when `lr` is unset.

### Checkpoints

`src/utils/checkpoint_utils.py` writes a single `checkpoint_<step>` torch
file with `params`, `ema_params1`, `opt_state`, `lr_scheduler`, `step`, `epoch`,
`dropout_rng`, and (if any) `grad_accum_buffers`. Auto-resume looks at
`config.output_dir` for the latest `checkpoint_*` if `resume` is unset.
HF repo ids are accepted in place of local paths and are resolved via
`_split_hf_path` → `huggingface_hub.snapshot_download`. `hf_repo_id` in YAML
mirrors the local `output_dir` to a HuggingFace repo after each save.

### Sampling steps

`SamplingConfig.sampling_method` is `"ode"` or `"sde"`. SDE adds per-step
hybrid churn `α = 1 - γ·h`; the last step is always ODE. Multiple sampling
configs are run in sequence via `sampling_configs_path` pointing at a YAML
list — see `src/configs/sampling_configs/*.yml`. Output dir per config is
`<output_dir>/<sampling_method>-stepsN-cfgC[-sccfgS]-ts_<sched>[-gammaG]-{uncond,cond}/`.

## Configs

- Training YAMLs in `src/configs/training_configs/` set every hyperparameter
  for a (model × dataset) pair. Field names mirror `src/configs/config.py::Config`.
- Anything not in YAML falls back to the `Config` class defaults (read
  `config.py` to find the source of truth — `apply_config_overrides` does the
  type coercion for `--config_override`).
- `pad_token: "pad"` vs `"eos"` controls both pad token id and which positions
  contribute to the L2/CE loss (`loss_mask` in `train_step.py`).

## Conventions worth knowing

- `src/` is added to `sys.path` by `train.py` / `eval.py` and by
  `scripts/launch.sh` (`PYTHONPATH=$(pwd)/src`), so internal imports use
  bare module paths like `from modules.model import ...`.
- `log_for_0` (in `src/utils/logging_utils.py`) gates logging to rank 0.
- All random ops that need reproducibility take an explicit `torch.Generator`
  threaded through `state.dropout_generator` — don't replace with global
  `torch.rand*`.

## Remote Dev Machine and HOPE Jobs

Use this runbook when an agent needs to inspect the remote training repo or
submit HOPE jobs from the MacBook.

### SSH and repo paths

From the MacBook, connect with the configured SSH alias:

```bash
ssh laionface-tokenizer
```

Use this SH02 3A path as the human-facing entry point for the ELF repo:

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro
```

Important path detail: on the dev machine this 3A path resolves physically to
the SH02 `native_mm` mirror. `pwd` may show the 3A path, while `pwd -P` and
`git rev-parse --show-toplevel` may show:

```bash
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/luoyaxin03/projects/elf-pro
```

Treat these as the same mounted worktree unless a command proves otherwise.
When syncing or editing files for a HOPE job, make sure the file is present in
the path used by that job's `worker.script` and by the wrapper script's
`PROJECT_DIR`.

### HOPE login and submission

Before submitting jobs, log in with the user's DaXiang/MIS identity and confirm
the login request in DaXiang:

```bash
hope login <misid>
```

Submit from the repo's `submit/` directory with `hope run`. Do not use
`hope submit`; this HOPE CLI does not support that subcommand in this
environment.

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro/submit
hope run <target_job>.hope
```

Example:

```bash
hope run submit_elf_l_llava_1024_pipeline_8gpu.hope
```

### Pre-submit path sanity check

Before submitting a new or edited job, inspect the HOPE file and wrapper:

```bash
grep -n "worker.script" submit/<target_job>.hope
grep -n "PROJECT_DIR=" submit/run_elf_l_hope.sh submit/run_elf_l_llava_1024_pipeline_hope.sh submit/run_elf_l_llava_instruct_hope.sh
grep -n "output_dir:" src/configs/training_configs/<target_config>.yml
```

For multimodal pipeline jobs, the active wrappers are expected to point at the
3A repo path. Some older text-only wrappers and submit files still contain the
`native_mm` path; if you edit only the 3A-looking path but the selected wrapper
`cd`s into `native_mm`, the worker may run stale code or miss the new YAML.
Resolve that mismatch before submitting by syncing both views or updating the
selected wrapper/submit file consistently.
