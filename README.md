# elf-pro

PyTorch port of [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938),
re-targeted as the baseline for the elf-pro paper track. This README documents
how **we** (`luoyaxin03` on Meituan dolphinfs) actually run the code — local
sanity checks on a small dev box, full training on the Shanghai HOPE H800-141G
queue. It is not the upstream paper-port README; expect mentions of dolphinfs
absolute paths, the shared `audio_jmh2_clone_yaxin` conda env, the
`hadoop-nlp-sh02` cluster, etc.

## Repo layout

```
elf-pro/
├── src/                       # all training / eval / model code
├── scripts/launch.sh          # local single-host launcher (NGPU=N for torchrun)
├── scripts/eval_ppl.py        # offline GPT-2-Large PPL on generated jsonl
├── scripts/download_data.sh   # one-shot OWT-T5 mirror downloader
├── data/                      # local mirrors of HF datasets (manually pre-downloaded)
│   └── embedded-language-flows/openwebtext-t5/main/
├── models/                    # local mirrors of HF models
│   └── t5-small/
├── outputs/                   # all checkpoints + launch logs (NOT under .git)
├── submit/                    # HOPE submission files (see submit/README.md)
└── docs/                      # research plan + code review
```

There are two on-disk copies of this repo on dolphinfs:

- **dev / source of truth** — `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/luoyaxin03/projects/elf-pro/`
  This is where you edit code and run local smoke tests.
- **HOPE worker mount** — `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro/`
  HOPE submission files `cd` into this path. Keep it in sync with the dev copy
  before every submit (see `submit/README.md`).

## Environment

We do **not** create a fresh conda env. The dev box and the HOPE worker both
use a pre-baked shared env on dolphinfs:

```
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/envs/audio_jmh2_clone_yaxin
```

Activate it from any node that has `/usr/local/conda` and the dolphinfs mount:

```bash
source /usr/local/conda/bin/activate \
    /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/envs/audio_jmh2_clone_yaxin
```

This env has Python 3.9 + torch 2.x + transformers + datasets pre-installed.
The only package that is occasionally missing is `muon-optimizer`; both the
local launcher and `submit/run_elf_l_hope.sh` install it on demand and also
ship a vendored copy at `submit/vendor/muon.py` as a fallback (it is added to
`PYTHONPATH` by `run_elf_l_hope.sh`).

`requirements.txt` / `pyproject.toml` are kept up-to-date for reference and
for clean-room rebuilds, but you usually do **not** need to run
`pip install -r requirements.txt` — the shared env is already populated.

WandB is **off** in every config we run on HOPE (`use_wandb: false`), because
the worker container has no outbound internet and the HOPE web UI already
covers job stdout / stderr. Locally you can flip it on with
`--config_override use_wandb=true` if you've got `wandb login` set up.

## Data & model mirrors (offline-first)

The HOPE worker container can **not** resolve `huggingface.co`. So:

- `data_path` and `encoder_model_name` / `tokenizer_name` in the YAML configs
  must be **dolphinfs absolute paths** when running on HOPE.
- The two mirrors we currently have:

  | Asset | Local path |
  | --- | --- |
  | OWT-T5 dataset (75 arrow shards) | `data/embedded-language-flows/openwebtext-t5/main/` |
  | t5-small encoder + tokenizer | `models/t5-small/` |

If those directories are empty (e.g. fresh clone of the repo on a new
dolphinfs zone), pre-fetch the dataset with:

```bash
bash scripts/download_data.sh /mnt/dolphinfs/.../elf-pro/data/embedded-language-flows/openwebtext-t5/main
```

This script retries up to 20× through the corp HTTP proxy
(`http://10.70.16.106:3128`) and verifies the 75-arrow file count. Run it from
a node that has internet (the dev box, not a HOPE worker).

For `models/t5-small/`, the simplest path is one-off `huggingface-cli download
t5-small --local-dir models/t5-small/` from the dev box, then commit the
folder lives under dolphinfs (it's ~250 MB).

## Local smoke test (dev box)

We have a single, dedicated smoke config that runs ELF-L on 2 local GPUs for
~100 steps. Goal: verify env / dataloader / forward+backward — nothing about
quality.

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/luoyaxin03/projects/elf-pro
source /usr/local/conda/bin/activate /mnt/dolphinfs/.../audio_jmh2_clone_yaxin

NGPU=2 bash scripts/launch.sh train src/configs/training_configs/smoke_local_2gpu.yml
```

`scripts/launch.sh` adds `src/` to `PYTHONPATH` and falls back to bare
`python` for `NGPU=1` (so you can attach `pdb`); for `NGPU>=2` or `NNODES>=2`
it uses `torchrun`. Output goes to `outputs/smoke_local_2gpu_ELF-L/`. Kill it
with Ctrl-C after you've seen 50–100 steps.

If you want to override a single config field without editing the YAML:

```bash
NGPU=2 bash scripts/launch.sh train src/configs/training_configs/smoke_local_2gpu.yml \
    --config_override use_compile=true \
    --config_override log_freq=5
```

CPU-only debugging: pass `--use_cpu` (entry-point flag, not a config override).

## HOPE training (full ELF-L baseline)

This is the actual training surface. Single-node 8gpu and 4-node 32gpu hope
files live under `submit/`:

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro/submit
hope submit submit_elf_l_8gpu.hope     # 1 node × 8 H800-141G  (≈4 h/epoch)
hope submit submit_elf_l_32gpu.hope    # 4 nodes × 8 = 32 GPUs (≈1 h/epoch)
```

The matching YAMLs are `train_owt_ELF-L_hope_8gpu.yml` and
`train_owt_ELF-L_hope_32gpu.yml`; both already have `data_path` / encoder /
tokenizer / `output_dir` rewritten to dolphinfs paths.

We do **not** have submit files for ELF-B / ELF-M. The full elf-pro main
claim runs on ELF-L only.

Before submitting (one-time), confirm:

1. **Code mirror**: `3A/multimodal/luoyaxin03/projects/elf-pro/` is in sync
   with `native_mm/...` (rsync from the dev box, exclude `.git` and `outputs/`).
2. **Dataset mirror**: `data/embedded-language-flows/openwebtext-t5/main/`
   has 75 `.arrow` files. The HOPE worker reads them via `LocalArrowDataset`,
   so no internet is needed.
3. **Encoder mirror**: `models/t5-small/` has `pytorch_model.bin`,
   `spiece.model`, and friends.
4. **`outputs/elf_l-owt-hope-8gpu/` writable** under the 3A path. The hope
   file sets `DFS_CLIENT_WRITE_ZONE=SH02` so writes hit the SH02 zone.

Job appears in the HOPE web UI as
`[hope/ml-easy-job]elf_pro_owt_ELF-L_8gpu@luoyaxin03` (or `..._32gpu...`).
Per-worker stdout is also tee-d to
`outputs/<run>/launch_logs/worker_<task_id>_<host>_<ts>.log`.

For everything else (resource knobs, monitoring, "what does the wrapper
script do"), see `submit/README.md`.

## Evaluation

For paper-track evaluation we need to **train** the checkpoints ourselves
(internet-locked clusters can't pull the upstream HF checkpoints
`embedded-language-flows/ELF-*-torch`). Once we have a `checkpoint_<step>`
file in `outputs/<run>/`:

```bash
# eval against our own trained checkpoint, on the dev box
NGPU=2 bash scripts/launch.sh eval \
    src/configs/training_configs/train_owt_ELF-L_hope_8gpu.yml \
    --checkpoint_path outputs/elf_l-owt-hope-8gpu/checkpoint_<step> \
    --config_override use_bf16=true --config_override use_compile=true
```

Eval writes generated samples to
`outputs/<run>/<sampling_method>-stepsN-cfgC[-sccfgS]-ts_<sched>[-gammaG]-{uncond,cond}/all_generated_*.jsonl`,
one file per rank. To get GPT-2-Large PPL afterwards (single-GPU friendly,
useful when ELF + GPT-2-Large together don't fit on the eval GPU):

```bash
python scripts/eval_ppl.py \
    --input outputs/<run>/<sampling_dir>/all_generated_*.jsonl \
    --batch_size 16
```

## Reference numbers (paper, for comparison)

These are the JAX paper numbers we are trying to reproduce. We have **not**
hit them yet — see `docs/elf_pro_research_plan.html` and the project memory
for current status.

| Model | Sampling | Gen. PPL ↓ | Entropy ↑ |
| --- | --- | --- | --- |
| ELF-B (105M) | 32-step SDE | 24.1 | 5.15 |
| ELF-M (342M) | 64-step SDE | 21.7 | 5.18 |
| ELF-L (652M) | 64-step SDE | 23.3 | 5.28 |

Default sampling configs:
`src/configs/sampling_configs/uncond_sampling_configs.yml` (SC-CFG=3, γ=1.5
@32-step / 1.0 @64-step), `cond_sampling_configs.yml` (64-step ODE, CFG=2,
SC-CFG=1).

## Conventions worth knowing

- `src/` is added to `sys.path` by `train.py` / `eval.py` and by
  `scripts/launch.sh` (`PYTHONPATH=$(pwd)/src`). Internal imports use bare
  paths like `from modules.model import ...`.
- `log_for_0` (in `src/utils/logging_utils.py`) gates logging to rank 0.
- All random ops that need reproducibility take an explicit `torch.Generator`
  threaded through `state.dropout_generator` — don't replace with global
  `torch.rand*`.
- `unwrap_model` in `src/utils/train_utils.py` strips `_orig_mod` (compile)
  and DDP `.module`. Keep it on the path before any checkpoint I/O.
- Effective LR is computed inside `train.py` as `blr * (global_batch *
  grad_accum) / 256`; YAML `lr` overrides this if set.

## See also

- `submit/README.md` — HOPE submission deep-dive (resource knobs, monitoring,
  rendezvous helper, wrapper script).
- `CLAUDE.md` — architecture notes for Claude Code agents.
- `docs/elf_pro_research_plan.html` — the elf-pro paper roadmap.
- `docs/elf_code_review.html` — improvement-point catalogue (P/S/L tags).
