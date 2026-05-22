# HOPE submission (Meituan Shanghai `hadoop-nlp-sh02`)

This directory has the HOPE / AFO submission files for training the ELF-L
baseline on OpenWebText on the Shanghai `shxs_training_cluster` H800-141G
queue.

| File | What it does |
| --- | --- |
| `submit_elf_l_8gpu.hope`  | 1 node × 8 H800-141G. **Wait — actually `workers = 2` in the file**, i.e. 2 nodes × 8 = 16 GPUs. Used as the current baseline run. |
| `submit_elf_l_32gpu.hope` | 4 nodes × 8 H800-141G = 32 GPUs. Scale-out variant; same recipe, ~4× less wall-clock per epoch. |
| `run_elf_l_hope.sh`       | Shared `worker.script` wrapper. Activates the `audio_jmh2_clone_yaxin` conda env, sets `PYTHONPATH` (incl. `submit/vendor` for the muon fallback), tee-s a per-worker launch log under `output_dir/launch_logs/`, ensures `muon-optimizer` is importable, then launches `torch.distributed.run` with rendezvous flags from `hope_run_torch_distribute.py`. |
| `hope_run_torch_distribute.py` | HOPE rendezvous helper. Parses `AFO_ENV_CLUSTER_SPEC` and prints `--nnodes/--nproc-per-node/--node_rank/--master_addr/--master_port` for `torch.distributed.run`. Copied verbatim from the standard `mm-pretrain-hfds` template. |
| `vendor/muon.py`          | Vendored copy of `muon-optimizer` so the worker can import `muon` even if the dolphinfs-shared env doesn't have it. `run_elf_l_hope.sh` puts `submit/vendor/` first on `PYTHONPATH`. |

The two YAML configs that go with these submission files live next to the
rest of the training configs:

- `src/configs/training_configs/train_owt_ELF-L_hope_8gpu.yml`
- `src/configs/training_configs/train_owt_ELF-L_hope_32gpu.yml`

They are derived from `train_owt_ELF-L.yml`. The diffs: `data_path`,
`encoder_model_name`, `tokenizer_name` and `output_dir` are rewritten to
**dolphinfs absolute paths** (the cluster has no internet, so HF auto-download
will not work), `use_wandb` is forced off, and `gradient_checkpointing` is on
so ELF-L (652M) fits at local batch 64.

> ⚠️ Note: there are **no** ELF-B / ELF-M hope files. We run the elf-pro
> main claim on ELF-L only.

## Two on-disk repo copies

The dev box edits + git operations happen under
`/mnt/dolphinfs/.../hadoop-nlp-sh02/native_mm/luoyaxin03/projects/elf-pro/`,
but every hope file `cd`s into the **3A mirror**:

```
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro
```

That is the path baked into `worker.script` and into `PROJECT_DIR` in
`run_elf_l_hope.sh`. It is **also** the path in `output_dir` of both YAMLs.

So whenever you change code on the dev box, mirror it to the 3A copy before
re-submitting:

```bash
rsync -av --exclude='.git' --exclude='outputs' \
    /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/luoyaxin03/projects/elf-pro/ \
    /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro/
```

## Prerequisites (one-time)

1. **Mirror code** to the 3A path (above). Repeat after every change.

2. **Mirror dataset to a path readable from the sh02 worker container.** The
   YAMLs default to:

       /mnt/dolphinfs/.../hadoop-nlp-sh02/native_mm/luoyaxin03/projects/elf-pro/data/embedded-language-flows/openwebtext-t5/main

   That directory needs 75 `.arrow` files. Pre-fetch from the dev box (which
   has internet) with `bash scripts/download_data.sh <target>` — it retries
   through the corp HTTP proxy until all 75 shards are present.

3. **Mirror the encoder.** `models/t5-small/` must contain
   `config.json`, `pytorch_model.bin`, `spiece.model`, `tokenizer.json`,
   `tokenizer_config.json`. Easiest: `huggingface-cli download t5-small
   --local-dir models/t5-small/` from the dev box.

4. **Make `outputs/` writable on the SH02 zone.** Both YAMLs write to
   `outputs/elf_l-owt-hope-8gpu/` (under the 3A mirror). Pre-create that
   directory once. The `afo.app.env.DFS_CLIENT_WRITE_ZONE = SH02` line in the
   hope file ensures writes go to the SH02 zone.

5. **WandB is off.** No login needed. If you ever flip `use_wandb=true`, the
   worker still has no internet — set `WANDB_API_KEY` and `WANDB_MODE=offline`
   so the run doesn't hang on auth.

## Submitting

From any node that has the HOPE CLI and the dolphinfs mount:

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro/submit
hope submit submit_elf_l_8gpu.hope        # 2-node baseline (see warning below)
# or
hope submit submit_elf_l_32gpu.hope       # 4-node scale-up
```

The job appears in the HOPE web UI as
`[hope/ml-easy-job]elf_pro_owt_ELF-L_8gpu@luoyaxin03` (or `..._32gpu...`).

> ⚠️ The "8gpu" file ships with `workers = 2`, i.e. it actually requests
> **2 nodes × 8 GPUs = 16 GPUs**. The filename is historical (it started life
> as a single-node 8-GPU recipe). If you want a literal 8-GPU run, edit
> `workers = 1` first; otherwise expect 16-GPU behavior with `local_batch =
> 512 / 16 = 32`.

## What `run_elf_l_hope.sh` does, step by step

1. `source` the shared conda env at
   `/mnt/dolphinfs/.../hadoop-hldy-nlp/3A/multimodal/yaxinluo/envs/audio_jmh2_clone_yaxin`.
   This is python 3.9 with torch / transformers / datasets pre-installed.
2. `cd` to the 3A repo mirror.
3. Set `PYTHONPATH=submit/vendor:src:$PYTHONPATH` so the vendored `muon.py`
   is importable even when the env's `muon-optimizer` is missing.
4. Read `output_dir` out of the YAML (via a tiny inline `python3 - <<PY`
   block) and `mkdir -p` `output_dir/launch_logs/`. Tee all stdout/stderr to
   `worker_<task_id>_<host>_<ts>.log` in there.
5. Print a probe block: python version, `import torch / transformers /
   datasets / muon` results, CUDA visibility. This is where most failures
   show up — read this block first when a job fails.
6. `ensure_muon`: if `import muon` still fails (vendor path lost,
   wrong `PYTHONPATH`, etc.), `pip install --no-cache-dir
   muon-optimizer==0.1.0`. A directory-based lock at `.muon_install.lock`
   serializes installs across the workers of one job.
7. `eval "HOPE_TRACKING_RANK=0 python3 -m torch.distributed.run \
   $(python3 submit/hope_run_torch_distribute.py) src/train.py --config
   <yml>"`. The helper resolves `--nnodes / --nproc-per-node / --node_rank
   / --master_addr / --master_port` from `AFO_ENV_CLUSTER_SPEC`.

If you need to override a config field at submit time (without editing the
YAML), append `--config_override key=value` to the `python3 -m
torch.distributed.run` line in `run_elf_l_hope.sh` — e.g. to flip wandb on:
`... src/train.py --config <yml> --config_override use_wandb=false`.

## Resource & hyperparameter summary

| Knob | 8gpu file (effectively 16 GPUs) | 32gpu file | Notes |
| --- | --- | --- | --- |
| Nodes (`workers`) | 2 | 4 | |
| GPUs / node (`worker.gcoresh800-141g`) | 8 | 8 | H800-141G |
| Total GPUs | 16 | 32 | |
| `global_batch_size` | 512 | 512 | Paper recipe; do not change without rescaling LR/warmup. |
| `grad_accum_steps` | 1 | 1 | Bump to 2 if 8gpu OOMs. |
| Per-device batch | 32 | 16 | `global_batch_size / world` (computed by `train.py`). |
| `epochs` | 5 | 5 | |
| `blr` | 1e-3 | 1e-3 | Effective LR is `blr * (GB * grad_accum) / 256 = 2e-3`. |
| `optimizer` | muon | muon | Auxiliary AdamW for non-matmul params. |
| `gradient_checkpointing` | true | true | Required for ELF-L at local_batch ≥ 32 to fit on H800-141G. |
| Wall-clock estimate | ~2 h / epoch | ~1 h / epoch | Linearly scaled from the README's 8× H200 estimate; H800-141G is comparable. Full 5-epoch baseline ~10 h on 16 GPUs / ~5 h on 32. |

## Adjusting GPU count

If you want a different topology than the two provided files:

- Single-node, fewer GPUs: edit `workers = 1` and `worker.gcoresh800-141g =
  N` (e.g. `4`) in the 8gpu hope file. `hope_run_torch_distribute.py` reads
  `nvidia-smi --list-gpus` at run time, so `nproc-per-node` follows
  automatically. Also update `global_batch_size` in the YAML so the
  per-device batch stays sensible.
- More nodes: bump `workers` in the 32gpu hope file (e.g. to `8` for 64
  GPUs). Same comment about `global_batch_size`.
- Different queue (e.g. H800-80G or A100): change `queue` and the
  `worker.gcoresh800-141g` resource key (HOPE uses the resource name to pick
  the GPU type — see the templates under
  `mm-pretrain-hfds/pretrain/template.hl.hope`).

## Monitoring

- **HOPE web UI / `hope log`** — stdout + stderr per worker. Useful for
  catching launch errors (CUDA OOM, dataset not found, NCCL hang at init).
- **Per-worker launch logs** — `outputs/<run>/launch_logs/worker_<task_id>_<host>_<ts>.log`,
  written by `run_elf_l_hope.sh` via `tee`. Same content as HOPE web UI but
  persisted on dolphinfs after the container exits.
- **WandB** — disabled (`use_wandb: false`). If you flip it on, the worker
  has no internet, so set `WANDB_MODE=offline` and rsync `wandb/` after the
  job.
- **Checkpoints** — `output_dir` on dolphinfs holds `checkpoint_<step>` files
  every `save_freq` epochs. Resume is automatic from the latest checkpoint
  if you re-submit the same hope file (controlled by `auto_resume` logic in
  `src/utils/checkpoint_utils.py`).
- **Tensorboard** — disabled in the hope file (`with.tensor.board = false`).

## Known failure modes (as of 2026-05-22)

- **`NameResolutionError` for `huggingface.co`** during dataset / encoder
  load. Root cause: a YAML field still pointing at a HF repo id, so
  `datasets` / `transformers` does a HEAD check against the hub. Fix: make
  sure all of `data_path`, `encoder_model_name`, `tokenizer_name` are
  dolphinfs absolute paths. The two `*_hope_*.yml` configs already have this.
- **`muon` import failure** despite the vendor fallback. Usually means
  `PYTHONPATH` got clobbered by an env hook. Add a `python3 -c "import sys;
  print(sys.path)"` to the probe block in `run_elf_l_hope.sh` to confirm.
- **OOM at local_batch=64** (would happen on a literal 1-node-8-GPU run).
  Either bump `grad_accum_steps` to 2 or run the file as 2-node-16-GPU
  (current default).

## Differences from `submit_face_enhance_luoyaxin03.hope`

For reference, the things this submission deliberately drops vs. the
face_enhance template the wrapper was originally cloned from:

- **No `pre_load.py` step.** That helper exists to warm
  `transformers_modules` cache for `AutoModelForCausalLM.from_pretrained` of
  a custom remote-code model. ELF-L's only HF-loaded weights are the
  `t5-small` encoder, which has no remote code.
- **No `env_configure.sh`.** The face_enhance variant only sets
  `CUDA_DEVICE_MAX_CONNECTIONS=1` + `NCCL_DEBUG=INFO` (rest is commented
  out); ELF-L is the target size and the defaults are fine. Add an
  `env_configure.sh` here if you start hitting NCCL issues at 32 GPUs.
- **No project-specific conda env.** The shared
  `audio_jmh2_clone_yaxin` env is enough; we don't `conda create` per
  project.
