# Stage 2 HOPE submission guide

This document is a self-contained handoff for an agent who has not seen the
prior session. Goal: take the stage-2 SigLIP2 + ELF-L instruct-tuning run
from the dev box (`hadoop-hldy-nlp` mount) to the Shanghai sh02 cluster
(`hadoop-nlp-sh02` mount) and submit it via HOPE.

You are running on the dev box at:
`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/projects/elf-pro`.

The cluster mirror lives at:
`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro`
(both mounts are visible from the dev box; you don't need to ssh anywhere).

## What's already set up

These files exist on the dev box and you should NOT recreate them:

- `submit/submit_elf_l_llava_instruct_32gpu.hope` — HOPE job spec (4 nodes × 8 H800-141G).
- `submit/run_elf_l_llava_instruct_hope.sh` — `worker.script` wrapper. Activates the
  shared conda env, sets HF offline + NCCL knobs, runs `submit/hope_run_torch_distribute.py`,
  invokes `torch.distributed.run src/train.py --config <yml>`.
- `submit/hope_run_torch_distribute.py` — generic HOPE rendezvous helper.
- `submit/sync_to_sh02.sh` — rsync helper. Read it first (`cat`) so you understand
  what it copies and where.
- `src/configs/training_configs/train_llava_siglip2_instruct_ELF-L_hope_32gpu.yml` —
  the cluster YAML. **All paths in this file are absolute under sh02.**
- `src/configs/training_configs/train_llava_siglip2_instruct_ELF-L_local.yml` —
  the dev-box YAML. **Don't submit this one to HOPE** — its paths point at hldy.
- Local stage-1 ckpt at `outputs/elf_l-llava-siglip2-warmup-local/checkpoint_4360`
  (this is the projector-warmed weights; stage 2 inits from it).

## What you need to do, in order

### 1. Confirm data is ready

The full LLaVA-1.5 mix665k image set must be on the dev box before sync. Verify:

```bash
DATA=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/data
ls "$DATA/playground/data/llava_v1_5_mix665k.json" || echo MISSING
for d in coco/train2017 gqa/images ocr_vqa/images textvqa/train_images vg/VG_100K vg/VG_100K_2; do
    n=$(ls "$DATA/playground/data/$d" 2>/dev/null | wc -l)
    echo "$d: $n files"
done
```

Healthy counts (approximate):

| dir | count |
| --- | --- |
| `coco/train2017` | ≥ 118000 |
| `gqa/images` | ≥ 148000 |
| `ocr_vqa/images` | ≥ 207000 |
| `textvqa/train_images` | ≥ 25000 |
| `vg/VG_100K` | ≥ 64000 |
| `vg/VG_100K_2` | ≥ 43000 |

If COCO or GQA is still missing or partial, **stop here** and ask the user.
Their downloads are tracked in `data/_logs/03_coco.log` and `data/_logs/04_gqa.log`,
and the proxy these go through is
`http_proxy=http://10.70.16.106:3128` (only needed for cocodataset.org and
cs.stanford.edu — HF works without it). Resume with `wget -c` against the same zip
files in `data/_zips/`.

### 2. Confirm stage-1 ckpt is present

```bash
ls -la /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/projects/elf-pro/outputs/elf_l-llava-siglip2-warmup-local/checkpoint_*
```

You should see at least one `checkpoint_<N>` file (~5 GB). If the directory is
empty, stage 1 hasn't run yet — see `scripts/launch_mm.sh stage1`. Don't proceed
to stage 2 without a stage-1 ckpt.

### 3. Sync everything to sh02

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/projects/elf-pro
bash submit/sync_to_sh02.sh --dry-run    # inspect first
bash submit/sync_to_sh02.sh              # actually sync
```

`sync_to_sh02.sh` rsyncs the following (read the script — don't trust this list
to stay in sync if the user edits the script):

1. **Code** — `projects/elf-pro/` minus `.git/`, `outputs/`, `__pycache__/`,
   `wandb/`. Re-run after any code edit.
2. **Stage-1 ckpt** — `outputs/elf_l-llava-siglip2-warmup-local/` →
   `outputs/elf_l-llava-siglip2-warmup/` on sh02 (note: dropped the `-local` suffix
   because the cluster YAML's `init_checkpoint` points at the no-`-local` name).
3. **mix665k JSON + 5 image subsets** — `data/playground/data/{*.json, coco, gqa, ocr_vqa, textvqa, vg}`.
   This is ~80 GB. First sync takes a while; subsequent syncs are incremental.
4. **Encoder + vision encoder weights** — `models/t5-small/` from `~/.cache/huggingface/hub/`,
   `models/siglip2-base-patch16-224/` from the project dir. Both must end up under
   `<sh02-project>/models/`.

Verify the sh02 side has what the YAML expects:

```bash
SH02=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro
ls "$SH02/data/playground/data/llava_v1_5_mix665k.json"
ls "$SH02/data/playground/data/coco/train2017" | head
ls "$SH02/outputs/elf_l-llava-siglip2-warmup/" | head      # should list checkpoint_*
ls "$SH02/models/t5-small/model.safetensors"
ls "$SH02/models/siglip2-base-patch16-224/model.safetensors"
ls "$SH02/submit/run_elf_l_llava_instruct_hope.sh"          # the wrapper that hope calls
ls "$SH02/src/configs/training_configs/train_llava_siglip2_instruct_ELF-L_hope_32gpu.yml"
```

If any of these are missing, sync didn't work. Look at sync_to_sh02.sh's stderr.

### 4. Submit the HOPE job

HOPE submission must happen from a node that has the `hope` CLI in `$PATH`
and access to both mounts. `which hope` should resolve.

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro/submit
hope submit submit_elf_l_llava_instruct_32gpu.hope
```

The job will appear in the HOPE web UI as
`[hope/ml-easy-job]elf_pro_llava_instruct_ELF-L_32gpu@luoyaxin03`.

**Do not edit any of the existing submit files just to try a different config.**
If you need a tweak, prefer either:
- Edit `train_llava_siglip2_instruct_ELF-L_hope_32gpu.yml` (re-sync after).
- Append `--config_override field=value` flags inside
  `run_elf_l_llava_instruct_hope.sh` at the `python3 -m torch.distributed.run`
  line (re-sync after).

### 5. Monitor

- HOPE web UI shows worker stdout. Useful for catching launch failures
  (CUDA OOM, dataset not found, NCCL hang at init).
- Each worker also tees its log to
  `<output_dir>/launch_logs/worker_<task_id>_<host>_<ts>.log` on sh02.
  `output_dir` for this run is
  `outputs/elf_l-llava-siglip2-instruct-hope-32gpu/`. The first few hundred
  lines of any log show the import probe block — that's where most failures
  surface.
- Checkpoints land in `output_dir/checkpoint_<step>` every `save_freq=0.2`
  epoch. With 5202 steps/epoch (665298 / 128), expect a checkpoint
  every ~1040 steps. Auto-resume picks up the latest if you re-submit.

## Things that will trip you up

These are gotchas the prior session ran into.

1. **Two repo copies.** The dev box edits live under `hadoop-hldy-nlp/.../yaxinluo/projects/elf-pro`.
   The cluster reads from `hadoop-nlp-sh02/.../luoyaxin03/projects/elf-pro`.
   The wrapper `run_elf_l_llava_instruct_hope.sh` `cd`s into the **sh02 mirror** —
   so any code change requires a re-sync. Don't be surprised if your edit
   does nothing until you re-run `sync_to_sh02.sh`.

2. **No internet on cluster workers.** Anything resolved via HF repo id will
   stall on a HEAD request and time out. Every weight + tokenizer in the YAML
   is already an absolute path under sh02 (`encoder_model_name`,
   `tokenizer_name`, `vision_encoder_model_name`, `init_checkpoint`,
   `data_path`, `image_root`). Don't change these to repo-id strings.

3. **SigLIP2 tokenizer is broken on transformers <4.45.** The conda env on
   sh02 has 4.41.x, which can't parse SigLIP2's GemmaTokenizer. Our code
   (`src/modules/vision_encoder.py`) calls `AutoImageProcessor`, *not*
   `AutoProcessor`, to avoid the tokenizer load entirely. Don't "fix" this
   to use `AutoProcessor` — it will crash inside `tokenization_siglip.py`.

4. **`mm_instruct` unfreezes everything in the ELF model.**
   `_configure_trainable_params` in `train.py` flips all 652M ELF params to
   `requires_grad=True` (verified — vision encoder + T5 are separate modules
   and stay frozen via their own paths). Activations are huge — keep
   `gradient_checkpointing: true` in the cluster YAML. With grad ckpt off
   the run will OOM on the first backward pass.

5. **`eval_freq=0` is intentional.** The default `run_generation` does
   *unconditional* sampling, which is meaningless for multimodal — there's no
   image and no prompt to condition on. The cluster YAML disables it. If you
   want real eval, add an `eval_data_path` pointing at a JSONL of image+prompt
   pairs (the code will then route to `test_generation_cond`).

6. **stage1 ckpt path naming.** The local dir is `…-warmup-local/`, but
   the sh02 mirror is `…-warmup/` (no `-local`). The cluster YAML's
   `init_checkpoint` points at the no-suffix name. `sync_to_sh02.sh` does the
   rename on copy. Don't manually rsync stage1 ckpts and forget this.

7. **`max_checkpoints_to_keep: 3` rolls older ckpts away.** If you want
   intermediate checkpoints preserved (e.g. for ablations), bump this in the
   YAML before submitting. The previous OWT run rolled `epoch 3` away because
   it was set to 3.

8. **`sampling_configs_path: null` does NOT mean no sampling.**
   `Config.sampling_configs` defaults to `[SamplingConfig()]` (one ODE / 50-step
   config), which is why disabling sampling needs `eval_freq=0`, not just a null
   sampling path. (See gotcha 5.)

## Sanity check before you submit

Run this from the sh02 dir to make sure the YAML parses and every absolute
path it references actually exists:

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro
PYTHONPATH=$(pwd)/src python3 -c "
import sys, os
sys.path.insert(0, 'src')
from configs.config import load_config_from_yaml
c = load_config_from_yaml('src/configs/training_configs/train_llava_siglip2_instruct_ELF-L_hope_32gpu.yml')
checks = [
    ('data_path',                c.data_path,                 os.path.isfile),
    ('image_root',               c.image_root,                os.path.isdir),
    ('encoder_model_name',       c.encoder_model_name,        os.path.isdir),
    ('vision_encoder_model_name',c.vision_encoder_model_name, os.path.isdir),
    ('init_checkpoint',          c.init_checkpoint,           os.path.isdir),
    ('output_dir parent',        os.path.dirname(c.output_dir), os.path.isdir),
]
for name, path, fn in checks:
    print(f'  {name}: {\"OK\" if fn(path) else \"MISSING\"}  {path}')
print(f'  global_batch_size: {c.global_batch_size}')
print(f'  grad_accum_steps:  {c.grad_accum_steps}')
print(f'  gradient_checkpointing: {c.gradient_checkpointing}')
print(f'  eval_freq: {c.eval_freq}  (should be 0 for multimodal)')
"
```

If anything reports `MISSING`, do NOT submit. Fix the sync (or the YAML) first.

## Resource notes

- 32 GPUs × H800-141G. Per-device memory usage on stage 1 (smaller setup,
  backbone frozen) was ~60 GB / 80 GB at batch=64/GPU. Stage 2 has the
  backbone trainable and uses grad ckpt; expect ~70-90 GB / 141 GB at
  batch=4/GPU (global=128). If `nvidia-smi` shows < 70 % VRAM after the first
  step, raise `global_batch_size` via override.
- `effective_batch = global_batch_size * grad_accum_steps`. The YAML defaults
  to 128 / 1 = 128, matching LLaVA-1.5 stage 2.
- Wall clock estimate: 5202 steps × ~1.0 s/step on 32 GPUs ≈ 1.5 hours/epoch.
  Stage 2 runs 1 epoch by default.

## If something fails

- **NCCL init hang**: usually means one worker's container is wedged. Read
  the `import probe` block in each worker's `launch_logs/*.log`. Check that
  every worker reached `torchrun start`. Resubmit if one node is dead.
- **`huggingface.co` resolve error**: a path in the YAML is still a repo id.
  Grep the run's launch_log for `huggingface.co`, fix the YAML, re-sync, re-submit.
- **OOM on first backward**: `gradient_checkpointing` is off. Re-enable it.
- **`muon` import failure**: the wrapper installs `muon-optimizer==0.1.0` on
  the fly if `submit/vendor/muon.py` is missing. Check that the vendor dir
  has the file (`ls submit/vendor/`); if empty, the install will run on every
  worker.
- **Run starts but loss is `nan` from step 1**: most likely `init_checkpoint`
  path is wrong and the model is on random init under a high LR. Check the
  log for `Initialized model weights from local; missing=N, unexpected=M`.
  `missing` should be 6 (the projector params, when loading from a
  text-only ckpt) or 0 (when loading from a stage-1 ckpt). `unexpected` should
  always be 0.
