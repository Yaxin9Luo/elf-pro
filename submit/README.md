# HOPE submission (Meituan Shanghai `hadoop-nlp-sh02`)

This directory has the HOPE/AFO submission files for training ELF-L on
OpenWebText on the Shanghai `shxs_training_cluster` H800-141G queue.

| File | What it does |
| --- | --- |
| `submit_elf_l_8gpu.hope` | 1 node x 8 H800-141G, single-host torchrun. Recommended first run for the ELF-L (652M) baseline. |
| `submit_elf_l_32gpu.hope` | 4 nodes x 8 H800-141G = 32 GPUs. Scale-out variant; same recipe, ~4x less wall-clock per epoch. |
| `hope_run_torch_distribute.py` | HOPE rendezvous helper. Parses `AFO_ENV_CLUSTER_SPEC` and prints `--nnodes/--nproc-per-node/--node_rank/--master_addr/--master_port` for `torch.distributed.run`. Copied verbatim from the standard `mm-pretrain-hfds` template. |

The two YAML configs that go with these submission files live next to the rest
of the training configs:

- `src/configs/training_configs/train_owt_ELF-L_hope_8gpu.yml`
- `src/configs/training_configs/train_owt_ELF-L_hope_32gpu.yml`

They are derived from `train_owt_ELF-B.yml`, with only `data_path` and
`output_dir` rewritten to dolphinfs-absolute paths (cluster has no internet,
so HuggingFace auto-download will not work).

## Prerequisites (one-time)

1. **Mirror the code to `hadoop-nlp-sh02`.** The Shanghai cluster reads from
   `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/...`. The hope file
   `cd`s into:

       /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro

   So either rsync this repo to that path, or symlink it. The same pattern is
   used by `submit_face_enhance_luoyaxin03.hope`. Example:

   ```bash
   # from a node that has both mounts visible
   mkdir -p /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects
   rsync -av --exclude=outputs --exclude='.git' \
       /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/projects/elf-pro/ \
       /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro/
   ```

   Re-run this whenever you change code locally. (A symlink between the two
   `dolphinfs` mounts works too if your gateway box has both mounts and the
   symlink is allowed by the platform.)

2. **Mirror the dataset to `hadoop-nlp-sh02` (only if the hldy path is not
   readable from the sh02 cluster).** Both YAMLs default to:

       /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/BERT_TRAINING_SERVICE/platform/dataset/embedded-language-flows/openwebtext-t5/main

   `afo.dolphinfs.otherusers` already includes `hadoop-hldy-nlp`, so this path
   *should* be readable from the sh02 worker container. Verify that with a
   quick interactive job before launching a full run; if it is not, copy the
   dataset to e.g.
   `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/data/openwebtext-t5/main`
   and update `data_path` in the two YAMLs to match.

3. **Make `outputs` writable.** Both YAMLs write checkpoints to
   `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro/outputs/...`.
   Pre-create that directory once. The `afo.app.env.DFS_CLIENT_WRITE_ZONE = SH02`
   line in the hope file ensures writes go to the SH02 zone.

4. **WandB key.** WandB is enabled (`use_wandb: true`). Either pre-`wandb login`
   on the worker user (the key persists in `~/.netrc` on dolphinfs home) or
   add `WANDB_API_KEY=...` to the worker.script before the `eval` line. If you
   want to disable WandB just override at submit time:
   `... src/train.py --config <yml> --config_override use_wandb=false`.

## Submitting

From any node that has the HOPE CLI:

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro/submit
hope submit submit_elf_l_8gpu.hope        # baseline
# or
hope submit submit_elf_l_32gpu.hope       # 4-node scale-up
```

The job appears in the HOPE web UI as
`[hope/ml-easy-job]elf_pro_owt_ELF-L_8gpu@luoyaxin03` (or `..._32gpu...`).

## Resource & hyperparameter summary

| Knob | 8gpu | 32gpu | Notes |
| --- | --- | --- | --- |
| Nodes (`workers`) | 1 | 4 | |
| GPUs / node (`worker.gcoresh800-141g`) | 8 | 8 | H800-141G |
| Total GPUs | 8 | 32 | |
| `global_batch_size` | 512 | 512 | Paper recipe; do not change without rescaling LR/warmup. |
| `grad_accum_steps` | 1 | 1 | Bump to 2 if 8gpu OOMs. |
| Per-device batch | 64 | 16 | `global_batch_size / world` (computed by `train.py`). |
| `epochs` | 5 | 5 | |
| `blr` | 1e-3 | 1e-3 | Effective LR is `blr * (GB * grad_accum) / 256 = 2e-3`. |
| `optimizer` | muon | muon | Auxiliary AdamW for non-matmul params. |
| Wall-clock estimate | ~4 h / epoch | ~1 h / epoch | Linearly scaled from the README's 8x H200 estimate; H800-141G is comparable. Full 5-epoch baseline ~20 h on 8 GPUs / ~5 h on 32. |

## Adjusting GPU count

If you want a different topology than the two provided files:

- Single-node, fewer GPUs: edit `worker.gcoresh800-141g` (e.g. `4`) in the 8gpu
  hope file. `hope_run_torch_distribute.py` will pick up `nvidia-smi --list-gpus`
  at run time, so `nproc-per-node` follows automatically. Also update
  `global_batch_size` in the YAML so the per-device batch stays sensible.
- More nodes: bump `workers` in the 32gpu hope file (e.g. to `8` for 64 GPUs).
  Same comment about `global_batch_size`.
- Different queue (e.g. H800-80G or A100): change `queue` and the
  `worker.gcoresh800-141g` resource key (HOPE uses the resource name to pick
  the GPU type — see the templates under
  `mm-pretrain-hfds/pretrain/template.hl.hope`).

## Monitoring

- **HOPE web UI / `hope log`** — stdout + stderr per worker. Useful for
  catching launch errors (CUDA OOM, dataset not found, NCCL hang at init).
- **WandB** — `wandb_project: elf`, run name set in the YAML
  (`elf_l-owt-hope-8gpu` / `elf_l-owt-hope-32gpu`). Loss curves, LR schedule,
  generation samples.
- **Checkpoints** — `output_dir` on dolphinfs holds `checkpoint_<step>` files
  every `save_freq` epochs. Resume is automatic from the latest checkpoint
  if you re-submit the same hope file (controlled by `auto_resume` logic in
  `src/utils/checkpoint_utils.py`).
- **Tensorboard** — disabled in the hope file (`with.tensor.board = false`)
  because WandB covers everything; flip to `true` if you want HOPE's built-in
  TB tab.

## Differences from `submit_face_enhance_luoyaxin03.hope`

For reference, the things this submission deliberately drops vs. the
face_enhance template:

- **No `pre_load.py` step.** That helper exists to warm
  `transformers_modules` cache for `AutoModelForCausalLM.from_pretrained` of a
  custom remote-code model. ELF-L's only HF-loaded weights are the `t5-small`
  encoder, which has no remote code.
- **No `env_configure.sh`.** The face_enhance variant only sets
  `CUDA_DEVICE_MAX_CONNECTIONS=1` + `NCCL_DEBUG=INFO` (rest is commented out);
  ELF-L is the target size and the defaults are fine. Add an `env_configure.sh` here
  if you start hitting NCCL issues at 32 GPUs.
- **No conda env activation beyond `base`.** The system `/usr/local/conda` base
  env + user-site pip install (`~/.local/lib/python3.9/site-packages`) is
  enough. If you migrate to a project-specific conda env, replace
  `source /usr/local/conda/bin/activate` with `... && conda activate <env>`.
