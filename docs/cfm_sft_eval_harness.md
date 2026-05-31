# CFM SFT Eval Harness

This harness is the fixed evaluation loop for text-only instruction SFT on the
Continuous Flow Matching ELF model. It is designed to answer CFM-specific
questions that normal AR-LM SFT eval does not isolate:

- Can the decoder recover text from clean target latents?
- Does the denoiser use the prompt condition, or only answer priors?
- Does high-noise denoising point toward the right target?
- Does the full reverse trajectory from pure noise stay stable?
- Which data slice breaks first: discrete short answers, natural short answers,
  or long answers?

## One-command Report

```bash
python3 scripts/analyze_cfm_ablation_diagnostics.py --min-n 5
```

Outputs:

- `eval_probes/diagnostics/sft_eval_harness_report.md`
- `eval_probes/diagnostics/sft_eval_harness_report.json`
- Compatibility copies:
  - `eval_probes/diagnostics/cfm_ablation_breakdown.md`
  - `eval_probes/diagnostics/cfm_ablation_breakdown.json`

The harness config is:

```text
eval_probes/sft_eval_harness_config.json
```

## Gates

### A: Discrete Short Answer

Examples:

- yes/no
- number
- single-word or label answer
- short JSON/list

Primary purpose:

Check whether high-noise flow can recover a unique target instead of drifting
to a nearby common answer.

### B: Natural Short Answer

Examples:

- short phrase
- one-sentence answer
- short refusal
- compact natural-language response

Primary purpose:

Check whether the model can follow a prompt and generate short natural
language, not just labels.

### C: Long Answer

Examples:

- multi-line answer
- long-form answer
- long target bucket
- bullet/reasoning/paragraph-style output

Primary purpose:

Check long-horizon reverse trajectory stability. For this gate, token exact is
not enough; future reports should add semantic similarity and structure/key-info
checks.

## Core Metrics

- `clean`: clean target latent -> decoder token accuracy.
- `t0.1 correct`: controlled denoise token accuracy with the correct prompt at
  high noise.
- `t0.1 zero` / `t0.1 shuffled`: condition controls. These should stay much
  worse than the correct prompt.
- `t0.1 gap`: `correct - max(zero, shuffled)`.
- `t_start=0 uniform`: full reverse trajectory from pure noise.
- `exact`: number of trajectory examples with token accuracy >= 0.999.

## Adding a New Checkpoint

1. Run the existing probe scripts against the checkpoint:

   ```bash
   bash scripts/probe_tulu3_short_answer.sh
   bash scripts/probe_tulu3_mixed_length.sh
   ```

2. Place probe outputs under a stable `eval_probes/<experiment>/` directory.

3. Add the experiment entry to `eval_probes/sft_eval_harness_config.json`:

   ```json
   {
     "label": "My New Experiment",
     "cfm": "eval_probes/my_experiment/cfm_sanity.json",
     "trajectory": "eval_probes/my_experiment/sampling_trajectory.json",
     "metadata": "eval_probes/dataset_metadata/my_experiment_metadata.jsonl"
   }
   ```

4. Re-run:

   ```bash
   python3 scripts/analyze_cfm_ablation_diagnostics.py --min-n 5
   ```

## Interpretation Rule

Do not treat clean decode or low-noise repair as SFT success. For CFM SFT, a
checkpoint is not instruction-following-ready until:

- clean decode is stable,
- correct condition beats zero/shuffled condition at high noise,
- A/B/C gates do not regress,
- pure-noise trajectory improves, and
- long-answer semantic/structure checks are acceptable.

## Gate Status Semantics

`gate_status` is computed per-bucket from the active threshold set:

- `pass` — every required metric is present and meets its threshold.
- `fail` — at least one required metric is below its threshold.
- `incomplete` — at least one required metric is missing entirely. This
  status is new; previously a missing metric was treated as a non-failure
  and could mask data gaps as `pass`.

### Per-gate thresholds

`gate_thresholds.per_gate` in `eval_probes/sft_eval_harness_config.json` lets
a gate override individual thresholds and skip metrics that are not
meaningful for it. Today `C_long_answer` skips `t01_correct_min` and
`trajectory_t0_uniform_min`, because long answers should not be judged by
token-exact accuracy. These are placeholders until semantic similarity and
structure / key-info metrics replace them; until then the C gate intentionally
reports `pass` from `clean_decode_min` and `t01_condition_gap_min` only.

## Data Coverage Section

The report now contains a `Metadata & Field Coverage` table for any
experiment that ships a `metadata` jsonl. Two columns matter:

- `match rate` — fraction of CFM examples joined to metadata via
  `(input.strip(), target.strip())`. Unmatched rows fall back to
  `source=unknown` and are excluded from `source_group` / `source` slices.
  A drop here means downstream slice mixes are biased.
- `missing fields` — counts of probe-artifact JSON keys that were absent.
  The harness skips those samples from the running mean instead of
  coercing them to `0.0`, so a non-zero count means the underlying probe
  run is incomplete.
