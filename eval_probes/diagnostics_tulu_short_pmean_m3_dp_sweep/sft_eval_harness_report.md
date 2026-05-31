# CFM SFT Eval Harness Report

This report is generated from fixed probe artifacts only. It is the evaluation harness for text-only Continuous Flow Matching SFT instruction experiments.
Slice metrics are sample-level macro means over probe examples, so they can differ slightly from token-weighted summaries inside the raw JSON files.

## Overall Metrics

| experiment | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t_start=0 uniform | exact | t_start=0 logit-tail | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tulu3 Short QA 10K baseline | 256 | 0.997 | 0.531 | 0.008 | 0.007 | 0.537 | 0.561 | 0.799 | 10/16 | 0.825 | 11/16 |
| Tulu3 Short QA 10K pmean=-3 | 256 | 0.999 | 0.632 | 0.026 | 0.033 | 0.664 | 0.684 | 0.890 | 11/16 | 0.808 | 9/16 |
| Tulu3 Short QA 10K pmean=-3 dp=0.35 | 256 | 0.971 | 0.518 | 0.003 | 0.001 | 0.505 | 0.490 | 0.654 | 7/16 | 0.592 | 6/16 |
| Tulu3 Short QA 10K pmean=-3 dp=0.50 | 256 | 0.993 | 0.667 | 0.008 | 0.004 | 0.653 | 0.691 | 0.762 | 8/16 | 0.772 | 8/16 |

## Metadata & Field Coverage

Metadata join uses `(input.strip(), target.strip())`. Rows that fail to match fall back to `source=unknown` and are not included in source-group / source slices, so a low match rate silently biases the breakdown. Missing-field counts cover JSON keys read by the harness; non-zero values mean the underlying probe artifact is incomplete and the corresponding metric was skipped (not coerced to 0).

| experiment | cfm rows | missing meta | match rate | missing fields |
|---|---:|---:|---:|---|
| Tulu3 Short QA 10K baseline | 256 | 0 | 1.000 | - |
| Tulu3 Short QA 10K pmean=-3 | 256 | 0 | 1.000 | - |
| Tulu3 Short QA 10K pmean=-3 dp=0.35 | 256 | 0 | 1.000 | - |
| Tulu3 Short QA 10K pmean=-3 dp=0.50 | 256 | 0 | 1.000 | - |


## Curriculum Gate Summary

Gate status uses configurable thresholds from `eval_probes/sft_eval_harness_config.json`. Current thresholds are intentionally aspirational and are meant to flag bottlenecks, not to claim model quality.
Status `incomplete` means the bucket has at least one required metric missing (no longer silently treated as pass).

Per-gate threshold overrides:
- `C_long_answer` skips metrics ['t01_correct_min', 'trajectory_t0_uniform_min'] — Long answers should not be judged by token-exact accuracy. token_acc / trajectory thresholds are intentionally relaxed; promote to semantic / structure metrics in a future revision.

| experiment | gate | n | status | clean | t0.1 correct | t0.1 gap | t0.3 correct | t0.5 correct | t_start=0 uniform | exact |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Tulu3 Short QA 10K baseline | A_discrete_short | 155 | fail | 1.000 | 0.532 | 0.525 | 0.532 | 0.560 | 0.812 | 9/12 |
| Tulu3 Short QA 10K baseline | B_natural_short | 96 | fail | 0.993 | 0.532 | 0.521 | 0.548 | 0.566 | 0.757 | 1/4 |
| Tulu3 Short QA 10K baseline | C_long_answer | 5 | pass | 1.000 | 0.479 | 0.466 | 0.507 | 0.527 | - | - |
| Tulu3 Short QA 10K pmean=-3 | A_discrete_short | 155 | fail | 1.000 | 0.664 | 0.635 | 0.694 | 0.709 | 0.935 | 9/12 |
| Tulu3 Short QA 10K pmean=-3 | B_natural_short | 96 | fail | 0.996 | 0.582 | 0.542 | 0.616 | 0.647 | 0.757 | 2/4 |
| Tulu3 Short QA 10K pmean=-3 | C_long_answer | 5 | pass | 1.000 | 0.630 | 0.563 | 0.647 | 0.625 | - | - |
| Tulu3 Short QA 10K pmean=-3 dp=0.35 | A_discrete_short | 155 | fail | 0.991 | 0.541 | 0.538 | 0.522 | 0.499 | 0.700 | 6/12 |
| Tulu3 Short QA 10K pmean=-3 dp=0.35 | B_natural_short | 96 | fail | 0.945 | 0.480 | 0.476 | 0.478 | 0.476 | 0.518 | 1/4 |
| Tulu3 Short QA 10K pmean=-3 dp=0.35 | C_long_answer | 5 | fail | 0.855 | 0.518 | 0.518 | 0.485 | 0.463 | - | - |
| Tulu3 Short QA 10K pmean=-3 dp=0.50 | A_discrete_short | 155 | fail | 0.995 | 0.694 | 0.687 | 0.679 | 0.709 | 0.755 | 7/12 |
| Tulu3 Short QA 10K pmean=-3 dp=0.50 | B_natural_short | 96 | fail | 0.990 | 0.627 | 0.619 | 0.620 | 0.665 | 0.784 | 1/4 |
| Tulu3 Short QA 10K pmean=-3 dp=0.50 | C_long_answer | 5 | pass | 0.957 | 0.599 | 0.590 | 0.488 | 0.625 | - | - |

## Fine-Grained Slices

Tables are sorted by low-noise-to-high-noise bottleneck metric `t=0.1:correct` ascending. Small groups below the `min_n` threshold are omitted.

## Worst t0.1 / t0.3 Controlled Denoise Examples

## Worst Pure-Noise Trajectory Examples

## Readout

- Clean decode is near-saturated across all current ablations, so the immediate bottleneck is not latent-to-token decoding.
- Synthetic 10K is already harder than Synthetic 128: pure-noise trajectory is no longer perfect, and the failures are concentrated in short discrete answers such as numbers and yes/no labels.
- Tulu Short 10K has weak controlled denoise at `t=0.1` even though outputs are short. This points to data/task distribution rather than answer length alone.
- Tulu Mixed 10K has better single-step denoise than Tulu Short but worse exact pure-noise trajectory for long answers, which separates local repair ability from long-horizon sampling stability.
- The next decision should be based on which slices dominate the weak `t=0.1:correct` groups: source mix, prompt templates, answer type, or length.
