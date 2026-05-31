# ELF CFM SFT probe artifacts

This directory stores lightweight probe inputs, scripts, and JSON/JSONL outputs
used for the text-only instruction-following diagnostics.

## Layout

- `cfm_sanity_*.json`, `sampling_trajectory_*.json`, and `sft_*jsonl`:
  full Tulu3 SFT probes from
  `outputs/elf_l-tulu3-sft-t5-1024-hope-32gpu-probe`.
- `short_qa_overfit_128/`: 128-example synthetic short-QA overfit probes.
- `short_qa_10k/`: synthetic short-QA 10K ablation probes.
- `tulu3_short_answer_clean_en_10k/`: Tulu3 short-answer 10K ablation probes.
- `tulu3_mixed_length_clean_en_10k/`: Tulu3 mixed-length 10K ablation probes.
- `dataset_reports/`: dataset filtering and token-length reports for the 10K
  ablation datasets.
- `dataset_metadata/`: stripped train metadata used to map probe rows back to
  source, prompt length, target length, and original sample id.
- `probe_inputs/`: JSONL inputs used for the random probe subsets.
- `diagnostics/`: generated fine-grained breakdowns from
  `scripts/analyze_cfm_ablation_diagnostics.py`.
- `sft_eval_harness_config.json`: fixed experiment and A/B/C gate
  configuration for the CFM SFT eval harness.
- `run_ar_instruction_baseline.py`: task-level autoregressive LM baseline for
  the same prompt/target probe JSONL files. It reports generation exact/prefix
  match, target containment, token F1, and target continuation NLL by the same
  A/B/C gates.

Example AR baseline:

```bash
python3 eval_probes/run_ar_instruction_baseline.py \
  --model gpt2-large \
  --prompts eval_probes/probe_inputs/tulu3_short_answer_clean_en_10k_probe_valid_random256_seed12345.jsonl \
  --batch_size 4
```

The corresponding analysis summary is in
`docs/elf_multimodal_scaling_plan.html`.

The full harness runbook is in `docs/cfm_sft_eval_harness.md`.
