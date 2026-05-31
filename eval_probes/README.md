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

The corresponding analysis summary is in
`docs/elf_multimodal_scaling_plan.html`.
