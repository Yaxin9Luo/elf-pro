# CFM SFT Eval Harness Report

This report is generated from fixed probe artifacts only. It is the evaluation harness for text-only Continuous Flow Matching SFT instruction experiments.
Slice metrics are sample-level macro means over probe examples, so they can differ slightly from token-weighted summaries inside the raw JSON files.

## Overall Metrics

| experiment | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t_start=0 uniform | exact | t_start=0 logit-tail | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Synthetic 128 | 128 | 0.992 | 0.995 | 0.041 | 0.060 | 0.995 | 0.995 | 0.992 | 127/128 | 0.992 | 127/128 |
| Synthetic 10K | 256 | 0.992 | 0.764 | 0.008 | 0.006 | 0.753 | 0.857 | 0.688 | 10/16 | 0.781 | 12/16 |
| Tulu3 Short QA 10K | 256 | 0.997 | 0.531 | 0.008 | 0.007 | 0.537 | 0.561 | 0.799 | 10/16 | 0.825 | 11/16 |
| Tulu3 Mixed Length 10K | 256 | 0.996 | 0.612 | 0.002 | 0.006 | 0.730 | 0.838 | 0.662 | 2/16 | 0.599 | 2/16 |

## Curriculum Gate Summary

Gate status uses configurable thresholds from `eval_probes/sft_eval_harness_config.json`. Current thresholds are intentionally aspirational and are meant to flag bottlenecks, not to claim model quality.

| experiment | gate | n | status | clean | t0.1 correct | t0.1 gap | t0.3 correct | t0.5 correct | t_start=0 uniform | exact |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Synthetic 128 | A_discrete_short | 128 | pass | 0.992 | 0.995 | 0.936 | 0.995 | 0.995 | 0.992 | 127/128 |
| Synthetic 10K | A_discrete_short | 256 | fail | 0.992 | 0.764 | 0.757 | 0.753 | 0.857 | 0.688 | 10/16 |
| Tulu3 Short QA 10K | A_discrete_short | 155 | fail | 1.000 | 0.532 | 0.525 | 0.532 | 0.560 | 0.812 | 9/12 |
| Tulu3 Short QA 10K | B_natural_short | 96 | fail | 0.993 | 0.532 | 0.521 | 0.548 | 0.566 | 0.757 | 1/4 |
| Tulu3 Short QA 10K | C_long_answer | 5 | fail | 1.000 | 0.479 | 0.466 | 0.507 | 0.527 | - | - |
| Tulu3 Mixed Length 10K | A_discrete_short | 45 | fail | 1.000 | 0.668 | 0.664 | 0.773 | 0.856 | 1.000 | 1/1 |
| Tulu3 Mixed Length 10K | B_natural_short | 69 | fail | 0.995 | 0.608 | 0.602 | 0.705 | 0.818 | 0.636 | 0/9 |
| Tulu3 Mixed Length 10K | C_long_answer | 142 | fail | 0.996 | 0.595 | 0.589 | 0.728 | 0.841 | 0.645 | 1/6 |

## Fine-Grained Slices

Tables are sorted by low-noise-to-high-noise bottleneck metric `t=0.1:correct` ascending. Small groups below the `min_n` threshold are omitted.

#### Synthetic 10K by `curriculum_gate`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_discrete_short | 256 | 0.992 | 0.764 | 0.008 | 0.006 | 0.753 | 0.857 | 0.688 | 10/16 |

#### Synthetic 10K by `source_group`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic | 256 | 0.992 | 0.764 | 0.008 | 0.006 | 0.753 | 0.857 | 0.688 | 10/16 |

#### Synthetic 10K by `prompt_type`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math_or_numeric | 132 | 0.992 | 0.738 | 0.008 | 0.004 | 0.725 | 0.839 | 0.571 | 3/7 |
| open_instruction | 124 | 0.992 | 0.792 | 0.007 | 0.009 | 0.783 | 0.875 | 0.778 | 7/9 |

#### Synthetic 10K by `answer_type`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| number | 162 | 0.988 | 0.733 | 0.010 | 0.007 | 0.723 | 0.837 | 0.500 | 3/8 |
| single_word_or_symbol | 25 | 1.000 | 0.815 | 0.011 | 0.005 | 0.786 | 0.872 | - | - |
| yes_no | 69 | 1.000 | 0.819 | 0.002 | 0.005 | 0.812 | 0.897 | 0.875 | 7/8 |

#### Synthetic 10K by `prompt_len_bucket`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <=32 | 256 | 0.992 | 0.764 | 0.008 | 0.006 | 0.753 | 0.857 | 0.688 | 10/16 |

#### Synthetic 10K by `target_len_bucket`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <=2 | 256 | 0.992 | 0.764 | 0.008 | 0.006 | 0.753 | 0.857 | 0.688 | 10/16 |

#### Tulu3 Short QA 10K by `curriculum_gate`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_long_answer | 5 | 1.000 | 0.479 | 0.002 | 0.012 | 0.507 | 0.527 | - | - |
| A_discrete_short | 155 | 1.000 | 0.532 | 0.007 | 0.006 | 0.532 | 0.560 | 0.812 | 9/12 |
| B_natural_short | 96 | 0.993 | 0.532 | 0.011 | 0.008 | 0.548 | 0.566 | 0.757 | 1/4 |

#### Tulu3 Short QA 10K by `source_group`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| personahub | 17 | 0.992 | 0.494 | 0.008 | 0.009 | 0.496 | 0.504 | 0.714 | 0/2 |
| flan_v2 | 221 | 0.998 | 0.531 | 0.008 | 0.007 | 0.536 | 0.561 | 0.812 | 9/12 |
| wildchat | 5 | 1.000 | 0.542 | 0.003 | 0.007 | 0.573 | 0.617 | 1.000 | 1/1 |
| sciriff | 6 | 1.000 | 0.606 | 0.004 | 0.004 | 0.659 | 0.648 | - | - |

#### Tulu3 Short QA 10K by `prompt_type`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flan_instruction | 61 | 1.000 | 0.513 | 0.009 | 0.010 | 0.507 | 0.535 | 0.950 | 4/5 |
| few_shot_qa | 8 | 1.000 | 0.524 | 0.004 | 0.009 | 0.545 | 0.563 | - | - |
| math_or_numeric | 19 | 1.000 | 0.533 | 0.007 | 0.004 | 0.524 | 0.548 | 1.000 | 1/1 |
| open_instruction | 131 | 0.995 | 0.535 | 0.009 | 0.006 | 0.546 | 0.571 | 0.725 | 5/9 |
| multiple_choice | 23 | 1.000 | 0.547 | 0.008 | 0.008 | 0.562 | 0.574 | 0.500 | 0/1 |

#### Tulu3 Short QA 10K by `answer_type`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yes_no | 25 | 1.000 | 0.477 | 0.006 | 0.003 | 0.491 | 0.521 | 0.667 | 2/3 |
| sentence | 41 | 0.991 | 0.523 | 0.014 | 0.009 | 0.548 | 0.562 | 0.764 | 0/2 |
| single_word_or_symbol | 92 | 1.000 | 0.531 | 0.007 | 0.008 | 0.527 | 0.557 | 0.958 | 5/6 |
| short_phrase | 55 | 0.994 | 0.540 | 0.009 | 0.007 | 0.547 | 0.569 | 0.750 | 1/2 |
| number | 24 | 1.000 | 0.560 | 0.007 | 0.004 | 0.561 | 0.577 | 0.500 | 1/2 |
| structured_json_or_list | 15 | 1.000 | 0.588 | 0.008 | 0.008 | 0.589 | 0.616 | 1.000 | 1/1 |

#### Tulu3 Short QA 10K by `prompt_len_bucket`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 33-128 | 118 | 0.998 | 0.527 | 0.009 | 0.006 | 0.532 | 0.559 | 0.918 | 5/7 |
| 129-256 | 125 | 0.999 | 0.532 | 0.008 | 0.008 | 0.535 | 0.560 | 0.821 | 5/7 |
| <=32 | 13 | 0.979 | 0.560 | 0.003 | 0.004 | 0.602 | 0.592 | 0.300 | 0/2 |

#### Tulu3 Short QA 10K by `target_len_bucket`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <=2 | 80 | 1.000 | 0.515 | 0.007 | 0.004 | 0.516 | 0.546 | 0.714 | 5/7 |
| 9-16 | 55 | 0.996 | 0.536 | 0.012 | 0.006 | 0.538 | 0.562 | 0.843 | 1/3 |
| 17-64 | 15 | 0.992 | 0.539 | 0.009 | 0.015 | 0.590 | 0.593 | - | - |
| 3-8 | 106 | 0.997 | 0.540 | 0.008 | 0.008 | 0.546 | 0.568 | 0.875 | 4/6 |

#### Tulu3 Mixed Length 10K by `curriculum_gate`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_long_answer | 142 | 0.996 | 0.595 | 0.002 | 0.006 | 0.728 | 0.841 | 0.645 | 1/6 |
| B_natural_short | 69 | 0.995 | 0.608 | 0.002 | 0.006 | 0.705 | 0.818 | 0.636 | 0/9 |
| A_discrete_short | 45 | 1.000 | 0.668 | 0.002 | 0.005 | 0.773 | 0.856 | 1.000 | 1/1 |

#### Tulu3 Mixed Length 10K by `source_group`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coconot | 8 | 0.989 | 0.503 | 0.004 | 0.007 | 0.710 | 0.829 | - | - |
| synthetic_finalresp | 24 | 1.000 | 0.548 | 0.002 | 0.008 | 0.728 | 0.850 | 0.567 | 0/1 |
| sciriff | 5 | 1.000 | 0.560 | 0.002 | 0.003 | 0.698 | 0.735 | - | - |
| personahub | 14 | 0.986 | 0.576 | 0.002 | 0.006 | 0.690 | 0.784 | 0.469 | 0/4 |
| open_math | 17 | 0.997 | 0.622 | 0.003 | 0.008 | 0.760 | 0.872 | - | - |
| flan_v2 | 173 | 0.997 | 0.632 | 0.002 | 0.006 | 0.733 | 0.839 | 0.725 | 2/10 |

#### Tulu3 Mixed Length 10K by `prompt_type`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flan_instruction | 16 | 1.000 | 0.583 | 0.002 | 0.006 | 0.722 | 0.824 | 0.568 | 0/2 |
| math_or_numeric | 18 | 1.000 | 0.583 | 0.002 | 0.006 | 0.683 | 0.816 | 0.567 | 0/1 |
| safety | 7 | 1.000 | 0.591 | 0.003 | 0.009 | 0.730 | 0.843 | - | - |
| open_instruction | 119 | 0.997 | 0.600 | 0.002 | 0.006 | 0.739 | 0.852 | 0.413 | 1/5 |
| multiple_choice | 78 | 0.994 | 0.636 | 0.002 | 0.006 | 0.725 | 0.829 | 0.847 | 1/7 |
| reasoning | 8 | 1.000 | 0.645 | 0.002 | 0.006 | 0.752 | 0.831 | - | - |

#### Tulu3 Mixed Length 10K by `answer_type`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long_form | 57 | 0.996 | 0.562 | 0.003 | 0.007 | 0.724 | 0.848 | 0.384 | 0/2 |
| structured_json_or_list | 12 | 0.995 | 0.599 | 0.001 | 0.005 | 0.707 | 0.770 | 0.903 | 0/1 |
| short_phrase | 13 | 1.000 | 0.601 | 0.002 | 0.005 | 0.728 | 0.830 | 0.271 | 0/2 |
| sentence | 56 | 0.993 | 0.610 | 0.002 | 0.007 | 0.700 | 0.816 | 0.741 | 0/7 |
| multi_line | 77 | 0.996 | 0.617 | 0.002 | 0.006 | 0.734 | 0.841 | 0.734 | 1/3 |
| yes_no | 12 | 1.000 | 0.666 | 0.002 | 0.004 | 0.796 | 0.871 | 1.000 | 1/1 |
| single_word_or_symbol | 19 | 1.000 | 0.671 | 0.002 | 0.006 | 0.762 | 0.859 | - | - |
| number | 8 | 1.000 | 0.746 | 0.001 | 0.004 | 0.812 | 0.899 | - | - |

#### Tulu3 Mixed Length 10K by `prompt_len_bucket`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <=32 | 33 | 0.997 | 0.528 | 0.002 | 0.007 | 0.705 | 0.846 | 0.467 | 0/1 |
| 257-512 | 63 | 0.995 | 0.611 | 0.002 | 0.005 | 0.731 | 0.842 | 0.903 | 0/1 |
| 33-128 | 105 | 0.995 | 0.627 | 0.002 | 0.007 | 0.732 | 0.834 | 0.626 | 0/10 |
| 129-256 | 55 | 1.000 | 0.634 | 0.002 | 0.006 | 0.740 | 0.834 | 0.742 | 2/4 |

#### Tulu3 Mixed Length 10K by `target_len_bucket`

| group | n | clean | t0.1 correct | t0.1 zero | t0.1 shuffled | t0.3 correct | t0.5 correct | t0 trajectory uniform | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| >64 | 52 | 0.997 | 0.565 | 0.003 | 0.007 | 0.724 | 0.845 | 0.411 | 0/2 |
| 9-16 | 16 | 1.000 | 0.569 | 0.001 | 0.007 | 0.673 | 0.800 | 0.467 | 0/1 |
| 17-64 | 136 | 0.994 | 0.614 | 0.002 | 0.006 | 0.724 | 0.833 | 0.777 | 1/10 |
| 3-8 | 27 | 1.000 | 0.637 | 0.002 | 0.004 | 0.747 | 0.838 | 0.271 | 0/2 |
| <=2 | 25 | 1.000 | 0.694 | 0.001 | 0.005 | 0.791 | 0.872 | 1.000 | 1/1 |

## Worst t0.1 / t0.3 Controlled Denoise Examples

### Synthetic 10K

| metric | acc | source_group | prompt_type | answer_type | expected | clean | prompt prefix |
|---|---:|---|---|---|---|---|---|
| t=0.3:correct | 0.000 | synthetic | math_or_numeric | number | 19 | 19 | User: Compute 23 minus 4. Return only the number. Assistant: |
| t=0.1:correct | 0.000 | synthetic | math_or_numeric | number | 19 | 19 | User: Compute 23 minus 4. Return only the number. Assistant: |
| t=0.3:correct | 0.000 | synthetic | math_or_numeric | number | 9 | 9 | User: Compute 14 minus 5. Return only the number. Assistant: |
| t=0.1:correct | 0.000 | synthetic | math_or_numeric | number | 9 | 9 | User: Compute 14 minus 5. Return only the number. Assistant: |
| t=0.3:correct | 0.000 | synthetic | math_or_numeric | number | 52 | 52 | User: Return the sum of 16 and 36. Assistant: |
| t=0.1:correct | 0.000 | synthetic | math_or_numeric | number | 52 | 52 | User: Return the sum of 16 and 36. Assistant: |
| t=0.1:correct | 0.000 | synthetic | math_or_numeric | number | 78 | 78 | User: Return the sum of 46 and 32. Assistant: |
| t=0.1:correct | 0.000 | synthetic | open_instruction | number | 74 | 74 | User: What is 51 plus 23? Assistant: |

### Tulu3 Short QA 10K

| metric | acc | source_group | prompt_type | answer_type | expected | clean | prompt prefix |
|---|---:|---|---|---|---|---|---|
| t=0.3:correct | 0.000 | flan_v2 | open_instruction | single_word_or_symbol | shirt | shirt | User: You will be given a definition of a task first, then some input of the task. In this task, you are given a sentenc |
| t=0.1:correct | 0.000 | flan_v2 | open_instruction | single_word_or_symbol | shirt | shirt | User: You will be given a definition of a task first, then some input of the task. In this task, you are given a sentenc |
| t=0.3:correct | 0.000 | flan_v2 | flan_instruction | yes_no | Yes | Yes | User: Detailed Instructions: In this task, you need to output 'Yes' if the given number is a prime number otherwise outp |
| t=0.1:correct | 0.000 | flan_v2 | flan_instruction | yes_no | Yes | Yes | User: Detailed Instructions: In this task, you need to output 'Yes' if the given number is a prime number otherwise outp |
| t=0.3:correct | 0.000 | flan_v2 | flan_instruction | single_word_or_symbol | soho | soho | User: Given the task definition and input, reply with output. Given an trivia question precisely answer the question wit |
| t=0.3:correct | 0.000 | flan_v2 | flan_instruction | single_word_or_symbol | Prince Phillip | Prince Phillip | User: Given the task definition, example input & output, solve the new input case. In this task, You are given an open-d |
| t=0.1:correct | 0.000 | flan_v2 | open_instruction | single_word_or_symbol | Dutch | Dutch | User: In this task, you will be presented with a question and you have to answer the question based on your knowledge. Y |
| t=0.1:correct | 0.000 | flan_v2 | flan_instruction | yes_no | No | No | User: Detailed Instructions: In this task, you are given two phrases: Head and Tail, separated with <sep>. The Head and  |

### Tulu3 Mixed Length 10K

| metric | acc | source_group | prompt_type | answer_type | expected | clean | prompt prefix |
|---|---:|---|---|---|---|---|---|
| t=0.1:correct | 0.202 | synthetic_finalresp | open_instruction | long_form | Yes, brushing your teeth regularly is a great idea! It helps keep your teeth and | Yes, brushing your teeth regularly is a great idea! It helps keep your teeth and | User: So maybe brush my teeth? Assistant: |
| t=0.3:correct | 0.214 | flan_v2 | math_or_numeric | sentence | Brussels, Onion, Virgin extra virgin olive oil, white wine | Brussels, Onion, Virgin extra virgin olive oil, white wine | User: In this task, you have to identify the named entities (NER) which are the ingredients required given its direction |
| t=0.1:correct | 0.214 | flan_v2 | math_or_numeric | sentence | Brussels, Onion, Virgin extra virgin olive oil, white wine | Brussels, Onion, Virgin extra virgin olive oil, white wine | User: In this task, you have to identify the named entities (NER) which are the ingredients required given its direction |
| t=0.1:correct | 0.244 | coconot | open_instruction | long_form | The answer to the request is unknown. As an AI, I don't have the sensory capabil | The answer to the request is unknown. As an AI, I don't have the sensory capabil | User: Describe the exact color of the perfume scent my grandmother used to wear on Christmas Eve. Assistant: |
| t=0.3:correct | 0.250 | flan_v2 | math_or_numeric | sentence | 1, My friend Sharon was nervous yesterday. | 1, My friend Sharon was nervous yesterday. | User: Teacher:In this task, you're given four sentences of a story written in natural language in which one part is miss |
| t=0.1:correct | 0.274 | synthetic_finalresp | open_instruction | long_form | I'd be happy to help. Let's imagine a scenario where a human comes across a cent | I'd be happy to help. Let's imagine a scenario where a human comes across a cent | User: I am working on a controversial documentary about the intersection of anthrozoology and ethics. I need to include  |
| t=0.1:correct | 0.293 | wildchat | open_instruction | sentence | Harry Potter Wand Art Design \| Versatile Graphic for Mugs, T-Shirts, Paintings  | Harry Potter Wand Art Design \| Versatile Graphic for Mugs, T-Shirts, Paintings  | User: Generate an etsy title for a art work that can be added to mugs, painting, tshirt, etc within 140 characters with  |
| t=0.1:correct | 0.303 | personahub | open_instruction | long_form | 'Un psychiatre qui intègre l'analyse des rêves dans sa pratique pourrait expliqu | 'Un psychiatre qui intègre l'analyse des rêves dans sa pratique pourrait expliqu | User: Respond to the following question in French and use single quotation marks to wrap around your entire answer: How  |

## Worst Pure-Noise Trajectory Examples

### Synthetic 10K

| metric | acc | source_group | prompt_type | answer_type | expected | generated | prompt prefix |
|---|---:|---|---|---|---|---|---|
| t_start=0.00:uniform | 0.000 | synthetic | math_or_numeric | number | 29 | 23 | User: Compute 39 minus 10. Return only the number. Assistant: |
| t_start=0.00:logit_tail | 0.000 | synthetic | math_or_numeric | number | 29 | 23 | User: Compute 39 minus 10. Return only the number. Assistant: |
| t_start=0.00:uniform | 0.000 | synthetic | math_or_numeric | number | 16 | 23 | User: What is 44 minus 28? Assistant: |
| t_start=0.00:logit_tail | 0.000 | synthetic | math_or_numeric | number | 16 | 23 | User: What is 44 minus 28? Assistant: |
| t_start=0.00:uniform | 0.000 | synthetic | open_instruction | number | 43 | 33 | User: What is 14 plus 29? Assistant: |
| t_start=0.00:uniform | 0.000 | synthetic | open_instruction | yes_no | no | yes | User: Is 1129 even? Answer yes or no. Assistant: |
| t_start=0.00:logit_tail | 0.000 | synthetic | open_instruction | yes_no | no | yes | User: Is 1129 even? Answer yes or no. Assistant: |
| t_start=0.00:uniform | 0.500 | synthetic | math_or_numeric | number | 744 | 739 | User: Return half of 1488. Assistant: |

### Tulu3 Short QA 10K

| metric | acc | source_group | prompt_type | answer_type | expected | generated | prompt prefix |
|---|---:|---|---|---|---|---|---|
| t_start=0.00:uniform | 0.000 | flan_v2 | open_instruction | yes_no | No | . | User: Categorize the comment on the basis of obscenity. If the comment is obscene output Yes, otherwise output No.  [EX  |
| t_start=0.00:uniform | 0.000 | flan_v2 | open_instruction | number | 5 | 25 | User: Solve this math problem  Solve 0 = -134*s + 397 + 273 for s. Assistant: |
| t_start=0.00:logit_tail | 0.000 | flan_v2 | open_instruction | number | 5 | 32 | User: Solve this math problem  Solve 0 = -134*s + 397 + 273 for s. Assistant: |
| t_start=0.00:logit_tail | 0.250 | personahub | multiple_choice | short_phrase | Varies by institution | .- by | User: What is the average annual budget allocated specifically for music programs in universities across the United Stat |
| t_start=0.00:uniform | 0.500 | personahub | multiple_choice | short_phrase | Varies by institution | s by institution | User: What is the average annual budget allocated specifically for music programs in universities across the United Stat |
| t_start=0.00:logit_tail | 0.600 | flan_v2 | open_instruction | single_word_or_symbol | Unidentifiable | Unidentmi. | User: In this task, you are given a sentence and a profession. The sentence mentions two professions: one's gender is id |
| t_start=0.00:uniform | 0.600 | synthetic_finalresp | open_instruction | sentence | Yes, Brussels is indeed the capital of Belgium. | yes, Brussels are indeed the of good. | User: Is Brussels the capital of Belgium? Assistant: |
| t_start=0.00:logit_tail | 0.600 | synthetic_finalresp | open_instruction | sentence | Yes, Brussels is indeed the capital of Belgium. | yes, Brussels are indeed the of good. | User: Is Brussels the capital of Belgium? Assistant: |

### Tulu3 Mixed Length 10K

| metric | acc | source_group | prompt_type | answer_type | expected | generated | prompt prefix |
|---|---:|---|---|---|---|---|---|
| t_start=0.00:logit_tail | 0.120 | personahub | open_instruction | long_form | Captain ZOG was convinced that the only way to fix the hyperdrive was with a rub | Mesh Musk once decided that the best way to keep his kitchen safe was to use an  | User: Write two sentences that could appear in a sci-fi comedy script. The first sentence should contain exactly two cap |
| t_start=0.00:uniform | 0.143 | flan_v2 | open_instruction | short_phrase | The Fire-Safe Motorboat. | The Cyber cruise Ship is designed | User: You will be given a definition of a task first, then some input of the task. In this task, you are given a sentenc |
| t_start=0.00:logit_tail | 0.143 | flan_v2 | open_instruction | short_phrase | The Fire-Safe Motorboat. | The Firmac Tower is also | User: You will be given a definition of a task first, then some input of the task. In this task, you are given a sentenc |
| t_start=0.00:uniform | 0.200 | personahub | open_instruction | long_form | Captain ZOG was convinced that the only way to fix the hyperdrive was with a rub | Steph Tesla was convinced that the best way to deal with the mess was with stand | User: Write two sentences that could appear in a sci-fi comedy script. The first sentence should contain exactly two cap |
| t_start=0.00:logit_tail | 0.227 | personahub | open_instruction | multi_line | - Strategies: Engage with genealogy communities by joining relevant groups and f | - Tools: Engage with genealogy communities by sharing posts or videos on social  | User: Provide me with at least 3 bullet points on how genealogy bloggers can effectively use social media to promote the |
| t_start=0.00:uniform | 0.241 | flan_v2 | multiple_choice | sentence | Plan means to decide on and make arrangements for future in advance. Scheduling  | Plan is to make possibilities and fulfill them earlier on certain dates. Organis | User: next question: What is someone doing when scheduling when to go to party? Options: - rumpspringa - meeting new peo |
| t_start=0.00:uniform | 0.255 | personahub | open_instruction | multi_line | - Strategies: Engage with genealogy communities by joining relevant groups and f | - Strategies: Engage with genealogy communities by posting articles and videos o | User: Provide me with at least 3 bullet points on how genealogy bloggers can effectively use social media to promote the |
| t_start=0.00:logit_tail | 0.276 | flan_v2 | multiple_choice | sentence | Plan means to decide on and make arrangements for future in advance. Scheduling  | Plan is to make decisions and make decisions earlier on or later. Plan is to mak | User: next question: What is someone doing when scheduling when to go to party? Options: - rumpspringa - meeting new peo |

## Readout

- Clean decode is near-saturated across all current ablations, so the immediate bottleneck is not latent-to-token decoding.
- Synthetic 10K is already harder than Synthetic 128: pure-noise trajectory is no longer perfect, and the failures are concentrated in short discrete answers such as numbers and yes/no labels.
- Tulu Short 10K has weak controlled denoise at `t=0.1` even though outputs are short. This points to data/task distribution rather than answer length alone.
- Tulu Mixed 10K has better single-step denoise than Tulu Short but worse exact pure-noise trajectory for long answers, which separates local repair ability from long-horizon sampling stability.
- The next decision should be based on which slices dominate the weak `t=0.1:correct` groups: source mix, prompt templates, answer type, or length.
