## Appendix J — Use of existing assets

> **Table 8:** List of models used in this work.

| Name | Source | License |
| --- | --- | --- |
| `Llama-3.1-8B-Instruct` | Grattafiori et al. (2024) | Meta Llama 3.1 Community License |
| `Llama-3.3-70B-Instruct` | Grattafiori et al. (2024) | Meta Llama 3.3 Community License |
| `Qwen2.5-7B-Instruct` | Yang et al. (2025) | Apache License 2.0 |
| `Gemma-2-9B-IT` | Riviere et al. (2024) | Gemma License (commercial-friendly terms of use) |
| `Mistral-Small-24B-Instruct-2501` | Mistral AI (2025) | Apache License 2.0 |

> **Table 9:** List of datasets used in this work.

| Dataset | Source | License |
| --- | --- | --- |
| LongFact | Wei et al. (2024b) | Apache License 2.0 |
| TriviaQA | Joshi et al. (2017) | Apache License 2.0 |
| HealthBench | Arora et al. (2025) | MIT License |
| SimpleQA | Wei et al. (2024a) | MIT License |

---

## Appendix K — Compute statement

Training a LoRA-based probe for `Llama-3.1-8B-Instruct` on the full annotated dataset (as specified in Section 4.1) with a batch size of 8 takes less than **2 hours on an H100 GPU**.
