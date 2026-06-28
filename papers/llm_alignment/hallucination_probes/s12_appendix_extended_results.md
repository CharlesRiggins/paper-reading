## Appendix G — Extended results: long-form hallucination detection

### G.1 LongFact++ evaluation, primary models

> **Table 4:** Extended results for Table 1, displaying evaluations on LongFact++.

| Dataset | Method | Llama-3.1-8B AUC ($\uparrow$) | R@0.1 ($\uparrow$) | Llama-3.3-70B AUC ($\uparrow$) | R@0.1 ($\uparrow$) |
| --- | --- | --- | --- | --- | --- |
| **LongFact++** (long-form) | Semantic entropy | 0.7082 | 0.2368 | 0.6757 | 0.2885 |
|  | Entropy | 0.7300 | 0.2900 | 0.7389 | 0.3701 |
|  | Perplexity | 0.7466 | 0.3400 | 0.7313 | 0.3424 |
|  | Linear probe | 0.8678 | 0.6207 | 0.8937 | 0.6971 |
|  | **LoRA probe** | **0.9036** | **0.7052** | **0.9265** | **0.7788** |

### G.2 LongFact and LongFact++ evaluation, secondary models

> **Table 5:** Results for secondary models, as referenced in Section 4.2.

| Dataset | Method | Gemma-2-9B AUC ($\uparrow$) | R@0.1 ($\uparrow$) | Qwen-2.5-7B AUC ($\uparrow$) | R@0.1 ($\uparrow$) | Mistral-Small-24B AUC ($\uparrow$) | R@0.1 ($\uparrow$) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LongFact** (long-form) | Linear probe | 0.8200 | 0.5362 | 0.8383 | 0.5432 | 0.8479 | 0.5752 |
|  | **LoRA probe** | **0.8733** | **0.6206** | **0.8947** | **0.6645** | **0.8894** | **0.6761** |
| **LongFact++** (long-form) | Linear probe | 0.8386 | 0.5560 | 0.8467 | 0.5549 | 0.8722 | 0.6278 |
|  | **LoRA probe** | **0.8860** | **0.6327** | **0.8961** | **0.6757** | **0.8893** | **0.6927** |

---

## Appendix H — Extended results: selective answering

> **Table 6:** Selective answering results for all models. Selective answering (Section 5.4) improves conditional accuracy, at the cost of decreasing the total number of questions attempted. The selective answering results displayed here are obtained using a probe threshold of $t{=}0.5$.

| Model | Conditional accuracy (%) — No intervention | Conditional accuracy (%) — Selective answering | Attempt rate (%) — No intervention | Attempt rate (%) — Selective answering |
| --- | --- | --- | --- | --- |
| `Llama-3.1-8B` | 19.7 | **48.8** | 10.1 | 2.2 |
| `Llama-3.3-70B` | 27.9 | **50.4** | 76.1 | 19.1 |
| `Mistral-Small-24B` | 18.6 | **37.6** | 29.5 | 7.6 |
| `Gemma-2-9B` | 9.1 | **23.2** | 59.8 | 9.2 |
| `Qwen-2.5-7B` | 5.5 | **11.3** | 79.4 | 11.2 |
