## 4 Long-form hallucination detection

### 4.1 Experimental setup

#### Models.

We primarily focus our analysis on two models ("*primary models*"): `Llama-3.1-8B-Instruct` and `Llama-3.3-70B-Instruct` (Grattafiori et al., 2024). We also replicate key results using three additional models ("*secondary models*"): `Gemma-2-9B-IT` (Riviere et al., 2024), `Qwen-2.5-7B-Instruct` (Yang et al., 2025), and `Mistral-Small-24B-Instruct-2501` (Mistral AI, 2025). All models studied in this paper are instruction-tuned models. For brevity, model names will henceforth exclude the "Instruct" or "IT" suffix.

#### Training data.

For our primary models, we train on a mixture of long-form and short-form data. We find that training on a mix of long-form and short-form data yields the best overall performance; training only on long-form data also works well (see Section 5.1). For each model $M$, we sample $n_{\text{LF}}^{M}$ long-form prompts from LongFact and LongFact++, and $n_{\text{SF}}^{M}$ short-form prompts from TriviaQA (Joshi et al., 2017), and then generate one completion per prompt. For `Llama-3.1-8B`, we use $n_{\text{LF}}^{M}{=}8{,}000$ and $n_{\text{SF}}^{M}{=}2{,}000$; for `Llama-3.3-70B`, we use $n_{\text{LF}}^{M}{=}8{,}000$ and $n_{\text{SF}}^{M}{=}1{,}000$. For secondary models (`Gemma`, `Qwen`, and `Mistral`), we use $n_{\text{LF}}^{M}{=}2{,}000$ and $n_{\text{SF}}^{M}{=}0$. Labels for long-form completions follow the pipeline in Section 3.1. For short-form completions, we extract and label only the single entity span corresponding to the answer of the trivia question (the "*answer entity span*"). For the results in this section, we train probes on our primary models using labeled generations from *all models* (primary and secondary), yielding a training corpus of $\sim$25,000 total samples. For more details on data generation, see Appendix C.2.

#### Evaluation.

Unless otherwise noted, detectors are always evaluated in the same-model setting: each probe is tested on generations from its own original model. All models share a common long-form test set of 1,000 LongFact and 1,000 LongFact++ prompts. We also evaluate performance on short-form completions using TriviaQA (Joshi et al., 2017). To test generalization, we evaluate on two held-out datasets: HealthBench (Arora et al., 2025), which contains unseen long-form medical-domain prompts, and MATH (Hendrycks et al., 2021b), which tests performance on an out-of-distribution mathematical reasoning task without discrete entities.

> **Table 1:** Detection performance on `Llama-3.1-8B` and `Llama-3.3-70B` across test sets of LongFact, HealthBench, TriviaQA, and MATH. We report AUC and recall at 10% false positive rate (R@0.1). Probes (linear, LoRA) outperform uncertainty-based baselines; LoRA is strongest across all settings. See Appendix G for evaluation on LongFact++ prompts, and results for secondary models.

| Dataset | Method | Llama-3.1-8B AUC ($\uparrow$) | R@0.1 ($\uparrow$) | Llama-3.3-70B AUC ($\uparrow$) | R@0.1 ($\uparrow$) |
| --- | --- | --- | --- | --- | --- |
| **LongFact** (long-form) | Perplexity | 0.7600 | 0.3616 | 0.7062 | 0.3011 |
|  | Entropy | 0.7415 | 0.2868 | 0.7118 | 0.3027 |
|  | Semantic entropy | 0.7189 | 0.2739 | 0.7138 | 0.3915 |
|  | Linear probe | 0.8535 | 0.5878 | 0.8667 | 0.6451 |
|  | **LoRA probe** | **0.8938** | **0.6801** | **0.9048** | **0.7228** |
| **HealthBench** (long-form, held-out) | Perplexity | 0.6506 | 0.2022 | 0.6446 | 0.2363 |
|  | Entropy | 0.6650 | 0.2535 | 0.6466 | 0.2377 |
|  | Semantic entropy | 0.6537 | 0.2411 | 0.6042 | 0.2575 |
|  | Linear probe | 0.8560 | 0.5843 | 0.8730 | 0.6479 |
|  | **LoRA probe** | **0.8960** | **0.6804** | **0.9057** | **0.7116** |
| **TriviaQA** (short-form) | Perplexity | 0.9021 | 0.7508 | 0.8121 | 0.5048 |
|  | Entropy | 0.9382 | 0.8628 | 0.8423 | 0.5524 |
|  | Semantic entropy | 0.9103 | 0.7500 | 0.8104 | 0.5525 |
|  | Linear probe | 0.9179 | 0.7649 | 0.9484 | 0.8590 |
|  | **LoRA probe** | **0.9651** | **0.9062** | **0.9827** | **0.9486** |
| **MATH** (reasoning, held-out) | Perplexity | 0.7143 | 0.1557 | 0.7802 | 0.4299 |
|  | Entropy | 0.7818 | 0.4481 | 0.6887 | 0.3178 |
|  | Semantic entropy | 0.8520 | 0.5767 | 0.7930 | 0.3981 |
|  | Linear probe | 0.8450 | 0.5739 | 0.8641 | 0.6877 |
|  | **LoRA probe** | **0.8845** | **0.6913** | **0.8750** | 0.6476 |

Our evaluation measures how well each method classifies individual entities as either supported or hallucinated. To do this, we assign a score to each entity using the *span-max* rule, where an entity's score is the maximum of any token within its span. In long-form tasks, we score all annotated entities, while for short-form QA, we score only the single entity corresponding to the answer of the question. For mathematical reasoning, which lacks entities, we score the entire completion by its maximum token score. The performance of this classification task is then measured by the area under the receiver operating characteristic curve (AUC) and recall at a 10% false positive rate (R@0.1). All specific labeling and scoring protocols are detailed in Appendix D.

### 4.2 Results

In long-form settings (LongFact and HealthBench), token-level probes markedly outperform baselines for both primary models (Table 1). Simple linear probes consistently achieve AUCs above **0.85**, and LoRA probes improve even further, pushing AUCs above **0.89**. In comparison, the uncertainty-based baselines all struggle, failing to exceed **0.76 AUC**.

In the short-form setting (TriviaQA), the baselines are stronger than in the long-form setting, yet probes still lead. Our LoRA probes consistently achieve greater than **0.96 AUC**, and linear probes also perform well.

Notably, our probes also achieve strong results on the MATH dataset. This out-of-distribution performance suggests our method captures signals of correctness that generalize beyond its original target of fabricated entities. An annotated example from the MATH dataset is provided in Appendix B.2.

We replicate the long-form results on the three secondary models, training each on only 2,000 annotated samples of its own long-form generations. The results are similar: LoRA probes again outperform linear probes, with AUCs ranging between **0.87–0.90** on LongFact generations. Full results for secondary models are displayed in Table 5.

While LoRA probe AUCs approach or exceed 0.9 in several settings, R@0.1 on long-form tops out around **0.7**, i.e., at 10% false positive rate, the detector recovers roughly two-thirds of hallucinated entities. These results underscore both the practical gains over standard uncertainty-based baselines, and also the remaining headroom before such methods can be used broadly in high-stakes contexts.
