## 5. Experiments

The experiments evaluate the proposed method from three angles. Section 5.2 examines whether RLDP can estimate input problem difficulty; Section 5.3 evaluates RLDP-AdaSwitch in terms of final answer accuracy and token efficiency; Section 5.4 analyzes reference sample size, layer selection, distance metrics, transfer, noise robustness, and other design choices.

### 5.1 Experimental Setups

**Models.** Experiments use Qwen3-4B, Qwen3-14B [43], and Llama-3.1-Nemotron-Nano-4B (abbreviated Llama-3.1-NN-4B) [2] to demonstrate robustness across LLMs.

**Datasets.** The evaluation covers five benchmarks:

- **OlympiadBench** [15]: International Mathematical Olympiad-style problems.
- **AIME** [41]: challenging competition-level mathematics.
- **MATH500** [27]: high-difficulty problems sampled from MATH.
- **LiveCodeBench** [19]: code generation tasks.
- **ScienceQA** [29]: question answering across scientific domains.

**Baselines.** The paper compares RLDP with:

1. **Zero-shot:** directly asks the LLM to estimate difficulty with a handcrafted prompt.
2. **Few-shot:** provides easy/hard examples before prediction.
3. **AG** [22]: estimates difficulty from output consistency.
4. **LLMs-Ranking** [42]: asks an LLM to compare difficulty across randomly grouped problems and aggregates comparisons into a global ranking.
5. **HaluSearch-Gen** [6]: fine-tunes an LLM to predict problem difficulty.
6. **Probe** [30, 21]: trains a representation-based linear probe for difficulty prediction.

**Evaluation metrics.** Difficulty perception is evaluated using classification accuracy and Macro-F1. The paper reports **E-Acc** as the proportion of easy problems perceived as easy and **H-Acc** as the proportion of hard problems perceived as hard. Adaptive reasoning performance is evaluated using final answer accuracy and total token cost, including both discrimination and reasoning. The accuracy–cost trade-off is summarized by token efficiency:

$$
\mathrm{TE}=\frac{\mathrm{Acc}}{\sqrt{\mathrm{Cost}}}.
$$

**Implementation details.** RLDP uses **10 easy–hard reference pairs** and reports performance over **10 random seeds**. For each model, RLDP uses a single discriminative layer selected as top-1 by LVD. Results for top-2 through top-5 layers appear in Appendix C.4.7.

### 5.2 The Main Results of Difficulty-Perception

Table 1 compares RLDP with baselines on problem difficulty perception. RLDP exhibits stable and effective performance across math reasoning, code generation, and QA. The paper emphasizes that on MATH500, most of RLDP's easy/hard accuracies remain in the **70%–80%** range, whereas many baselines collapse toward predicting nearly all examples as one difficulty class.

On Olympiad with Qwen3-14B, RLDP reaches a Macro-F1 of **67.39%**, outperforming all baselines and demonstrating clear separation between difficulty levels. The authors attribute this robustness to the use of hidden representations associated with different difficulty levels inside Transformer layers.

#### Table 1. Difficulty-perception performance (E-Acc / H-Acc / Macro-F1)

| Model | Method | Olympiad | AIME | MATH500 | LiveCodeBench | ScienceQA |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | Zero-Shot | 8.84 / 98.15 / 35.04 | 1.27 / 99.55 / 38.26 | 30.95 / 92.96 / 39.98 | 53.91 / 82.07 / 67.34 | 95.75 / 2.07 / 34.55 |
| Qwen3-4B | Few-Shot | 34.89 / 88.55 / 52.34 | 14.20 / 95.53 / 48.79 | 59.16 / 81.31 / 55.46 | 79.70 / 58.93 / 68.98 | 58.11 / 61.21 / 59.65 |
| Qwen3-4B | AG | 99.86 / 27.16 / 62.26 | 93.02 / 35.94 / 58.25 | 99.74 / 18.31 / 61.67 | — | 96.69 / 22.31 / 53.00 |
| Qwen3-4B | LLMs-Ranking | 0.00 / 100.00 / 26.21 | 0.00 / 100.00 / 36.99 | 29.63 / 95.77 / 39.51 | — | 94.70 / 4.13 / 36.37 |
| Qwen3-4B | HaluSearch-Gen | 50.56 / 69.39 / 56.97 | 28.42 / 79.26 / 52.52 | 76.32 / 59.09 / 62.40 | 77.88 / 65.52 / 71.59 | 95.03 / 32.88 / 60.10 |
| Qwen3-4B | Probe | 73.48 / 45.71 / 59.63 | 55.58 / 65.78 / 60.56 | 88.16 / 34.55 / 61.48 | 64.52 / 64.71 / 64.46 | 92.25 / 34.52 / 60.05 |
| Qwen3-4B | **Ours** | **67.04 / 60.79 / 62.75** | **61.18 / 62.92 / 61.42** | **74.57 / 74.92 / 64.52** | **77.67 / 65.89 / 71.81** | **70.21 / 54.09 / 61.68** |
| Qwen3-14B | Zero-Shot | 10.45 / 97.93 / 33.57 | 0.50 / 99.70 / 34.85 | 34.88 / 95.66 / 37.88 | 65.31 / 80.67 / 72.83 | 76.12 / 20.55 / 44.01 |
| Qwen3-14B | Few-Shot | 62.46 / 80.00 / 66.17 | 49.15 / 81.05 / 64.40 | 63.05 / 86.36 / 52.57 | 91.59 / 50.00 / 69.48 | 52.26 / 55.15 / 53.70 |
| Qwen3-14B | AG | 96.42 / 34.48 / 67.05 | 96.25 / 39.64 / 64.37 | 99.51 / 15.56 / 60.63 | — | 97.42 / 34.93 / 62.51 |
| Qwen3-14B | LLMs-Ranking | 0.00 / 100.00 / 23.20 | 0.00 / 100.00 / 34.35 | 33.90 / 100.00 / 37.78 | — | 88.79 / 10.27 / 40.33 |
| Qwen3-14B | HaluSearch-Gen | 73.27 / 50.00 / 61.22 | 41.67 / 79.55 / 59.58 | 88.62 / 57.14 / 68.02 | 91.26 / 66.67 / 78.64 | 95.01 / 22.73 / 52.69 |
| Qwen3-14B | Probe | 79.21 / 52.96 / 65.72 | 65.08 / 65.98 / 65.50 | 93.50 / 42.86 / 68.31 | 84.37 / 58.22 / 71.51 | 94.61 / 22.50 / 52.36 |
| Qwen3-14B | **Ours** | **67.29 / 75.26 / 67.39** | **66.54 / 66.41 / 66.17** | **65.83 / 84.86 / 54.17** | **77.60 / 75.28 / 74.22** | **71.35 / 55.29 / 62.67** |
| Llama-3.1-NN-4B | Zero-Shot | 4.27 / 99.03 / 43.69 | 6.15 / 96.31 / 49.72 | 20.62 / 88.32 / 46.18 | 53.87 / 59.89 / 56.84 | 52.95 / 49.97 / 51.45 |
| Llama-3.1-NN-4B | Few-Shot | 13.25 / 92.53 / 48.77 | 3.33 / 98.16 / 48.23 | 27.65 / 85.61 / 49.52 | 86.58 / 34.09 / 57.40 | 23.79 / 74.34 / 45.58 |
| Llama-3.1-NN-4B | AG | 97.56 / 20.00 / 44.46 | 93.08 / 38.77 / 46.31 | 98.05 / 36.04 / 65.69 | — | 72.65 / 1.53 / 27.98 |
| Llama-3.1-NN-4B | LLMs-Ranking | 0.00 / 100.00 / 39.54 | 0.00 / 100.00 / 45.45 | 0.00 / 100.00 / 30.26 | — | 0.00 / 100.00 / 33.33 |
| Llama-3.1-NN-4B | HaluSearch-Gen | 30.00 / 69.89 / 49.81 | 23.08 / 64.80 / 43.67 | 34.62 / 66.67 / 48.09 | 52.83 / 48.62 / 50.70 | 33.94 / 69.58 / 50.18 |
| Llama-3.1-NN-4B | Probe | 39.20 / 72.04 / 55.59 | 19.49 / 85.05 / 52.23 | 66.15 / 64.83 / 65.24 | 81.51 / 47.98 / 58.75 | 50.33 / 59.61 / 54.87 |
| Llama-3.1-NN-4B | **Ours** | **54.09 / 58.60 / 54.93** | **59.75 / 60.98 / 52.28** | **69.55 / 70.75 / 69.64** | **65.49 / 58.92 / 59.40** | **53.60 / 60.62 / 57.06** |

> **Takeaway 3.** RLDP enables stable and low-cost difficulty perception via hidden representations. It achieves balanced difficulty perception across datasets and models without output generation or additional training.

### 5.3 The Main Results of Adaptive Reasoning

The adaptive reasoning experiments apply the Table 1 difficulty predictions to route inference. If a problem is perceived as easy, it is handled using the fast mode; otherwise, the slow mode is triggered. Fast/slow implementations are specified in Appendix C.2. The experiments record answer accuracy, average token cost including both perception and reasoning, and token efficiency.

RLDP-AdaSwitch has a clear advantage in token efficiency. On Qwen3-4B, it reaches an average token efficiency of **1.85**, approximately **2.00×** AG and **1.34×** LLMs-Ranking, while also outperforming training-based approaches in average TE. Probe is a strong baseline because it also uses representations and avoids generation overhead for perception, but it requires supervised training. Rollout-based methods have low token efficiency because of repeated sampling.

#### Table 2. Adaptive reasoning performance under difficulty-aware routing

Each cell reports **Acc / Tokens / TE**; Avg. TE is shown in the final column.

| Model | Method | Olympiad | AIME | MATH500 | LiveCodeBench | ScienceQA | Avg. TE |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | Random (50%) | 75.00 / 5672 / 1.00 | 75.00 / 7273 / 0.88 | 75.00 / 2554 / 1.48 | 75.00 / 7073 / 0.89 | 75.00 / 325 / 4.16 | 1.68 |
| Qwen3-4B | AG | 63.49 / 17540 / 0.48 | 67.90 / 25854 / 0.42 | 60.35 / 10579 / 0.59 | — | 61.16 / 767 / 2.21 | 0.93 |
| Qwen3-4B | LLMs-Ranking | 100.00 / 15088 / 0.81 | 100.00 / 21398 / 0.68 | 97.89 / 6354 / 1.23 | — | 52.10 / 347 / 2.79 | 1.38 |
| Qwen3-4B | HaluSearch-Gen | 84.36 / 6221 / 1.07 | 89.80 / 9563 / 0.92 | 81.67 / 2302 / 1.70 | 81.55 / 6515 / 1.01 | 66.44 / 224 / 4.43 | 1.83 |
| Qwen3-4B | Probe | 73.00 / 4572 / 1.08 | 83.30 / 7767 / 0.95 | 67.06 / 1578 / 1.69 | 82.36 / 7082 / 0.98 | 67.26 / 224 / 4.49 | 1.84 |
| Qwen3-4B | **Ours** | **80.65 / 5430 / 1.10** | **81.75 / 7375 / 0.95** | **86.36 / 2540 / 1.72** | **82.95 / 6526 / 1.03** | **77.05 / 297 / 4.47** | **1.85** |
| Qwen3-14B | Random (50%) | 75.00 / 5451 / 1.02 | 75.00 / 6790 / 0.91 | 75.00 / 2486 / 1.50 | 75.00 / 5226 / 1.04 | 75.00 / 251 / 4.72 | 1.84 |
| Qwen3-14B | AG | 67.67 / 17936 / 0.51 | 70.09 / 25135 / 0.44 | 57.64 / 11633 / 0.53 | — | 67.47 / 981 / 2.15 | 0.91 |
| Qwen3-14B | LLMs-Ranking | 100.00 / 14695 / 0.82 | 100.00 / 20236 / 0.70 | 100.00 / 6057 / 1.28 | — | 55.14 / 530 / 2.39 | 1.30 |
| Qwen3-14B | HaluSearch-Gen | 74.29 / 4393 / 1.12 | 88.91 / 8482 / 0.97 | 78.89 / 1997 / 1.77 | 83.11 / 4407 / 1.25 | 61.37 / 149 / 5.02 | 2.03 |
| Qwen3-14B | Probe | 75.80 / 4416 / 1.15 | 82.89 / 6815 / 1.00 | 69.64 / 1550 / 1.78 | 79.11 / 4202 / 1.22 | 61.25 / 141 / 5.15 | 2.06 |
| Qwen3-14B | **Ours** | **87.67 / 5763 / 1.16** | **83.34 / 6790 / 1.01** | **92.89 / 2845 / 1.74** | **87.64 / 5135 / 1.22** | **77.65 / 227 / 5.15** | **2.06** |
| Llama-3.1-NN-4B | Random (50%) | 75.00 / 7345 / 0.88 | 75.00 / 4304 / 0.80 | 75.00 / 3375 / 1.29 | 75.00 / 9996 / 0.75 | 75.00 / 1628 / 1.86 | 1.12 |
| Llama-3.1-NN-4B | AG | 60.27 / 24910 / 0.38 | 69.66 / 32586 / 0.39 | 67.97 / 16458 / 0.53 | — | 50.77 / 6416 / 0.63 | 0.48 |
| Llama-3.1-NN-4B | LLMs-Ranking | 100.00 / 35252 / 0.53 | 100.00 / 29648 / 0.58 | 100.00 / 13413 / 0.86 | — | 100.00 / 3961 / 1.59 | 0.89 |
| Llama-3.1-NN-4B | HaluSearch-Gen | 85.12 / 8978 / 0.90 | 80.32 / 10599 / 0.78 | 82.92 / 3782 / 1.35 | 74.31 / 9709 / 0.75 | 84.79 / 2081 / 1.86 | 1.13 |
| Llama-3.1-NN-4B | Probe | 84.99 / 8675 / 0.91 | 91.76 / 11751 / 0.85 | 85.69 / 3820 / 1.39 | 73.99 / 7589 / 0.85 | 79.81 / 1656 / 1.96 | 1.19 |
| Llama-3.1-NN-4B | **Ours** | **79.33 / 7546 / 0.91** | **80.44 / 8954 / 0.85** | **85.36 / 3389 / 1.47** | **79.46 / 9524 / 0.81** | **80.31 / 1649 / 1.98** | **1.21** |

> **Takeaway 4.** RLDP-AdaSwitch yields superior token efficiency via allocation of reasoning effort. By identifying difficulty and selectively allocating slow reasoning, it improves token efficiency while maintaining competitive accuracy.

### 5.4 Ablation Studies

#### 5.4.1 Number of Reference Pairs

RLDP is robust to reference-set size. Increasing reference pairs from 1 to 10 improves difficulty perception across datasets, but gains quickly saturate. This indicates that large reference sets are unnecessary.

#### 5.4.2 Layer-wise Perception Analysis

Effective difficulty perception emerges after the embedding layer and closely follows the LVD criterion. Appendix Table 9 shows that the embedding layer has no meaningful discriminative power and collapses to all-hard predictions. Across models, layer-wise F1 trends align with LVD profiles, confirming that LVD identifies difficulty-informative layers.

#### 5.4.3 Distance Metrics

RLDP is sensitive to the distance metric. Euclidean distance performs worst; cosine and normalized Euclidean distances improve moderately. Mahalanobis distance is the most stable and accurate overall because it accounts for anisotropic feature scaling.

| Model | Distance | Olympiad Avg/Std | AIME Avg/Std | MATH500 Avg/Std |
|---|---|---:|---:|---:|
| Qwen3-4B | Euclidean | 0.5780 / 0.0688 | 0.5680 / 0.0687 | 0.5711 / 0.1092 |
| Qwen3-4B | Cosine | 0.6259 / 0.0198 | 0.5917 / 0.0558 | 0.6251 / 0.0276 |
| Qwen3-4B | Class-Normalized Euclidean | 0.6263 / 0.0169 | 0.5912 / 0.0581 | 0.6225 / 0.0282 |
| Qwen3-4B | **Mahalanobis** | **0.6275 / 0.0183** | **0.6142 / 0.0565** | **0.6452 / 0.0282** |
| Qwen3-14B | Euclidean | 0.6512 / 0.0338 | 0.6306 / 0.0353 | 0.5927 / 0.0310 |
| Qwen3-14B | Cosine | 0.6713 / 0.0171 | 0.6615 / 0.0273 | 0.5375 / 0.0325 |
| Qwen3-14B | Class-Normalized Euclidean | 0.6778 / 0.0177 | 0.6636 / 0.0285 | 0.5364 / 0.0274 |
| Qwen3-14B | **Mahalanobis** | **0.6739 / 0.0217** | **0.6617 / 0.0124** | **0.5417 / 0.0322** |
| Llama-3.1-NN-4B | Euclidean | 0.5126 / 0.0958 | 0.4759 / 0.0890 | 0.6605 / 0.0772 |
| Llama-3.1-NN-4B | Cosine | 0.5544 / 0.0450 | 0.5210 / 0.0541 | 0.6834 / 0.0599 |
| Llama-3.1-NN-4B | Class-Normalized Euclidean | 0.5493 / 0.0482 | 0.5204 / 0.0547 | 0.6842 / 0.0569 |
| Llama-3.1-NN-4B | **Mahalanobis** | **0.5493 / 0.0478** | **0.5228 / 0.0499** | **0.6964 / 0.0596** |

#### 5.4.4 Cross-dataset and Cross-class Reference Transfer

The paper evaluates whether references and test sets can come from different datasets. For Qwen3-4B, using Olympiad as the reference set gives AIME performance comparable to in-domain AIME references.

| Reference set | Test: Olympiad E/H/F1 | Test: AIME E/H/F1 |
|---|---:|---:|
| Olympiad | 67.04 / 60.79 / 62.75 | 44.72 / 77.56 / 60.23 |
| AIME | 74.51 / 50.99 / 61.87 | 61.18 / 62.92 / 61.42 |

For inter-class generalization within MATH500, references are swapped between **Intermediate Algebra** and **Precalculus**. Performance remains comparable to in-class references.

| Reference set | Intermediate Algebra E/H/F1 | Precalculus E/H/F1 |
|---|---:|---:|
| Intermediate Algebra | 58.82 / 66.88 / 56.49 | 66.29 / 37.50 / 51.68 |
| Precalculus | 54.26 / 67.69 / 56.61 | 62.39 / 45.00 / 52.88 |

#### 5.4.5 Reference-set Noise Robustness

The paper injects mislabeled samples into Qwen3-4B reference sets. RLDP degrades only slightly at low noise, declines gradually as noise increases, and clearly fails only near a 50% noise ratio.

| Method | Olympiad E/H/F1 | AIME E/H/F1 | MATH500 E/H/F1 |
|---|---:|---:|---:|
| RLDP (5/10 clean) | 53.24 / 47.57 / 49.23 | 52.91 / 51.37 / 51.19 | 41.03 / 45.90 / 35.98 |
| RLDP (7/10 clean) | 60.88 / 60.26 / 59.04 | 56.46 / 52.35 / 53.49 | 63.53 / 63.28 / 54.48 |
| RLDP (8/10 clean) | 65.28 / 61.64 / 62.17 | 59.60 / 61.94 / 60.16 | 70.71 / 70.33 / 60.67 |
| RLDP (9/10 clean) | 62.89 / 60.33 / 60.20 | 61.77 / 62.20 / 61.30 | 74.13 / 74.26 / 63.99 |
| RLDP (10/10 clean) | 67.04 / 60.79 / 62.75 | 61.18 / 62.92 / 61.42 | 74.57 / 74.92 / 64.52 |

#### 5.4.6 Probe under Limited Data

Probe performance drops substantially as training data is reduced. Under the fair low-resource setting of only 10 reference pairs, Probe falls below RLDP. On Olympiad, RLDP reaches **62.75%** F1 compared with **47.08%** for Probe using 10 pairs.

| Method | Olympiad E/H/F1 | AIME E/H/F1 | MATH500 E/H/F1 |
|---|---:|---:|---:|
| Probe (70%) | 73.48 / 45.71 / 59.63 | 55.58 / 65.78 / 60.56 | 88.16 / 34.55 / 61.48 |
| Probe (30%) | 69.27 / 45.35 / 57.16 | 48.87 / 64.65 / 56.72 | 89.47 / 27.80 / 58.98 |
| Probe (10 pairs) | 29.58 / 82.04 / 47.08 | 31.90 / 77.33 / 53.40 | 53.64 / 76.07 / 50.65 |
| **RLDP (10 pairs)** | **67.04 / 60.79 / 62.75** | **61.18 / 62.92 / 61.42** | **74.57 / 74.92 / 64.52** |

#### 5.4.7 Effect of Using Multiple Discriminative Layers

The main experiments use the single top-1 LVD layer. The appendix varies top-n layers from 1 to 5. Additional layers produce only marginal and inconsistent changes; sometimes they help slightly, sometimes they degrade performance. Because optimal layer count varies across models and datasets, the paper adopts top-1 for simplicity and robustness.

For example, on Qwen3-4B, top-1 gives 62.75 / 61.42 / 64.52 Macro-F1 on Olympiad / AIME / MATH500; top-5 gives 62.93 / 61.37 / 64.55, a negligible change. On Qwen3-14B, top-1 is much stronger than top-2 through top-5 for several datasets, showing the sensitivity introduced by multi-layer aggregation.
