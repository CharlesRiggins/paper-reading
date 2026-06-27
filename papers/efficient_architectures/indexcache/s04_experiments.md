## 4. Experiments

### 4.1 Setup

#### Model

The main experiments use a DSA model obtained by two-stage training from `GLM-4.7-Flash`, a 30B-A3B MoE model with MLA and 47 layers. Its evaluation performance is comparable to the original `GLM-4.7-Flash` reported in the `GLM-5` technical report.

#### Training-free IndexCache

The greedy pattern search uses per-token validation loss on SFT data. Calibration uses batch size **768** and context length **200K**. All candidate pattern evaluations use the same batches so that measured loss differences isolate the effect of indexer sharing.

#### Training-aware IndexCache

A full DSA pipeline from the base model is expensive, so the paper initializes directly from `GLM-4.7-Flash` and trains it into a DSA model on SFT data at **200K** context length. Training uses a **1,000-step** dense warm-up followed by a **4,000-step** sparse training phase. This shorter pipeline closely matches full DSA training and is sufficient to evaluate sharing-aware distillation.

#### Evaluation

Evaluation covers five long-context benchmarks: `MRCR v2`, `GraphWalks`, `LongBench v2`, `RULER`, and `AA-LCR`. It also covers four general and reasoning benchmarks: `AIME 2025`, `GPQA-Diamond`, `LiveCodeBench v6`, and `IFBench`. Appendix D specifies temperature, top-$p$, top-$k$, context budget, and truncation rules.

### 4.2 End-to-End Inference Speedup

End-to-end inference is measured on the 30B DSA model served with `dp_attention` enabled (`dp_size=8`) in `SGLang`, running on an NVIDIA H100 node. The comparison is original DSA versus IndexCache at retention ratios **1/2** and **1/4**. Three metrics are reported across 10K, 60K, 120K, and 200K context lengths: prefill latency, per-request decode throughput under single concurrency, and full-KV-cache decode throughput.

| Metric | Config | 10K | 60K | 120K | 200K |
|---|---:|---:|---:|---:|---:|
| Prefill time (s) ↓ | DSA | 0.57 | 3.38 | 8.57 | 19.5 |
| Prefill time (s) ↓ | + IndexCache (1/2) | 0.47 | 2.86 | 6.57 | 13.7 |
| Prefill time (s) ↓ | + IndexCache (1/4) | **0.45** | **2.59** | **5.66** | **10.7** |
| Decode throughput, per request (tok/s) ↑ | DSA | 73.5 | 67.0 | 63.0 | 58.0 |
| Decode throughput, per request (tok/s) ↑ | + IndexCache (1/2) | 84.5 | 80.0 | 77.0 | 73.0 |
| Decode throughput, per request (tok/s) ↑ | + IndexCache (1/4) | **91.0** | **89.5** | **88.0** | **86.0** |
| Decode throughput, full KV cache (tok/s) ↑ | DSA | 2700 | 613 | 341 | 197 |
| Decode throughput, full KV cache (tok/s) ↑ | + IndexCache (1/2) | 3070 | 750 | 431 | 253 |
| Decode throughput, full KV cache (tok/s) ↑ | + IndexCache (1/4) | **3310** | **840** | **498** | **297** |

#### Prefill

IndexCache delivers substantial prefill acceleration, and the gain grows with context length. At 200K tokens, `IndexCache (1/4)` reduces prefill latency from **19.5s** to **10.7s**, a **1.82×** speedup, by eliminating 75% of indexer computations that dominate the prefill phase. Even at 10K, where the indexer accounts for a smaller fraction of compute, the method still gives about **1.27×** speedup. The paper expects larger gains at contexts beyond 200K.

#### Decode

Decode throughput improves significantly at long contexts because DSA must run a per-token indexer pass over the full context. At 200K tokens, DSA decodes at **58 tok/s**, while `IndexCache (1/4)` reaches **86 tok/s**, a **1.48×** speedup. When the KV cache is fully saturated, `IndexCache (1/4)` improves total decode throughput by **22–51%**, with 200K throughput increasing from **197** to **297 tok/s**.

The larger `GLM-5` model shows similar trends. `IndexCache (1/4)` yields at least **1.3×** improvement in both prefill latency and decode throughput beyond 100K context. Overall, IndexCache is most valuable exactly where DSA's remaining indexer cost becomes largest: long-context inference.

### 4.3 Training-Free IndexCache Results

Training-free IndexCache is evaluated on the same 30B DSA model at retention ratios **1/2**, **1/4**, and **1/8**. Each ratio is tested with uniform interleaving and with the greedy-searched pattern from Appendix B.

| Config | Long Avg | G&R Avg | MRCR | GW | LB2 | RULER | LCR | AIME | GPQA | LCB | IFB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original DSA | **50.2** | 74.6 | 24.5 | 49.6 | 45.5 | 87.9 | 43.6 | 91.0 | 77.6 | 71.4 | 58.4 |
| 1/2 Unif. IndexCache | 47.4 | 74.3 | 22.0 | 46.6 | 46.0 | 83.6 | 38.6 | 92.2 | 76.4 | 69.7 | 59.0 |
| + Search pattern | **50.3** | 74.4 | 24.7 | 49.5 | 46.3 | 87.8 | 43.2 | 91.9 | 76.3 | 71.3 | 58.2 |
| 1/4 Unif. IndexCache | 43.0 | 73.8 | 17.7 | 37.2 | 43.1 | 79.2 | 37.8 | 91.3 | 75.7 | 69.4 | 58.9 |
| + Search pattern | **49.9** | **74.9** | **25.1** | 47.4 | 45.7 | 87.6 | **43.8** | **92.6** | **78.6** | 70.0 | 58.3 |
| 1/8 Unif. IndexCache | 35.3 | 70.0 | 12.9 | 33.1 | 37.7 | 68.8 | 24.0 | 89.1 | 74.1 | 58.7 | 58.0 |
| + Search pattern | 46.1 | 73.7 | 21.7 | 43.8 | 42.3 | 82.0 | 40.8 | 90.7 | 76.5 | 69.6 | 58.1 |

#### Searched patterns close the gap on long-context tasks

Uniform interleaving causes significant long-context degradation at aggressive retention ratios. Long Avg drops by **2.8** points at 1/2 retention and **7.2** points at 1/4 retention. Greedy-searched patterns largely remove this deficit: Long Avg recovers to **50.3** at 1/2 and **49.9** at 1/4, both comparable to the original DSA value of **50.2**. This confirms that which indexer layers are retained matters more than the raw count.

At 1/8 retention, quality loss becomes non-negligible even with search. The searched pattern still improves Long Avg from **35.3** to **46.1**, but the gap to original DSA remains meaningful. Thus IndexCache can remove 75% of indexers safely in this setting, while removing 87.5% is too aggressive for training-free use.

#### Long chain-of-thought reasoning capabilities are preserved

Across all configurations except uniform 1/8, G&R Avg stays within one point of the **74.6** baseline. The 1/4 searched pattern even improves over DSA on `AIME 2025` (**92.6** vs. **91.0**) and `GPQA-Diamond` (**78.6** vs. **77.6**). The paper interprets this as evidence that removing redundant indexer computation does not trade away general reasoning ability and may mildly regularize inference.

### 4.4 Training-Aware IndexCache Results

Training-aware IndexCache uses the multi-layer distillation loss from Section 3.2 at retention ratios **1/2** and **1/4**, both with uniform interleaving. Two ablations are run at 1/2 retention: replacing uniform interleaving with the greedy-searched pattern, and removing the cross-layer distillation loss so each indexer is trained only for its own layer. The DSA baseline here is trained with the shortened DSA pipeline, so its numbers differ slightly from Table 2.

| Config | Long Avg | G&R Avg | MRCR | GW | LB2 | RULER | LCR | AIME | GPQA | LCB | IFB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original DSA | 51.0 | 74.2 | 24.7 | 49.1 | 46.9 | 87.3 | 47.0 | 88.8 | 79.4 | 70.5 | 57.9 |
| 1/2 Unif. IndexCache | **51.6** | **74.5** | 23.8 | **50.2** | **47.2** | 87.0 | **49.8** | 89.3 | 76.7 | **72.2** | **59.9** |
| w/ searched pattern | 50.6 | 73.6 | 23.9 | 48.1 | 47.1 | **87.5** | 46.6 | **89.6** | 78.6 | 68.5 | 57.7 |
| w/o cross-layer loss | 49.8 | **74.5** | 24.6 | 48.3 | 45.0 | 87.1 | 44.0 | 88.8 | **79.4** | 71.7 | 58.0 |
| 1/4 Unif. IndexCache | 50.6 | 74.1 | 23.7 | 48.1 | 46.9 | 86.1 | 48.4 | 89.3 | 78.0 | 70.5 | 58.7 |

#### Training-aware IndexCache matches the DSA baseline

Uniform IndexCache at 1/2 retention reaches Long Avg **51.6**, surpassing the baseline **51.0**, while G&R Avg stays comparable (**74.5** vs. **74.2**). At 1/4 retention, both Long Avg and G&R Avg are within roughly **0.4** aggregate points of baseline. This confirms that DSA can be trained to adapt to cross-layer sharing.

#### Pattern sensitivity vanishes with training

In the training-free setting, searched patterns are essential. In the training-aware setting, however, uniform interleaving at 1/2 retention performs on par with or slightly above the searched pattern: Long Avg **51.6** vs. **50.6**, G&R Avg **74.5** vs. **73.6**. The explanation is that retraining lets `S` layers adapt to inherited indices while retained indexers learn selections that generalize across served layers. This eliminates the layer-specific sensitivity observed when weights are frozen.

#### Cross-layer distillation provides a meaningful benefit

Removing the cross-layer loss drops Long Avg from **51.6** to **49.8**, with `AA-LCR` falling from **49.8** to **44.0**. This validates the multi-layer distillation objective: training each retained indexer toward the centroid of served-layer attention distributions produces a top-$k$ set that generalizes across layers rather than overfitting to one.

### 4.5 Scaling Experiment

The paper also evaluates training-free IndexCache on `GLM-5`, a production-scale 744B-parameter model with 40B active parameters that uses DSA by default.

| Config | Long Avg | MRCR v2 | GraphWalks | LongBench v2 | RULER | AA-LCR |
|---|---:|---:|---:|---:|---:|---:|
| Original DSA | 78.4 | 71.1 | **92.7** | 64.5 | **97.7** | 66.2 |
| 1/2 Unif. IndexCache | 78.1 | **72.8** | 90.2 | 65.1 | 97.6 | 64.6 |
| + Searched pattern | **78.7** | 72.3 | 90.8 | **66.0** | 97.3 | 67.2 |
| 1/4 Unif. IndexCache | 72.7 | 65.8 | 74.9 | 62.2 | 96.2 | 64.6 |
| + Searched pattern | 78.0 | 70.8 | 90.3 | 63.7 | 97.6 | **67.6** |

The trends mirror the 30B results. Uniform interleaving degrades at aggressive retention, while the searched pattern recovers quality. At 1/2 retention, the searched pattern slightly exceeds baseline Long Avg (**78.7** vs. **78.4**). At 1/4 retention, it remains within **0.4** points (**78.0** vs. **78.4**). The paper also reports that 1/2-retention IndexCache is nearly identical to original `GLM-5` across the Artificial Analysis Index.

The authors plan to apply training-aware IndexCache to the production-scale model. Since training-free IndexCache already matches baseline quality, they expect sharing-aware training to further stabilize deployment-time efficiency gains.
