## 5. Realistic Dataset (Wikipedia) Results

The main experiment moves from synthetic language labels to natural document-level topical labels. A `254M`-parameter model is trained from scratch on **3.7B English Wikipedia tokens**. This setup uses `STEM.Biology` as a controllable proxy for a sensitive target domain, not a direct CBRN evaluation.

### 5.1 Dataset, labels, and baselines

Wikipedia's ORES `articletopic` classifier assigns each article one of 64 topical labels; the highest classifier score is selected. `STEM.Biology` is the forget class and accounts for **3.7%** of all training tokens. The remaining data is retain content.

The evaluation separates three sets, reserving **5,000 articles per subcategory**:

| Set | Topics | Role |
|---|---|---|
| Forget | Biology | target loss should increase after removal |
| Related retain | Earth & Environment, Medicine & Health, Chemistry | measures collateral damage to biology-adjacent knowledge |
| General retain | Culture, Geography, History & Society | measures broad non-target utility |

All labelled biology data becomes $\mathbf{D}_{\text{forget}}$. Non-biology data is split 90% unlabelled and 10% confident retain. SGTM reserves **1/32** attention heads and **64/4096** MLP units in each block for the forget set. Results are averaged across three runs and use post-hoc logit calibration.

The baselines have the same training-step compute budget:

- **Weak filter:** remove only articles labelled biology.
- **Strict filter:** additionally remove Earth & Environment, Medicine & Health, and Chemistry.
- **No filter:** standard training on the original mixture.

### 5.2 Retain/forget results

Unlike `TinyStories`, the authors do not add an artificial error rate. Natural noise arises because article-level labels cannot isolate every biology fact inside non-biology documents.

On both general and biology-adjacent retain axes, Figure 1 shows SGTM has higher biology loss at a fixed retain loss than weak and strict filtering. Strict filtering forgets more strongly than weak filtering because it deletes more nearby source content, but also damages related capability more:

- On **general** knowledge, filtered models can end with lower loss than the no-filter model because the fixed budget is redistributed to more non-biology material. SGTM ends slightly worse; Appendix B converts this to a roughly **5–6%** general compute penalty.
- On **biology-adjacent** knowledge, SGTM's final retain loss sits between weak and strict filters while its biology forget loss is higher than both. This is the intended trade-off: retain more useful medicine/chemistry/environment content than strict filtering, but learn less target biology from it than weak filtering.

### 5.3 Robustness to adversarial fine-tuning

A high target loss could reflect shallow suppression rather than actual removal. The authors therefore full-parameter fine-tune each post-removal model on a **50/50 mixture of forget and retain data**, measuring how long it takes to recover the biology loss of the no-filter baseline. Each step consumes **260k forget tokens**.

| Starting model | Fine-tuning to baseline biology loss | Interpretation |
|---|---:|---|
| `RMU` post-training unlearning | **50 steps / 13M** forget tokens | rapidly recoverable |
| Weak filter | **325 steps / 85M** forget tokens | closer to relearning from filtered pretraining |
| SGTM | **350 steps / 92M** forget tokens | **7×** RMU's steps; matches strict filtering |
| Strict filter | **350 steps / 92M** forget tokens | strongest filtering comparator |

The result supports, but does not prove, deeper removal: SGTM's recovery time matches a model that never saw broad adjacent data under strict filtering. Appendix E controls for SGTM's somewhat worse initial general loss using undertrained filtering checkpoints. The later SGTM trajectory remains similar to a retain-matched weak filter and stays ahead of strict filtering for roughly the first 100 steps, so the effect cannot be explained solely by generic degradation. The same appendix also notes a nuance: SGTM initially relearns faster than weak filtering, suggesting some target knowledge was removed superficially even though a substantial component remains slow to recover.

### 5.4 Per-token loss distribution

Mean loss could be high because of a handful of catastrophic token failures while most biology remains intact. Figure 6(b) tests this directly. After ablation, SGTM shifts biology-token probability mass toward **moderately higher losses** relative to filtering and no-filter baselines rather than creating a heavy tail of rare extreme losses. Together with the re-finetuning result, this is evidence that the effect is broadly distributed across target tokens; it is still an indirect proxy for downstream capability removal.
