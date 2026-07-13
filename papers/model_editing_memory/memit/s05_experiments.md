## 5. Experiments

### 5.1. Models and Baselines

The experiments use two autoregressive LLMs: `GPT-J` (6B) and `GPT-NeoX` (20B). MEMIT is compared against three baselines:

- **FT-W**: naive fine-tuning with weight decay to reduce forgetting.
- **MEND**: a hypernetwork-based model editor that edits multiple facts together.
- **ROME**: a direct rank-one model editing method applied sequentially, one fact at a time.

`SERAC` is not included because public code was unavailable at the time. Implementation details are given in Appendix B.

### 5.2. MEMIT Scaling

#### 5.2.1. Editing 10K Memories in `zsRE`

The first scaling test uses `zsRE`, a question-answering dataset from which the authors extract **10,000 real-world facts**. Since `zsRE` does not include generation tasks, evaluation uses prediction metrics only. **Efficacy** measures top-1 recall on the original prompt, **Paraphrase** measures the same on paraphrases, **Specificity** measures top-1 accuracy on unrelated facts that should not change, and **Score** is the harmonic mean of those three scores.

**Table 1: 10,000 `zsRE` edits on `GPT-J` (6B).**

| Editor | Score ↑ | Efficacy ↑ | Paraphrase ↑ | Specificity ↑ |
|---|---:|---:|---:|---:|
| `GPT-J` | 26.4 | 26.4 (±0.6) | 25.8 (±0.5) | 27.0 (±0.5) |
| `FT-W` | 42.1 | 69.6 (±0.6) | 64.8 (±0.6) | 24.1 (±0.5) |
| `MEND` | 20.0 | 19.4 (±0.5) | 18.6 (±0.5) | 22.4 (±0.5) |
| `ROME` | 2.6 | 21.0 (±0.7) | 19.6 (±0.7) | 0.9 (±0.1) |
| `MEMIT` | **50.7** | **96.7 (±0.3)** | **89.7 (±0.5)** | **26.6 (±0.5)** |

MEMIT performs best at 10,000 edits: most new memories are recalled, paraphrase generalization remains high, and specificity is close to the unedited model. Fine-tuning is stronger than MEND and ROME at this scale, likely because it applies one aggregate objective rather than many sequential updates. ROME collapses specificity under sequential mass editing.

#### 5.2.2. `COUNTERFACT` Scaling Curves

The second scaling test uses `COUNTERFACT`, a collection of **21,919** factual statements. The paper filters conflicts that violate Eqn. 5, then inserts $n$ counterfactuals for:

$$
n\in\{1,2,3,6,10,18,32,56,100,178,316,562,1000,1778,3162,5623,10000\}.
$$

The metrics follow Meng et al. [29]:

- **Efficacy Success (ES)**: the new object $o_i$ has higher probability than the true object $o_i^c$ under the original prompt.
- **Paraphrase Success (PS)**: the same comparison under rephrased prompts.
- **Neighborhood Success (NS)**: specificity on semantically related but unedited subjects, where the correct original answer should remain more probable.
- **Editing Score (S)**: harmonic mean of ES, PS, and NS.
- **Reference Score (RS)**: semantic consistency of free-form generations with reference Wikipedia text about the new object.
- **Generation Entropy (GE)**: fluency measure based on bi-gram and tri-gram entropy, used to detect repetition collapse.

Figure 5 shows that ROME works up to about $n=10$ but degrades starting at $n=32$. MEND works for $n=1$ but rapidly declines at $n=6$, loses efficacy before $n=1000$, and at $n=10000$ has little effect on the model. MEMIT performs best at large $n$. At small $n$, ROME has stronger generalization but lower specificity, suggesting that its hard equality constraint yields robust but less localized edits compared with MEMIT’s soft minimization.

**Table 2: Numerical results on `COUNTERFACT` for 10,000 edits.**

| Model / Editor | S ↑ | ES ↑ | PS ↑ | NS ↑ | GE ↑ | RS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `GPT-J` | 22.4 | 15.2 (0.7) | 17.7 (0.6) | 83.5 (0.5) | 622.4 (0.3) | 29.4 (0.2) |
| `GPT-J` + `FT-W` | 67.6 | **99.4 (0.1)** | 77.0 (0.7) | 46.9 (0.6) | 293.9 (2.4) | 15.9 (0.3) |
| `GPT-J` + `MEND` | 23.1 | 15.7 (0.7) | 18.5 (0.7) | **83.0 (0.5)** | 618.4 (0.3) | 31.1 (0.2) |
| `GPT-J` + `ROME` | 50.3 | 50.2 (1.0) | 50.4 (0.8) | 50.2 (0.6) | 589.6 (0.5) | 3.3 (0.0) |
| `GPT-J` + `MEMIT` | **85.8** | 98.9 (0.2) | **88.6 (0.5)** | 73.7 (0.5) | **619.9 (0.3)** | **40.1 (0.2)** |
| `GPT-NeoX` | 23.7 | 16.8 (1.9) | 18.3 (1.7) | 81.6 (1.3) | 620.4 (0.6) | 29.3 (0.5) |
| `GPT-NeoX` + `MEMIT` | **82.0** | **97.2 (0.8)** | **82.2 (1.6)** | 70.8 (1.4) | 606.4 (1.0) | **36.9 (0.6)** |

At 10,000 `COUNTERFACT` edits, MEMIT achieves **85.8** editing score on `GPT-J` and **82.0** on `GPT-NeoX`. FT-W has excellent efficacy but damages generation quality, with GE falling from 622.4 to 293.9 and RS from 29.4 to 15.9. MEMIT keeps fluency close to baseline while improving semantic consistency above the original model.

Runtime analysis shows that MEND is fastest at **98.25 seconds** for 10,000 edits. FT-W takes about **29 minutes**. MEMIT and ROME are slower at **7.44 hours** and **12.29 hours**, respectively. The authors note that MEMIT’s implementation computes $z_i$ vectors serially even though those optimizations are embarrassingly parallel.

### 5.3. Editing Different Categories of Facts

The authors examine the 27 `COUNTERFACT` relation categories with at least 300 cases each. MEMIT achieves better overall scores than FT and MEND in every category. Some relations are harder than others; for example, changing the sport an athlete plays is difficult for all editors. Even on difficult relations, MEMIT outperforms the alternatives.

The paper highlights a trade-off between generalization and specificity. MEND visibly suffers from this trade-off, and FT consistently fails to maintain specificity. MEMIT has a better joint score across both dimensions, though it still shows some trade-off for relations such as `P127` (“product owned by company”) and `P641` (“athlete plays sport”).

### 5.4. Editing Different Categories of Facts Together

To test whether MEMIT scaling depends on the diversity of memories edited together, the authors sample mixed relation sets $\mathcal{E}_{mix}$ from `COUNTERFACT`. Four scenarios combine relations with similar or different subject/object types. In all four cases, MEMIT performance on the mixed set is close to the average of each relation edited separately. This supports the claim that MEMIT’s scaling is neither helped nor harmed substantially by relation diversity, at least up to the tested scale of 700 mixed facts.
