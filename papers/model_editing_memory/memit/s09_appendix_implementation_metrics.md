## B. Implementation Details

### B.1. Fine-Tuning with Weight Decay

The fine-tuning baseline updates layer 21 of `GPT-J`, which ROME found to work best for single edits. Instead of a hard $L_\infty$-norm constraint, the authors use a soft weight decay regularizer. The optimal regularization depends strongly on the number of edits, so the weight decay is tuned for the $n=10000$ case. Figure 9 shows that $5\times10^{-4}$ gives the best trade-off between generalization and specificity.

`FT-W` runs for at most 25 optimization steps with learning rate $5\times10^{-4}$. Early stopping is triggered when loss reaches $10^{-2}$. For 10,000 edits on `GPT-J`, FT takes **1,716.21 seconds**, or approximately **0.48 hours**. The authors intentionally avoid tuning FT-W over multiple layers; Table 2 already shows that one layer can achieve near-perfect efficacy but poor specificity, indicating enough edit capacity and a localization problem rather than insufficient optimization power.

### B.2. Model Editing Networks with Gradient Decomposition (`MEND`)

`MEND` makes concurrent edits by accumulating gradients from all edit examples, then passing the combined gradient through a hypernetwork. The experiments use the `GPT-J` MEND hypernetwork trained by Meng et al. [29]. At inference time, the learning-rate scale remains at the default value of 1.0. `MEND` is the fastest method, taking **98.25 seconds** for 10,000 updates on `GPT-J`.

### B.3. Rank-One Model Editing (`ROME`)

The ROME baseline uses the default open-source hyperparameters. For `GPT-J`, edits are applied at layer 5, optimization runs for 20 steps, weight decay is 0.5, the KL factor is 0.0625, and learning rate is $5\times10^{-1}$. ROME samples 10 prefixes of length 5 and 10 prefixes of length 10. Covariance statistics are collected in `fp32` on `Wikitext` with 100,000 samples. ROME takes **44,248.26 seconds**, or approximately **12.29 hours**, for 10,000 edits—about 4 seconds per edit.

### B.4. Mass-Editing Memory in a Transformer (`MEMIT`)

For `GPT-J`, MEMIT uses:

- Edited layer range: $\mathcal{R}=\{3,4,5,6,7,8\}$.
- Covariance adjustment: $\lambda=15000$.
- Covariance samples: 100,000 `Wikitext` samples in `fp32`.
- $\delta_i$ optimization: 25 steps, learning rate $5\times10^{-1}$.
- Norm clamp: $\|\delta_i\|_2 < \frac{3}{4}\|h_i^L\|$.

For `GPT-NeoX`, MEMIT uses:

- Edited layer range: $\mathcal{R}=\{6,7,8,9,10\}$.
- Covariance adjustment: $\lambda=20000$.
- Covariance samples: 50,000 `Wikitext` samples collected in `fp16` and stored in `fp32`.
- $\delta_i$ optimization: 20 steps, learning rate $5\times10^{-1}$.
- Norm clamp: $\|\delta_i\|_2 < \frac{3}{10}\|h_i^L\|$.

MEMIT can precompute and cache $z_i$ values because all memories are inserted in parallel. If $z_i$ vectors are already computed, MEMIT takes **3,226.35 seconds**, or **0.90 hours**, for 10,000 `GPT-J` updates. Computing all 10,000 $z_i$ vectors takes **23,546.65 seconds**, or **6.54 hours**, in the authors’ serial implementation. This part is embarrassingly parallel and could be batched.

## C. Evaluation Metrics

### C.1. For `zsRE`

The `zsRE` evaluation reports three probability tests plus their harmonic mean.

**Efficacy** is top-1 recall under the exact prompt seen by the editor:

$$
\mathbb{E}_i\left[o_i=\operatorname{argmax}_{x_E}\mathbb{P}_G[x_E\mid p(s_i,r_i)]\right]. \tag{21}
$$

**Paraphrase** is top-1 recall under paraphrases:

$$
\mathbb{E}_i\left[\mathbb{E}_{p\in\mathrm{paraphrases}(s_i,r_i)}\left[o_i=\operatorname{argmax}_{x_E}\mathbb{P}_G[x_E\mid p]\right]\right]. \tag{22}
$$

**Specificity** is top-1 accuracy on neighborhood prompts that should keep the original correct answer $o_i^c$:

$$
\mathbb{E}_i\left[\mathbb{E}_{p\in\mathrm{neighborhood\ prompts}(s_i,r_i)}\left[o_i^c=\operatorname{argmax}_{x_E}\mathbb{P}_G[x_E\mid p]\right]\right]. \tag{23}
$$

**Score** is the harmonic mean of Efficacy, Paraphrase, and Specificity.

### C.2. For `COUNTERFACT`

`COUNTERFACT` provides prompts and texts for evaluating rewrites. The probability metrics are:

**Efficacy Success (ES)**:

$$
\mathbb{E}_i[\mathbb{P}_G[o_i\mid p(s_i,r_i)]>\mathbb{P}_G[o_i^c\mid p(s_i,r_i)]]. \tag{24}
$$

**Paraphrase Success (PS)**:

$$
\mathbb{E}_i\left[\mathbb{E}_{p\in\mathrm{paraphrases}(s_i,r_i)}[\mathbb{P}_G[o_i\mid p]>\mathbb{P}_G[o_i^c\mid p]]\right]. \tag{25}
$$

**Neighborhood Success (NS)**:

$$
\mathbb{E}_i\left[\mathbb{E}_{p\in\mathrm{neighborhood\ prompts}(s_i,r_i)}[\mathbb{P}_G[o_i\mid p]<\mathbb{P}_G[o_i^c\mid p]]\right]. \tag{26}
$$

**Editing Score (S)** is the harmonic mean of ES, PS, and NS.

Generation quality is measured with:

- **Reference Score (RS)**: prompt $G$ with subject $s$, compute TF-IDF vectors for $G(s)$ and reference Wikipedia text about $o$, then use cosine similarity.
- **Generation Entropy (GE)**: detects repetition using bi-gram and tri-gram entropy:

$$
-\left(\frac{2}{3}\sum_k f_2(k)\log_2 f_2(k)+\frac{4}{3}\sum_k f_3(k)\log_2 f_3(k)\right). \tag{27}
$$

Here $f_n(\cdot)$ is the $n$-gram frequency distribution.
