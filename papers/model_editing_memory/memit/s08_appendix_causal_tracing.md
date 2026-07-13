## A. Causal Tracing

Appendix A details the causal tracing procedure used to select the MLP layer range for MEMIT. The method follows Meng et al. [29]. Starting from **501** true factual statements that `GPT-J` predicts correctly, the authors corrupt the subject-token encodings with Gaussian noise, then measure how restoring specific hidden states or module outputs improves the probability of the correct fact.

The noise standard deviation is $3\sigma$, where $\sigma^2$ is the empirically observed variance of embedding activations. Each measurement averages across the 501 statements and 10 noise samples. Three restoration settings are used:

1. Restore individual hidden states $h_t^l$.
2. Restore 10-layer windows of MLP outputs $m_t^l$.
3. Restore 10-layer windows of attention outputs $a_t^l$.

The measurements show that `GPT-J` has a causal structure similar to the `GPT-2 XL` structure reported by ROME. A strong causal effect appears in the earliest attention layers at the last subject token, likely because `GPT-J` recognizes and chunks the subject name there. However, path-dependent experiments suggest that this attention activity is not the decisive mediator of factual recall.

In the main paper’s Figure 3, the authors focus on the last subject token and compare causal effects under two interventions. Red bars freeze attention modules at the last subject token in their corrupted state; green bars freeze MLP modules. Freezing attention barely shifts the curve, but freezing MLPs creates a large gap. This gap identifies the layers where MLP computation is most important for factual recall. MEMIT chooses the layer range $\mathcal{R}$ from this high-gap region.
