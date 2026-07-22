## 4. Synthetic Dataset (`TinyStories`) Results

The controlled experiment removes one language from a bilingual model. English is the retain domain and Spanish is the forget domain, so the authors know the ground-truth label of every example and can independently vary labeler error.

### 4.1 Setup

A `64M`-parameter model trains for one epoch on **1.2B tokens**: `466M` English `TinyStories` tokens plus a **718M-token** Spanish translation created with `Claude 3 Haiku`. The dataset follows a roughly Chinchilla-optimal setup. One percent of each language (`12M` tokens total) is held out for testing.

Under perfect labeling, all Spanish examples enter $\mathbf{D}_{\text{forget}}$. English is split randomly: 75% goes to $\mathbf{D}_{\text{unlabeled}}$ and 25% to confidently retained $\mathbf{D}_{\text{retain}}$. To simulate a false-negative target classifier, the authors move a selected fraction of Spanish data from $\mathbf{D}_{\text{forget}}$ into $\mathbf{D}_{\text{unlabeled}}$. This fraction is the **undiscovered forget percentage**, equivalently the classifier's false-negative rate on target data.

Only **1 of 32** attention heads and **64 of 2048** MLP units per block are designated for forgetting. Since the languages have distinct tokens, embeddings are updated using both domains. The filtering baseline simply discards $\mathbf{D}_{\text{forget}}$ and trains for one epoch on the rest.

All reported English and Spanish losses are **logit-calibrated** after ablation. A learned per-logit bias prevents implausibly low target-token probabilities from producing misleadingly extreme loss spikes; Appendix I gives the objective.

### 4.2 Retain/forget trade-off under false negatives

Figure 3 reports loss while the undiscovered Spanish fraction rises. SGTM has lower retain loss and higher forget loss than the prior activation-gradient Gradient Routing variant at every tested discovery rate. Filtering degrades much faster on forgetting when even a small number of Spanish examples escape the filter. Both localization methods decline more slowly as unlabeled target data rises, though filtering retains English more efficiently.

Figure 4(a) makes the essential Pareto comparison at **1%** artificial target miss rate. It traces checkpoints for SGTM, Gradient Routing, perfect filtering, 99% filtering, and no filtering:

- SGTM's final retain loss is slightly worse than the 99%-filter model, roughly comparable to that filter after 80% of its training.
- At any fixed retain loss, however, SGTM has higher Spanish forget loss than 99% filtering and prior routing.
- SGTM is not meant to beat **perfect** filtering—if target data were truly excluded, a method cannot do better than never learning it—but it approaches the perfect filter's final forget loss.

The experiment therefore isolates the claimed advantage: it appears only once there are false negatives, precisely where a real filtering pipeline cannot be ideal.

### 4.3 Leakage: converting loss to equivalent target exposure

High target loss alone does not say how much missed target content reached non-forget parameters. The paper defines **leakage** through a standard-training calibration curve:

1. Train ordinary models on all retain data plus varying quantities of Spanish data.
2. Map each resulting Spanish loss to the number of Spanish tokens needed to obtain it under standard training.
3. For an SGTM run, divide that *equivalent standard-training exposure* by the number of unlabelled Spanish tokens actually seen.

For example, with 20% undiscovered Spanish data, SGTM sees **144M** unlabelled Spanish tokens but attains a Spanish loss matching an ordinary model trained on only **965k** Spanish tokens. Data filtering has leakage 1 by construction: every escaped target token is trained normally.

Across `8M`, `18M`, `34M`, and `64M` models, with $d_{\text{forget}}=64$ and $h_{\text{forget}}=1$ fixed, Figure 5 finds lower leakage in larger models. For the `64M` model, leakage remains **0.005–0.02** with up to 40% undiscovered target data. This is encouraging evidence, but it is a small-scale extrapolation, not evidence that frontier models have the same scaling trend.

### 4.4 Gradient norms and absorption

To test absorption, the authors take a perfectly labelled SGTM model and evaluate samples as if all were unlabelled—no masks in either pass. They compare per-sample relative gradient norms

$$
\frac{|\nabla_\theta|}{|\theta|}
$$

for forget and retain parameter sets.

Figure 4(b) shows clear specialization: Spanish examples preferentially update forget parameters, while English examples preferentially update retain parameters. In particular, forget weights receive much stronger updates from unlabelled Spanish than from English. This supports the proposed self-reinforcing localization mechanism.

The retain set also receives slightly stronger updates from English than Spanish, but with substantially weaker separation. The authors flag a limitation: gradient magnitude alone does not determine acquired information; gradient direction and alignment may matter. Thus the norm plot supports absorption but does not fully explain the low leakage values.
