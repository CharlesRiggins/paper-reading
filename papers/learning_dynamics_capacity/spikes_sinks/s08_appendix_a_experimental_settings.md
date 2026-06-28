## Appendix A Experimental Settings

All models in Section 4 are trained on the `DCLM` dataset (Li et al., 2024) using a shared codebase and a common baseline recipe. Unless otherwise noted, we keep the training pipeline fixed and vary only the factors under study in each ablation. The default recipe closely follows the Llama-style pretraining setup (Touvron et al., 2023a). With some additional details that we couldn't find from the original paper, we refer to the `torchtitan` (Liang et al., 2025b) and `Olmo` (Olmo et al., 2025) codebase. Table 9 reports the baseline architecture and optimization hyperparameters used across experiments.

> **Table 9:** Baseline training configurations.

| Hyperparameter | Value |
| --- | --- |
| Layers | 32 |
| Hidden size | 4096 |
| Attention heads | 32 |
| Head dimension | 128 |
| Intermediate size | 11008 |
| Vocabulary size | 32000 |
| Optimizer | AdamW |
| $\beta_{1},\beta_{2}$ | 0.9, 0.95 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Base learning rate | $3.0\times10^{-4}$ |
| LR schedule | Cosine decay |
| Warmup steps | 2,000 |
| Training steps | 5,000 |
| Batch size | 2M tokens |
| Min LR ratio | 0.1 |

**Datasets.** During evaluation, we randomly sample text from the `C4` corpus (Raffel et al., 2020), with a total budget of up to $1024\times 4096$ tokens. We tokenize the sampled text and partition it into fixed-length chunks of $\{64,256,1024,2048,4096\}$ tokens, selecting the chunk size to match the configured context window for each model.

**Sink Ratio.** For an input sequence of length $T$, let $A^{l,h}\in[0,1]^{T\times T}$ denote the (causal) self-attention matrix at layer $l$ and head $h$, where $A^{l,h}_{t,k}$ is the attention weight from query position $t$ to key position $k$. Following Gu et al. (2025), we define the importance score of position $k$ as the attention it receives on average:

$$
\alpha^{l,h}_{k}\coloneqq\frac{1}{T}\sum_{t=1}^{T}A^{l,h}_{t,k}.
$$

*(Equation 27)*

A head is classified as a sink head if its maximum importance over the first half of positions exceeds a threshold $\epsilon$:

$$
s_{\epsilon}=\frac{1}{LH}\sum_{l=1}^{L}\sum_{h=1}^{H}\mathds{1}\!\left(\max_{1\leq k\leq\lfloor T/2\rfloor}\alpha^{l,h}_{k}>\epsilon\right).
$$

*(Equation 28)*

Finally, we report the model-level sink ratio by averaging $s_{\epsilon}$ over evaluation sequences. In our experiments, we use $\epsilon=0.3$ and $T=64$ consistently.
