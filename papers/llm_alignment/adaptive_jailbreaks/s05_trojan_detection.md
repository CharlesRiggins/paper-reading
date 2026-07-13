## 5. Adaptive Attacks for Trojan Detection

The paper connects jailbreaking to universal trojan-string detection in poisoned LLMs. A universal trojan string is a short suffix that can be appended to many inputs and cause a model that normally refuses harmful requests to comply. This resembles the jailbreak setting because both tasks search for short trigger sequences that systematically alter model behavior.

The experiment uses the SaTML 2024 Trojan Detection Competition. Rando and Tramèr showed that backdoors can be implanted during RLHF by poisoning a small percentage of preference data with a universal suffix. The competition provided five poisoned `Llama-2-7B` models, each with a different hidden trigger, a reward model that scores safety of prompt-response pairs, and a set of harmful requests. The goal was to find 5–15 token triggers acting as universal jailbreaks.

The authors adapt random search by restricting the token search space using model-pair information. For a reference model $M_r$ and suspicious model $M_s$, they compare token embeddings or related vectors and rank tokens by the $\ell_2$ distance between corresponding representations. If $\pi^{rs}$ is the ordering by decreasing distance, the ranking satisfies

$$
\pi^{rs}(i)<\pi^{rs}(j)\;\Longrightarrow\;\left\|v_i^r-v_i^s\right\|_2\geq \left\|v_j^r-v_j^s\right\|_2,
\quad i,j=0,\ldots,32000.
$$

The candidate pool is then

$$
\operatorname{top-}k(M_r,M_s)=\{t_i\in T:\pi^{rs}(i)\leq k\}.
$$

The hypothesis is that trigger tokens for the poisoned models are disproportionately likely to lie among tokens with large representation differences. The paper also notes a tokenizer complication: decoding a token-index sequence into text and re-encoding it may not recover the original sequence.

> Table 5: SaTML trojan competition results. Lower reward-model scores indicate more successful triggers; the total score is summed across five target models.

| Method | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| No trigger | 2.78 | 2.55 | 2.05 | 3.34 | 1.94 | 12.66 |
| 3rd place | -5.98 | -5.20 | -4.63 | -4.51 | 0.42 | -19.89 |
| 2nd place | -5.73 | -6.46 | -4.84 | -4.93 | -7.26 | -29.21 |
| RS on selected tokens (ours) | **-6.30** | **-6.98** | **-5.52** | -4.70 | -6.73 | **-30.22** |
| Ground truth trojans | -11.96 | -7.20 | -5.76 | -4.93 | -7.63 | -37.48 |

Without triggers, the poisoned models produce safe answers and receive high safety scores. The authors' restricted-token random search gets the best score among competition submissions, winning on 3 of 5 models and achieving the best total score. The exact ground-truth trojans remain stronger, but the gap is small enough to show that adaptive candidate selection is highly effective.

The section reinforces the paper's general lesson: automated search is useful, but task-specific adaptivity—here, narrowing the candidate token pool—is what turns a generic algorithm into a strong attack.
