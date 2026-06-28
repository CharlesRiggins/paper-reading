## Appendix I — Impact on model outputs and behavior

### I.1 Quantitative analysis

We evaluate the impact of LoRA fine-tuning on model outputs using three complementary metrics:

- **KL divergence** quantifies distributional changes by computing the average KL divergence between the original model ($\pi_{ref}$) and the LoRA-adapted model ($\pi_{\theta}$) across token positions: $\mathcal{L}_{KL}=\frac{1}{T}\sum_{t=1}^{T}D_{KL}(\pi_{\theta}(\cdot|\mathbf{q},\mathbf{t})\,||\,\pi_{ref}(\cdot|\mathbf{q},\mathbf{t}))$. We generate completions from the original model on 750 prompts from Arena-Hard-Auto (Li et al., 2024), and, over these completions, compute the average token-wise KL divergence between the original model distribution and the modified model distribution.
- **Win rate** measures generation quality via `GPT-4.1` pairwise comparisons on Arena-Hard-Auto (Li et al., 2024), with mean and confidence intervals obtained from bootstrap resampling.
- **MMLU accuracy** (Hendrycks et al., 2021a) evaluates knowledge retention using standard zero-shot chain-of-thought prompting. We use Inspect (AI Security Institute, 2024) to run evaluations.

Table 7 provides comprehensive win-rate results across different regularization strengths, showing that regularization values $\lambda$ of $0.50$ or higher tend to preserve model quality.

Figure 12 demonstrates that KL-regularized probes achieve superior trade-offs compared to LM-regularized probes. KL regularization creates smooth, predictable behavior as $\lambda_{KL}$ increases, while LM regularization exhibits erratic patterns—higher $\lambda_{LM}$ does not consistently reduce KL divergence and can even increase it through overfitting.

> **Table 7:** Win rates on Arena-Hard-Auto for `Llama-3.1-8B` variants, as judged by `GPT-4.1`. Each win rate is the mean estimate from a bootstrap analysis (100 resamples of the battle outcomes). The CI represents the corresponding 90% percentile confidence interval.

| Variant | $\lambda$ | Win rate (%) | CI (%) |
| --- | --- | --- | --- |
| Baseline | $-$ | 50.0 | $(-0.0\,/\,+0.0)$ |
| LoRA $\lambda_{LM}$ |  |  |  |
|  | $0.01$ | 34.4 | $(-2.0\,/\,+2.1)$ |
|  | $0.05$ | 39.0 | $(-1.8\,/\,+1.7)$ |
|  | $0.10$ | 43.3 | $(-1.9\,/\,+1.9)$ |
|  | $0.20$ | 42.0 | $(-1.8\,/\,+2.0)$ |
|  | $0.50$ | 47.2 | $(-1.8\,/\,+1.5)$ |
|  | $0.90$ | 48.3 | $(-2.0\,/\,+2.4)$ |
|  | $0.99$ | 50.4 | $(-2.2\,/\,+2.4)$ |
|  | $0.999$ | 48.7 | $(-2.1\,/\,+2.1)$ |
|  | $0.9999$ | 48.2 | $(-1.9\,/\,+1.9)$ |
| LoRA $\lambda_{KL}$ |  |  |  |
|  | $0.00$ | 35.9 | $(-2.1\,/\,+2.2)$ |
|  | $0.01$ | 32.5 | $(-1.9\,/\,+1.8)$ |
|  | $0.05$ | 39.7 | $(-2.0\,/\,+2.0)$ |
|  | $0.10$ | 45.3 | $(-1.9\,/\,+2.1)$ |
|  | $0.20$ | 44.7 | $(-2.3\,/\,+2.2)$ |
|  | $0.50$ | 52.8 | $(-2.0\,/\,+2.1)$ |
|  | $0.90$ | **53.3** | $(-1.7\,/\,+1.8)$ |
|  | $0.99$ | 52.4 | $(-2.1\,/\,+2.0)$ |
|  | $0.999$ | 48.9 | $(-1.7\,/\,+1.7)$ |
|  | $0.9999$ | 46.3 | $(-2.0\,/\,+2.5)$ |

> **Figure 12:** Trade-off between hallucination detection (AUC) and distributional shift (KL divergence). Each point corresponds to a different regularization strength $\lambda$.

> **Figure 13:** Trade-off between hallucination detection (AUC) and LM loss on model generations. However, LM loss is not ultimately the metric we care about. Minimizing LM loss can result in overfitting and distribution shift.

### I.2 Qualitative analysis

After training LoRA probes with minimal regularization, we anecdotally observe changes in the model's output distribution that suggest increased epistemic caution and reduced propensity for hallucination.

Note that this is distinct from the layer selection used in the rest of the paper, where we attach the probe head to $\lfloor 0.95\times\text{num\_layers}\rfloor$. Empirically, we only observe these qualitative behavioral changes when optimizing probes on the last layer.

> **Figure 14:** Example of hallucination detection affecting generation behavior. The baseline `Llama-3.3-70B` confidently states an incorrect referee name. The LoRA-augmented model exhibits an interesting behavior: it still provides an incorrect answer but immediately acknowledges its uncertainty with "but I cannot confirm this, a more reliable source would be needed."

> **Figure 15:** The baseline model confidently provides an incorrect answer ("Lynyrd Skynyrd" is not even a valid anagram of "O UGLY NINE"). The LoRA-augmented model correctly expresses inability to solve the anagram, rather than guessing an incorrect answer.

> **Figure 16:** The baseline generation contains potentially life-threatening errors: the lithium dosing (600–900 mg/day) is underdosed for acute mania, and the claim that lithium is "generally considered compatible with breastfeeding" overstates its safety profile. The LoRA-augmented model is appropriately cautious about breastfeeding risks with lithium and provides safer, albeit less detailed, guidance.

> **Figure 17:** The baseline model confidently cites multiple specific studies with full details. The LoRA model initially cites studies but then acknowledges it cannot verify these citations when attempting to provide references.
