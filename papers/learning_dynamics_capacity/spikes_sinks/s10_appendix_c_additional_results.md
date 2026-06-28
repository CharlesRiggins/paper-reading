## Appendix C Additional Empirical Results

**Open-Source Models.** While our primary analysis focused on `Llama-2-7B`, we validate the universality of our findings across the diverse set of open-source models detailed in Table 10. These models span multiple families (`Llama 2`, `Llama 3`, `Qwen2.5`, `Qwen3`), depths (28 to 48 layers), and parameter counts (7B to 14B).

> **Table 10:** List of open-source models evaluated in Appendix C. We observe consistent massive activations and attention sinks phenomena across the following diverse model families and sizes.

| Model Family | Model Name | Layers | Dimensions | Heads | Huggingface Model Id |
| --- | --- | --- | --- | --- | --- |
| Llama | `Llama-2-7B` | 32 | 4096 | 32 | meta-llama/Llama-2-7b-hf |
| | `Llama-2-13B` | 40 | 5120 | 40 | meta-llama/Llama-2-13b-hf |
| | `Llama-3-8B` | 32 | 4096 | 32 | meta-llama/Meta-Llama-3-8B |
| Qwen | `Qwen2.5-7B` | 28 | 3584 | 28 | Qwen/Qwen2.5-7B |
| | `Qwen2.5-14B` | 48 | 5120 | 40 | Qwen/Qwen2.5-14B |
| | `Qwen3-8B` | 36 | 4096 | 32 | Qwen/Qwen3-8B |
| | `Qwen3-14B` | 40 | 5120 | 40 | Qwen/Qwen3-14B |

##### Universality of Step-Up/Step-Down Dynamics.

Figure 7 visualizes the top-3 coordinate magnitudes through the residual stream for all 12 evaluated models. We observe two consistent behaviors across all architectures:

1. **Massive Activations:** Every model exhibits activation spikes orders of magnitude larger than the baseline variance.
2. **Feed-Forward Block Driven Origin:** Comparing the "post-residuals" (top panels) and "block outputs" (bottom panels) plots in Figure 7 confirms that these spikes are not merely accumulated residual error. Instead, they originate abruptly at specific "step-up" blocks and are neutralized by subsequent "step-down" blocks, matching the mechanism described in Section 3.

> **Figure 7:** Top-3 coordinate magnitudes after (top panels) and before (bottom panels) the residual branch for 12 open-source models. All models exhibit the characteristic step-up and step-down behavior driven by one or few early blocks and then the late blocks.

##### Universality of Frobenius Norm Outliers.

To confirm that these activations arise from the directional quadratic amplification mechanism, we analyze the Frobenius norms of the quadratic-form matrices $\mathbf{U}_{k}$. As shown in Figure 8, channels exhibiting massive activations correspond to $\mathbf{U}_{k}$ matrices with exceptionally large Frobenius norms.

For example, in `Llama-3-8B`, distinct spikes are visible at channels 788, 1384, and 4062, which align with the massive activation spikes observed in Figure 7. This confirms that the alignment of attention sinks with high-gain quadratic directions in the MLP is a structural invariant across the Llama model families.

> **Figure 8:** Frobenius norms $\|\mathbf{U}_{k}\|_{F}$ for the quadratic forms of Llama models. Spike channels align with $\mathbf{U}_{k}$ matrices that exhibit substantially larger norms than typical channels. These high-norm coordinates appear exclusively in step-up and step-down blocks.
