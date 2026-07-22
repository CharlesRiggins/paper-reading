## Appendix A. Gradient Routing Variants

The paper uses **Gradient Routing** in two senses: broadly, a framework for routing target knowledge to a chosen parameter subset; narrowly, the earlier implementation by Cloud et al. that masks activation gradients. In this appendix $\theta_{\text{joint}}$ denotes parameters updated by both domains but not erased after training.

All routing variants reserve the same Wikipedia dimensions: $h_{\text{forget}}=1$ of 32 attention heads and $d_{\text{forget}}=64$ of 4096 MLP units.

| Method | What is masked | Main consequence |
|---|---|---|
| Gradient Routing | activation gradients | blocks backward flow through selected activations, thereby changing gradients for remaining parameters; leaves unmasked layers and down-projections exposed to target updates |
| Activation Masking | retain parameters/activations in the target forward pass | behaves like deterministic data-dependent dropout; affects loss and gradients of all parameters and still permits down-projection updates |
| SGTM | retain **parameter gradients** on forget examples | ordinary gradient is computed, then forbidden parameter updates are zeroed; default has no $\theta_{\text{joint}}$ |
| SGTM (Joint Projection) | SGTM except $W_2,b_2,W_O,b_O$ are joint | improves final retain loss but weakens localization |
| SGTM (Joint Attention) | only first MLP layer is split; attention and second MLP layer joint | closer to `Memorization Sinks`; likewise improves retained loss but weakens forget/retain trade-off |

> **Figure 7.** On Wikipedia biology removal, standard Gradient Routing and Activation Masking trace curves similar to weak filtering but farther right, hence are less compute-efficient. Original Gradient Routing is particularly costly: its final loss resembles weak filtering after only about **20%** of training. The joint SGTM variants improve final retain loss, but their full trade-off is worse than default SGTM and both filtering baselines.

## Appendix B. Compute penalty

Raw cross-entropy losses are nonlinear in training resources; moving from loss 10 to 9 does not cost the same amount as moving from 3 to 2. The authors therefore express results as **equivalent baseline compute**.

### B.1 Scaling-law conversion

They train standard Wikipedia models of `34M`, `64M`, `125M`, and `254M` parameters under Chinchilla-style scaling, using 20 tokens per parameter. On biology, biology-adjacent, and general test sets separately, they fit

$$
\ell = \alpha C^{-\beta}, \qquad C \approx 6 \times \text{parameters} \times \text{tokens},
$$

where $C$ is FLOPs. A method's endpoint loss is mapped to the compute that an unfiltered baseline would need to reach the same loss. The reported penalty is the relative shortfall from full baseline compute.

### B.2 Results

| Evaluation subset | SGTM | Weak filter | Strict filter | Reading |
|---|---:|---:|---:|---|
| Biology / forget | **99%** | 95% | 96% | all methods leave the model at a loss a standard model could reach using under 5% of full compute; SGTM forgets slightly more |
| Biology-adjacent retain | **38%** | 7% | 67% | SGTM preserves more nearby medicine/chemistry/environment knowledge than strict filtering while forgetting more target biology |
| General retain | **5–6%** | -10% | -15% | SGTM's gradient operations slow general learning; filtering's negative penalty is an artifact of reallocating fixed compute to extra general data |

The appendix emphasizes that a high equivalent penalty on biology is not direct evidence of safety: loss is only a proxy, and a frontier model might still learn consequential behavior from very few missed examples. Rather, the analysis makes the cost explicit: SGTM sacrifices some compute efficiency for robustness to label noise and reduced collateral loss on adjacent domains.
