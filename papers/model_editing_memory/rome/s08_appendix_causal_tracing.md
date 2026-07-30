## Appendix B. Causal Tracing

### B.1. Experimental Settings

The causal-tracing study uses 1,000 prompts for facts that `GPT-2 XL` already knows. The authors greedily generate from `COUNTERFACT` templates, retain cases where the correct object appears before any competing capitalized word, and truncate immediately before the object. Mean clean probability for the correct next token is **27.0%**.

Subject embeddings are corrupted with Gaussian noise:

$$
\epsilon\sim\mathcal N(0,\nu^2I),\qquad \nu=3\sigma_t,
$$

where $\sigma_t$ is the empirical standard deviation of token embeddings. Ten noise samples are run per prompt. Corruption lowers the correct-object probability to **8.47%** on average.

Restoring one clean hidden state at a time can raise the average probability to **19.5%** at the strongest last-subject-token location. When grouped by layer, the main peak is around layer 15, where restoration raises the average to **15.0%**.

### B.2. Separating MLP and Attention Effects

Single MLP or attention outputs usually have negligible effects because factual information accumulates across layers. The experiment therefore restores windows of ten module outputs centered at layer $l_*$:

$$
[l_*-4,\ldots,l_*+5].
$$

For MLP windows, the strongest individual restoration reaches **23.6%**, and the aggregate maximum is centered near layer 17 at the final subject token. For attention windows, the maximum is centered near layer 32 at the final prompt token. The confidence intervals in Figure 7 confirm that both peaks differ significantly from non-peak regions.

### B.3. Other Model Sizes and Architectures

The same experiment is run on:

- `GPT-2 Medium` (334M);
- `GPT-2 Large` (774M);
- `GPT-J` (6B, 28 layers);
- `GPT-NeoX` (20B, 44 layers).

Noise is rescaled to match each embedding space: $\nu=0.025$ for `GPT-J` and $\nu=0.03$ for `GPT-NeoX`. Despite architectural and depth differences, every model shows an early site at a subject token with strong MLP involvement and a late attention-dominated prediction site.

In the EleutherAI models, early attention effects are more visible. The paper suggests that shallower stacks may need to concentrate subject-reading attention into fewer layers. Nevertheless, the cross-model stability of the main pattern explains why ROME transfers from `GPT-2 XL` to `GPT-J`.

### B.4. Robustness, Exceptions, and Further Insights

The final subject token is typical but not universal. For “Windows Media Player,” the decisive lookup occurs on “Windows,” not “Player.” For “Mitsubishi Electric,” “Mitsubishi” dominates; for “Madame de Montesson,” the title “Madame” is decisive. Low-confidence factual guesses may have no dominant subject-token MLP site.

The authors test several corruption variants:

- corrupting an additional token after the subject;
- matching a full multivariate Gaussian to token-embedding mean and covariance;
- sampling uniform noise in the range $\pm3\sigma_t$.

All preserve the qualitative early-site MLP pattern, though weaker noise produces a smaller total effect and makes mediation harder to measure. The method needs corruption strong enough to create a meaningful gap between clean and baseline behavior.

The path-specific MLP-severing intervention also remains stable under expanded corruption. Early restored states cease to matter when downstream MLP outputs are frozen, while later states retain their effects.

Finally, the appendix compares causal tracing to **Integrated Gradients (IG)** using 50-step Gauss–Legendre quadrature. IG saliency maps are scattered and emphasize local sensitivity; they do not reveal the systematic importance of the final subject token or middle-layer MLPs. The distinction illustrates that gradient salience and causal mediation answer different questions: sensitivity to a local perturbation is not the same as a component’s ability to restore behavior under a controlled counterfactual.
