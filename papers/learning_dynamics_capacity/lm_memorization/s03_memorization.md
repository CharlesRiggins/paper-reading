## 2. Memorization, Intended and Unintended

Let a training algorithm $L$ map a dataset $x\sim X$ to a model $\hat{\theta}=L(x)$. Training transfers some information from $x$ into $\hat{\theta}$, but the paper distinguishes information specific to that sample from information the model should learn in general. A language model trained on “What is $2^{100}$?” should be credited for performing arithmetic rather than treated as having memorized the whole example.

The definition is designed to satisfy three requirements:

1. **Separation from generalization.** Unintended memorization must exclude intended knowledge of the underlying process.
2. **Sample-level measurement.** It must evaluate a realization $x$ and a realized model $\hat{\theta}$, not merely distributions over them.
3. **Independence from the training algorithm.** It should depend on the final model and the target sample, because these are often all that are available.

### 2.1. Warm-up: a statistical view of memorization

For random variables, total memorization is the mutual information between a dataset and the trained model:

$$
\operatorname{mem}(X,\hat{\Theta}) = I(X;\hat{\Theta}) = H(X)-H(X\mid\hat{\Theta}).
$$

Let $\Theta$ denote a prior describing the underlying data-generating model. The information about $X$ that remains after fixing $\Theta$ is the portion that cannot be explained by generalization. The paper defines **unintended memorization** as

$$
\operatorname{mem}_{U}(X,\hat{\Theta},\Theta)
= I(X;\hat{\Theta}\mid\Theta)
= H(X\mid\Theta)-H(X\mid\Theta,\hat{\Theta}),
$$

and **intended memorization** (generalization) as the residual:

$$
\operatorname{mem}_{I}(X,\hat{\Theta},\Theta)
= \operatorname{mem}(X,\hat{\Theta})-\operatorname{mem}_{U}(X,\hat{\Theta},\Theta).
$$

For an i.i.d. dataset $X=(X_1,\ldots,X_n)$, Proposition 1 establishes

$$
\sum_{i\in[n]}\operatorname{mem}_{U}(X_i,\hat{\Theta},\Theta)
\leq \operatorname{mem}_{U}(X,\hat{\Theta},\Theta)
\leq H(\hat{\Theta}).
$$

Thus summing per-example values lower-bounds total unintended memorization, while the information content of the trained model upper-bounds it. The result explains why unintended memorization grows with data but cannot outgrow capacity. The distributional definition follows Brown et al. [8], but cannot directly evaluate one observed model/sample pair because it requires conditional probabilities over datasets given a model.

### 2.2. Measuring unintended memorization with Kolmogorov complexity

To obtain an instance-level quantity, the paper moves to compression. Kolmogorov complexity $H^K(x)$ is the length of the shortest description that generates string $x$ in a fixed computational model. Conditional complexity $H^K(x\mid\theta)$ is the shortest description of $x$ when $\theta$ is given as side information. Algorithmic mutual information is

$$
I^K(x;\theta)=H^K(x)-H^K(x\mid\theta).
$$

For a target model $\hat{\theta}$ and reference model $\theta$, the proposed per-sample quantities are

$$
\operatorname{mem}^{K}_{U}(x,\theta,\hat{\theta})
=H^K(x\mid\theta)-H^K(x\mid\theta,\hat{\theta}),
$$

$$
\operatorname{mem}^{K}_{I}(x,\theta,\hat{\theta})
=\operatorname{mem}^{K}(x,\hat{\theta})-\operatorname{mem}^{K}_{U}(x,\theta,\hat{\theta}).
$$

The algorithmic form approximates the earlier statistical notion in expectation when the model and samples can be represented in finite numbers of bits. It preserves the desired sample-level and algorithm-independent interpretation.

### 2.3. Estimating Kolmogorov complexity with compression

Exact Kolmogorov complexity is uncomputable [32], so the authors approximate it with effective compression schemes. Their primary choice is **arithmetic coding**, whose code length can be computed from language-model likelihoods. Other schemes—such as optimized prompts [45] or prefixes [11]—could be substituted.

For a target model, $H^K(x\mid\hat{\theta})$ is approximated by $-\log p(x\mid\hat{\theta})$. When both target and reference are available, the best available coding distribution gives

$$
H^K(x\mid\theta,\hat{\theta})
\approx -\log\max\{p(x\mid\theta),p(x\mid\hat{\theta})\}.
$$

For random strings, the exact uniform generating distribution is the reference. For text, the reference is a larger same-family model trained on a much broader superset of the target model’s data. Although likelihood is used to evaluate code lengths, this is not the ordinary likelihood-based memorization definition: it is part of a compression algorithm and may depend on its decoding choices.

> **Table 1.** Across depths and widths, estimated capacity averages **3.83 bpp** in `float32` and **3.51 bpp** in `bfloat16`; doubling precision does not double useful memorization capacity.
