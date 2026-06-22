## Setting LoRA hyperparameters

One barrier to LoRA adoption is the necessity to choose optimal hyperparameters, which are different from ones optimized for FullFT. This section shows that the problem is not as daunting as it first appears and discusses findings related to hyperparameter choice.

### Optimal learning rate and rank

Following Hu et al., the authors consider the following parametrization for LoRA:

$$
W' = W + \frac{\alpha}{r}BA
$$

Here $r$ is the LoRA rank, $\alpha$ is the LoRA scaling factor, and $A$ and $B$ are the LoRA weight matrices of rank $r$. The authors use $\alpha = 32$ for the experiments in this article, following standard practice from other implementations.

The $1/r$ scaling factor makes the optimal learning rate approximately independent of rank. In fact, a stronger condition holds: the learning curve is exactly the same at the beginning of training, regardless of rank. This effect is striking, and in the experiments the closeness of the learning curves for different ranks initially made the authors worry that a bug caused the rank parameter to be ignored. It follows that in a short training regime, the optimal LR is also independent of rank. However, as shown above in the plots of learning rate versus loss, Figure 2, optimal LR has some rank-dependence in the longer-training regime.

![](https://thinkingmachines.ai/blog/lora/svg/fig8.svg)

Figure 8 looks at differences in learning curves early in training for different ranks with the same learning rate. On the left, it shows the learning curves. On the right, it shows the difference between rank 16 and 256, which grows over time. Strangely, it is negative, though tiny, for the first few steps, so that part of the curve is missing from the plot.

The authors partly explain this result by looking at the expected update to the LoRA matrix after the very first training update. The LoRA product $BA$ can be treated as the sum of $r$ rank-1 outer products:

$$
BA = \sum_{i=1}^r b_i a_i^T = \sum_{i=1}^r \Delta_i,
\quad \Delta_i = b_i a_i^T.
$$

Here, $\partial \text{Loss}/\partial \Delta_i$ is the same for all $i$; however, the gradients $\partial \text{Loss}/\partial b_i$ and $\partial \text{Loss}/\partial a_i$ depend on the initialization. For example, $\partial \text{Loss}/\partial b_i$ depends on $a_i$. Since the initialization of $a_i$ and $b_i$ does not depend on rank, $\mathbb{E}[\Delta_i]$ is the same for all $i$ and does not depend on rank. At the first step of training, the expected update from each term is equal and independent of rank. It follows that $(1/r)\sum_{i=1}^r \Delta_i$ is a sample average of $r$ terms with the same expectation, so the expectation of the average — the change to the adapter $(1/r)BA$ — does not depend on rank.

### Parametrization invariances

There are four hyperparameters potentially applicable to LoRA:

1. The scale factor $\alpha$, which appears in $\alpha/r$.
2. The learning rate for the down-projection matrix $A$, $LR_A$.
3. The learning rate for the up-projection matrix $B$, $LR_B$.
4. The initialization scale of matrix $A$, $\text{init}_A$. For a random initialization, this is the standard deviation of $A$’s initial elements. Matrix $B$ is initialized to zero, so there is no need to define $\text{init}_B$.

Having to tune four different parameters may seem overwhelming. However, invariances in the training dynamics mean that two of these are redundant, and learning behavior is determined by two. The authors show this invariance by noting that when training with Adam and $\varepsilon = 0$, the optimization process is invariant to the following two-parameter transformation. This can be extended to $\varepsilon > 0$, but the scaling must account for gradients being scaled by the same factor.

For $p, q > 0$:

- $\alpha \to \frac{1}{pq} \cdot \alpha$
- $\text{init}_A \to p \cdot \text{init}_A$
- $LR_A \to p \cdot LR_A$
- $LR_B \to q \cdot LR_B$

Since two degrees of freedom out of the four do not affect the learning process, the effective parameter space is 2D. The authors choose one basis for this 2D space because it lends itself to a straightforward interpretation:

- $\alpha \cdot \text{init}_A \cdot LR_B$. This determines the scale of initial updates, or equivalently, the initial slope of the learning curve. Since $B$ is initialized to zero, $LR_A$ and the initial updates to $A$ are irrelevant.
- $\text{init}_A / LR_A$. Since Adam updates the elements of $A$ by approximately $LR_A$ at each step, this timescale parameter determines the number of steps it takes to significantly transform $A$ away from its initial state.

Some proposals from previous work on LoRA can be reinterpreted in terms of this basis.

`LoRA+`, from LoRA+: Efficient Low Rank Adaptation of Large Models (Hayou et al., 2024), proposes to use different LRs on $A$ and $B$, with a higher rate for $B$. Expressed in the basis above, increasing $LR_B$ is equivalent to increasing $\text{init}_A/LR_A$ so that $A$ changes on a longer timescale.

Unsloth’s LoRA Hyperparameter Guide recommends using higher values of $\alpha$ for high-rank LoRA, for example by avoiding the $1/r$ scaling. This is also equivalent to increasing $\text{init}_A/LR_A$. When $\alpha$ is increased, $LR_A$ and $LR_B$ need to be lowered in compensation to get the same update size. This in turn simply makes $LR_A$ smaller relative to $\text{init}_A$.

In the experiments, the authors used the standard parametrization from the Hugging Face `peft` library, PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods (Mangrulkar et al., 2022), proposed by Hu et al.: a uniform distribution for $A$ with scale $1/\sqrt{d_{in}}$, zero initialization for $B$, the same LR for both, and $\alpha = 32$. They were unable to improve on these hyperparameters in their experimentation.

### Optimal learning rates for LoRA vs. FullFT

The experiments showed that the optimal LR for LoRA is consistently **10x** the one used for FullFT in the same application, for both supervised learning and reinforcement learning. This appears in every U-shaped plot of performance, whether loss or reward, charted against learning rate. This observation should make it more straightforward to transfer learning hyperparameters from FullFT to LoRA.

The authors do not yet have an adequate theoretical explanation for this observation. One possible derivation starts from the facts that optimal LoRA LR is invariant to rank and that full-rank LoRA is directly comparable to FullFT. This analysis suggests an LR ratio of the model’s hidden size divided by $2 \cdot \alpha$, which does not match the empirical result of the optimal ratio being fixed at 10 independent of the base model.

For empirical analysis, the authors conducted an LR sweep of 14 different `Llama` and `Qwen` models for both LoRA and FullFT on the `Tulu3` dataset. From those sweeps, they fit a function that predicts the optimal learning rate based on the model’s hidden size and an indicator of whether it is `Llama` or `Qwen`. The functional form was:

$$
\text{LR} = M_{\text{LoRA}} \cdot \left(\frac{2000}{\text{hidden size}}\right)^{\text{model pow} + \text{LoRA pow}}
$$

Where:

- $M_{\text{LoRA}}$ is a multiplier applied when LoRA is used, and is 1 if using FullFT.
- $\text{model pow}$ is an exponent adjustment, calculated separately for each model source, `Llama` and `Qwen`.
- $\text{LoRA pow}$ is an additional exponent adjustment for LoRA.
- $\text{hidden size}$ is the dimension of the residual stream of the model.

A predicted learning rate was scored by using linear interpolation to predict the loss, based on the data from the sweep, and rating the parameters by summing the predicted loss over the 14 problems. The optimization found a multiplier of **9.8** for LoRA over FullFT, and different dependence on hidden size for `Qwen3` and `Llama` models. However, LoRA LRs had the same dependence on hidden size as FullFT LRs; the optimization found $\text{LoRA pow} = 0$.

### Learning rates in short and long runs

The typical initialization of LoRA creates an implicit schedule of change in the effective learning rate. This leads to differences between short and long training runs, and some differences in the shape of learning curves compared to FullFT.

At the start of training, $B$ is initialized to zero. While $B$ is very small, changes in $A$ have negligible effects on the adapter $BA$ that is added to the original network weights. As $B$ grows larger, updates to $A$ start to have a bigger impact on the network outputs, with the effective learning rate increasing over the course of training as $B$ approaches $A$ in scale. The authors found that by the end of the full training runs on the `Tulu3` and `OpenThoughts3` datasets, the $B$ matrices ended up with larger spectral norms than the $A$ matrices.

This implies that the optimal LR should be set higher for shorter training runs. Preliminary evidence suggests an optimal multiplier around **15x** over FullFT for short runs, based on anecdotal evidence that the higher multiplier is effective under roughly 100 steps, converging to the 10x multiplier for longer runs.
