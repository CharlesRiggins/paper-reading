## Discussion

The authors move beyond the empirical results to discuss broader considerations related to LoRA performance and applicability that would be of interest to both researchers and builders.

First, they examine in more depth the main result, namely the two conditions under which LoRA performs similarly to full fine-tuning:

1. LoRA is applied to all layers of the network, especially the MLP/MoE layers that house most of the parameters.
2. LoRA works well when it is not capacity constrained, i.e. when the number of trainable parameters exceeds the amount of information to be learned, which can be estimated in terms of dataset size.

When condition (1) is satisfied, LoRA gives similar learning dynamics to FullFT at the very start of training. Then, under condition (2), LoRA continues to look like FullFT until it starts reaching capacity limits.

### Why LoRA might be needed on all layers

As shown earlier, if LoRA is placed only on the attention layers, learning is slower even in the tiny-data regime.

One possible explanation comes from thinking about the empirical neural tangent kernel (**eNTK**) as an approximation of what happens when a small amount of fine-tuning is performed, following Malladi et al., A Kernel-Based View of Language Model Fine-Tuning (Malladi et al., 2022). eNTK is based on dot products of gradients, specifically gradients

$$
g_i = \frac{\partial}{\partial \theta} \log p(\text{token}_i \mid \text{prefix}_i),
$$

and

$$
K(i, j) = g_i \cdot g_j.
$$

As a consequence, the layers with the most parameters will typically have the most influence on the kernel. The paper also points out that the eNTK for LoRA is approximately the same as that for full fine-tuning when all layers are trained. Thus:

$$
\text{LoRA training} \approx \text{eNTK(LoRA)} \approx \text{eNTK(FullFT)} \approx \text{FullFT}.
$$

The approximation $\text{eNTK(LoRA)} \approx \text{eNTK(FullFT)}$ only holds when LoRA is applied to the layers that contain most of the parameters making up the dot products.

### How much capacity is needed by supervised and reinforcement learning?

Past work, Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws (Allen-Zhu and Li, 2024), has shown that neural networks can store **2 bits per parameter**. These results pertain to the maximum amount of information absorbed in the long-training limit, not to compute efficiency or the rate of learning.

The 2-bits-per-parameter result relied on synthetic datasets cleverly constructed to contain a precise amount of information. It is not as straightforward to estimate the information content required for a realistic learning problem. One classic observation is that when minimizing log-loss, the total log-loss measured during the first epoch of training provides a measurement of the dataset’s description length. That is, it gives an upper bound for the number of bits required to memorize the dataset. LLM datasets usually have a loss of around **1 bit**, or 0.69 nats, per token, depending on dataset and model size.

This estimate measures the capacity required to perfectly memorize the dataset, which overestimates the actual capacity needed for “generalizable” learning that reduces log-loss on test data. Measuring the capacity requirements of supervised learning and how these interact with the number of trainable parameters is an open question for future work.

For RL, the authors claim that policy gradient algorithms learn roughly 1 bit of information per episode, given that there is a single reward value at the end of the episode. This is not a fundamental property of RL, as other algorithms could conceivably learn much more from each episode. For example, model-based RL algorithms train the learning agent to predict observations and build a world model, potentially extracting more information per episode. The claim of 1 bit per episode may only apply narrowly to policy gradient algorithms.

The bits-counting argument can be sharpened in information-theoretic terms. Consider an episode, consisting of a trajectory $\tau$ and final reward, as a message, i.e. a noisy channel, that provides some information about the unknown reward function $R$. Conditioning on the current policy and training history, the authors examine the mutual information between the policy gradient estimator and $R$. The REINFORCE update is $G = S \cdot \text{Adv}$ with $S = \nabla \log p_\theta(\tau)$. $S$ is independent of $R$ given the history, so the only $R$-dependent component is the scalar advantage.

By the data processing inequality:

$$
I(G ; R \mid \text{history}) \leq I((S, \text{Adv}) ; R \mid \text{history}) = I(\text{Adv} ; R \mid S, \text{history}) \leq H(\text{Adv}).
$$

If the advantage is quantized into $B$ bins, then $H(\text{Adv}) \lesssim \log(B)$. That is, the number of bits of useful information gleaned per episode is $O(1)$, independent of model size. These bits tell us which member of a discrete set of reward functions, or equivalently optimal-policy classes, we are in. This analysis of mutual information mirrors what is used in some theoretical analysis of optimization algorithms, such as Information Complexity of Black-Box Convex Optimization: A New Look via Feedback Information Theory (Raginsky and Rakhlin, 2009). This estimate is an upper bound on the information absorbed by training; the actual amount learned depends on the policy initialization and other details. For example, if training initializes with a policy that gets no reward, then the entropy of the advantage is zero rather than $\log(B)$, and it will not learn anything.

### Compute efficiency advantage of LoRA

The experiments measured learning progress against the number of training steps, but compute efficiency may also matter. The authors calculate that LoRA takes slightly more than $\frac{2}{3}$ of the FLOPs that full fine-tuning does per pass. As a result, LoRA will often outperform FullFT on compute efficiency overall.

They derive this $\frac{2}{3}$ ratio by analyzing the FLOPs used in the forward-backward pass on a given weight matrix. These operations account for the vast majority of FLOPs in neural network models. The notation is:

- $W \in \mathbb{R}^{N \times N}$ is a weight matrix.
- $x \in \mathbb{R}^N$ is an input vector.
- $y = Wx \in \mathbb{R}^N$ is an output vector.
- $\bar{x}, \bar{y} \in \mathbb{R}^N$ are the gradients of the loss with respect to $x$ and $y$, computed in the backward pass.
- $\bar{W} \in \mathbb{R}^{N \times N}$ is the gradient of the loss with respect to $W$.

Full fine-tuning performs the following operations.

Forward:

1. $y = Wx$, requiring $N^2$ multiply-adds.

Backward:

1. $\bar{x} = W^T \bar{y}$, requiring $N^2$ multiply-adds.
2. $\bar{W} \mathrel{+}= x \bar{y}^T$, requiring $N^2$ multiply-adds.

The forward pass requires $N^2$ multiply-adds, and the backward pass requires another $2 \cdot N^2$, for $3N^2$ total. Training, which requires both, thus uses 3 times the FLOPs of forward-only inference.

With LoRA, $W$ is replaced by $W + BA$, where $B \in \mathbb{R}^{N \times R}$ and $A \in \mathbb{R}^{R \times N}$, with $R \ll N$. Since only $\bar{A}$ and $\bar{B}$ are updated, the third step of updating $\bar{W}$ is replaced with a much cheaper operation. $A$ and $B$ are $N \cdot R$ matrices, so the full forward-backward computation on each requires $3NR$ multiply-adds instead of $3N^2$ for $W$. The total for both is $6NR$. The forward-backward pass on $Wx$ and $\bar{x}$ is still performed, equivalent to the first two steps of FullFT. The total number of multiply-adds is $2N^2 + 6NR$. With $R \ll N$, this is slightly more than $\frac{2}{3}$ of $3N^2$.

If LoRA performance were plotted over FLOPs instead of training steps, it would show a clear advantage over FullFT. This analysis omits FLOPs used for attention, which could be significant in long-context settings.

### Open questions

There are several questions related to the results that the authors would like to see investigated in the future:

- Sharpening predictions of LoRA performance and the precise conditions under which it matches full fine-tuning. The blog roughly characterizes the regime of equal performance and can estimate the required capacity in terms of tokens or episodes, but cannot yet make accurate forecasts.
- The theoretical understanding of LoRA learning rates and training dynamics is limited. A fuller theory that explains the ratio between LoRA and FullFT learning rates would be valuable.
- How do LoRA variants such as `PiSSA`, from PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models (Meng, Wang & Zhang, 2024), perform when measured according to this article’s methodology?
- There are various options for applying LoRA to MoE layers. LoRA users would benefit from an investigation into how well they perform, and how compatible each approach is with methods like tensor parallelism and expert parallelism that are important for large MoE models.
