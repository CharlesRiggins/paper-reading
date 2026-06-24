## 2. Background

### 2.1 Knowledge Distillation for Autoregressive Large Language Models

Knowledge distillation transfers knowledge from a larger teacher model to a smaller student model by training the student to mimic the teacher's behavior [10, 13, 25]. The core insight is that the teacher's **soft probability distribution** over classes contains richer information than hard labels alone, as it reveals the teacher's learned similarities between classes.

For auto-regressive language models, given a dataset $\mathcal{S}=\{(x,y^\star)\}$ where $x$ denotes an input and $y^\star$ is the corresponding reference output, both teacher $p_T$ and student $p_S$ define token-level distributions over vocabulary $\mathcal{V}$. **Traditional supervised distillation** minimizes a divergence $D$ between teacher and student distributions averaged over a fixed dataset:

$$
\mathcal{L}_{\text{Supervised Distillation}}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{S}}[D(p_{T}\|p_{S})(y|x)], \tag{2}
$$

where $D(p_{T}\|p_{S})(y|x)=\frac{1}{|y|}\sum_{n=1}^{|y|}D\bigl(p_{T}(\cdot|y_{<n}) \,\|\, p_{S}(\cdot|y_{<n})\bigr)$ is the trajectory-averaged token-level divergence. This is **off-policy**: it trains on a fixed reference sequence $y$ rather than the student's own outputs.

**On-policy distillation** instead samples trajectories from the student and matches the teacher on those samples:

$$
\mathcal{L}_{\text{On-Policy Distillation}}(\theta)=\mathbb{E}_{x\sim\mathcal{S}}\bigl[\mathbb{E}_{\hat{y}\sim p_{S}(\cdot|x)}[D(p_{T}\|p_{S})(\hat{y}|x)]\bigr]. \tag{3}
$$

This approach connects distillation to **imitation learning** [24], where the student iteratively improves by learning from the teacher's guidance on its own outputs, combining the on-policy relevance of reinforcement learning with the dense reward signal of supervised learning, thereby mitigating exposure bias while maintaining computational efficiency.

### 2.2 Reinforcement Learning with Verifiable Rewards

Reinforcement learning with verifiable rewards (**RLVR**) has emerged as a popular approach for post-training LLMs, particularly on tasks with easily verifiable outcomes such as mathematics and coding, using algorithms like **Proximal Policy Optimization** (PPO) [26] and **Group Relative Policy Optimization** (GRPO) [27].

GRPO trains by sampling a group of $G$ responses $\{o_1, o_2, \ldots, o_G\}$ from the current policy $\pi_\theta$ for each problem $x$. Each response $o_i$ receives a binary reward $r_i \in \{0,1\}$ indicating correctness. The method then assigns advantages to all tokens $k=1,\ldots,|o_i|$ within response $o_i$ using a **group-normalized reward**:

$$
A_{i}=\frac{r_{i}-\text{mean}(\{r_{j}\}_{j=1}^{G})}{\text{std}(\{r_{j}\}_{j=1}^{G})}. \tag{4}
$$

This formulation can be understood through the **value function lens**: $\text{mean}(\{r_j\}_{j=1}^G)$ serves as a $G$-sample Monte Carlo estimate of the value function $V(x)$, while the sparse binary reward $r_i$ represents the (undiscounted) state-action value $Q(x, o_i)$. Critically, **all tokens within a response share the same advantage**, as the reward signal is provided only at the sequence level.

The GRPO objective incorporates a clipped surrogate loss to moderate policy updates, along with a reverse KL penalty to prevent excessive deviation from a reference policy:

$$
\begin{split}\mathcal{L}_{\text{GRPO}}(\theta)=\mathbb{E}_{\begin{subarray}{c}x\sim\mathcal{S}\\ o_{1},\ldots,o_{G}\sim\pi_{\theta}(\cdot|x)\end{subarray}}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{n=1}^{|o_{i}|}\\ \min\left(\rho_{i}^{n}A_{i},\text{clip}\left(\rho_{i}^{n},1-\varepsilon,1+\varepsilon\right)A_{i}\right)\\ -\beta D_{\text{KL}}[\pi_{\theta}(\cdot|x)\|\pi_{\text{ref}}(\cdot|x)]\Bigg]\end{split} \tag{5}
$$

where $\rho_{i}^{n}=\frac{\pi_{\theta}(o_{i}^{n}|x,o_{i}^{<n})}{\pi_{\theta_{\text{old}}}(o_{i}^{n}|x,o_{i}^{<n})}$ is the importance-sampling ratio.

While RLVR methods have demonstrated strong empirical performance, they face **two key limitations**: (1) the reward signal is **sparse**, providing only sequence-level feedback rather than token-level guidance on where errors occur; and (2) when all sampled responses receive identical rewards (all correct or all incorrect), the **advantages become zero**, preventing any policy update despite the computational cost of sampling.
