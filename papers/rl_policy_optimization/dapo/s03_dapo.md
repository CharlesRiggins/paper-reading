## 3. DAPO

The paper proposes **Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)**. For each question $q$ paired with answer $a$, DAPO samples a group of outputs $\{o_i\}_{i=1}^{G}$ and optimizes the policy with a token-level clipped objective:

$$
\mathcal{J}_{\mathrm{DAPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D},\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}\left[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\ \mathrm{clip}\left(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\right)\hat{A}_{i,t}\right)\right]
$$

subject to dynamic-sampling eligibility:

$$
0<\left|\{o_i\mid \texttt{is\_equivalent}(a,o_i)\}\right|<G.
$$

The ratio and advantage are:

$$
r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})},\qquad
\hat{A}_{i,t}=\frac{R_i-\mathrm{mean}(\{R_i\}_{i=1}^{G})}{\mathrm{std}(\{R_i\}_{i=1}^{G})}.
$$

The full algorithm is summarized later in this section. The key technical ingredients are Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, Overlong Reward Shaping, and dataset transformation.

### 3.1 Raise the Ceiling: Clip-Higher

In initial experiments with naive PPO [21] or GRPO [38], the authors observe **entropy collapse**: policy entropy quickly decreases during training, and sampled responses within certain groups become nearly identical. This indicates limited exploration and premature determinism, which can hinder scaling.

DAPO introduces **Clip-Higher** to address this issue. PPO-Clip uses clipping over the importance sampling ratio to restrict the trust region and improve RL stability. The authors identify that the **upper clip** can restrict policy exploration: making an already-likely “exploitation” token more probable is easy, while uplifting an unlikely “exploration” token is tightly bounded.

For example, with $\varepsilon=0.2$ and positive advantage $\hat{A}_{i,t}>0$, consider two actions whose old-policy probabilities are $\pi_{\theta_{\mathrm{old}}}(o_i\mid q)=0.01$ and $0.9$. Their clipped upper probabilities are $0.012$ and $1.08$ respectively, since the bound is $\pi_{\theta_{\mathrm{old}}}\cdot(1+\varepsilon)$. High-probability exploitation tokens are effectively unconstrained, while low-probability exploration tokens can only increase by a very small absolute amount. Empirically, the authors find that the mean probability of up-clipped tokens is low, $\pi_\theta(o_i\mid q)<0.2$, supporting the intuition that the upper clipping threshold restricts the probability increase of exploration tokens.

DAPO decouples the lower and upper clipping ranges as $\varepsilon_{\mathrm{low}}$ and $\varepsilon_{\mathrm{high}}$:

$$
\mathrm{clip}\left(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\right).
$$

The paper increases $\varepsilon_{\mathrm{high}}$ to leave more room for low-probability tokens to increase. Figure 2 shows that this adjustment enhances policy entropy and facilitates more diverse sampling. $\varepsilon_{\mathrm{low}}$ is kept unchanged because increasing it can suppress token probabilities toward zero and collapse the sampling space.

Figure 3 reports two supporting curves: the mean up-clipped probability and the proportion of prompts whose sampled group has accuracy 1. Both curves motivate the need for asymmetric clipping and dynamic sampling.

### 3.2 The More the Merrier: Dynamic Sampling

Existing RL algorithms suffer from a **gradient-decreasing problem** when prompts have group accuracy equal to 1. In GRPO, if all outputs $\{o_i\}_{i=1}^{G}$ for a prompt are correct and receive the same reward, the group advantage becomes zero. Zero advantage yields zero policy gradients, shrinking batch-gradient magnitude and increasing sensitivity to noise, which degrades sample efficiency.

Empirically, the number of samples with accuracy equal to 1 continues to increase during training, as shown in Figure 3(b). This means that the effective number of prompts in each batch decreases over time, increasing gradient variance and dampening useful training signals.

DAPO therefore **over-samples and filters out prompts with accuracy equal to 1 or 0**, keeping only groups with non-zero relative signal:

$$
0<\left|\{o_i\mid \texttt{is\_equivalent}(a,o_i)\}\right|<G.
$$

Before training on each batch, the system keeps sampling until the batch is filled with prompts whose group accuracy is neither 0 nor 1. The sampling cost for each batch is dynamic, but the authors argue that this does not necessarily hurt wall-clock efficiency: in synchronized systems, generation time is often dominated by long-tail samples, and Figure 6 shows that dynamic sampling reaches the same performance faster because fewer training steps are required.

### 3.3 Rebalancing Act: Token-Level Policy Gradient Loss

Original GRPO computes a **sample-level loss**: it averages token losses within each sample, then aggregates across samples. Consequently, each sample receives equal weight in the final loss, regardless of length. In long-CoT RL, this introduces several challenges.

First, tokens in longer high-quality responses contribute less to the overall loss, making it harder for the model to learn useful reasoning patterns from those responses. Second, the authors observe that overly long samples often contain low-quality patterns such as gibberish and repetitive text. Sample-level loss reduction does not penalize such undesirable long-sample patterns strongly enough, which can lead to unhealthy increases in entropy and response length, as shown in Figure 4.

DAPO replaces sample-level reduction with **token-level policy gradient loss**:

$$
\mathcal{J}_{\mathrm{DAPO}}(\theta)=\mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\ \mathrm{clip}\left(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\right)\hat{A}_{i,t}\right)\right].
$$

Under this reduction, longer sequences can have more influence on the gradient update than shorter sequences. From the token perspective, any generation pattern that increases or decreases reward is promoted or suppressed equally, regardless of the response length in which it appears.

### 3.4 Hide and Seek: Overlong Reward Shaping

In RL training, generation usually has a maximum length, and overlong samples are truncated. The authors find that improper reward shaping for truncated samples introduces reward noise and significantly disrupts training.

By default, one might assign punitive reward to truncated samples. This can be noisy because a sound reasoning process may be penalized solely for being too long, which can confuse the model about whether its reasoning process is valid. To study this noise, the authors first apply **Overlong Filtering**, which masks the loss of truncated samples. Figure 5 shows that this stabilizes training and improves performance.

DAPO then introduces **Soft Overlong Punishment**, a length-aware penalty for truncated or near-truncated samples. If the response length exceeds a predefined expected maximum, the method defines a punishment interval: inside that interval, longer responses receive larger penalties. The penalty is added to the original rule-based correctness reward, signaling the model to avoid excessive length without turning every near-correct long solution into a noisy hard negative.

The length penalty is:

$$
R_{\mathrm{length}}(y)=
\begin{cases}
0, & |y|\le L_{\mathrm{max}}-L_{\mathrm{cache}},\\
\frac{(L_{\mathrm{max}}-L_{\mathrm{cache}})-|y|}{L_{\mathrm{cache}}}, & L_{\mathrm{max}}-L_{\mathrm{cache}}<|y|\le L_{\mathrm{max}},\\
-1, & L_{\mathrm{max}}<|y|.
\end{cases}
$$

#### Algorithm 1. DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization

| Step | Operation |
|---:|---|
| Input | Initial policy model $\pi_\theta$; reward model $R$; task prompts $\mathcal{D}$; hyperparameters $\varepsilon_{\mathtt{low}}$, $\varepsilon_{\mathtt{high}}$. |
| 1 | For each step $1,\ldots,M$, sample a batch $\mathcal{D}_b$ from $\mathcal{D}$. |
| 2 | Update the old policy model: $\pi_{\theta_{\mathrm{old}}}\leftarrow\pi_\theta$. |
| 3 | For each question $q\in\mathcal{D}_b$, sample $G$ outputs $\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)$. |
| 4 | Compute rewards $\{r_i\}_{i=1}^{G}$ for sampled outputs using $R$. |
| 5 | Filter outputs and add remaining groups to the dynamic sampling buffer according to Equation 11. |
| 6 | If buffer size $n_b<N$, continue sampling. |
| 7 | For each $o_i$ in the buffer, compute $\hat{A}_{i,t}$ for each token using Equation 9. |
| 8 | For iteration $1,\ldots,\mu$, update $\pi_\theta$ by maximizing the DAPO objective. |
| Output | Optimized policy $\pi_\theta$. |

### 3.5 Dataset Transformation

The dataset is sourced from the web and official competition homepages through web scraping and manual annotation. Math answers often appear in heterogeneous formats—expressions, formulas, numbers—which makes comprehensive parsing rules difficult and error-prone.

To provide accurate rule-based rewards and minimize formula-parser errors, the authors adopt an AIME-inspired answer transformation strategy: select and transform answers into integers, which are easy to parse. For instance, if an original answer has the form $\frac{a+\sqrt{b}}{c}$, the LLM is instructed to modify the question so that the expected answer becomes $a+b+c$.

After selection and transformation, the authors obtain **DAPO-Math-17K**, consisting of **17K prompts**, each paired with an integer answer.
