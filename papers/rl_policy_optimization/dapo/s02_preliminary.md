## 2. Preliminary

### 2.1 Proximal Policy Optimization (PPO)

`PPO` [21] introduces a clipped surrogate objective for policy optimization. By constraining policy updates within a proximal region of the previous policy through clipping, PPO stabilizes training and improves sample efficiency. PPO updates the policy by maximizing:

$$
\mathcal{J}_{\mathrm{PPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D},\,o_{\le t}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}\left[\min\left(\frac{\pi_{\theta}(o_t\mid q,o_{<t})}{\pi_{\theta_{\mathrm{old}}}(o_t\mid q,o_{<t})}\hat{A}_t,\ \mathrm{clip}\left(\frac{\pi_{\theta}(o_t\mid q,o_{<t})}{\pi_{\theta_{\mathrm{old}}}(o_t\mid q,o_{<t})},1-\varepsilon,1+\varepsilon\right)\hat{A}_t\right)\right].
$$

Here, $(q,a)$ is a question-answer pair from the data distribution $\mathcal{D}$, $\varepsilon$ is the clipping range for the importance sampling ratio, and $\hat{A}_t$ is an advantage estimator at time step $t$. Given value function $V$ and reward function $R$, $\hat{A}_t$ is computed using **Generalized Advantage Estimation (GAE)** [22]:

$$
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},
$$

where

$$
\delta_l=R_l+\gamma V(s_{l+1})-V(s_l),\qquad 0\le \gamma,\lambda\le 1.
$$

### 2.2 Group Relative Policy Optimization (GRPO)

Compared with PPO, `GRPO` removes the value function and estimates advantage in a group-relative manner. For a question-answer pair $(q,a)$, the behavior policy $\pi_{\theta_{\mathrm{old}}}$ samples a group of $G$ responses $\{o_i\}_{i=1}^{G}$. The advantage for the $i$-th response is obtained by normalizing group-level rewards $\{R_i\}_{i=1}^{G}$:

$$
\hat{A}_{i,t}=\frac{r_i-\mathrm{mean}(\{R_i\}_{i=1}^{G})}{\mathrm{std}(\{R_i\}_{i=1}^{G})}.
$$

Like PPO, GRPO uses a clipped objective plus a directly imposed KL penalty:

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D},\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\ \mathrm{clip}(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_{i,t}\right)-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\right)\right],
$$

where

$$
r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}.
$$

A key implementation detail is that GRPO computes the objective at the **sample level**: it first averages token losses within each generated sequence, then averages losses across samples. The paper later argues in Section 3.3 that this reduction choice matters substantially for long-CoT RL.

Figure 2 compares AIME accuracy and actor-model entropy before and after applying **Clip-Higher**. The figure motivates the later DAPO modifications by showing that naive policy optimization can rapidly lose entropy during RL training.

### 2.3 Removing KL Divergence

The KL penalty regulates divergence between the online policy and a frozen reference policy. In the RLHF setting [23], the goal of RL is to align model behavior while avoiding excessive drift from the initial model.

For long-CoT reasoning model training, however, the model distribution may need to diverge significantly from the initial model. The paper therefore argues that this restriction is unnecessary for its setting and excludes the KL term from DAPO.

### 2.4 Rule-based Reward Modeling

Reward models often suffer from **reward hacking** [24–29]. Instead of relying on a learned reward model, the paper directly uses final accuracy on verifiable tasks as the outcome reward:

$$
R(\hat{y},y)=
\begin{cases}
1, & \texttt{is\_equivalent}(\hat{y},y),\\
-1, & \text{otherwise}.
\end{cases}
$$

Here, $y$ is the ground-truth answer and $\hat{y}$ is the predicted answer. The authors note that this rule-based approach has proven effective for activating base-model reasoning ability across automated theorem proving [30–33], computer programming [34–37], and mathematics competition [2].
