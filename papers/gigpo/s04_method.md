## 4. Training LLM Agents with GiGPO

While group-based RL algorithms [18; 15] have proven highly effective for training LLMs in single-turn tasks, their extension to multi-step agent settings faces critical challenges in credit assignment. Vanilla GRPO treats each trajectory as a whole and computes a single relative advantage for the entire episode, which fails to provide actionable feedback for individual steps. A natural remedy is to roll out multiple single-step actions for each state $\bm{s}_t$ via $\pi_{\theta_{\text{old}}}$, but this approach quickly becomes impractical due to the substantial overhead of extra LLM forward passes and the difficulty of evaluating rewards for hypothetical actions never actually executed.

To overcome these challenges, the authors propose **Group-in-Group Policy Optimization (GiGPO)**. GiGPO begins by sampling groups of trajectories under identical tasks and initial environment states. It then introduces a **two-level grouping structure**: preserving episode-level grouping for holistic performance comparison, while dynamically constructing an additional set of step-level groups by retroactively aggregating actions encountering the same environment states. This "group-in-group" construction yields two complementary advantages:

1. **Episode relative advantages** capture the holistic effectiveness of each trajectory, providing a stable, global training signal.
2. **Step relative advantages** zoom in on which actions outperform their peers within the same state, endowing the gradient with fine-grained credit.

---

### 4.1 Episode Relative Advantages

The episode-level relative advantages represent the coarse-grained component of GiGPO and mirror the naive application of GRPO at the trajectory level. The agent's policy $\pi_{\theta_{\text{old}}}$ is rolled out in the environment to collect $N$ complete trajectories under a fixed task $x$ and identical initial states. This yields a group of trajectories $\{\bm{\tau}_i\}_{i=1}^N$, each denoted as $\bm{\tau}_i = \{(\bm{s}^{(i)}_1, \bm{a}^{(i)}_1, r^{(i)}_1), \dots, (\bm{s}^{(i)}_T, \bm{a}^{(i)}_T, r^{(i)}_T)\}$ with identical initial states $\bm{s}^{(1)}_1 = \bm{s}^{(2)}_1 = \dots = \bm{s}^{(N)}_1$. The total return $R(\bm{\tau}_i) = \sum_t r^{(i)}_t$ measures how effectively the agent completes the task.

The trajectories and their returns are organized into an episode-level group:

$$G^E = \Bigl\{\bigl(\bm{\tau}_1, R(\bm{\tau}_1)\bigr), \bigl(\bm{\tau}_2, R(\bm{\tau}_2)\bigr), \dots, \bigl(\bm{\tau}_N, R(\bm{\tau}_N)\bigr)\Bigr\}$$

The episode relative advantage $A^E(\bm{\tau}_i)$ is computed by normalizing the total return with the group's mean and a normalization factor:

$$A^E(\bm{\tau}_i) = \frac{R(\bm{\tau}_i) - \text{mean}\bigl(\{R(\bm{\tau}_j)\}_{j=1}^N\bigr)}{F_{\text{norm}}\bigl(\{R(\bm{\tau}_j)\}_{j=1}^N\bigr)}$$

In GRPO [18], the default normalization factor is the standard deviation ($F_{\text{norm}} = \text{std}$). However, this may introduce a **difficulty bias** [19], where trajectories from low-variance groups (e.g., very easy or hard tasks) receive disproportionately large gradients. In the context of LLM agents with very long horizons, this effect tends to emerge frequently. As an alternative, the paper also considers a fixed normalization factor $F_{\text{norm}} = 1$, which yields an unbiased Leave-One-Out estimator [16] (see Appendix C for details).

---

### 4.2 Step Relative Advantages

While the episode relative advantage offers a macro, trajectory-wide signal, it cannot distinguish between the contributions of individual actions within the trajectory. To obtain fine-grained feedback, step-level groups must be formed: for the same state, different actions are gathered and their outcomes compared.

#### Anchor State Grouping

As all trajectories $\{\bm{\tau}_1, \dots, \bm{\tau}_N\}$ arise from the same task $x$ and identical initial conditions, many environment states naturally recur across episodes and even across time steps within a single trajectory. This redundancy is leveraged by identifying and grouping identical states across trajectories. Let $\mathcal{U} = \{\tilde{\bm{s}}_1, \tilde{\bm{s}}_2, \dots, \tilde{\bm{s}}_U\}$ denote the set of all distinct environment states appearing in the trajectory group. Each unique state $\tilde{\bm{s}} \in \mathcal{U}$ is treated as an implicit **anchor** and used to gather all matching occurrences. For each anchor state $\tilde{\bm{s}}$, a step-level group is constructed:

$$G^S(\tilde{\bm{s}}) = \bigl\{\bigl(\bm{a}^{(i)}_t, r^{(i)}_t\bigr) \; \bigm| \; \bm{s}^{(i)}_t = \tilde{\bm{s}}, \; 1 \leq i \leq N, \; 1 \leq t \leq T\bigr\}$$

Unlike per-state rollout, this procedure incurs **no extra rollouts**: it is entirely offline and requires only lightweight key-based grouping using hashmaps.

#### Relative Advantage Computation

Although each tuple $(\bm{a}^{(i)}_t, r^{(i)}_t)$ contains an immediate reward $r^{(i)}_t$, it may be sparse in long-horizon tasks. To better capture long-term impact, a **discounted return** is associated with each step. With discount factor $\gamma \in (0,1]$, the discounted return $R_t^{(i)}$ is:

$$R_t^{(i)} = \sum\nolimits_{k=t}^{T} \gamma^{k-t} \, r_k^{(i)}$$

This quantity captures the future impact of action $\bm{a}^{(i)}_t$ on subsequent rewards. The step-level group becomes:

$$G^S(\tilde{\bm{s}}) = \bigl\{\bigl(\bm{a}^{(i)}_t, R^{(i)}_t\bigr) \; \bigm| \; \bm{s}^{(i)}_t = \tilde{\bm{s}}, \; 1 \leq i \leq N, \; 1 \leq t \leq T\bigr\}$$

The **step relative advantage** for each $\tilde{\bm{s}} \sim \mathcal{U}$ and each action $\bm{a}_t^{(i)}$ in $G^S(\tilde{\bm{s}})$ is:

$$A^S(\bm{a}_t^{(i)}) = \frac{R_t^{(i)} - \text{mean}\left(\bigl\{R_t^{(j)} \bigm| (\bm{a}_t^{(j)}, R_t^{(j)}) \in G^S(\tilde{\bm{s}})\bigr\}\right)}{F_{\text{norm}}\left(\bigl\{R_t^{(j)} \bigm| (\bm{a}_t^{(j)}, R_t^{(j)}) \in G^S(\tilde{\bm{s}})\bigr\}\right)}$$

$A^S$ provides micro credit assignment and fine-grained feedback on the relative quality of individual actions taken from the same state.

#### How Step-Level Group Works

Consider two trajectories in WebShop: in $\tau_1$, the agent first selects the 2nd Item (incorrect), then returns to the previous page and selects the 1st Item (correct), successfully completing the task. Due to temporal discounting, the earlier action (2nd Item) receives a lower discounted return than the later correct one (1st Item). In $\tau_2$, the agent clicks Next Page, ultimately failing. By aggregating these actions into the same step-level group based on their shared anchor state, GiGPO computes their relative advantages and yields a clear preference ordering: $A^S(\text{1st Item}) > A^S(\text{2nd Item}) > A^S(\text{Next Page})$. This ranking successfully captures fine-grained distinctions in long-term utility missed by prior group-based RL methods.

---

### 4.3 Group-in-Group Policy Optimization

The two levels of advantage signals are combined into a single **group-in-group advantage**:

$$A(\bm{a}_t^{(i)}) = A^E(\bm{\tau}_i) + \omega \cdot A^S(\bm{a}_t^{(i)})$$

where $\omega \in \mathbb{R}_{\geq 0}$ is a weighting coefficient that balances episode relative advantage and step relative advantage. The clipped policy optimization objective of GiGPO is:

$$\begin{aligned}
\mathcal{J}_{\mathrm{GiGPO}}(\theta) &= \mathbb{E}_{\begin{subarray}{c}x \sim p(X) \\ \{\bm{\tau}_i\}_{i=1}^{N} \sim \pi_{\theta_{\text{old}}}\end{subarray}} \biggl[ \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} \min\Bigl(\rho_{\theta}(\bm{a}_t^{(i)}) A(\bm{a}_t^{(i)}), \, \text{clip}\bigl(\rho_{\theta}(\bm{a}_t^{(i)}), 1 \pm \epsilon\bigr) A(\bm{a}_t^{(i)})\Bigr) \biggr] \\
&\quad - \beta \mathbb{D}_{\mathrm{KL}}\bigl(\pi_{\theta}(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x)\bigr)
\end{aligned}$$

where $\rho_{\theta}(\bm{a}_t^{(i)}) = \frac{\pi_{\theta}(\bm{a}_t^{(i)} \mid \bm{s}_t^{(i)}, x)}{\pi_{\theta_{\text{old}}}(\bm{a}_t^{(i)} \mid \bm{s}_t^{(i)}, x)}$ is the importance sampling ratio, $\beta$ controls the strength of the KL penalty encouraging proximity to a reference policy $\pi_{\text{ref}}$.
