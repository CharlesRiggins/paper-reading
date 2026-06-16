## 3. Preliminaries

### Problem Setup

The paper considers a general setting in which an LLM agent interacts with an environment to complete multi-step tasks based on a task description $x \sim p(X)$. At each discrete time step $t = 1,2,\dots,T$, the agent observes a state $\bm{s}_t \in \mathcal{S}$ and generates a textual action $\bm{a}_t \in \mathcal{V}^n$, where $\mathcal{V}$ denotes the token vocabulary and $n$ is the maximum generation length. The environment then returns a scalar reward $r_t \in \mathbb{R}$ and the next state $\bm{s}_{t+1}$.

A full episode consists of a trajectory:

$$\bm{\tau} = \{(\bm{s}_1, \bm{a}_1, r_1), (\bm{s}_2, \bm{a}_2, r_2), \dots, (\bm{s}_T, \bm{a}_T, r_T)\}$$

The agent's behavior is governed by an LLM policy $\pi_\theta(\bm{a}_t | \bm{s}_t, x)$, parameterized by $\theta$, which defines a distribution over outputs conditioned on the current state $\bm{s}_t$ and the task prompt $x$. In many realistic scenarios, the environment may provide sparse or delayed rewards (e.g., success and failure signals at the end of an episode) or weak feedback signals for intermediate steps. As the agent generates $T$ consecutive textual actions $(\bm{a}_1, \dots, \bm{a}_T)$, each potentially spanning thousands of tokens, it becomes particularly challenging to assign credit to individual tokens over the course of an episode.

### Group-based RL

Recent RL works converge on a simple recipe for training LLMs: for a given task description $x$, the LLM samples a group of $N$ candidate trajectories $\{\bm{\tau}_1, \bm{\tau}_2, \dots, \bm{\tau}_N\}$, each corresponding to one full episode rollout under $\pi_{\theta_{\text{old}}}$. Each trajectory $\bm{\tau}_i$ receives a scalar reward $R(\bm{\tau}_i)$ reflecting the overall quality or success of the generated outcome. Instead of estimating advantages using separate value functions like PPO [42], group-based RL methods compute advantages purely based on the statistics internal to the sampled group:

$$A(\bm{\tau}_i) = \text{GroupComputation}(\{R(\bm{\tau}_i)\}_{i=1}^N)$$

For example, in **GRPO** [18], the advantage of each trajectory is estimated by normalizing its reward with respect to the group's mean and standard deviation. This design is highly memory-efficient and can scale effectively to the large batch sizes and model sizes typical in modern LLM training, making it a practical and scalable choice for large-scale RL training.
