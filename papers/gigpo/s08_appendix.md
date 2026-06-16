## 8. Appendix

### A. Open Source Codebase: verl-agent

As part of the new assets released with this work, the authors propose **verl-agent** ([github.com/langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)), a highly scalable RL training framework for long-horizon, multi-turn LLM agent training.

Built upon the veRL framework [73], verl-agent extends it with key features:
1. **Step-wise multi-turn interaction paradigm** — avoids concatenating full interaction histories, ensuring efficient memory control and scalability for very long-horizon optimization
2. **Customizable memory module** — developers can flexibly determine which historical information to include at each step
3. **Parallel and group-based environments** — gym-style interface supporting high-throughput rollouts
4. **Broad model compatibility** — supports Qwen3 [74], Qwen2.5, LLaMA3.2, and LoRA-based fine-tuning [75]
5. **Multimodal (vision-language) agent support** — e.g., Qwen2.5-VL
6. **Diverse suite of environments** — Search (tool use), ALFWorld, WebShop, Sokoban, Gym Cards
7. **Comprehensive RL algorithm support** — GiGPO, GRPO, PPO, DAPO, RLOO, etc.

Unlike RAGEN/Search-R1 which concatenate full history at every step (leading to rapidly expanding context), verl-agent adopts a step-wise multi-turn rollout with flexible per-step input construction and memory control.

---

### B. Broader Impacts

GiGPO improves training stability and agent performance without increasing computational or memory overhead. The algorithm holds promise for virtual assistants, web automation, educational tools, and embodied AI systems. Its critic-free and scalable design lowers the barrier to training effective multi-step agents. While GiGPO is a methodological contribution with no direct downstream deployment, the improved agent training techniques may indirectly enable more autonomous behavior. Appropriate safeguards and responsible usage should be considered.

---

### C. Unbiasedness

Setting $F_{\text{norm}} = 1$ leads to an unbiased estimator up to a constant scaling factor:

The standard REINFORCE Leave-One-Out (RLOO) [16; 17] is defined as:

$$A^{\mathrm{RLOO}}(\tau_i) = R(\tau_i) - \frac{1}{N-1}\sum_{j \neq i} R(\tau_j)$$

Relating $A^E(\tau_i)$ (with $F_{\text{norm}} = 1$) and $A^{\mathrm{RLOO}}(\tau_i)$ by introducing a scaling factor of $\frac{N}{N-1}$:

$$\frac{N}{N-1}A^E(\tau_i) = \frac{N}{N-1}R(\tau_i) - \frac{N}{N-1} \cdot \frac{1}{N}\sum_{j=1}^{N} R(\tau_j)$$

$$= \frac{N}{N-1}R(\tau_i) - \frac{1}{N-1}R(\tau_i) - \frac{1}{N-1}\sum_{j \neq i} R(\tau_j)$$

$$= R(\tau_i) - \frac{1}{N-1}\sum_{j \neq i} R(\tau_j) = A^{\mathrm{RLOO}}(\tau_i)$$

Thus, $F_{\text{norm}} = 1$ corresponds to a rescaled RLOO advantage. Scaling by a constant does not affect policy gradient dynamics (absorbed into learning rate).

---

### D. Pseudo Code

**Algorithm 1: Training LLM Agents with GiGPO**

1. **Require:** Initial policy $\pi_{\theta_{\text{old}}}$, task distribution $p(X)$, discount factor $\gamma$, weighting $\omega$, clipping parameter $\epsilon$, KL penalty $\beta$, group size $N$
2. **for** each training iteration **do**
3. &nbsp;&nbsp;&nbsp;&nbsp;Update the old policy model: $\theta_{\text{old}} \leftarrow \theta$
4. &nbsp;&nbsp;&nbsp;&nbsp;// Multi-step rollout phase
5. &nbsp;&nbsp;&nbsp;&nbsp;Sample task $x \sim p(X)$ and initialize $N$ identical environments
6. &nbsp;&nbsp;&nbsp;&nbsp;**for** $t = 1$ to $T$ **do**
7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample actions $\{\bm{a}_t^{(i)} \sim \pi_{\theta_{\text{old}}}(\cdot \mid \bm{s}_t^{(i)}, x)\}_{i=1}^{N}$
8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Execute actions, observe rewards $\{r_t^{(i)}\}_{i=1}^{N}$ and next state $\{\bm{s}_{t+1}^{(i)}\}_{i=1}^{N}$
9. &nbsp;&nbsp;&nbsp;&nbsp;**end for**
10. &nbsp;&nbsp;&nbsp;&nbsp;// Grouping phase
11. &nbsp;&nbsp;&nbsp;&nbsp;Compute episode relative advantages $A^E(\tau_i)$ via Equation (3)
12. &nbsp;&nbsp;&nbsp;&nbsp;*Build step-level groups $G^S(\tilde{\bm{s}})$ via the anchor states*
13. &nbsp;&nbsp;&nbsp;&nbsp;*Compute step relative advantages $A^S(\bm{a}_t^{(i)})$ via Equation (7)*
14. &nbsp;&nbsp;&nbsp;&nbsp;// Policy update phase
15. &nbsp;&nbsp;&nbsp;&nbsp;*Combine advantages: $A(\bm{a}_t^{(i)}) = A^E(\tau_i) + \omega A^S(\bm{a}_t^{(i)})$*
16. &nbsp;&nbsp;&nbsp;&nbsp;Update policy $\theta$ by maximizing objective $\mathcal{J}_{\mathrm{GiGPO}}(\theta)$
17. **end for**

Compared to vanilla GRPO, the italicized parts are GiGPO's additions. Building $G^S(\tilde{\bm{s}})$ is implemented via hash tables with minimal overhead. Step advantage computation and combination involve only simple arithmetic operations.

---

### E. Experiment Details

#### E.1 Details of Training

**ALFWorld:** Max prompt length 2048 tokens, max response 512 tokens, 50 env steps/episode. LR: 1e-6 (actor), 1e-5 (critic for PPO only). Reward: 10 (success), 0 (failure), -0.1 (invalid action). Group size 8, 16 groups per rollout = 128 environments. Rollout temp 1.0, validation temp 0.4. Mini-batch 256, KL coeff 0.01. $\omega = 1$, $\gamma = 0.95$.

**WebShop:** Max prompt length 4096 tokens, max response 512 tokens, 15 env steps/episode. Same LR, reward, and rollout config as ALFWorld. Mini-batch 64, KL coeff 0.01. $\omega = 1$, $\gamma = 0.95$.

**Search-Augmented QA:** Max prompt 4096 tokens, max response 512 tokens, 4 max turns. LR 1e-6. Reward: 1 (success), 0 (failure), -0.01 (invalid action). Train data size 256, group size 5. Rollout temp 1.0, validation temp 0.0. Mini-batch 512, KL coeff 0.001. $\omega = 1$, $\gamma = 0.95$.

**Computing:** ALFWorld/WebShop: 2×H100 (1.5B), 4×H100 (7B), 150 iterations. Search QA: 4×H100 (3B), 8×H100 (7B), 200 iterations.

#### E.2 Prompts

All prompt templates use Python-style string formatting with placeholders like `{task_description}`, `{step_count}`, `{current_observation}`. The agent outputs reasoning within `<think> </think>` tags and actions within `<action> </action>` tags. Historical information is used with history length 2 for ALFWorld/WebShop and full history for search-augmented QA. The search agent additionally uses `<search> </search>` for queries, `<information> </information>` for retrieved evidence, and `<answer> </answer>` for answers.

#### E.3 Performance on Vision-Language Agents

Additional experiments in VLM settings with Qwen2.5-VL-3B-Instruct [78] on two interactive game environments:

| Method | Sokoban [6×6] | EZPoints |
|--------|---------------|----------|
| Qwen2.5-VL (prompting) | 11.7 | 3.1 |
| GRPO | 67.1±4.7 | 86.9±3.4 |
| GiGPO w/ std | 76.9±2.7 | **100.0±0.0** |
| GiGPO w/o std | **81.0±3.6** | **100.0±0.0** |

GiGPO significantly outperforms both prompting and GRPO, achieving 81.0% on Sokoban and 100% on EZPoints, demonstrating ability to generalize beyond language-only settings.

#### E.4 Orthogonality to Single-Turn Group-Based RL

GiGPO remains orthogonal to other single-turn group-based RL advances. Integrating DAPO's dynamic sampling and clip-higher techniques into GiGPO yields **GiGPO_dynamic**:

| Method | WebShop Score | Success Rate |
|--------|---------------|--------------|
| GRPO | 75.8±3.5 | 56.8±3.8 |
| DAPO | 84.6±2.9 | 66.1±3.2 |
| **GiGPO_dynamic** | **87.5±1.6** | **75.0±3.5** |

GiGPO_dynamic further outperforms DAPO, confirming it can effectively benefit from and amplify such improvements.

#### E.5 Sensitivity Analysis on $\omega$

| $\omega$ | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 |
|----------|-----|-----|-----|-----|-----|-----|-----|-----|
| Score | 76.2 | 79.6 | 82.4 | 83.5 | **84.9** | 83.5 | 82.6 | 77.0 |
| Success Rate | 56.6 | 63.1 | 65.8 | 67.2 | **68.3** | 67.4 | 66.5 | 56.3 |

GiGPO is relatively insensitive to $\omega$ within $[0.4, 1.2]$, demonstrating robustness. The optimum is at $\omega = 0.8$. Excessive emphasis on step-level signals ($\omega > 1.2$) suppresses useful trajectory-level guidance.

---

### F. Reasoning Behavior

The paper presents a complete multi-turn trajectory showing the reasoning behavior of a GiGPO-trained Qwen2.5-7B-Instruct agent on ALFWorld (task: "heat some egg and put it in countertop"). The agent demonstrates structured step-by-step reasoning within `<think>` tags:

- **Step 1:** Identifies need to find an egg → goes to fridge 1
- **Step 2:** Observes fridge is closed → opens fridge 1
- **Step 3:** Opens fridge, finds no egg → plans to check other locations
- Subsequent steps show systematic exploration and task completion

This emergent reasoning behavior illustrates how GiGPO's fine-grained credit assignment enables the agent to develop coherent, goal-directed planning capabilities.
