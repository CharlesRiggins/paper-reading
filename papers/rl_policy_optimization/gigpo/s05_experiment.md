## 5. Experiment

This section presents empirical evaluations of GiGPO across a variety of agentic tasks, demonstrating: (1) the strong ability of GiGPO in training LLM agents; (2) ablation study results; (3) the dynamic trend of step-level groups $G^S(\tilde{\bm{s}})$ over training; (4) the computational budget of GiGPO.

### 5.1 Experiment Setup

**Benchmarks.** Three categories of tasks are used:

- **ALFWorld** [5]: An embodied environment for multi-step decision-making with 3,827 task instances across six categories — Pick & Place, Examine in Light, Clean & Place, Heat & Place, Cool & Place, and Pick Two & Place. Each episode involves multi-turn text-based interaction.
- **WebShop** [22]: A web-based interactive shopping environment with over 1.1 million products and 12k user instructions, requiring the agent to search, navigate, and purchase items on a simulated HTML website.
- **Search-augmented QA**: Multi-turn tool calling on single-hop QA datasets (NQ [62], TriviaQA [63], PopQA [64]) and multi-hop QA datasets (HotpotQA [65], 2Wiki [66], MuSiQue [67], Bamboogle [68]).

**Baselines.**
- Closed-source LLMs: GPT-4o [1], Gemini-2.5-Pro [2]
- Prompting agents: ReAct [29], Reflexion [30]
- RL training: PPO [42] (with critic), RLOO [16; 17], GRPO [18]
- For search-augmented QA: R1-Instruct, Search-R1 [58], ZeroSearch [59], StepSearch [69]

**Training details.** Base models: Qwen2.5-1.5B/3B/7B-Instruct [3]. Weighting coefficient $\omega = 1$ with no further tuning. Rollout group size $N = 8$ for ALFWorld/WebShop, $N = 5$ for search-augmented QA. For search, E5 [70] is used as the retriever. Similarity-based GiGPO is incorporated, where anchor state grouping uses longest matching subsequence with threshold 0.9.

---

### 5.2 Performance on ALFWorld and WebShop

| Type | Method | ALFWorld All (1.5B) | ALFWorld All (7B) | WebShop Score (1.5B) | WebShop Succ. (1.5B) | WebShop Score (7B) | WebShop Succ. (7B) |
|------|--------|---------------------|--------------------|-----------------------|-----------------------|---------------------|---------------------|
| Prompting | GPT-4o | 48.0 | — | 31.8 | 23.7 | — | — |
| Prompting | Gemini-2.5-Pro | 60.3 | — | 42.5 | 35.9 | — | — |
| Prompting | Qwen2.5 | 4.1 | 14.8 | 23.1 | 5.2 | 26.4 | 7.8 |
| Prompting | ReAct | 12.8 | 31.2 | 40.1 | 11.3 | 46.2 | 19.5 |
| Prompting | Reflexion | 21.8 | 42.7 | 55.8 | 21.9 | 58.1 | 28.8 |
| RL | PPO (with critic) | 54.4±3.1 | 80.4±2.7 | 73.8±3.0 | 51.5±2.9 | 81.4±3.1 | 68.7±5.1 |
| RL | RLOO | 69.7±2.5 | 75.5±4.6 | 73.9±5.6 | 52.1±6.7 | 80.3±3.2 | 65.7±4.0 |
| RL | GRPO | 72.8±3.6 | 77.6±5.2 | 75.8±3.5 | 56.8±3.8 | 79.3±2.8 | 66.1±3.7 |
| RL | **GiGPO w/ std** | **86.7±1.7** | **90.8±1.3** | 83.1±1.6 | 65.0±3.2 | 84.4±2.9 | 72.8±3.2 |
| RL | **GiGPO w/o std** | **86.1±4.7** | **90.2±2.3** | **83.5±1.8** | **67.4±4.5** | **86.2±2.6** | **75.2±3.8** |

Key findings:
- Closed-source LLMs offer only moderate performance; Gemini-2.5-Pro reaches 60.3% on ALFWorld and 35.9% on WebShop.
- Prompt-only agents (ReAct, Reflexion) show marginal improvements but underperform.
- RL training brings substantial gains: PPO improves ALFWorld to 54.4% (1.5B) and 80.4% (7B), but requires a separate critic network.
- **GiGPO consistently surpasses GRPO and RLOO**: GiGPO w/o std beats GRPO by **13.3%** on ALFWorld and **10.6%** on WebShop at 1.5B, and by **12.6%** and **9.1%** at 7B.
- The normalization factor $F_{\text{norm}}$ is task-dependent: $F_{\text{norm}} = 1$ yields higher success on difficult tasks (Look, Pick2, WebShop), while $F_{\text{norm}} = \text{std}$ can still be beneficial when reward variance is stable.
- GiGPO enables agents to exhibit emergent reasoning behavior (see Appendix F).

---

### 5.3 Performance on QA Tasks

| Method | NQ† | TriviaQA⋆ | PopQA⋆ | HotpotQA† | 2Wiki⋆ | MuSiQue⋆ | Bamboogle⋆ | Avg. |
|--------|-----|-----------|--------|-----------|--------|-----------|-------------|------|
| **3B** | | | | | | | | |
| R1-Instruct | 27.0 | 53.7 | 19.9 | 23.7 | 29.2 | 7.2 | 29.3 | 27.1 |
| Search-R1 | 34.1 | 54.5 | 37.8 | 32.4 | 31.9 | 10.3 | 26.4 | 32.5 |
| ZeroSearch | 41.4 | 57.4 | 44.8 | 27.4 | 30.0 | 9.8 | 11.1 | 31.7 |
| StepSearch | — | — | — | 34.5 | 32.0 | 17.4 | 34.4 | — |
| **GiGPO** | **42.0** | **59.5** | **42.4** | **36.9** | **37.0** | **12.6** | **64.1** | **42.1** |
| **7B** | | | | | | | | |
| R1-Instruct | 21.0 | 44.9 | 17.1 | 20.8 | 27.5 | 6.0 | 19.2 | 22.4 |
| Search-R1 | 39.3 | 61.0 | 39.7 | 37.0 | 40.1 | 14.6 | 36.8 | 38.5 |
| ZeroSearch | 43.6 | 61.8 | 51.5 | 34.6 | 35.2 | 18.4 | 27.8 | 39.1 |
| StepSearch | — | — | — | 38.6 | 36.6 | 22.6 | 40.0 | — |
| **GiGPO** | **46.4** | **64.7** | **46.1** | **41.6** | **43.6** | **18.9** | **68.9** | **47.2** |

Key findings:
- GiGPO achieves **42.1%** at 3B and **47.2%** at 7B, significantly outperforming Search-R1, ZeroSearch, and StepSearch.
- GiGPO is markedly more tool-efficient: the 7B model requires only ~0.9 calls on average for single-hop tasks and ~1.6 calls for multi-hop tasks, matching OTC's superior performance.
- This efficiency stems from GiGPO's ability to identify and suppress redundant queries in multi-turn decision-making.

---

### 5.4 Ablation Study

An ablation study was conducted comparing:
- **GiGPO w/o std** ($F_{\text{norm}} = 1$)
- **GiGPO w/ std** ($F_{\text{norm}} = \text{std}$)
- **GiGPO w/o $A^S$** (without step relative advantages)
- **GiGPO w/o $A^E$** (without episode relative advantages)

Results (Qwen2.5-1.5B-Instruct):

- Eliminating either component of the two-level advantage significantly degrades performance.
- Removing $A^E$ leads to a substantial drop across all tasks — the policy loses the stable, trajectory-wide signal for long-range coherence.
- Removing $A^S$ results in pronounced declines, particularly on complex tasks (Cool, Pick2, WebShop), where precise per-step credit assignment is critical.
- The performance gap between GiGPO w/ std and w/o std is comparatively minor relative to structural ablations — the combination of episode- and step-level signals is the **primary driver** of performance gains.

---

### 5.5 Dynamics of Step-Level Group

The distribution of step-level group sizes was tracked throughout training (Qwen2.5-1.5B-Instruct on ALFWorld):

- Step-level groups of size 1 ($|G^S(\tilde{\bm{s}})| = 1$) account for only **< 35%** throughout training — over 65% of states recur across trajectories.
- At iteration 10, large group sizes $|G^S(\tilde{\bm{s}})| \geq 10$ account for over 20%, reflecting behavioral redundancy (immature policies produce invalid actions or fall into loops).
- By iteration 75, extreme group sizes decrease: $10 \leq |G^S(\tilde{\bm{s}})| < 50$ drops from 16.2% to 12.1%, and $|G^S(\tilde{\bm{s}})| \geq 50$ drops from 5.6% to 3.1%.
- At iteration 140, the distribution concentrates around group sizes of **6–8**, aligning with $N = 8$ — all 8 trajectories now behave consistently, and the agent has learned a coherent, robust policy (success rate > 80%).

---

### 5.6 Computational Budget

Training time breakdown per iteration (Qwen2.5-1.5B-Instruct on ALFWorld):

| Component | Time (s) |
|-----------|----------|
| Rollouts (shared with GRPO) | — |
| Old & ref prob computation (shared) | — |
| Policy update (shared) | — |
| **Total shared components** | **362.83** |
| Anchor state grouping (hashmap lookups) | 0.01 |
| Step-relative advantage computation | 0.53 |
| **Total GiGPO additions** | **0.54** |

The additional GiGPO-specific components account for **< 0.002%** of the total per-iteration training time. GiGPO shares the same GPU memory usage and LLM rollout costs as GRPO — both are critic-free and operate with a single actor LLM. This demonstrates that GiGPO preserves the same high computational efficiency as GRPO.
