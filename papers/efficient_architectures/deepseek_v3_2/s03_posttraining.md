## 3. Post-Training

After continued pre-training, post-training creates the final DeepSeek-V3.2, also employing sparse attention in the same way as the sparse continued pre-training stage. The pipeline includes **specialist distillation** and **mixed RL training**.

### Specialist Distillation

For each task, a specialized model is developed exclusively for that domain, fine-tuned from the same pre-trained DeepSeek-V3.2 base checkpoint. The framework encompasses six specialized domains: **mathematics**, **programming**, **general logical reasoning**, **general agentic tasks**, **agentic coding**, and **agentic search** — all supporting both thinking and non-thinking modes. Each specialist is trained with large-scale RL computing.

Different models generate training data for long chain-of-thought reasoning (thinking mode) and direct response generation (non-thinking mode). Models trained on distilled data achieve performance levels only marginally below domain-specific specialists, with the gap effectively eliminated through subsequent RL training.

### Mixed RL Training

DeepSeek-V3.2 adopts **Group Relative Policy Optimization (GRPO)** as the RL training algorithm. Reasoning, agent, and human alignment training are merged into **one RL stage**, balancing performance across diverse domains while circumventing catastrophic forgetting issues. Reward signals:
- **Reasoning and agent tasks**: Rule-based outcome reward, length penalty, and language consistency reward.
- **General tasks**: Generative reward model with per-prompt evaluation rubrics.

#### DeepSeek-V3.2 and DeepSeek-V3.2-Speciale

DeepSeek-V3.2 integrates reasoning, agent, and human alignment data distilled from specialists, undergoing thousands of steps of continued RL training. **DeepSeek-V3.2-Speciale** is an experimental variant trained exclusively on reasoning data with a **reduced length penalty** during RL, additionally incorporating the dataset and reward method from DeepSeekMath-V2 to enhance mathematical proof capabilities.

### 3.1 Scaling GRPO

GRPO optimizes the policy model $\pi_{\theta}$ by maximizing on a group of responses $\{o_1, \cdots, o_G\}$ sampled from the old policy $\pi_{\mathrm{old}}$:

$$\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\mathrm{old}}(\cdot|q)} \Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}\left(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon\right)\hat{A}_{i,t}\right) - \beta\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}(o_{i,t}) \ \middle\| \ \pi_{\mathrm{ref}}(o_{i,t})\right)\Bigg]$$

where $r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t}|q, o_{i,<t})}{\pi_{\mathrm{old}}(o_{i,t}|q, o_{i,<t})}$ is the importance sampling ratio. $\hat{A}_{i,t} = R_i - \text{mean}(\boldsymbol{R})$ is the advantage estimated by normalizing the outcome reward within each group.

#### Unbiased KL Estimate

The K3 estimator is corrected to obtain an **unbiased KL estimate** using the importance-sampling ratio:

$$\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}(o_{i,t}) \ \middle\| \ \pi_{\mathrm{ref}}(o_{i,t})\right) = \frac{\pi_{\theta}(o_{i,t}|q, o_{i,<t})}{\pi_{\mathrm{old}}(o_{i,t}|q, o_{i,<t})}\left(\frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta}(o_{i,t}|q, o_{i,<t})} - \log\frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta}(o_{i,t}|q, o_{i,<t})} - 1\right)$$

This eliminates systematic estimation errors. When $\pi_{\theta} \ll \pi_{\mathrm{ref}}$, the original K3 estimator assigns disproportionately large, unbounded weights, resulting in noisy gradient updates that degrade sample quality. In practice, different domains benefit from varying KL regularization strengths — for mathematics, applying a relatively weak KL penalty or even omitting it can yield improved performance.

#### Off-Policy Sequence Masking

RL systems typically generate a large batch of rollout data split into mini-batches for multiple gradient updates, inherently introducing off-policy behavior. Training-inference inconsistency further exacerbates the degree of off-policyness.

To stabilize training, **negative sequences that introduce significant policy divergence** are masked, as measured by the KL divergence between $\pi_{\mathrm{old}}$ and $\pi_{\theta}$. A binary mask $M$ is introduced:

$$M_{i,t} = \begin{cases} 0 & \hat{A}_{i,t} < 0, \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\log\frac{\pi_{\mathrm{old}}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta}(o_{i,t}|q, o_{i,<t})} > \delta \\[4.3pt] 1 & \text{otherwise} \end{cases}$$

where $\delta$ controls the policy divergence threshold. Only sequences with **negative advantages** are masked. The intuition: the model benefits most from learning from its own mistakes, but highly off-policy negative samples can mislead or destabilize optimization.

#### Keep Routing

MoE models face discrepancies between inference and training frameworks that can result in **inconsistent expert routing** for identical inputs, causing abrupt shifts in the active parameter subspace and destabilizing optimization. To mitigate this, the expert routing paths used during sampling in the inference framework are preserved and enforced during training. This has been adopted since DeepSeek-V3-0324 and was found **crucial for RL training stability** of MoE models.

#### Keep Sampling Mask

Top-p and top-k sampling enhance generation quality but introduce a mismatch between the action spaces of $\pi_{\mathrm{old}}$ and $\pi_{\theta}$, violating importance sampling principles. The truncation masks from sampling $\pi_{\mathrm{old}}$ are preserved and applied to $\pi_{\theta}$ during training, ensuring identical action subspaces. Combining top-p sampling with Keep Sampling Mask effectively **preserves language consistency** during RL training.

### 3.2 Thinking in Tool-Use

#### 3.2.1 Thinking Context Management

DeepSeek-R1 demonstrated that incorporating a thinking process significantly enhances complex problem-solving ability. However, replicating DeepSeek-R1's strategy of discarding reasoning content upon the second round of messages results in **significant token inefficiency**, forcing the model to redundantly re-reason through the entire problem.

A tailored context management strategy:

| Event | Reasoning Content |
|-------|------------------|
| **New user message** | Historical reasoning content is discarded |
| **Tool-related messages** only (e.g., tool outputs) | Reasoning content is **retained throughout** the interaction |
| When reasoning traces are removed | History of tool calls and results remains preserved |

Note: Agent frameworks that simulate tool interactions via user messages (e.g., Roo Code, Terminus) may not fully benefit. Non-thinking models are recommended for such architectures.

#### 3.2.2 Cold-Start

Given the availability of reasoning data (non-agentic) and non-reasoning agentic data, a straightforward integration strategy uses **carefully designed prompting**. The model is assumed to have sufficient ability to follow explicit instructions, enabling seamless incorporation of tool execution within reasoning.

Distinct task prompts are associated with different system prompts:
- **Reasoning data**: System prompt explicitly asks the model to reason before the final answer, using `<think></think>` tags.
- **Non-reasoning agentic data**: System prompt contains tool-call guidance.
- **Reasoning + agentic combined**: System prompt instructs the model to incorporate multiple tool calls within its reasoning process.

Although reasoning-in-tool-use patterns may initially lack robustness, the model occasionally generates desired trajectories, providing a basis for subsequent RL stages.

#### 3.2.3 Large-Scale Agentic Tasks

A diverse set of RL tasks is crucial for enhancing model robustness. The agent tasks used:

| Task Type | # Tasks | Environment | Prompt Source |
|-----------|---------|-------------|---------------|
| Code Agent | 24,667 | Real | Extracted |
| Search Agent | 50,275 | Real | Synthesized |
| General Agent | 4,417 | Synthesized | Synthesized |
| Code Interpreter | 5,908 | Real | Extracted |

##### Search Agent

A **multi-agent pipeline** based on DeepSeek-V3.2:
1. Sample informative **long-tail entities** across diverse domains from large-scale web corpora.
2. A question-construction agent explores each entity using search tools with configurable depth and breadth, consolidating information into QA pairs.
3. Multiple answer-generation agents with heterogeneous configurations produce diverse candidate responses.
4. A verification agent with search capabilities validates all answers through multiple passes, retaining only samples where the ground-truth is correct and all candidates are verifiably incorrect.
5. Data spans multiple languages, domains, and difficulty levels. Augmented with filtered instances from existing helpful RL datasets.

A generative reward model scores responses based on multi-dimensional rubrics, enabling optimization for both **factual reliability and practical helpfulness**.

##### Code Agent

Mined **millions of issue-PR pairs from GitHub**, rigorously filtered using heuristic rules and LLM-based judgments. Must contain: a reasonable issue description, a correlated gold patch, and a test patch for validation. An automated environment-setup agent handles package installation, dependency resolution, and test execution. An environment is deemed successfully built only when applying the gold patch results in a non-zero count of **false-to-positive (F2P) test cases** and a zero count of **pass-to-fail (P2F) test cases**. Tens of thousands of reproducible issue resolution environments were built across Python, Java, JavaScript, TypeScript, C, C++, Go, and PHP.

##### Code Interpreter Agent

Uses Jupyter Notebook as a code interpreter to address complex reasoning tasks spanning mathematics, logic, and data science, each requiring code execution to arrive at a solution.

##### General Agent

An automatic **environment-synthesis agent** synthesizes 1,827 task-oriented environments (4,417 tasks total) — hard to solve but easy to verify. The workflow:

1. Given a task category and a sandbox with bash + search tools, the agent generates/retrieves relevant data from the Internet and stores it in a sandbox database.
2. The agent synthesizes a set of **task-specific tools**, each implemented as a function.
3. The agent proposes a simple task with solution and verification functions. The solution function is restricted to invoking tool functions or logical computations (cannot access the database directly). The agent iteratively increases difficulty, updating solutions and verification functions, augmenting the toolset as needed.

The result: thousands of ⟨environment, tools, task, verifier⟩ tuples. RL is performed, retaining only instances with non-zero pass@100. Example: a trip-planning task with constraints on cities, hotels, restaurants, and attractions, where the combinatorial space is large but verification is straightforward.
