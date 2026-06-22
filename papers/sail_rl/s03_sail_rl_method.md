## 3. SAIL-RL

SAIL-RL is proposed to improve both the **effectiveness** and **efficiency** of RL for MLLMs. The framework introduces a dual-reward mechanism that guides models on **how** and **when** to think:

- a **Thinking Reward** to guarantee reasoning quality;
- a **Judging Reward** to enable adaptive reasoning;
- a **Cascading Reward System** that combines the rewards multiplicatively so that they act in strict synergy.

The method keeps the common two-stage post-training recipe—LongCoT SFT followed by RL tuning—but changes the reward structure from answer-only optimization to process-aware and routing-aware optimization.

### 3.1 Thinking Reward: How to Think

The paper begins from the premise that “sound reasoning leads to correct answers.” Instead of rewarding only the final answer, SAIL-RL introduces a **Thinking Reward** that evaluates the intermediate reasoning process with LLM-based judge models. The reward is designed to guide the model toward clear, coherent, grounded, and faithful reasoning traces.

#### 3.1.1 Logical Coherence Reward

The **Logical Coherence Reward** evaluates whether the model can think clearly. It measures the internal logical integrity of the reasoning process and checks whether arguments are well structured and correctly executed.

The judge model applies two sequential checks:

1. **Structural Soundness:** whether the problem is properly decomposed and formulated, e.g. into valid equations or logical steps;
2. **Deductive Soundness:** whether subsequent steps are free of contradictions, calculation errors, or logical fallacies.

Failure in either check yields $d_1=0$, while passing both checks yields $d_1=1$.

#### 3.1.2 Factual Grounding Reward

The **Factual Grounding Reward** evaluates whether the model is thinking truthfully rather than hallucinating. It penalizes unsupported statements by requiring each reasoning step to be factually grounded.

The judge performs hierarchical fact-checking across three evidence sources:

1. **Visual Grounding:** verify claims against the provided image;
2. **Textual Grounding:** check consistency with the input query;
3. **World Knowledge:** consult general knowledge only when verification is not possible from the image or text.

Any contradiction at any stage yields $d_2=0$; otherwise $d_2=1$.

#### 3.1.3 Answer Consistency Reward

The **Answer Consistency Reward** evaluates whether the model thinks faithfully. It ensures that the final answer is a direct and logical conclusion derived strictly from the preceding reasoning.

The judge verifies that the reasoning trace fully justifies the answer and explicitly checks for disconnects, reliance on unstated information, or unsupported leaps in logic. Any failure yields $d_3=0$; otherwise $d_3=1$.

The overall thinking reward is the average of the three binary dimensions:

$$
R_{\text{think}} = \frac{1}{3}\sum_{i=1}^{3} d_i.
$$

### 3.2 Judging Reward: When to Think

SAIL-RL also guides MLLMs on **when** to think. The goal is to apply detailed reasoning only for complex problems while giving direct responses to simple ones, balancing efficiency and effectiveness.

The model is required to output a thinking decision before answering. This decision is evaluated against ground-truth complexity labels. The reward $d_{\text{judge}}$ is binary:

- $d_{\text{judge}}=1$ if the model’s decision aligns with the ground truth, i.e. choosing thinking mode for complex tasks or no-thinking mode for simple tasks;
- $d_{\text{judge}}=0$ otherwise.

Thus:

$$
R_{\text{judge}} = d_{\text{judge}}.
$$

This mechanism penalizes both **under-reasoning** on complex tasks and **over-thinking** on simple tasks. By optimizing this reward, SAIL-RL encourages adaptive reasoning so that rigorous thinking is triggered only when necessary.

### 3.3 Cascading Reward System

The core reasoning signal is formulated as a multiplicative interaction among the judging decision, thinking quality, and final answer accuracy. The paper’s total reward is:

$$
R_{\text{total}} = \alpha \cdot \left(R_{\text{judge}} \cdot R_{\text{think}} \cdot R_{\text{answer}}\right) + (1-\alpha) \cdot R_{\text{format}},
\tag{1}
$$

where $\alpha=0.9$ is a balancing coefficient that prioritizes logical rigor while still enforcing response-format compliance.

This cascading product acts like a logical **AND gate**. Rewards are granted only when the judgment, reasoning, and answer are all correct. The design imposes a zero-tolerance penalty to prevent reward hacking:

- If a model guesses correctly on a complex problem but skips reasoning, $R_{\text{judge}}=0$ nullifies the reward.
- If a model selects thinking but generates irrelevant or empty reasoning, $R_{\text{think}}=0$ nullifies the reward.
- If the reasoning is good but the answer is wrong, $R_{\text{answer}}=0$ blocks the main reward.

The system also incorporates standard rewards. $R_{\text{answer}}$ evaluates response correctness and $R_{\text{format}}$ regularizes structural compliance. This eliminates shortcuts and forces the model to align reasoning behavior with task complexity.

#### Figure 3: Reward-system overview

Figure 3 illustrates a response evaluated across four dimensions: **Format**, **Answer**, **Thinking**, and **Judging**. The nuanced semantic rewards for Thinking and Judging are provided by Gemini acting as a reward-judger.

The figure’s example concerns a graph question asking whether the period of a red curve is larger, equal, or smaller than that of a blue curve. The model correctly identifies that the task requires thinking and produces a parseable response, but its reasoning contains a factual error about the relationship between period and frequency. As a result, the Thinking Reward is $(1+1+0)/3=2/3$, the Judging Reward is 1, the Format Reward is 1, and the Answer Reward is 0 because the final answer is incorrect.

### 3.4 Post-training Strategy

SAIL-RL uses a two-stage post-training strategy to instill and refine reasoning behavior.

### 3.4.1 Stage 1: LongCoT SFT

The first stage builds the model’s ability to sequentially:

1. judge a problem’s complexity;
2. generate a step-by-step reasoning process when needed;
3. derive a final answer.

To support this **Judge-Think-Answer** paradigm, the authors construct a LongCoT dataset where each sample is structured as a sequence of judgment, reasoning, and answer. A strong teacher model generates a judgment (`<judge>`) about whether the problem requires complex reasoning, followed by a detailed thinking process (`<think>`) leading to the ground-truth answer enclosed in a `\boxed{}` tag. This explicit structure is intended to prevent the model from bypassing reasoning on complex tasks.

The model is fine-tuned with standard next-token prediction over the full sequence. The objective is:

$$
\mathcal{L}_{\text{LongCoT-SFT}} = -\frac{1}{|\mathcal{D}_{\text{CoT}}|}
\sum_{(I,J,T,A)\in \mathcal{D}_{\text{CoT}}}
\log P_\theta(J \circ T \circ A \mid I),
\tag{2}
$$

where:

- $I$ is the input;
- $J$ is the judgment text;
- $T$ is the reasoning process;
- $A$ is the final answer;
- $\circ$ denotes concatenation.

This objective teaches the model to first judge the problem’s nature, then produce a corresponding reasoning trace, and finally output the answer in the correct format.

### 3.4.2 Stage 2: RL tuning with the reward system

LongCoT SFT provides a strong generative template, but RL further incentivizes the model to optimize both **how to think** and **when to think**. To ensure stable RL training and reduce reward hacking, the authors curate a dataset of verifiable tasks.

The RL-stage data construction includes:

- converting multiple-choice questions into free-response formats;
- applying difficulty-based filtering to remove trivial or unsolvable samples;
- training on problems in an optimal difficulty range.

The authors optimize the SFT model using the **DAPO** algorithm with the proposed dual-reward system $R_{\text{total}}$. They remove the KL divergence penalty and adjust to a higher clipping value to encourage exploration.
