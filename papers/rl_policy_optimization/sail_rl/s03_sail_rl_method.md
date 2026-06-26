## 3. SAIL-RL

We propose **SAIL-RL** to improve both the effectiveness and efficiency of RL for MLLMs. SAIL-RL introduces a dual-reward mechanism that guides models on *how* and *when* to think. Specifically, our design features a **Thinking Reward** to guarantee reasoning quality and a **Judging Reward** to enable adaptive reasoning. Crucially, these are unified via a **Cascading Reward System**, a multiplicative formulation that ensures both rewards work in strict synergy.

### 3.1 Thinking Reward: How to Think

As the saying goes, "sound reasoning leads to correct answers." To improve response quality, a model is required to learn *how* to think by constructing clear and coherent reasoning paths. Beyond outcome-only supervision in conventional RL-tuning, we introduce a **Thinking Reward** that comprehensively evaluates reasoning quality with LLM-based judge models. This reward is integrated into RL tuning to guide models toward producing higher-quality reasoning across multiple dimensions.

**Logical Coherence Reward.** We first introduce the Logical Coherence Reward, which evaluates whether a model can think clearly. This dimension measures the internal logical integrity of the reasoning process, ensuring arguments are both well-structured and meticulously executed. To this end, the judge model applies two sequential checks: (i) *Structural Soundness*, assessing whether the problem is properly decomposed and formulated (e.g., into valid equations or logical steps); and (ii) *Deductive Soundness*, verifying that subsequent steps are free of contradictions, calculation errors, or logical fallacies. Failure in either check yields a score of $d_1 = 0$, while success in both yields $d_1 = 1$.

**Factual Grounding Reward.** We then introduce the Factual Grounding Reward to evaluate whether the model is thinking truthfully rather than hallucinating. This reward penalizes unsupported statements by requiring each step in the reasoning process to be factually grounded. To this end, the judge model performs a hierarchical fact-check across three sources: (i) *Visual Grounding*, verifying claims against the provided image; (ii) *Textual Grounding*, checking consistency with the input query; and (iii) *World Knowledge*, consulted only when verification is not possible from the first two sources. Any contradiction at any stage yields a score of $d_2 = 0$; otherwise, $d_2 = 1$.

**Answer Consistency Reward.** We further introduce the Answer Consistency Reward to evaluate whether the model thinks faithfully. This dimension ensures that the final answer is a direct and logical conclusion derived strictly from the preceding reasoning. The judge model verifies that the reasoning trace fully justifies the answer, explicitly checking for disconnects, reliance on unstated information, or unsupported leaps in logic. Any failure results in a score of $d_3 = 0$; otherwise, the score is $d_3 = 1$. Finally, the overall thinking reward $\mathcal{R}_{\mathrm{think}}$ is computed as the average of these three dimensions:

$$\mathcal{R}_{\mathrm{think}} = \frac{1}{3} \sum_{i=1}^{3} d_i$$

### 3.2 Judging Reward: When to Think

We further guide MLLMs on *when* to think: applying detailed reasoning only for complex problems while giving direct responses to simple ones. The goal is to balance efficiency and effectiveness by adapting the reasoning to task difficulty. To this end, we introduce a **Judging Reward**, which incentivizes the model to determine whether reasoning is necessary before generating a response.

Specifically, the model is required to output a thinking decision before answering. This decision is evaluated against ground-truth complexity labels. The reward $d_{\mathrm{judge}}$ is binary: it is set to 1 if the model's decision aligns with the ground truth (i.e., choosing thinking mode for complex tasks or no-thinking mode for simple tasks); otherwise, it is 0. This simple mechanism penalizes both under-reasoning on complex tasks and over-thinking on simple tasks. By optimizing $\mathcal{R}_{\mathrm{judge}} = d_{\mathrm{judge}}$, SAIL-RL enables adaptive reasoning, ensuring that rigorous thinking is triggered only when necessary.

Figure 3 provides an overview of the SAIL-RL reward system. The system evaluates a model's response across four dimensions: Format, Answer, Thinking, and Judging. The nuanced semantic rewards for Thinking and Judging are provided by Gemini acting as a reward-judger.

### 3.3 Cascading Reward System

We formulate the core reasoning signal as a multiplicative interaction between its constituent components. As shown in Eq. (1), the joint success of the judging decision ($\mathcal{R}_{\mathrm{judge}}$), the thinking quality ($\mathcal{R}_{\mathrm{think}}$), and the final answer accuracy ($\mathcal{R}_{\mathrm{answer}}$) is integrated via a cascading product:

$$\mathcal{R}_{\mathrm{total}} = \alpha \cdot (\mathcal{R}_{\mathrm{judge}} \cdot \mathcal{R}_{\mathrm{think}} \cdot \mathcal{R}_{\mathrm{answer}}) + (1 - \alpha) \cdot \mathcal{R}_{\mathrm{format}}, \tag{1}$$

where $\alpha = 0.9$ is a balancing coefficient that prioritizes logical rigor while ensuring adherence to response format. This structure functions as a logical AND gate, ensuring that rewards are granted only when the judgment, reasoning, and answer are all correct. This design imposes a zero-tolerance penalty to prevent reward hacking. For example, if a model attempts a lucky guess on a complex problem by skipping reasoning, the term $\mathcal{R}_{\mathrm{judge}}$ becomes 0, nullifying the entire reward even if the answer happens to be correct. Similarly, if it selects thinking but generates irrelevant or empty reasoning, $\mathcal{R}_{\mathrm{think}}$ becomes 0. Note that we also incorporate standard rewards, where $\mathcal{R}_{\mathrm{answer}}$ evaluates response correctness and $\mathcal{R}_{\mathrm{format}}$ acts as a regularizer for structural compliance. This mechanism eliminates the possibility of obtaining rewards through shortcuts, forcing the model to strictly align its reasoning behaviors with the complexity of the task. The prompts of thinking reward and judging reward are provided in Appendix C.

### 3.4 Post-training Strategy

We employ a two-stage post-training strategy to instill and refine the reasoning capabilities of SAIL-RL. We first utilize SFT to establish a structured reasoning format, followed by RL to optimize the reasoning quality and efficiency using our proposed reward system.

**Stage 1: LongCoT SFT.** The first stage builds the model's foundational ability to sequentially judge a problem's complexity, generate a step-by-step reasoning process, and derive a final answer. To support our "Judge-Think-Answer" paradigm, we construct a LongCoT dataset where each sample is structured as a sequence of judgment, reasoning, and answer. Specifically, we utilize a strong teacher model to generate a judgment (`<judge>`) on whether the problem requires complex reasoning, followed by a detailed thinking process (`<think>`) leading to the ground-truth answer enclosed in a `\boxed{}` tag. This explicit structure prevents the model from bypassing reasoning on complex tasks. We fine-tune the base model using a standard next-token prediction loss over the full sequence. The objective function is defined as:

$$\mathcal{L}_{\mathrm{LongCoT-SFT}} = -\frac{1}{|D_{\mathrm{CoT}}|} \sum_{(I, J, T, A) \in D_{\mathrm{CoT}}} \log P_{\theta}(J \circ T \circ A \mid I) \tag{2}$$

where $I$ is the input, $J$ is the judgment text, $T$ the reasoning process, $A$ the final answer, and $\circ$ denotes concatenation. This training objective teaches the model to first judge the problem's nature, then produce a corresponding reasoning trace, and finally output the answer in the correct format.

**Stage 2: RL Tuning with Reward System.** While LongCoT SFT provides a strong generative template, RL further incentivizes the model to optimize *how* to think (quality) and *when* to think (efficiency). To ensure stable RL training and prevent reward hacking, we curate a dataset on verifiable tasks. We convert multiple-choice questions into free-response formats and apply difficulty-based filtering to remove trivial or unsolvable samples. This ensures the model learns from problems within an optimal difficulty range. We optimize the SFT model using the DAPO [Yu et al., 2025] algorithm with our proposed dual-reward system ($\mathcal{R}_{\mathrm{total}}$). We remove the KL divergence penalty and adjust the higher clip to encourage exploration.
