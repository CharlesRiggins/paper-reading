## A. Implementation Details

### A.1 Data Curation

**LongCoT Data Curation.** We develop a holistic data pipeline to build our high-quality LongCoT dataset. The goal is to instill a meta-cognitive skill: first judging a problem's complexity, then executing the appropriate reasoning path. To this end, every sample in our dataset is meticulously structured in a **judge-think-answer** format. We first collect diverse datasets, ranging from complex, logical problems (e.g., VisualWebInstruct [14], MathV360K [29]) to simple, perception-based questions (e.g., LLaVA-CoT [35]). All collected data is then processed through a unified pipeline to fit our judge-think-answer format. The key steps are as follows:

- **Data Cleaning:** We perform a cleaning pass to remove extraneous content, such as system prompts and conflicting hints. We then deduplicate the entire dataset based on unique image and question pairs to ensure diversity.

- **Conditional Annotation:** Each sample is annotated based on its complexity. For complex problems requiring reasoning, we use a guided-prompting strategy to generate a detailed chain-of-thought for the `<think>` section, and the `<judge>` tag is set to indicate that thinking is necessary. For simple perceptual tasks, the `<judge>` tag is set to indicate that the question can be answered directly, and the `<think>` section is intentionally populated with an empty string (`\n\n`). The final answer for all samples is standardized within the `<boxed>` tag.

- **Quality Filtering:** Finally, the annotated dataset undergoes a rigorous filtering workflow. This includes a redundancy filter, which penalizes trivial reasoning by measuring the token overlap between the thought process and the final answer, and a length-balancing step on the reasoning chains to ensure a varied representation of complexity.

This pipeline results in **400K** high-quality LongCoT samples, each designed to train the model on when and how to reason.

**RL Data Curation.** We construct a diverse dataset for the RL stage, balanced between specialized STEM problems and general-purpose QA. The STEM domain is curated from a wide array of public benchmarks in fields like Math [30, 23], Puzzles [5], Science [33, 19], OCR [3], and Counting [15]. This data undergoes a rigorous two-stage filtering pipeline to optimize for training stability. The first stage mitigates reward hacking by reformatting multiple-choice questions into an open-ended, free-response format. The second stage implements a difficulty-based curriculum filtering, using our SFT model's pass@4 score to retain only problems within an optimal difficulty range by removing the easiest (pass@4=1) and hardest (pass@4=0) instances. To ensure broad capabilities, we incorporate **20K** General QA samples from LLaVA-OneVision [16]. This subset is filtered primarily for quality and diversity, preserving a wide range of conversational and factual questions.

The RL dataset comprises **70K** samples, evenly split between **50K** challenging STEM problems and **20K** General QA instances, creating a comprehensive training environment for both specialized reasoning and general interaction.

### A.2 Training Strategy

Our model builds upon **SAIL-VL2** [39], which integrates the **AimV2** [11] visual encoder with the **Qwen3-7B** LLM. We use **VeOmni** [22] and **VeRL** [28] for the SFT and RL stages, respectively. All experiments are conducted on **64 NVIDIA A100 GPUs** using the AdamW optimizer with a cosine learning rate schedule. **Gemini-2.5-Pro** is employed as our reward judger.

**LongCoT SFT Stage.** In the first stage, we fine-tune all parameters of the model for one epoch on our 400K-sample LongCoT dataset. For this SFT stage, we set the maximum sequence length to **20K**, the global batch size to **1024**, and the learning rate to **$1 \times 10^{-6}$**.

**RL Stage.** Subsequently, we optimize the SFT model for three epochs on our 70K-sample mixed RL dataset using the **DAPO** [40] algorithm, guided by our proposed SAIL-RL reward system. We set the maximum sequence length to **20K**, consisting of **16K** for the input and **4K** for the output. The policy learning rate is set to **$1 \times 10^{-6}$** with a global PPO batch size of **256**. For each sample, we rollout **5** times to estimate the advantage. To encourage exploration and stabilize training, we remove the standard KL divergence and dynamically adjust the clipping value $\varepsilon$ within the range of **[0.20, 0.28]**.
