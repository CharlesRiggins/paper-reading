## Abstract

Inference scaling empowers LLMs with unprecedented reasoning ability, with reinforcement learning as the core technique to elicit complex reasoning. However, key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report), and the community still struggles to reproduce their RL training results.

The paper proposes **Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)** and fully open-sources a state-of-the-art large-scale RL system that reaches **50 points on AIME 2024** using the `Qwen2.5-32B` base model. Unlike previous works that withhold training details, DAPO identifies four key techniques that make large-scale LLM RL succeed: **Clip-Higher**, **Dynamic Sampling**, **Token-Level Policy Gradient Loss**, and **Overlong Reward Shaping**. The training code is built on `verl`, and the authors also release a carefully curated and processed dataset so that the community can reproduce and extend large-scale LLM RL.

Figure 1 reports AIME 2024 scores of DAPO on the `Qwen2.5-32B` base model: DAPO outperforms the previous SoTA `DeepSeek-R1-Zero-Qwen-32B` using **50%** of the training steps, with the x-axis denoting gradient update steps.

## 1. Introduction

Test-time scaling such as OpenAI's o1 [1] and DeepSeek's R1 [2] brings a profound paradigm shift to Large Language Models (LLMs) [3–7]. Test-time scaling enables longer **Chain-of-Thought (CoT)** thinking and induces sophisticated reasoning behaviors, making models superior on competitive mathematics and coding tasks such as `AIME` and `Codeforces`.

The central technique driving this revolution is large-scale **Reinforcement Learning (RL)**, which elicits complex reasoning behaviors such as self-verification and iterative refinement. However, the actual algorithms and key recipes for scalable RL training remain largely hidden from technical reports of existing reasoning models [1, 2, 8–11]. This paper aims to reveal significant obstacles in large-scale RL training and open-source a scalable RL system with fully open-sourced algorithm, training code, and dataset that provides democratized solutions with industry-level RL results.

The authors experiment with `Qwen2.5-32B` [12] as the pretrained model for RL. In their initial `GRPO` run, they obtain only **30 points on AIME**, far below DeepSeek's RL result of **47 points**. Their analysis shows that naive `GRPO` suffers from key issues such as **entropy collapse**, reward noise, and training instability. The broader community has encountered similar challenges in reproducing DeepSeek's results [13–19], suggesting that critical training details may have been omitted from the R1 paper and are required to build an industry-level, large-scale, reproducible RL system.

To close this gap, the paper releases an open-source state-of-the-art system for large-scale LLM RL. The system achieves **50 points on AIME 2024** using `Qwen2.5-32B`, outperforming `DeepSeek-R1-Zero-Qwen-32B` [2], which reports **47 points**, while using only **50%** of the training steps. DAPO is introduced as the algorithmic core, with details in Section 3.

The paper highlights four techniques that make RL effective in the long-CoT scenario:

1. **Clip-Higher** promotes system diversity and avoids entropy collapse.
2. **Dynamic Sampling** improves training efficiency and stability.
3. **Token-Level Policy Gradient Loss** is critical when optimizing long CoT responses.
4. **Overlong Reward Shaping** reduces reward noise and stabilizes training.

The implementation is based on `verl` [20]. By releasing the full RL system, including training code and data, the authors aim to expose valuable practical insights into large-scale LLM RL and benefit the broader research community.
