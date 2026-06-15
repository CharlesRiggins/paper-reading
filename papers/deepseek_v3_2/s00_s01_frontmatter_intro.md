# DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

arXiv: 2512.02556 | DeepSeek-AI (2025)

## Abstract

We introduce **DeepSeek-V3.2**, a model that harmonizes high computational efficiency with superior reasoning and agent performance. The key technical breakthroughs are:

1. **DeepSeek Sparse Attention (DSA)**: An efficient attention mechanism that substantially reduces computational complexity while preserving model performance in long-context scenarios.
2. **Scalable Reinforcement Learning Framework**: By implementing a robust RL protocol and scaling post-training compute, DeepSeek-V3.2 performs comparably to GPT-5. The high-compute variant, **DeepSeek-V3.2-Speciale**, surpasses GPT-5 and exhibits reasoning proficiency on par with Gemini-3.0-Pro, achieving **gold-medal performance** in both the 2025 IMO and IOI.
3. **Large-Scale Agentic Task Synthesis Pipeline**: A novel synthesis pipeline that systematically generates training data at scale, facilitating scalable agentic post-training with substantial improvements in generalization and instruction-following robustness.

## 1. Introduction

The release of reasoning models marked a pivotal moment in LLM evolution, catalyzing a substantial leap in overall performance across verifiable fields. However, a distinct divergence has emerged: while the open-source community continues to make strides, the performance trajectory of closed-source proprietary models (GPT-5, Claude-4.5-Sonnet, Gemini-3.0-Pro) has accelerated at a significantly steeper rate. Rather than converging, the **performance gap between closed-source and open-source models appears to be widening**.

The authors identify three critical deficiencies limiting open-source models in complex tasks:

1. **Architecture**: Predominant reliance on vanilla attention mechanisms severely constrains efficiency for long sequences, posing a substantial obstacle to both scalable deployment and effective post-training.
2. **Resource allocation**: Insufficient computational investment during the post-training phase, limiting performance on hard tasks.
3. **AI agents**: Marked lag in generalization and instruction-following capabilities compared to proprietary counterparts.

To address these limitations, DeepSeek-V3.2 introduces:

- **DSA**: A highly efficient attention mechanism that substantially reduces computational complexity while preserving long-context performance.
- **Scalable RL protocol**: A stable framework that allocates a post-training computational budget **exceeding 10% of the pre-training cost**.
- **Agentic task synthesis pipeline**: Cold-start via DeepSeek-V3 methodology to unify reasoning and tool-use, then large-scale synthesis generating **over 1,800 distinct environments and 85,000 complex prompts**.

DeepSeek-V3.2 achieves similar performance with Kimi-K2-Thinking and GPT-5 across multiple reasoning benchmarks, while significantly advancing agentic capabilities on long-tail agent tasks. **DeepSeek-V3.2-Speciale** achieves **gold-medal performance in IOI 2025, ICPC World Final 2025, IMO 2025, and CMO 2025**.
