# The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks

## Abstract

We study two recurring phenomena in Transformer language models: **massive activations**, in which a small number of tokens exhibit extreme outliers in a few channels, and **attention sinks**, in which certain tokens attract disproportionate attention mass regardless of semantic relevance. Prior work observes that these phenomena frequently co-occur and often involve the same tokens, but their functional roles and causal relationship remain unclear. Through systematic experiments, we show that the co-occurrence is largely an architectural artifact of modern Transformer design, and that the two phenomena serve related but distinct functions. Massive activations operate globally: they induce near-constant hidden representations that persist across layers, effectively functioning as implicit parameters of the model. Attention sinks operate locally: they modulate attention outputs across heads and bias individual heads toward short-range dependencies. We identify the pre-norm configuration as the key choice that enables the co-occurrence, and show that ablating it causes the two phenomena to decouple.

---

## 1 Introduction

Transformer-based (Vaswani et al., 2017) large language models (LLMs) have achieved unprecedented success across a wide range of tasks (Radford et al., 2018, 2019; Brown et al., 2020; OpenAI, 2022; Achiam et al., 2023; Touvron et al., 2023a, b; Grattafiori et al., 2024; Qwen et al., 2025; Yang et al., 2025), yet many aspects of their internal computations remain poorly understood. In this paper, we study two phenomena that reliably co-occur in decoder-only, pre-norm Transformers (Radford et al., 2018; Xiong et al., 2020):

- **Massive activations**, in which a handful of tokens exhibit extreme outliers in a few hidden channels (Sun et al., 2024; Yu et al., 2024a).
- **Attention sinks**, in which a small number of tokens attract disproportionate attention mass across many heads and layers (Xiao et al., 2024b).

Both phenomena have significant practical implications for quantization (Xiao et al., 2023; Yu et al., 2024a; Son et al., 2024), pruning (Ma et al., 2023; Sandoval-Segura et al., 2026; Shin et al., 2025), KV-cache management (Ge et al., 2024; Su and Yuan, 2025; Wu and Tu, 2024), and long-context inference (Huang et al., 2023; Fu et al., 2025; Xiao et al., 2024a), among others. Understanding how these two phenomena relate is therefore both theoretically and practically important.

Prior work (Sun et al., 2024; Kaul et al., 2024; Queipo-de-Llano et al., 2025) has suggested that the co-occurrence is driven by the overlap of involved tokens, but existing explanations remain largely descriptive. Here, we move beyond description to provide a mechanistic account of how and why this overlap emerges in pretrained LLMs. Our core finding is that the co-occurrence is not an inherent property of Transformers, but a predictable consequence of specific architectural and training choices.

We advance three central claims:

**First**, normalization is a key architectural component bridging the relationship between massive activations and attention sinks. Changing the normalization configuration can suppress massive activations while preserving attention sinks. Mechanistically, massive activations interact with normalization to produce near-constant hidden representations within a forward pass, effectively serving as implicit parameters that can be exploited to generate attention sinks.

**Second**, attention sinks are primarily driven by the dimensionality of the attention space and by the training context-length distribution. We further show that sinks provide a mechanism to dynamically modulate attention output across heads, biasing certain heads toward short-range dependencies that capture local sentence structure.

**Third**, each phenomenon can be independently suppressed without degrading language-modeling performance, suggesting that their overlap reflects incidental architectural interactions rather than a functional necessity.

Together, our results clarify the causal relationship between massive activations and attention sinks and show how alternative design choices can mitigate either of the phenomena. For readability, we refer to the tokens and channels that exhibit massive activations as **spike tokens** and **spike channels**, and to the tokens and attention heads affected by attention sinks as **sink tokens** and **sink heads**.
