## 3. Related Work

### 3.1 Cognitive Paradigms in MLLMs

Recent MLLMs broadly follow two regimes: **System 1** and **System 2**. System 1 generalist models [8, 9] emphasize fast perception via direct instruction tuning, but often struggle on multi-step derivations. System 2 models target deliberative reasoning; inspired by `DeepSeek-R1` [10], multimodal systems such as `Gemini-2.5-Pro` [11] and `Kimi-VL-Thinking` [12] use RL to internalize long chain-of-thought (CoT), improving visual math and logic at the cost of higher inference latency. This motivates routing between regimes [13], where *when* and *how* to switch remains an open problem.

### 3.2 Multimodal Reinforcement Learning

RL has been extended from LLM reasoning [14, 10] to MLLMs [6, 7, 4, 15, 16], but faces two bottlenecks: **efficiency** (overthinking) and **effectiveness** (incorrect reasoning). Prior work often tackles them separately, e.g., bi-mode optimization for gating and visual process rewards for verification [17, 18]. We instead couple them via dual rewards that jointly learn *when* to reason and *how* to reason.

---

### References (Related Work)

[8] Hurst et al., 2024. GPT-4o System Card.

[9] Bai et al., 2025. Qwen-VL: A Versatile Vision-Language Model.

[10] Guo et al., 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.

[11] Comanici et al., 2025. Gemini 2.5 Pro.

[12] Team et al., 2025a. Kimi-VL-Thinking.

[13] Zhang et al., 2025. Routing between reasoning regimes.

[14] OpenAI, 2024. Learning to Reason with LLMs.

[15] Wang et al., 2025a.

[16] Fan et al., 2025.

[17] Wang et al., 2025b.

[18] Luo et al., 2025.
