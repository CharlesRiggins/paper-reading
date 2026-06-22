## 2. Related Work

### 2.1 Cognitive paradigms in MLLMs

Recent MLLMs broadly follow two regimes: **System 1** and **System 2**.

**System 1** generalist models emphasize fast perception and direct instruction-following. Examples cited include GPT-4o-style and Qwen2.5-VL-style MLLMs, which are effective for broad visual recognition and general multimodal interaction but often struggle on multi-step derivations. These models are typically optimized for prompt-to-answer behavior rather than explicit deliberative reasoning.

**System 2** models target deliberative reasoning. Inspired by DeepSeek-R1, multimodal systems such as **Gemini-2.5-Pro** and **Kimi-VL-Thinking** use RL to internalize long chain-of-thought behavior. This improves visual math and logic performance, but it also increases inference latency and can create inefficiency when every input is routed through a long reasoning process.

The paper uses this distinction to motivate adaptive routing between cognitive regimes. Prior work has begun to study fast/slow or direct/thinking mode switching, but deciding **when** and **how** to switch remains an open problem.

### 2.2 Multimodal reinforcement learning

RL has been extended from LLM reasoning to MLLMs. The cited work includes training and self-improvement approaches for complex vision-language reasoning, visual process rewards, and multimodal RL frameworks. However, the authors argue that multimodal RL faces two bottlenecks:

1. **Efficiency bottleneck:** models overthink simple tasks, wasting tokens and sometimes introducing harmful noise;
2. **Effectiveness bottleneck:** models can produce incorrect or hallucinated reasoning traces, even when final answers are rewarded.

Prior work often addresses these bottlenecks separately. Some methods focus on gating or bi-mode optimization to reduce overthinking, while others introduce visual process reward models to verify reasoning. SAIL-RL instead couples the two objectives through **dual rewards** that jointly learn **when to reason** and **how to reason**.

### 2.3 Positioning of SAIL-RL

The paper positions SAIL-RL as a unified post-training strategy rather than a single-purpose reasoning enhancer. Compared with methods that only improve reasoning traces, SAIL-RL also learns whether reasoning should be invoked at all. Compared with methods that only route between thinking and non-thinking modes, SAIL-RL explicitly supervises logical coherence, factual grounding, and answer consistency inside the thinking mode.

The core design claim is that reasoning quality and reasoning efficiency should be optimized together. This motivates the cascading reward system introduced in Section 3, where successful reward requires the judgment decision, reasoning quality, and final answer correctness to all be aligned.
