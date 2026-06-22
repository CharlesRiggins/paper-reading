# Guiding MLLMs in When and How to Think via Dual-Reward RL Tuning

## Abstract

The paper introduces **SAIL-RL**, a reinforcement-learning post-training framework intended to enhance reasoning capabilities of multimodal large language models (MLLMs) by teaching them **when** and **how** to think. Existing approaches are limited by outcome-only supervision, which rewards correct answers without ensuring sound reasoning, and by uniform thinking strategies, which can lead to overthinking on simple tasks and underthinking on complex ones.

SAIL-RL addresses these issues with a dual reward system. The **Thinking Reward** evaluates reasoning quality through factual grounding, logical coherence, and answer consistency, while the **Judging Reward** adaptively determines whether deep reasoning or direct answering is appropriate. Experiments on SAIL-VL2 at 4B/8B scales show improvements on reasoning and multimodal understanding benchmarks, competitive performance against closed-source models such as GPT-4o, and substantial hallucination reduction.

### Figure 1: Performance overview

Figure 1 compares **SAIL-VL2-8B-Thinking**—SAIL-VL2 post-trained with SAIL-RL—against other large vision models. On mathematical reasoning benchmarks, the figure reports SAIL-VL2-8B-Thinking scores of **38.3** on DynaMath, **63.8** on LogicVista, **65.1** on MathVerse, **49.4** on MathVision, **80.9** on MathVista, and **58.2** on WeMath. On general multimodal benchmarks, it reports **63.3/66.1** on MMMU-val/MMMU depending on the table/figure context, **90.4** on MMBench, **86.0** on MME, **93.6** on ChartQA, **87.4** on AI2D, and **61.5** on hallucination-oriented evaluation.

## 1. Introduction

Multimodal Large Language Models (MLLMs) are moving from basic visual recognition toward complex reasoning and holistic understanding. This evolution is driven by training paradigms that define how models internalize intelligence from large-scale multimodal data. While supervised fine-tuning (SFT) established the foundation for multimodal instruction following, the field has increasingly shifted toward hybrid post-training frameworks that integrate SFT with reinforcement learning (RL).

RL for MLLMs is also undergoing a paradigm shift. Rather than only aligning model outputs with human preferences, recent approaches commonly follow a “thinking before speaking” paradigm: guided by a special `\think` token, the model first generates a structured reasoning trace before producing the final answer. Long reasoning chains can serve as an internal knowledge source, allowing the model to extract salient cues that improve answer accuracy and strengthen overall capability. However, the paper identifies two fundamental challenges that remain unresolved.

### 1.1 Answers without sound reasoning

Conventional RL methods often rely on **outcome-only supervision**, where rewards are determined by final-answer correctness and the quality of reasoning is ignored. The authors argue that this creates two problems.

First, incoherent or redundant reasoning traces can prevent the model from extracting useful cues, leading to inaccurate answers and exacerbating hallucinations. The paper emphasizes the intuition that the model must “think well to answer right”; if reasoning contains factual errors, the final answer may be less robust even when it appears correct. Figure 2 illustrates that conventional MLLMs can sometimes produce correct answers despite factual errors in the reasoning chain.

Second, during optimization, a model may occasionally reach the correct answer through flawed or fabricated reasoning paths. Outcome-only rewards reinforce such spurious alignments as positive examples. The authors call this form of optimization artifact **false correctness**, because the model appears successful under answer accuracy while its reasoning process is untrustworthy.

### 1.2 Overthinking the easy, underthinking the hard

Most existing methods apply the same reasoning process to all tasks regardless of complexity. This uniformity causes **overthinking** on simple problems: for trivial queries such as object color recognition, models may generate redundant reasoning chains, increasing computational cost and introducing noisy intermediate statements. Figure 2 shows an overthinking example in which a model applies unnecessary analysis to a simple handwritten-number recognition task.

The same rigidity also causes **underthinking** on complex problems. When a problem genuinely requires multi-step inference, a static or shallow reasoning strategy may fail to allocate sufficient cognitive effort, producing incomplete analysis and inaccurate answers. The authors frame this as a mismatch between MLLM inference behavior and human cognition, where humans naturally adjust effort based on task difficulty.

### 1.3 SAIL-RL: dual reward post-training

To address these challenges, the paper proposes **SAIL-RL**, a post-training framework for MLLMs. It follows a standard two-stage paradigm—CoT-augmented SFT followed by RL tuning—but introduces a dual reward system that supervises both reasoning quality and reasoning efficiency.

The **Thinking Reward** moves beyond outcome-only supervision by directly assessing the reasoning process. It evaluates:

1. **Logical coherence**, to maintain step-by-step validity;
2. **Factual grounding**, to mitigate hallucinations by checking claims against visual/textual/world evidence;
3. **Trace-to-answer consistency**, to ensure the final answer is faithfully derived from the reasoning trace.

The **Judging Reward** enhances adaptivity by enabling the model to decide when deep reasoning is necessary. The model learns a direct-answer mode for simple tasks and a full-reasoning mode for complex ones. This allows cognitive resources to be allocated more efficiently while aligning model behavior more closely with human effort allocation.

### 1.4 Claimed empirical impact

The authors evaluate SAIL-RL by building **SAIL-VL2-Thinking** on top of SAIL-VL2. With the dual reward system, SAIL-VL2-Thinking delivers consistent gains over the base model and conventional RL-style approaches. The paper reports state-of-the-art open-source results on multiple reasoning benchmarks at the 8B scale, leading results on OpenCompass, competitive accuracy on general multimodal understanding tasks, and reduced hallucination tendencies.

Together, the authors position SAIL-RL as a principled post-training framework that strengthens both the **quality** and **adaptivity** of reasoning in MLLMs.

### Figure 2: Limitations of current MLLMs in reasoning

Figure 2 highlights two failure modes:

- **Over-Thinking:** The model applies a complex reasoning process to a simple recognition problem, producing unnecessary analysis that can lead to an incorrect answer.
- **Flawed-Thinking / lucky success:** The model reaches a correct final answer through a reasoning process that contains factual or logical errors.

The figure motivates the need for both process-level reasoning supervision and adaptive routing between direct answering and deliberate reasoning.
