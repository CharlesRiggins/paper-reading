# Author Rebuttal to Reviewer Rb3e — Say When You Don’t Know

Source: OpenReview forum export PDF  
Forum: https://openreview.net/forum?id=oOILbzcS44  
Submission Number: 8440  
Reviewer: Rb3e (us)  
Date: 28 Jul 2026, 02:40 (modified: 28 Jul 2026, 21:05)

---

## Original text

We appreciate the reviewer's constructive comments, and we would like to respond to the reviewer's concern as follows.

**Q1: General-capability preservation.** Can the authors report MMLU, GSM8K, and at least one instruction-following benchmark (e.g., MT-Bench/AlpacaEval) before and after GRPO training? This is critical to verify that calibration post-training does not cause capability degradation. If degradation is observed, how should the "without degrading reasoning accuracy" claim be qualified?

**Answer:** Thank you for raising this important point. We agree that calibration-oriented post-training should be evaluated not only on uncertainty and calibration metrics, but also on general capabilities.

We therefore ran rebuttal-time evaluations before and after GRPO training on three general-purpose benchmarks: MMLU for broad knowledge and reasoning, GSM8K for mathematical reasoning, and IFEval for instruction following. We chose IFEval because it is a standard instruction-following benchmark with deterministic, rule-based scoring, which avoids the additional judge variance of MT-Bench or AlpacaEval during the rebuttal period. The results show a mixed picture. On MMLU and GSM8K, both model families are preserved and in fact improve slightly after GRPO (MMLU: +2.36 for Llama, +1.05 for Qwen; GSM8K: +2.04 for Llama, +2.50 for Qwen). On IFEval, however, both families degrade substantially (−19.96 for Llama, −14.42 for Qwen). The original phrase "without degrading reasoning accuracy" should therefore be qualified: our method preserves broad knowledge and mathematical reasoning, but it does not preserve instruction-following behavior under the current GRPO recipe, and general-capability preservation is model- and benchmark-dependent rather than guaranteed.

We will revise the claim accordingly and add this as a limitation. In particular, we will avoid any unconditional broad-capability preservation claim, and instead state that preserving general capabilities requires additional constraints, such as stronger KL regularization, mixed general-instruction replay, or multi-objective checkpoint selection based on both calibration and general-benchmark performance.

### Experimental Setup

| Component | Setting |
|---|---|
| Models | Llama base vs. Llama after GRPO; Qwen base vs. Qwen after GRPO |
| General knowledge / reasoning | MMLU accuracy |
| Mathematical reasoning | GSM8K exact match, corrected flexible extraction |
| Instruction following | IFEval prompt-level loose accuracy |
| GSM8K protocol note | Corrected benchmark-native evaluation with longer generation; we do not use the earlier strict-format Qwen numbers, which were format-confounded |

### Results

| Model Family | Benchmark | Metric | N | Before GRPO | After GRPO | Delta |
|---|---|---|---:|---:|---:|---:|
| Llama | MMLU | Accuracy | 14,042 | 63.17 | 65.53 | +2.36 |
| Llama | GSM8K | Exact match | 1,319 | 85.06 | 87.10 | +2.04 |
| Llama | IFEval | Prompt loose acc. | 541 | 78.93 | 58.96 | −19.96 |
| Qwen | MMLU | Accuracy | 14,042 | 68.81 | 69.86 | +1.05 |
| Qwen | GSM8K | Exact match | 1,319 | 85.37 | 87.87 | +2.50 |
| Qwen | IFEval | Prompt loose acc. | 541 | 74.12 | 59.70 | −14.42 |

All numbers are percentages. GSM8K uses the corrected flexible-extraction score.

### Interpretation

These results indicate that the calibration-oriented GRPO objective introduces a capability trade-off, and that the trade-off is qualitatively consistent across the two model families: MMLU and GSM8K are preserved (and slightly improved), while IFEval drops sharply for both. The IFEval degradation—roughly 14 to 20 points—suggests that training the model to expose uncertainty through a special token can interfere with general instruction-following behavior when no explicit instruction-following preservation objective is included.

**Q2: Comparison with end-to-end agentic tool-use training.** In a setting where an agent is trained end-to-end with tool access and task-success reward, the model implicitly learns when to call tools—which already requires some form of self-awareness. Have the authors compared explicit self-assessment training against this implicit alternative? If end-to-end agentic training achieves comparable calibration and downstream performance, what is the unique value of training explicit signals (beyond interpretability)?

**Answer:** Thank you for raising this point. We agree that end-to-end agentic tool-use training is an important related direction, but we view it as a different setting rather than a direct baseline for our main question.

In end-to-end tool-use training, the model is trained in an expanded action space: it must learn when to call a tool, how to format the tool call, how to consume tool outputs, and how to optimize final task success under a particular tool protocol. This conflates several factors: uncertainty estimation, tool-call formatting, retrieval/tool quality, multi-turn interaction dynamics, and final-answer generation.

Our goal is narrower and complementary: we train the model to expose a task-agnostic self-assessment signal inside its own response. The `<uncertain>` marker is not itself a tool call. It is a reasoning-time risk signal that can be consumed by different downstream controllers, including retrieval, regeneration, abstention, or human review. This separation lets us study whether the model can make latent failure risk observable without forcing it to learn a specific JSON/tool schema or a full multi-turn agent policy.

The unique value of explicit self-assessment is therefore not only interpretability. It also provides:

- a controller-agnostic signal that can be reused across interventions;
- a thresholdable trigger for retrieval-budget control;
- a way to evaluate exposed uncertainty directly, independent of tool success;
- lower training complexity than full tool-use policy learning;
- and a cleaner analysis of whether the model is surfacing failure risk rather than merely learning a task-specific tool policy.

We therefore do not claim that explicit self-assessment replaces end-to-end agentic training. Instead, it can serve as a lightweight and modular uncertainty interface for such systems. We will clarify this distinction and list direct comparison with fully agentic tool-use training as an important future direction.

**Q3: Incremental contribution over Self-RAG.** Self-RAG already trains retrieve/critique tokens. What specifically does the during-reasoning `<uncertain>` marker add beyond Self-RAG's critique tokens? A head-to-head comparison or a clearer articulation of the design difference would strengthen the novelty claim.

**Answer:** Thank you for raising this. We agree that Self-RAG is the closest prior work and that the distinction should be made clearer. Self-RAG trains reflection tokens mainly as part of a retrieval-and-critique workflow: the model learns whether to retrieve, whether retrieved passages are relevant/supportive, and how useful the resulting response is. These tokens are therefore tied to an external retrieval pipeline and are supervised through offline critic-generated annotations on augmented trajectories. Our `<uncertain>` marker has a different role. It is not a retrieval-action token or a passage-critique token. It is trained as a reasoning-time risk exposure signal: the model should emit it when its current reasoning state is likely to lead to failure, even before any retrieval has happened. The reward explicitly distinguishes four cases: correct/no emit > correct/emit > wrong/emit > wrong/no emit. Thus, the key pressure is to penalize silent failure more than uncertain failure. This trains the model to expose likely errors, rather than only to decide whether a retrieved passage is relevant or whether an answer is supported. The training signal is also different. Self-RAG relies on offline reflection-token annotations from a critic/stronger model and then distills these annotations with a next-token objective. In contrast, our marker is learned from task outcome through GRPO-style post-training, without requiring token-level uncertainty labels. This makes the marker an outcome-aligned self-assessment signal rather than a supervised critique/action label. Empirically, we already include Self-RAG as an adaptive-RAG baseline, but we agree that the novelty discussion should be clearer. In the revision, we will explicitly position `<uncertain>` as complementary to Self-RAG: Self-RAG learns when and how to use retrieval/critique tokens, whereas our method studies when the generator should expose internal uncertainty during reasoning, which can then be used by retrieval, abstention, regeneration, or other controllers.

**Q4: Terminology.** Could the authors add brief inline definitions (or at least citations at first occurrence) for GRPO and ECE?

**Answer:** Thank you for pointing this out. We agree and will add inline definitions and citations at first occurrence. In the revision, we will define GRPO as Group Relative Policy Optimization, a reinforcement-learning post-training method that estimates relative advantages from a group of sampled responses for the same prompt, following [1]. We will also define ECE as Expected Calibration Error, the standard weighted average gap between empirical accuracy and predicted confidence across confidence bins, following [2]. We will add these definitions where the terms first appear, rather than assuming familiarity.

[1] Shao, Zhihong, et al. "Deepseekmath: Pushing the limits of mathematical reasoning in open language models." arXiv preprint arXiv:2402.03300 (2024).

[2] Guo, Chuan, et al. "On calibration of modern neural networks." International conference on machine learning. PMLR, 2017.
