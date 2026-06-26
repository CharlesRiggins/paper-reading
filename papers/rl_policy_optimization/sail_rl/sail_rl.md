# NeurIPS 2026 Review — SAIL-RL

## Summary

This paper proposes SAIL-RL, a dual-reward RL post-training framework for multimodal LLMs that teaches models when and how to think. The Thinking Reward evaluates reasoning quality along three binary dimensions — logical coherence, factual grounding (cross-referencing the image, question, and world knowledge), and answer consistency — averaged into $\mathcal{R}_{\mathrm{think}}$. The Judging Reward is a binary signal that checks whether the model's decision to think or not matches the task's true complexity. These are fused via a multiplicative cascading reward $\mathcal{R}_{\mathrm{total}} = \alpha(\mathcal{R}_{\mathrm{judge}} \cdot \mathcal{R}_{\mathrm{think}} \cdot \mathcal{R}_{\mathrm{answer}}) + (1-\alpha)\mathcal{R}_{\mathrm{format}}$ that acts as a logical AND gate, granting reward only when judgment, reasoning, and answer are all correct. Training proceeds in two stages: LongCoT SFT to establish a judge-think-answer format, then DAPO-based RL with the dual-reward system. Built on SAIL-VL2, the resulting SAIL-VL2-8B-Thinking achieves 59.3 average on multimodal reasoning benchmarks (surpassing GPT-4o at 54.8) and 80.8 on general multimodal understanding, establishing SOTA among open-source 8B models.

---

## Strengths And Weaknesses

The paper is well-motivated and the core idea is clean. The cascading multiplicative reward (Eq. 1) is the highlight: treating judge/think/answer as a logical AND chain elegantly prevents reward hacking where a model compensates for flawed reasoning with a lucky answer. The ablation in Figure 6a backs this up — multiplicative beats additive by +2.9 average with a clear mechanistic rationale. The discrete-vs-continuous reward finding (Figure 6b, +3.2) is a genuinely useful practical contribution, tying LLM-as-judge calibration noise to RL advantage estimation instability. Experiments are thorough: main results (Tables 1-2), single-dimension baselines (Table 3), per-reward ablations (Tables 4-6), mechanism analysis (Figure 6), and robustness across reward models (Table 7) and architectures (Table 8). Hitting 59.3 average on reasoning at 8B and beating GPT-4o is a strong headline result, and fully disclosing reward prompts in Appendix C helps reproducibility.

That said, several issues hold me back from a higher score. First and most concerning: the Judging Reward depends on "ground-truth complexity labels," but the paper never says where these come from. Reading Appendix C.1 carefully, the reward model (Gemini) itself independently decides whether a question "requires reasoning" — meaning the ground-truth label and the reward signal are produced by the same model. This is a potential circularity that the paper does not acknowledge. If the policy is being trained to match Gemini's subjective difficulty assessment, the "meta-cognitive routing" claim is weaker than presented.

Second, the paper's central motivation is eliminating "false correctness" (correct answers via flawed reasoning), yet this phenomenon is never directly measured. Table 4 reports aggregate accuracy gains, but aggregate gains cannot distinguish whether improvements come from better reasoning or simply more correct answers. The core problem the paper claims to solve is empirically unverified.

Third, Table 6 shows a 99.8% thinking trigger rate on MathVision. The model essentially always thinks on hard benchmarks, which raises the question of whether the Judging Reward has learned genuine adaptive routing or collapsed to a trivial "always think on hard data" heuristic. Without a trigger-rate breakdown on a genuinely mixed-difficulty sample, the adaptivity claim is overstated.

Fourth, $\alpha = 0.9$ is introduced without any sensitivity analysis. Fifth, all three Thinking Reward dimensions rest on a single LLM judge; while Table 7 shows robustness across judges, the absolute performance ceiling imposed by judge capability goes undiscussed. Finally, the gains are demonstrated on the authors' own SAIL-VL2 base, and the cross-architecture transfer in Table 8 uses a different reward setup (answer-only baseline comparison) that is not directly apples-to-apples with the main SAIL-RL configuration.

---

## Quality

3

---

## Clarity

3

---

## Significance

3

---

## Originality

3

---

## Questions

1. Where do the ground-truth complexity labels for the Judging Reward come from? Appendix C.1 suggests the reward model (Gemini) independently determines `requires_reasoning`, which means the ground truth and the reward signal originate from the same source. If confirmed, the policy is being optimized to mimic Gemini's subjective difficulty assessment rather than learning true meta-cognitive routing. This would undermine the validity of the "when to think" component. Please clarify the label source and, if labels are indeed judge-generated, discuss how you prevent this circularity. A convincing answer here could raise my score; confirmation without mitigation would lower it.

2. The paper motivates itself by targeting "false correctness" (correct answer via flawed reasoning), but this is never directly measured. Please report, for both the answer-only baseline and SAIL-RL, the fraction of correct answers where the reasoning is also fully valid (i.e., $d_1=d_2=d_3=1$). Without this, the paper's central claim is empirically unsupported. If SAIL-RL substantially increases the "correct-and-sound" rate relative to "correct-by-luck," my confidence in the contribution increases significantly.

3. Table 6 shows a 99.8% trigger rate on MathVision — the model almost always thinks on hard benchmarks. This looks like the Judging Reward may have collapsed to a trivial "always think on hard data" heuristic rather than learning fine-grained adaptive routing. What is the trigger rate on a genuinely mixed-difficulty sample (not a benchmark skewed toward hard problems)? If the model cannot discriminate within a heterogeneous set, the "adaptive reasoning" claim is overstated and my score would decrease.

4. $\alpha = 0.9$ is used throughout with no justification or sensitivity analysis. Please report performance across at least $\alpha \in \{0.7, 0.8, 0.9, 0.95, 1.0\}$. If there is a performance cliff, the method's robustness is weaker than presented.

5. Table 8 claims cross-architecture generalization, but the comparison is against an "answer-only" baseline rather than the full SAIL-RL configuration used in the main results. Can you run the complete dual-reward SAIL-RL on at least one non-SAIL-VL2 backbone (e.g., Qwen3-VL-8B) and report the full reward breakdown? This would verify that the gains are not an artifact of the SAIL-VL2 base.

---

## Limitations

The authors acknowledge two limitations (noisy out-of-domain rewards; memory overhead from reasoning traces), which is good. However, they do not discuss the dependence on LLM-as-judge quality and the potential circularity in the Judging Reward's ground-truth labels (see Questions). These are central methodological caveats, not minor ones, and should be addressed explicitly.

---

## Rating

5

---

## Confidence

4

---

## Ethical Concerns

- [x] NO or VERY MINOR ethics concerns only

---

## Paper Formatting Concerns

No major formatting or anonymity concerns.

---

## Code Of Conduct Acknowledgement

- [x] Yes

---

## Responsible Reviewing Acknowledgement

- [x] Yes
