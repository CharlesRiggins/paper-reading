# NeurIPS 2026 Review Form (Draft)

Paper: *Say When You Don't Know: Training LLMs to Expose Self-Assessment*

---

## Summary

This paper tackles LLM self-assessment by reframing it as an "exposure" problem: instead of estimating how reliable a response is from the outside after generation, the model should be trained to say when it's unsure, right in its own response. The authors study two complementary signals under a unified GRPO post-training framework: (1) a verbalized confidence score emitted after the final answer, trained with a Brier-style reward, and (2) a `<uncertain>` marker emitted mid-reasoning whenever the model hits a risky step, trained with an asymmetric reward that punishes silent failure hardest. Both are evaluated on five factual QA benchmarks against baselines like P(True), temperature scaling, SFT, Self-RAG, FLARE, and DRAGIN. The results show both signals cut overconfident errors sharply (ECE 0.383→0.049, strict epistemic errors 88.6%→3.9%) and work well as adaptive RAG triggers (best: 41.6% EM at 48.1% trigger rate). A mechanism analysis using KL divergence, CKA, and activation patching suggests the two methods leave different internal traces: verbalized confidence sharpens an existing confidence structure while preserving representation geometry, whereas the `<uncertain>` marker reshapes late-layer representations. An appendix theorem shows GRPO can only reweight existing trajectories, not create new ones.

---

## Strengths And Weaknesses

> **Strengths.**
>
> - **Thoughtful reward design.** The `<uncertain>` reward asymmetry—penalizing silent failure hardest—effectively converts hidden failures into explicit control signals, and the ablation (Table 10) confirms the design matters.
> - **Support-preserving theorem.** The trajectory-reweighting analysis clarifies what GRPO can and cannot do: reweight existing trajectories, not create new ones—a timely contribution to the RL capability debate.
> - **Clean baseline results.** Table 3 shows GRPO is the only method simultaneously winning on ECE, overconfident-wrong rate, and EM, whereas P(True), temperature scaling, and SFT each trade off different metrics.
>
> **Weaknesses.**
>
> **(W1) Limited novelty.** Verbalized confidence has clear precedents (Lin et al. 2022; Kadavath et al. 2022; Yona et al. 2024), and the `<uncertain>` marker is conceptually a learned control token already explored by Self-RAG and backtracking-token work. The incremental contribution beyond these prior works is not clearly articulated.
>
> **(W2) Undefined terminology.** GRPO and ECE are used throughout without inline definition or citation at first occurrence, making the paper hard to follow for readers unfamiliar with RL post-training or calibration literature.
>
> **(W3) No general-capability evaluation.** All evaluation is on five factual QA datasets. The claim that calibration "does not degrade reasoning accuracy" is only verified on the narrow trained task—no MMLU/GSM8K/HumanEval/MT-Bench results to rule out capability degradation from GRPO post-training, despite parameter drift in broadly shared layers.
>
> **(W5) Narrow scope.** Single model family for main results (Qwen relegated to appendix), single during-reasoning marker, no scaling analysis, and no error bars or significance tests (checklist item 7: "No"). Results are limited to factual QA.

---

## Quality

2 (not good)

---

## Clarity

2 (not good)

---

## Significance

2 (not good)

---

## Originality

2 (not good)

---

## Questions

1. **General-capability preservation.** Can the authors report MMLU, GSM8K, and at least one instruction-following benchmark (e.g., MT-Bench/AlpacaEval) before and after GRPO training? This is critical to verify that calibration post-training does not cause capability degradation. If degradation is observed, how should the "without degrading reasoning accuracy" claim be qualified?

2. **Comparison with end-to-end agentic tool-use training.** In a setting where an agent is trained end-to-end with tool access and task-success reward, the model implicitly learns when to call tools—which already requires some form of self-awareness. Have the authors compared explicit self-assessment training against this implicit alternative? If end-to-end agentic training achieves comparable calibration and downstream performance, what is the unique value of training explicit signals (beyond interpretability)?

3. **Incremental contribution over Self-RAG.** Self-RAG already trains retrieve/critique tokens. What specifically does the during-reasoning `<uncertain>` marker add beyond Self-RAG's critique tokens? A head-to-head comparison or a clearer articulation of the design difference would strengthen the novelty claim.

4. **Terminology.** Could the authors add brief inline definitions (or at least citations at first occurrence) for GRPO and ECE?

---

## Limitations

Partially addressed. The authors are upfront about the factual-QA-only scope and the single-marker design (Section 7.1), and the broader-impact statement honestly notes the over-trust risk. However, the absence of general-capability evaluation is a critical omission not mentioned in the limitations section—this should be acknowledged, especially given that the mechanism analysis shows parameter drift in broadly shared layers.

---

## Rating

3 (Borderline reject)

---

## Confidence

4

---

## Ethical Concerns

- [x] NO or VERY MINOR ethics concerns only

---

## Paper Formatting Concerns

No major formatting or anonymity concerns noted.

---

## Code Of Conduct Acknowledgement

- [x] Yes

---

## Responsible Reviewing Acknowledgement

- [x] Yes
