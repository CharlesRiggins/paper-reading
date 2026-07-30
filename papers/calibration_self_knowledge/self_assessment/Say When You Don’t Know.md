# NeurIPS 2026 Official Review (Reviewer Rb3e)

Paper: *Say When You Don't Know: Training LLMs to Expose Self-Assessment*  
Submission: 8440  
OpenReview: https://openreview.net/forum?id=oOILbzcS44  
Reviewer: Rb3e (Ran Wang)  
Dates: 26 Jun 2026, 11:40 (modified: 29 Jul 2026, 02:22)

Synced to the submitted OpenReview text, then revised after PC note on Contribution Type mismatch. **Contribution Type corrected: General → Use-inspired**; Strengths/Weaknesses, Questions, Limitations, and Originality re-aligned to Use-inspired criteria. See also [`ac_contribution_type_reply.md`](./ac_contribution_type_reply.md).

---

## Summary

This paper tackles LLM self-assessment by reframing it as an "exposure" problem: instead of estimating how reliable a response is from the outside after generation, the model should be trained to say when it's unsure, right in its own response. The authors study two complementary signals under a unified GRPO post-training framework: (1) a verbalized confidence score emitted after the final answer, trained with a Brier-style reward, and (2) a `<uncertain>` marker emitted mid-reasoning whenever the model hits a risky step, trained with an asymmetric reward that punishes silent failure hardest. Both are evaluated on five factual QA benchmarks against baselines like P(True), temperature scaling, SFT, Self-RAG, FLARE, and DRAGIN. The results show both signals cut overconfident errors sharply (ECE 0.383→0.049, strict epistemic errors 88.6%→3.9%) and work well as adaptive RAG triggers (best: 41.6% EM at 48.1% trigger rate). A mechanism analysis using KL divergence, CKA, and activation patching suggests the two methods leave different internal traces: verbalized confidence sharpens an existing confidence structure while preserving representation geometry, whereas the `<uncertain>` marker reshapes late-layer representations. An appendix theorem shows GRPO can only reweight existing trajectories, not create new ones.

---

## Contribution Type

Use-inspired: The main contribution is in framing or designing approaches to meet the needs of a specific real-world application. (This often involves, e.g., engaging with domain experts.)

---

## Strengths And Weaknesses

> **Strengths.**
>
> - **Application-matched reward design.** The `<uncertain>` reward asymmetry—penalizing silent failure hardest—directly targets a deployment failure mode (overconfident wrong answers that never trigger intervention). The ablation (Table 10) confirms this design choice matters, and the adaptive-RAG results show the signal can be consumed by a concrete downstream controller.
> - **Useful mechanism analysis for method design.** The KL / CKA / patching analysis helps explain *why* end-of-answer verbalized confidence and mid-reasoning markers behave differently—valuable for deciding which signal to use under different application constraints.
> - **Clean calibration and RAG-trigger results.** Table 3 shows GRPO is the only method simultaneously winning on ECE, overconfident-wrong rate, and EM, whereas P(True), temperature scaling, and SFT each trade off different metrics. The adaptive-RAG evaluation is a meaningful use-facing metric beyond pure calibration.
>
> **Weaknesses.**
>
> **(W1) Weak use-case grounding.** Under Use-inspired criteria, the central question is whether the work addresses a pre-existing external need and matches design/metrics to that need. The paper motivates "risky failures in real-world applications," but the concrete use case remains generic (LLM overconfidence / factual QA + adaptive RAG on standard benchmarks). There is little evidence of a specific stakeholder need, domain constraint, or non-standard real-world dataset that shaped the task framing. As a result, it is hard to judge whether the proposed signals and metrics are the right ones for a real deployment setting, versus an interesting methodological study framed as use-inspired.
>
> **(W2) Design only loosely matched to deployment needs.** If the intended application is reliable self-assessment for retrieval / abstention / human review, several application-facing pieces are missing: cost/latency of triggered retrieval, false-trigger vs missed-failure trade-offs under realistic budgets, comparison to commonly used non-ML or heuristic interventions, and evaluation beyond academic factual QA. Adaptive RAG is a step in the right direction, but still stays within standard ML benchmark protocols.
>
> **(W3) Capability side effects undercut the use claim.** A use-inspired reliability method must not quietly break other deployed behaviors. The paper claims calibration "does not degrade reasoning accuracy," but originally reported no general-capability evaluation despite parameter drift in shared layers. Rebuttal-time results strengthen this concern: MMLU/GSM8K are preserved, but IFEval drops sharply (−19.96 Llama, −14.42 Qwen). For any real application that depends on instruction following, this is a serious deployment risk and should qualify the paper's claims.
>
> **(W4) Clarity for an ML audience and evaluation completeness.** GRPO and ECE appear without inline definition/citation at first use. Main results focus on a single model family (Qwen in appendix), with no error bars or significance tests (checklist item 7: "No"). These issues limit how confidently practitioners could adopt or reproduce the approach.

---

## Quality

2: not good

---

## Clarity

2: not good

---

## Significance

2: not good

---

## Originality

3: good

---

## Questions

1. **Concrete use case.** What is the specific real-world application this work is designed for (beyond generic LLM overconfidence)? Are there external stakeholders or domain constraints that shaped the choice of signals, rewards, or metrics? If the intended use is adaptive retrieval / abstention, can the authors state the operating constraints (budget, latency, cost of missed failures vs false triggers) and show that the method is matched to them?

2. **General-capability / instruction-following preservation.** The rebuttal IFEval drops (~15–20 points) indicate a clear capability trade-off. How should the "without degrading reasoning accuracy" claim be revised for a use-inspired reliability method? What mitigation (stronger KL, mixed instruction replay, multi-objective checkpointing) would be required before deployment?

3. **Incremental value over Self-RAG in the target application.** Self-RAG already trains retrieve/critique tokens tied to a retrieval workflow. In the authors' intended use case, what does the during-reasoning `<uncertain>` marker uniquely enable that Self-RAG (or a simple heuristic trigger) does not—e.g., controller-agnostic risk exposure, earlier intervention, or better budget control? A clearer application-level comparison would strengthen the contribution.

4. **Terminology.** Could the authors add brief inline definitions (or at least citations at first occurrence) for GRPO and ECE?

---

## Limitations

Partially addressed. The authors note the factual-QA-only scope and single-marker design (Section 7.1), and the broader-impact statement notes over-trust risk. Under a Use-inspired reading, two omissions are more critical: (i) the lack of a clearly specified external use case / deployment evaluation, and (ii) general-capability side effects—especially instruction-following degradation now reported in the rebuttal. Both should be acknowledged as limitations of the current evidence for real-world applicability.

---

## Rating

3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.

---

## Confidence

4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

---

## Ethical Concerns

NO or VERY MINOR ethics concerns only

---

## Paper Formatting Concerns

No.

---

## Code Of Conduct Acknowledgement

Yes

---

## Responsible Reviewing Acknowledgement

Yes
