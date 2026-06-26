# Review — Self-Compacting Language Model Agents (OpenReview: 18449)

NeurIPS 2026 review. Summary and Strengths/Weaknesses kept concise and casual; scores are numbers only; no bold.

## Summary

The paper proposes SELFCOMPACT, a training-free scaffold that decides when to compact an LM agent's rolling context. Rather than firing summarization at a fixed token threshold, it pairs an inline summarization tool (invoked by the model itself) with a lightweight rubric that returns a COMPRESS/CONTINUE verdict at periodic probe intervals based on trajectory structure — e.g., whether a sub-task has resolved or the trajectory is converging — instead of token count. Both the rubric judge and the summarizer are the same underlying model, so no fine-tuning or external verifier is needed. The method is evaluated on competition math (4 Qwen models across 3 benchmarks) and agentic search (3 deployed agents across 3 benchmarks), with a cost analysis built on KV-cache reuse, plus ablations, an oracle headroom analysis, and qualitative trajectory cases.

## Strengths And Weaknesses

This is a clean, well-engineered paper on a genuinely practical problem. The motivating observation lands: fixed-interval compaction is content-agnostic and can wipe verified facts mid-derivation, and Figure 1 makes the failure concrete. The part I found most valuable is the cost analysis built on KV-cache reuse — it explains why adding a rubric probe per checkpoint is nearly free, and why per-question cost actually drops 30–70% on agentic search despite the extra calls. The experimental design is fair too: token budgets are matched between fixed-interval and SELFCOMPACT within ±3k, so the gains aren't bought with more compute. The ablation (tool-only vs tool+rubric) is the central evidence for the contribution, and the oracle headroom analysis (52.9% vs 45.5%) is a refreshingly honest way to show how far adaptive timing still has to go.

Originality is the main concern. The contribution narrows to "gate a summarization tool with a hand-written rubric on trajectory structure." The authors note LangChain shipped a summary tool concurrently, and the delta is the rubric. The rubric is also task-specific — math and search get different gate sets hand-designed per domain — and cross-task generalization is never tested, so it's hard to tell whether the approach scales or whether each new task needs a fresh rubric.

The more substantive weakness is a failure mode the paper surfaces but doesn't fully own. Case C shows the rubric firing COMPRESS on a wrong-but-plausible candidate: the structural gates all pass, the summary locks in the wrong identity, and every subsequent search runs from a prefix asserting it. This is arguably worse than the no-compaction baseline because it forecloses the recovery path. Yet this lock-in failure isn't mentioned in §6 Limitations — it only shows up in Appendix E. For a method pitched on "better timing," a timing error more damaging than the baseline's is exactly the case to discuss up front.

Finally, evaluation is open-weight only. The authors speculate frontier models may have enough metacognition to skip the rubric, but don't test it, so the central reframing ("when-to-compact as a scaffold-supplied metacognitive capability") rests on the models that may be precisely the ones lacking it.

## Quality

3

## Clarity

3

## Significance

3

## Originality

2

## Questions

1. Case C shows the rubric firing on a wrong candidate and locking it in via the summary. Across the BrowseComp-Plus test set, can the authors report how often SELFCOMPACT turns a baseline-correct answer wrong (correct→wrong regressions on the lock-in path) versus how often fixed-interval does? If SELFCOMPACT is more damaging than fixed-interval on these regressions, my Rating would drop further; if the rate is negligible, it would ease my main concern.

2. The rubric is hand-designed per task (math gates vs search gates). Is there a unified template, or evidence that the gate design transfers to a third task without re-engineering? A positive answer would raise my Originality and Significance scores.

3. The ablation is tool-only vs tool+rubric. Could the authors decompose further — e.g., rubric-gated timing paired with a constant/random summary, to isolate how much gain comes from timing vs summarization quality? This bears directly on the Quality assessment.

4. Even a small-scale check on whether a frontier model still benefits from the rubric would address the concern that the method solves a problem only weaker models have; it would clarify the Significance ceiling.

5. The weak oracle (skip-if-correct) reaches 52.9% while SELFCOMPACT sits at 45.5%. How do the authors attribute this ~7.4-point gap — to the rubric judging wrongly, or to the fire rule being too conservative? This would clarify whether the ceiling lies in the judgment or the policy.

## Limitations

Partially. The authors are upfront about the open-weight-only scope and the training-free framing, and they flag RL as a natural extension. However, the error-lock-in failure mode from Case C — where a wrong candidate gets preserved by the summary and forecloses recovery — is not surfaced in §6; it appears only in Appendix E. For a "better timing" method this is exactly the negative result to own in the main limitations. I'd also like the task-specific rubric design flagged as a generalization limitation rather than just an engineering detail.

## Rating

3

## Confidence

3

## Ethical Concerns

[x] NO or VERY MINOR ethics concerns only

## Paper Formatting Concerns

No major formatting issues observed.

## Code Of Conduct Acknowledgement

[x] Yes

## Responsible Reviewing Acknowledgement

[x] Yes
