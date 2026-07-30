# Reply to AC — Contribution Type Correction (Reviewer Rb3e)

Reply **to the AC only** (not to the authors), under the PC note:  
https://openreview.net/forum?id=oOILbzcS44&noteId=GdSx6sJs25

Paper: 8440 — *Say When You Don't Know: Training LLMs to Expose Self-Assessment*

---

## Suggested OpenReview text

To the AC:

I have corrected the Contribution Type from General to Use-inspired to match the authors, and re-aligned the review discussion with Use-inspired criteria.

Main changes:

1. **Reframed W1.** Instead of penalizing limited algorithmic novelty (a General lens), I now evaluate whether the paper grounds a real external use case and matches design/metrics to that use case. Under this lens, the motivation remains somewhat generic (LLM overconfidence / adaptive RAG on standard factual QA), with limited evidence of a specific pre-existing application need or deployment-facing evaluation.

2. **Reweighted originality.** Per Use-inspired guidelines, originality need not require wholly new methods. I softened the novelty critique, raised Originality from 2 to 3, and now focus on whether the framing and reward design are justified by application needs.

3. **Strengthened deployment-capability concerns.** For a use-inspired reliability/calibration claim, side effects on deployed behavior matter more. I retained and sharpened the general-capability critique; the authors’ rebuttal IFEval drops (~15–20 points) reinforce this limitation.

4. **Rewrote strengths and questions.** Strengths now emphasize application-matched reward design and RAG-trigger utility rather than the GRPO support theorem as an RL-debate contribution. Questions now ask for a concrete use case, operating constraints, and application-level value over Self-RAG.

5. **Scores / rating.** Originality 2→3; Quality, Clarity, Significance remain 2; overall Rating remains 3 (Borderline reject), because under the corrected criteria the use-case grounding and deployment-side evaluation are still insufficient.

Happy to clarify further if needed.
