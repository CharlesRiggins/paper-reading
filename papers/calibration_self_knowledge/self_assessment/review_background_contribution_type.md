# Review Background — Contribution Type Correction

Paper: *Say When You Don't Know: Training LLMs to Expose Self-Assessment*  
Submission: 8440  
OpenReview: https://openreview.net/forum?id=oOILbzcS44  
Reviewer: Rb3e (Ran Wang)  
Venue: NeurIPS 2026 Main Track

This note records the background for our review revision after the Program Chairs flagged a Contribution Type mismatch. It is local documentation only; the authoritative review text lives in [`Say When You Don’t Know.md`](./Say%20When%20You%20Don%E2%80%99t%20Know.md).

---

## Timeline

| Date | Event |
|---|---|
| 30 Apr 2026 | Authors submit paper; declare **Contribution Type: Use-inspired** |
| 26 Jun 2026 | Our review (Rb3e) submitted with **Contribution Type: General** |
| 28 Jul 2026 | Authors post rebuttal to Rb3e (Q1–Q4) |
| ~29–30 Jul 2026 | PC posts Contribution Type mismatch note; requests review update + AC-only reply |
| 30 Jul 2026 | Local review revised to **Use-inspired**; AC reply drafted |

---

## What went wrong

Authors and reviewer selected different Contribution Types:

| Party | Declared type |
|---|---|
| Authors ([`openreview_submission_original.md`](./openreview_submission_original.md)) | **Use-inspired** — framing/design for a specific real-world application |
| Our original review | **General** — default / most submissions |

NeurIPS 2026 evaluates Quality / Clarity / Significance / Originality **differently by Contribution Type**. A General lens over-weights algorithmic novelty; a Use-inspired lens asks whether the use case is real, whether design/metrics match that use case, and whether the work could impact a concrete application.

PC note ([`pc_contribution_type_comment.md`](./pc_contribution_type_comment.md)):

- Correct the contribution type in the review and **re-align the assessment** with Use-inspired criteria.
- Reply to the note **to the AC only** (not authors), briefly describing how the review changed.
- **"No change" is not allowed.**
- If the reviewer does not respond, the AC should **downweight** this review.

Official guidance: [NeurIPS 2026 Reviewer Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines) (Use-Inspired section).

---

## Use-inspired criteria (what we re-aligned to)

From the official guidelines:

- **Quality:** Is the use case real and meaningful (not artificially constructed)? Is the design matched to the use case? Non-standard real-world datasets are expected when justified.
- **Clarity:** Write for an ML audience; explain application jargon.
- **Significance:** Impact on an important use case and/or the NeurIPS community; simple practical ideas can be valuable. Consider prior work outside ML and non-ML baselines.
- **Originality:** Need not mean wholly novel methods; novel combination or framing for the use case is enough, but should analyze why design choices work.

---

## How the review changed

Full revised text: [`Say When You Don’t Know.md`](./Say%20When%20You%20Don%E2%80%99t%20Know.md)  
AC reply draft: [`ac_contribution_type_reply.md`](./ac_contribution_type_reply.md)

| Item | Before (General) | After (Use-inspired) |
|---|---|---|
| Contribution Type | General | Use-inspired |
| W1 | Limited novelty | Weak use-case grounding |
| W2 | Undefined terminology | Design only loosely matched to deployment needs |
| W3 | No general-capability eval | Capability side effects (incl. rebuttal IFEval drops) undercut the use claim |
| W4 | Narrow scope | Clarity + evaluation completeness for practitioners |
| Strengths | Reward design; GRPO support theorem; clean baselines | Application-matched reward; mechanism analysis for design choice; calibration + RAG-trigger results |
| Originality | 2 | **3** |
| Quality / Clarity / Significance | 2 / 2 / 2 | unchanged |
| Rating | 3 (Borderline reject) | unchanged |
| Questions | Capability; agentic tool-use; Self-RAG novelty; terminology | Concrete use case; capability/IFEval; application-level value vs Self-RAG; terminology |

**Rationale for keeping Rating 3:** correcting the type changes the *reasons*, not the overall verdict. Under Use-inspired criteria the paper still lacks a clearly specified external use case and deployment-facing evaluation; rebuttal IFEval degradation further weakens the reliability/deployment claim.

---

## Author rebuttal (context, separate from PC issue)

Source: [`openreview_rebuttals_original.md`](./openreview_rebuttals_original.md)

Authors answered our original Q1–Q4:

1. **General capability:** Added rebuttal-time MMLU / GSM8K / IFEval. MMLU & GSM8K slightly improve; **IFEval drops sharply** (−19.96 Llama, −14.42 Qwen). They agree to qualify the “without degrading reasoning accuracy” claim.
2. **Agentic tool-use:** Argue explicit self-assessment is complementary / controller-agnostic, not a direct substitute for end-to-end tool policy learning.
3. **Self-RAG:** Position `<uncertain>` as reasoning-time risk exposure via GRPO outcomes, vs Self-RAG’s retrieval/critique tokens from offline critic labels.
4. **Terminology:** Will add GRPO / ECE definitions and citations.

The PC Contribution Type note is **orthogonal** to the rebuttal thread. We incorporated the IFEval finding into the revised Use-inspired W3 / Questions because it is directly relevant under deployment-oriented criteria. A separate author-facing rebuttal response (if any) can still be written later.

---

## Related local files

| File | Role |
|---|---|
| [`openreview_submission_original.md`](./openreview_submission_original.md) | Author abstract + declared Contribution Type |
| [`Say When You Don’t Know.md`](./Say%20When%20You%20Don%E2%80%99t%20Know.md) | Our review (revised) |
| [`openreview_rebuttals_original.md`](./openreview_rebuttals_original.md) | Author rebuttal to Rb3e |
| [`pc_contribution_type_comment.md`](./pc_contribution_type_comment.md) | PC mismatch notification |
| [`ac_contribution_type_reply.md`](./ac_contribution_type_reply.md) | Draft reply to AC only |
| [`review_background_contribution_type.md`](./review_background_contribution_type.md) | This background note |

---

## Suggested OpenReview actions

1. Update the review form: Contribution Type → Use-inspired; paste revised Strengths/Weaknesses, Questions, Limitations; set Originality to 3.
2. Reply **to the AC only** under the PC note, using the text in [`ac_contribution_type_reply.md`](./ac_contribution_type_reply.md).
3. Optionally later: post a separate response to the authors’ rebuttal (not required by the PC note).
