## E. Qualitative Trajectories

To make the abstract claims about timing concrete, we walk through four real SELFCOMPACT trajectories on BrowseComp-Plus: three on MiniMax-M2.5 (Cases A–C) and one on GLM-4.7-Flash (Case D). Each case shows the rubric's parsed verdict at the iterations that matter, the prefix length when the verdict was issued, and the resulting trajectory outcome.

**How to read these traces.** Iteration numbers are scaffold `token_lengths_each_step` indices, i.e. the running count of LLM calls in the trajectory. At each rubric probe the model returns four answers (C1 closed-unit, C2 summarizable, C3 progress, N1 stuck) plus quoted evidence; the fire rule from Box B is COMPRESS iff $\text{C1} = Y \land \text{C2} = Y \land \text{C3} = Y \land \text{N1} = N$, otherwise CONTINUE. We annotate each verdict with a short reason code: `first_principles_pass` means all four conditions held; `fp_fail: X=Y` means the listed answer was the gate that flipped the verdict (e.g. N1=Y suppresses an otherwise-passing fire because the model is stuck and summarization would mask it); `threshold_30pct` means the token-pct backstop forced a COMPRESS regardless of the rubric. Prefix-token counts are the running prompt size (system prompt + question + trajectory so far) at the moment the probe was issued.

### E.1 Case A: Good Fire (Correct Answer)

**Question.** "As of December 2022, I am looking for the name of a museum along a river that was added to the National Register of Historic Places in 1988. The museum is named after the last name of the individual who paid just over $500 for land rights..." The gold answer is **Cozens Ranch Museum**.

**What happened.** The agent issued nineteen searches and one `get_document` call. The rubric was probed at every iteration; verdicts 1–11 all returned CONTINUE. The first COMPRESS fired at iteration 15 once all four conditions held simultaneously, collapsing a 39.5k-token prefix to an 810-token summary (compression ratio ≈ **49×**). The model committed the correct final answer two iterations later.

**Case A trace (selected rubric verdicts):**

- **iter 11** (prefix 28k tokens). C1=Y, C2=N, C3=N, N1=Y ⇒ **CONTINUE**.
  Rubric reasoning: "key facts about the land purchase and NRHP dates, but missing the specific birth dates for the children... the negative search results showing we can't find these dates".

- **iter 15** (prefix 39.5k tokens). C1=Y, C2=Y, C3=Y, N1=N ⇒ **COMPRESS**.
  Rubric reasoning: "(C2) 1. 'Cozens Ranch Museum' is the museum name | 2. 'added to the National Register of Historic Places in 1988' | 3. 'located along the Fraser River' | 4. 'paid a little more...'".

  Summary fires. Prefix 39,511 tokens → summary 810 tokens. The model then re-asked one confirmation query and emitted the correct answer.

**Take-away.** The rubric did not fire on the first iteration at which a candidate appeared (iter 11, C1 = Y). It held out three more iterations until the dispersed evidence about the NRHP date and river location had been collected and could be cited verbatim (C2 flipped N → Y). A fixed-interval 30% trigger (58,982 tokens for MiniMax-M2.5) would have fired later, at a state with the same answer but a larger discardable prefix.

### E.2 Case B: Rubric Correctly Suppresses (Correct Answer)

**Question.** A multi-hop band-identification puzzle requiring the agent to triangulate a singer born in June between 1927 and 1930 whose love song was covered by a band formed in 2001–2004. The gold answer is **JadaL**.

**What happened.** The trajectory ran for 99 iterations and the running prefix repeatedly crossed the 30% threshold (58,982 tokens). Despite this, the rubric returned CONTINUE on 25 high-token probes (> 40k), explicitly because C1 = N: the model was mid-search every time the threshold backstop would have fired. The rubric only allowed three compressions total across the trajectory, all at points where searches had genuinely closed.

**Case B trace (rubric blocks fixed-interval-style fires):**

- **iter 18** (prefix 52.7k tokens, above threshold). C1=N, C2=N, C3=N, N1=Y ⇒ **CONTINUE**.
  Rubric reasoning: "The last assistant message shows open-ended reasoning: 'Let me think about this more systematically. I need to identify famous singers born in June between 1927–1930. Let me search more specifically:' — this is mid-thought, not a closed unit."

- **iter 20** (prefix 57.8k tokens, near threshold). C1=N, C2=N, C3=N, N1=Y ⇒ **CONTINUE**.
  Rubric reasoning: "still searching for the singer's birth date... without reaching a closed unit."

- **iter 38** (prefix 47.3k tokens after one earlier fire). C1=N, C2=N, ... ⇒ **CONTINUE**.
  Rubric reasoning: "after searching for musicians born June 1927, I received search results about famous musicians but haven't analyzed them or reached a conclusion."

**Take-away.** A fixed-interval policy would have fired four to five more times across this trajectory, each one cutting an active multi-search reasoning thread mid-stride. The rubric's **C1 = closed-unit check is the load-bearing gate** here: it lets the prefix grow past the static threshold whenever the model is genuinely mid-search, and only releases compression at a thought boundary. The model still reaches the correct answer with three (vs. ≥ 6) compactions and a **~25% smaller cumulative prefill bill**.

### E.3 Case C: Failure Mode (Incorrect Answer)

**Question.** A description of a Boer War officer with a specific physical appearance, captured at Mafeking, with land in a specific cemetery. The gold answer is **Frederik Christoffel Eloff**.

**What happened.** At iteration 5, with only 14.9k tokens of context, the model had already converged on a candidate and the rubric fired COMPRESS with all four conditions met. But the candidate was wrong: **Sarel Johannes Eloff**, a different member of the same family. The summary preserved the wrong candidate as "essential information", and every subsequent search ran from a prefix that asserted Sarel Eloff as identity — locking the model in. Ten summaries fired across the trajectory, and the final answer was Stephanus Petrus Erasmus Eloff (yet another sibling). The predicted surname is correct; the first name is not.

**Case C trace (rubric fires confidently on a wrong candidate):**

- **iter 5** (prefix 14.9k tokens). C1=Y, C2=Y, C3=Y, N1=N ⇒ **COMPRESS**.
  Rubric reasoning (C2 essential facts, verbatim): "1. 'Very tall, spare and upright, youthful looking...' — describes Sarel Johannes Eloff | 2. 'Received receipt for one revolver, horse, saddle...'".

  Summary fires. Prefix 14,883 tokens → summary 833 tokens. The summary opens with "From the search results, I identified Sarel Johannes Eloff (1870–1944) as the person matching the description."

- **Subsequent iterations.** Searches conditioned on the summary return more facts about Sarel's family, which the rubric again recognises as "new concrete facts" and re-fires on (iter 11, 12, etc.). The wrong identity is never revisited.

**Take-away.** The rubric checks structural conditions (closed-unit, citable facts, progress, not-stuck) but **not factual correctness** of the candidate. When the model converges early on a plausible-but-wrong identity, C2 happily lists verbatim quotes about that wrong identity and fires COMPRESS. The summary then erases the alternative leads the model would have needed to recover, locking in the error. This points at a natural extension that we leave to future work: pairing the structural rubric with a candidate-aware verifier that tests, before firing, whether the candidate it would preserve survives a counter-example search.

### E.4 Case D: GLM-4.7-Flash, Single Well-Timed Fire (Correct Answer)

**Question.** "In February 2017, an article was published about an animal that suffered injury after an accident involving a car on a road that was built by a person who got married 199 years before the accident..." The gold answer is the post-accident weight of the leopard, **37 kg**.

**What happened.** The agent issued 14 searches and 4 `get_document_bcp` calls across 23 iterations under SELFCOMPACT with the `first_principles_threshold` variant (rubric + 0.30 token-pct backstop). The rubric was probed at three iterations and fired COMPRESS exactly once, at iteration 12, collapsing a 31,199-token prefix to a 668-token summary (compression ratio ≈ **47×**). The model committed the correct final answer shortly after. End-to-end cost: $0.011 rollout + $0.007 tool, well below the per-question average for the no-compaction baseline.

**Case D trace (rubric verdicts and the single fire):**

- **iter 8** (prefix 20.4k tokens). C1=Y, C2=N, C3=N, N1=Y ⇒ **CONTINUE** (`fp_fail: N1=Y`).
  Reason: the model has just closed a search loop (C1 = Y) but no progress facts to cite (C2 = N, C3 = N), and recent searches duplicated (N1 = Y). Compressing here would freeze a stuck state.

- **iter 10** (prefix 25.8k tokens). C1=N, C2=N, C3=N, N1=N ⇒ **CONTINUE** (`fp_fail: C1,C2,C3`).
  The model is mid-thought ("Let me search for French..."); no closed unit yet.

- **iter 12** (prefix 31.2k tokens). C1=Y, C2=Y, C3=Y, N1=N ⇒ **COMPRESS** (`first_principles_pass`).
  C2 essential facts (verbatim): "1. leopard hit by car in Bainskloof Pass | 2. broken back and internal injuries | 3. vet records report 37 kg post-accident weight...".

  Summary fires. Prefix 31,199 tokens → summary 668 tokens (`triggered_by=selfcheck:first_principles_pass`). The summary opens "Essential Information: In February 2017 there was an article about a leopard that was hit by a car in Bainskloof Pass...," preserving the candidate weight verbatim. The model emits the correct answer two iterations later.

**Take-away.** Case D mirrors Case A on a different model and a different rubric configuration (GLM-4.7-Flash with thr-pct backstop at 0.30 · 128,000 = 38,400 tokens). The two intermediate probes were correctly suppressed — once by N1 (stuck), once by C1 (mid-thought) — so the rubric did not misfire on a prefix that would have looked compressible to a token-only policy. The single COMPRESS at iter 12 landed at the first moment all four conditions actually held, well below the 38,400-token threshold backstop, demonstrating that **C1 + C2 together can fire compaction earlier than a fixed-token trigger would**.

---

## NeurIPS Paper Checklist

The paper includes a complete NeurIPS Paper Checklist covering 16 items (Claims, Limitations, Theory Assumptions, Reproducibility, Open Access, Experimental Settings, Statistical Significance, Compute Resources, Code of Ethics, Broader Impacts, Safeguards, Licenses, New Assets, Crowdsourcing, IRB, and LLM Usage). All answers are either [Yes], [No], or [N/A] with 1–2 sentence justifications. Key answers include:

- **Claims** [Yes]: The three contributions stated in the abstract/introduction are matched by experimental results in §4.1 and §4.2.
- **Limitations** [Yes]: §6 discusses evaluation only on open-weight models, task-specific prompt engineering, and training-free scope.
- **Reproducibility** [Yes]: Full scaffold, prompts, model lists, sampling parameters, and cost accounting are documented in Appendices A–C.
- **Open Access** [No]: Code is not released, but all benchmarks are public, all models are open-weight or OpenRouter-accessible, and prompts are reproduced verbatim.
- **LLM Usage** [Yes]: LLMs are the central object of study; SELFCOMPACT is an inference-time scaffold where the same model $\pi$ serves as generator, rubric judge, and summarizer.
