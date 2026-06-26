## 3. Our Approach: SELFCOMPACT

**Setup.** Given a natural language prompt $x$, a language model $\pi(\cdot \mid x)$ generates a continuation $y = (y_1, y_2, \ldots)$ autoregressively, with each token conditioned on the prompt and all preceding tokens. We equip the language model with a **summarization tool** $\mathcal{S} : (x, y_{1:t}) \mapsto \tilde{y}$, which takes the original prompt $x$ and a (possibly partial) continuation $y_{1:t}$ and produces a condensed version $\tilde{y}$. Generation then resumes from the new context $(x, \tilde{y})$. Crucially, the summarizer is the same model $\pi$ — no external verifier or auxiliary model is required.

**SELFCOMPACT.** SELFCOMPACT pairs two inference-time elements: (i) the summarization tool $\mathcal{S}$ exposed to the model, and (ii) a lightweight **rubric** that decides, at periodic probe intervals, whether $\mathcal{S}$ should fire. Both elements are needed: exposing the tool alone — letting $\pi$ emit `<summarize>` whenever it wants — yields uneven, often poorly-timed firing across models: some call the tool reflexively at unhelpful moments, others rarely at all. The rubric closes that gap by translating sub-task has resolved" and "the trajectory is not stuck mid-derivation" into concrete, cite-able conditions $\pi$ can verify against the trajectory in front of it. SELFCOMPACT runs on a single inference engine with no fine-tuning.

**Algorithm 1: SELFCOMPACT — rubric-gated inference-time context compaction.**

**Input:** prompt $x$; model $\pi$; probe interval $N$; step budget $T$; rubric prompt $P_R$; summarizer prompt $P_S$.

**Notation:** $\mathcal{C} \gets \mathcal{C} \circ m$ appends message $m$ to context $\mathcal{C}$; $z \sim \pi(\cdot \mid \mathcal{C})$ samples response $z$ from $\pi$ given $\mathcal{C}$; $\text{pop}(\mathcal{C}, m)$ pops (removes) message $m$ from $\mathcal{C}$ (assuming that it's the last item in $\mathcal{C}$).

1. $\mathcal{C} \gets x$; // Overall context and its KV cache
2. **for** $t = 1, \dots, T$ **do**
3. $\quad y_t \sim \pi(\cdot \mid \mathcal{C}); \;\; \mathcal{C} \gets \mathcal{C} \circ y_t$;
4. $\quad$ **if** $y_t$ is a final answer **then return** $y_t$;
5. $\quad$ **if** $t \bmod N = 0$ **then**
6. $\qquad \mathcal{C} \gets \mathcal{C} \circ P_R$; // rubric probe; KV prefix reused
7. $\qquad r_t \sim \pi(\cdot \mid \mathcal{C}), \; r_t \in \{\text{COMPRESS}, \text{CONTINUE}\}; \;\; \mathcal{C} \gets \mathcal{C} \circ r_t$;
8. $\qquad$ **if** $r_t = \text{COMPRESS}$ **then**
9. $\qquad\quad \mathcal{C} \gets \mathcal{C} \circ P_S$; // KV prefix still reused
10. $\qquad\quad \tilde{y} \sim \pi(\cdot \mid \mathcal{C})$; // summarize trajectory
11. $\qquad\quad \mathcal{C} \gets x \circ \tilde{y}$; // hard reset; resume from summary
12. $\qquad$ **else**
13. $\qquad\quad \text{pop}(\mathcal{C}, r_t)$; $\text{pop}(\mathcal{C}, P_R)$; // revert; trajectory unchanged
14. $\qquad$ **end**
15. $\quad$ **end**
16. **end**
17. **return** $y_T$;

**Summarizer design.** To maximize KV-cache reuse, we implement $\mathcal{S}$ by appending a summarizer prompt (Box A for Math; Box B for Search Agents) as a user-role message: the cache of $y_{1:t}$ at trigger time is preserved across the call, so $\mathcal{S}$ pays prefill only on the appended instruction, not on the full trajectory. A saving we quantify in the cost analysis below. The scaffold appends the instruction to $(x, y_{1:t})$, $\pi$ generates the summary $\tilde{y}$, and $\tilde{y}$ then replaces $y_{1:t}$; decoding resumes from the original question and the summarized content $(x, \tilde{y})$.

**Triggering the summarizer.** At intervals $t \in \{N, 2N, 3N, \ldots\}$, the scaffold appends a rubric prompt $P_R$ as a user-role message to $(x, y_{1:t})$ and samples from $\pi(x, y_{1:t}, P_R)$ a binary verdict $r_t \in \{\text{COMPRESS}, \text{CONTINUE}\}$. These intervals are measured in tokens of the reasoning trace (math) or in tool calls (agentic search).

As with $\mathcal{S}$, the probe is appended rather than substituted so that the KV cache of $y_{1:t}$ is preserved across the call, keeping the judge near-free relative to the trajectory length. The probe enumerates a small set of conditions, each requiring verbatim evidence quoted from the trajectory, so $\pi$ can check them locally. The rubric is task-specific: the conditions for competition math differ from those for agentic search, and we defer the exact wording to Appendix A (math) and Appendix B (search). On a COMPRESS verdict, the summarizer fires:

$$
\tilde{y} \leftarrow \mathcal{S}(x, y_{1:t}) \quad \text{whenever } t \equiv 0 \pmod{N} \text{ and } r_t = \text{COMPRESS};
$$

otherwise generation resumes from $(x, y_{1:t})$ unchanged.

**Workflow.** Our experiments compare three regimes that share the same generation loop but differ in how the transcript is compressed: **no compaction** (the trajectory accumulates untruncated), **fixed-interval compaction** (summarize-and-replace at every probe interval), and **SELFCOMPACT** (summarize-and-replace only when the rubric returns COMPRESS, otherwise continue). On agentic search this yields a search–judge–summarize–search loop, with the agent continuing against $\tilde{y}$ rather than the full history after each compression.

**Cost analysis.** Write $L = |y_{1:t}|$ for the pre-compression trajectory length and $\ell = |\tilde{y}|$ for the summary length, both in tokens. SELFCOMPACT adds at most two LLM calls per checkpoint: the rubric probe always, the summarizer only on a COMPRESS verdict. Both append to $(x, y_{1:t})$, so the KV cache of $y_{1:t}$ is reused across the call and the $\mathcal{O}(L^2)$ re-prefill that a naive re-encode would incur is avoided. The rubric probe is essentially free: a short verdict $r_t$ generated on top of the cached prefix. The summarizer is the only real overhead — $\mathcal{O}(L\ell)$ to generate $\tilde{y}$ plus a one-time $\mathcal{O}(\ell^2)$ prefill of the new $(x, \tilde{y})$ prefix — paid once per fire and amortized over every subsequent decoded token, which now attends to the $\ell$-token summary rather than the $L$-token history, dropping per-token attention from $\mathcal{O}(L)$ to $\mathcal{O}(\ell)$. Empirically the trade is favorable: on agentic search SELFCOMPACT adds 5–9 accuracy points at 30–70% lower per-question token cost than the no-compaction baseline (§4.2); on competition math it leads on 11 of 12 benchmark/model cells under a token budget matched to fixed-interval summarization (§4.1). Appendix C states the per-question cost formula in full, defines how $(N_{\text{uncached}}, N_{\text{cached}}, N_{\text{out}})$ are accumulated from each trajectory, lists the OpenRouter prices, and works out the cached-vs-prefill break-even (compaction wins iff $L / \ell > 10$, which the search summarizer reaches at 20–80×).
