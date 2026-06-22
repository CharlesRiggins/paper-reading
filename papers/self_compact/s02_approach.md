# Self-Compacting Language Model Agents — Approach

## 3. Our Approach: SELFCOMPACT

### 3.1 Setup

Given a natural-language prompt $x$, a language model $\pi(\cdot \mid x)$ generates a continuation

$$
y = (y_1, y_2, \ldots)
$$

autoregressively, with each token conditioned on the prompt and all preceding tokens. The scaffold equips the language model with a summarization tool

$$
S : (x, y_{1:t}) \to \tilde{y},
$$

which takes the original prompt and a possibly partial continuation and produces a condensed version $\tilde{y}$. Generation then resumes from the new context

$$
(x, \tilde{y}).
$$

The key design choice is that the summarizer is the same model $\pi$. SELFCOMPACT does not require an external verifier, an auxiliary summarizer, supervised fine-tuning, or reinforcement learning.

### 3.2 Two inference-time elements

SELFCOMPACT pairs two components:

1. **Summarization tool $S$** exposed to the model.
2. **Rubric prompt $P_R$** that decides at probe intervals whether $S$ should fire.

The paper emphasizes that both are required. If only the tool is exposed, models vary widely: some call it at bad times, others rarely or never. If only the rubric exists, it can judge but cannot act. The combination translates high-level notions such as “a sub-task has resolved” and “the trajectory is not stuck mid-derivation” into local, cite-able conditions that the same model can verify from the current trajectory.

### 3.3 Algorithm 1: rubric-gated inference-time compaction

The paper’s algorithm can be reconstructed as follows.

**Inputs**: prompt $x$; model $\pi$; probe interval $N$; step budget $T$; rubric prompt $P_R$; summarizer prompt $P_S$.

**Notation**: $C \leftarrow C \circ m$ appends message $m$ to context $C$; $z \sim \pi(\cdot \mid C)$ samples a response; `pop(C, m)` removes message $m$ from the context, assuming it is the last item.

```text
Algorithm 1: SELFCOMPACT: rubric-gated inference-time context compaction

1:  C ← x                                      // overall context and KV cache
2:  for t = 1, ..., T do
3:      y_t ~ π(· | C); C ← C ◦ y_t
4:      if y_t is a final answer then return y_t
5:      if t mod N = 0 then
6:          C ← C ◦ P_R                         // rubric probe; KV prefix reused
7:          r_t ~ π(· | C), r_t ∈ {COMPRESS, CONTINUE}; C ← C ◦ r_t
8:          if r_t = COMPRESS then
9:              C ← C ◦ P_S                     // KV prefix still reused
10:             ỹ ~ π(· | C)                    // summarize trajectory
11:             C ← x ◦ ỹ                       // hard reset; resume from summary
12:         else
13:             pop(C, r_t); pop(C, P_R)         // revert; trajectory unchanged
14:         end
15:     end
16: end
17: return y_T
```

The important implementation detail is that probes and summary instructions are appended to the existing trajectory rather than replacing it during the call. This preserves KV-cache reuse for the long prefix during rubric judgment and summarization.

### 3.4 Summarizer design

To maximize KV-cache reuse, the scaffold implements $S$ by appending a summarizer prompt as a user-role message. At trigger time, the cached prefix $y_{1:t}$ is preserved across the summarizer call, so the summarizer pays prefill only on the appended instruction rather than re-encoding the entire trajectory.

The scaffold then:

1. appends the summarizer instruction to $(x, y_{1:t})$;
2. samples the summary $\tilde{y}$ from $\pi$;
3. replaces $y_{1:t}$ with $\tilde{y}$;
4. resumes decoding from $(x, \tilde{y})$.

The exact prompt differs by task. For math, the summarizer preserves key insights, intermediate calculations, and any final answer. For search agents, the summarizer extracts concrete, explicitly observed facts that directly contribute to the original question.

### 3.5 Triggering the summarizer

At intervals

$$
t \in \{N, 2N, 3N, \ldots\},
$$

the scaffold appends the rubric prompt $P_R$ to $(x, y_{1:t})$ and samples a binary verdict

$$
r_t \in \{\text{COMPRESS}, \text{CONTINUE}\}.
$$

The interval is measured differently by domain:

| Domain | Probe interval unit |
|---|---|
| Competition math | Tokens / refinement rounds in the reasoning trace |
| Agentic search | Tool-call or ReAct-round boundaries |

The rubric prompts are task-specific. Math uses conditions around whether a final answer exists, whether the model is stuck, and whether it has an exact next step. Search uses conditions around closed units, summarizability into 3–5 cite-able facts, progress since the last compression, and not being stuck.

The paper writes the firing rule generically as:

$$
\tilde{y} \leftarrow S(x,y_{1:t})
\quad \text{whenever } t \equiv 0 \pmod N \text{ and } r_t = \text{COMPRESS};
$$

otherwise generation resumes from $(x,y_{1:t})$ unchanged.

### 3.6 Workflow and compared regimes

The experiments compare three regimes that share the same generation loop but differ in transcript compression:

| Regime | Compression behavior |
|---|---|
| No compaction | The trajectory accumulates untruncated until answer or budget. |
| Fixed-interval compaction | Summarize-and-replace at every probe interval or token threshold. |
| SELFCOMPACT | Summarize-and-replace only when the rubric returns COMPRESS; otherwise continue from the unchanged prefix. |

On agentic search, this yields a search–judge–summarize–search loop. After each compression, the agent continues against $\tilde{y}$ rather than the full history.

### 3.7 Cost analysis

Let

- $L = |y_{1:t}|$ be the pre-compression trajectory length in tokens;
- $\ell = |\tilde{y}|$ be the summary length in tokens.

SELFCOMPACT adds at most two LLM calls per checkpoint:

1. the rubric probe, always;
2. the summarizer, only if the rubric returns COMPRESS.

Both calls append to $(x,y_{1:t})$, so the KV cache of $y_{1:t}$ is reused and the naive $O(L^2)$ re-prefill is avoided. The rubric probe is almost free because it generates only a short verdict on top of a cached prefix.

The summarizer is the main overhead:

$$
O(L\ell)
$$

to generate $\tilde{y}$, plus a one-time

$$
O(\ell^2)
$$

prefill of the new $(x,\tilde{y})$ prefix. This cost is amortized over later tokens, which now attend to $\ell$ summary tokens rather than the original $L$-token history. The per-token attention burden drops from $O(L)$ to $O(\ell)$.

### 3.8 Empirical cost trade-off

The paper reports that the trade is favorable in both domains:

- On agentic search, SELFCOMPACT adds 5–9 accuracy points at 30–70% lower per-question token cost than no compaction.
- On competition math, SELFCOMPACT leads on 11 of 12 benchmark/model cells under a token budget matched to fixed-interval summarization.

Appendix C gives the full per-question cost formula and token bookkeeping. It also states a cached-vs-prefill break-even condition: compaction wins when

$$
\frac{L}{\ell} > 10,
$$

and the search summarizer empirically reaches shrinkage ratios of about $20\times$–$80\times$.
