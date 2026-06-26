## B. Agentic-Search Experimental Setup

### B.1 Models and Serving

The three deployed agents (`GLM-4.7-Flash`, `MiniMax-M2.5`, `Mimo-V2-Flash`) are accessed through OpenRouter via the AggAgent runtime [Lee et al., 2026]. We use the unmodified `react_agent_selfcheck` scaffold; the rubric and summarizer prompts are reproduced verbatim in Box B and Box B.

### B.2 Benchmarks

BrowseComp [Wei et al., 2025], BrowseComp-Plus [Chen et al., 2025], and DeepSearchQA [Gupta et al., 2026]. Following Sun et al. [2025] and Lee et al. [2026], we sub-sample 150 questions per benchmark.

### B.3 Sampling Parameters

All three models share the same decoding configuration: temperature 1.0, top-p 0.95, max output tokens per call 10,000, parallel-tool-calls disabled, and a per-trajectory cap of 100 LLM calls. Per-model context windows, the 30%-of-context trigger used by FIXED-INTERVAL SUMMARY and the rubric's token-pct backstop, and model-specific decoding flags are listed in Table 5.

### B.4 Scaffold Details

A single trajectory is the standard ReAct loop: the agent emits a tool call (search and either visit for BrowseComp/DeepSearchQA or `get_document_bcp` for BrowseComp-Plus), receives the result, reasons, and either issues another tool call or commits a final answer. For BrowseComp and DeepSearchQA, search is backed by the Serper API (https://serper.dev/) and visit fetches and extracts page content via crawl4ai (https://github.com/unclecode/crawl4ai); for BrowseComp-Plus, search and `get_document_bcp` resolve against the benchmark's released corpus. The rubric in Box B is appended to a copy of the trajectory at each round boundary so the probe does not pollute the rolling cache. COMPRESS fires only when four gates pass simultaneously: (ROUND) iteration $\geq 3$, (TOKENS) running prompt length $\geq 40{,}000$, (CAP) total summaries so far < 1, and (PERIOD) $\geq 2$ rounds elapsed since the last probe. The token-pct backstop forces a COMPRESS once the prompt crosses $0.30 \cdot \text{ctx\_window}$ regardless of the rubric.

**Table 5: Per-model context windows, 30%-of-context summarization trigger, and decoding flags for the three OpenRouter agents.** GLM-4.7-Flash keeps the thinking trace visible to the rubric probe via `enable_thinking=True` and `clear_thinking=False`.

| Model | Ctx window | 30% trigger | Model-specific flags |
|-------|-----------|-------------|---------------------|
| GLM-4.7-Flash | 128,000 | 38,400 | enable_thinking=True, clear_thinking=False |
| MiniMax-M2.5 | 196,608 | 58,982 | reasoning_split=True |
| Mimo-V2-Flash | 262,144 | 78,643 | (default OpenRouter body) |

### B.5 Search Rubric Probe

**(appended to $(x, y_{1:t})$ at every round boundary)**

```
You are about to decide whether to compress your conversation history into a summary that REPLACES the full history above. After compression, you continue research from only [system, original_question, assistant_summary, user_continue]. Compression is irreversible: anything not preserved in the summary is gone.

Compression is safe ONLY when ALL FOUR of the following hold:

(C1) the trajectory has reached a closed unit (not mid-thought),

(C2) the essential information is reducible to 3–5 cite-able facts without loss,

(C3) something has progressed since the last compression,

(C4) the model is NOT stuck (N1).

C1 CLOSED-UNIT: The most recent assistant message is a closed unit — a completed tool call whose result is now visible, or a completed sub-analysis with a clear stopping point. It is NOT mid-sentence reasoning ("Let me now check...", "I should next look at..."), and not a half-formulated query. If Y, quote the closing fragment of the last assistant message. If N, quote the open fragment that shows the trajectory is mid-thought.

C2 SUMMARIZABLE: You can write 3–5 essential facts (with verbatim citations from the trajectory) that future-you needs to continue research after compression. Each fact must be a single concrete statement: a name, date, URL, quoted claim, or resolved sub-question. Answer N if the trajectory's value is dispersed across many small inferences (e.g., a list of dead-end queries needed to avoid retries, negative results that constrain hypothesis space) that would be lost without the dispersal. If Y, list the 3–5 facts numbered, each with a verbatim citation in quotes, separated by " | ". If N, name in one sentence the class of information that would be lost.

C3 PROGRESS: Since the most recent compression (or since the start of the conversation if none), you have either obtained a new concrete fact (name, date, URL, or quoted claim) OR refined the sub-question being pursued. If Y, name the new fact or refined sub-question. If N, state that you are returning the same state you compressed from.

N1 STUCK: At least 3 of your last 4 search queries returned no new URL or fact (i.e., were duplicates or returned already-known content). If you have made fewer than 4 searches total, answer N. If Y, name 1 distinct strategy you have NOT yet tried (different tool, different query type, different angle on the question). If N, name one new URL or fact obtained recently.

Answer C1, C2, C3, N1 honestly. Each Y answer requires verbatim evidence quoted from the trajectory above; answers without evidence default to N.

Output: exactly 4 lines, no preamble or trailing text.
C1: Y/N -- <evidence>
C2: Y/N -- <if Y: 1. fact "citation" | 2. fact "citation" |
3. fact "citation"; if N: <class of info lost>>
C3: Y/N -- <evidence>
N1: Y/N -- <evidence>
Fire rule: COMPRESS iff C1 = Y ∧ C2 = Y ∧ C3 = Y ∧ N1 = N; otherwise CONTINUE.
```

### B.6 Search Summarizer Prompt

**(S instruction appended to $(x, y_{1:t})$)**

```
You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history above and the user's original question to provide a summary that will REPLACE the full conversation history when you continue.

## Task Guidelines

## 1. Information Analysis:

• Carefully analyze the conversation history to identify truly useful information.

• Focus on information that directly contributes to answering the question.

• Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated in the conversation.

• If information is missing or unclear, do NOT include it in your summary.

## 2. Summary Requirements:

• Extract only the most relevant information that is explicitly present in the conversation.

• Synthesize information from multiple exchanges when relevant.

• Only include information that is certain and clearly stated in the conversation.

• Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed from the conversation.
```

---

## C. Cost Analysis of Summarization

### C.1 Cost Model

For all OpenRouter-hosted agents we charge each question with the additive token-cost:

$$
\mathrm{COST}(q) = \frac{1}{10^6} \Big( p_{\mathrm{cache}} \cdot N_{\mathrm{prompt}} + p_{\mathrm{out}} \cdot N_{\mathrm{out}} \Big),
$$

where $N_{\mathrm{prompt}}$ is the cumulative prompt tokens (`usage.prompt_tokens`) summed across every LLM call in the trajectory — assistant turns, the rubric probe, and the summarizer call — and $N_{\mathrm{out}}$ is the cumulative completion tokens (`usage.completion_tokens`) over the same set of calls. We charge every prompt token at the cache rate $p_{\mathrm{cache}}$ rather than separately tracking which tokens were a cache hit and which were a first-time prefill. This is a conservative single-rate approximation: for trajectories with many turns over a slowly growing prefix, the first-time prefill tokens (cumulative $\approx N_{\mathrm{out}}$) are a small fraction of $N_{\mathrm{prompt}}$, so the effective per-token rate sits close to $p_{\mathrm{cache}}$ even when provider-side caching misses occasionally. Per-1M token prices are listed in Table 6.

### C.2 Calculation Procedure

For each of the 150 sampled questions per benchmark, we (i) load the trajectory's per-call usage records logged by the AggAgent runtime, (ii) accumulate $(N_{\mathrm{prompt}}, N_{\mathrm{out}})$ as defined above, (iii) apply the model's prices from Table 6, and (iv) report the mean across the 150 questions in Table 4 (units: USD per question).

**Table 6: OpenRouter prices in USD per 1,000,000 tokens.**

| Model | $p_{\mathrm{in}}$ | $p_{\mathrm{cache}}$ | $p_{\mathrm{out}}$ |
|-------|-------------------|----------------------|---------------------|
| GLM-4.7-Flash | 0.07 | 0.01 | 0.40 |
| MiMo-V2-Flash | 0.10 | 0.01 | 0.30 |
| MiniMax-M2.5 | 0.30 | 0.03 | 1.20 |

### C.3 Where Compaction Saves Cost

Under the single-rate cache billing above, the savings from compaction come entirely from shortening every subsequent call's prompt. After a fire that collapses a $T$-token prefix to a $\tilde{t}$-token summary, every later call pays $p_{\mathrm{cache}} \cdot \tilde{t}$ instead of $p_{\mathrm{cache}} \cdot T$, a $T/\tilde{t}$ ratio per call. Empirically WebResummer collapses a 50–100k trajectory into $\tilde{t} \approx 1{-}3\text{k}$, giving **20–80× shrinkage** of the post-compact prompt. The rubric probe and summarizer both append to the live prefix so the running prompt is preserved during the probe itself; the probe contributes only its own ~60-token verdict to $N_{\mathrm{out}}$, and the summarizer contributes $\tilde{t}$ tokens once, both negligible relative to the future-call savings amortized across the rest of the trajectory.

**Math.** The Qwen runs are served locally with vLLM, so cost collapses to a wall-clock trade-off rather than a dollar one. We report the per-question budget tag [Xk] in Table 1: the average sum of generated tokens (rounds × num_tokens) plus refinement tokens across the $n = 16$ samples. FIXED-INTERVAL SUMMARY's budget is matched to SELFCOMPACT within ±3k tokens so all four rows in each model block compete at identical compute.

---

## D. Trajectory Token Consumption

### D.1 Token Lifecycle

A trajectory under SELFCOMPACT looks like $Q \; A_1 \; A_2 \; \cdots \; A_k \; S \; A_1' \; A_2' \; \cdots$, where $Q$ is the original question (plus system prompt), each $A_t$ is one assistant turn (model output for tool call or final answer), and $S$ is a summary the model emits when the rubric fires COMPRESS. Each token a model produces is billed three different ways over its lifetime:

- **Output** — when the model first emits the tokens (one charge).
- **Prefill** (uncached input) — on the immediately next LLM call, when the tokens are appended to the prefix and seen by the provider for the first time (one charge).
- **Cached input** — on every subsequent LLM call until the prefix is reset (one charge per reuse).

Concretely, a single $A_t$ at round $t$ in a trajectory of $R$ rounds contributes once to Output, once to Prefill, and $R - t$ times to Cached. A summary $S$ emitted at compaction is billed once as output, once as prefill on the first post-compact call, then as cached input until the next compaction (or end of trajectory).

### D.2 Per-Question Token Totals

Table 7 reports the cumulative prompt and completion tokens summed across all LLM calls (assistant turns, rubric probes, summarizer calls) for each (model, method) cell on BrowseComp-Plus. Under the single-rate cache billing in Appendix C, each row's cost in Table 4 equals $(p_{\mathrm{cache}} \bar{N}_{\mathrm{prompt}} + p_{\mathrm{out}} \bar{N}_{\mathrm{out}}) / 10^6$ with the prices in Table 6. We hold $N_{\mathrm{out}}$ fixed at SELFCOMPACT's value per model so that across the three methods the only varying quantity is the cumulative prompt tokens, which makes the savings from compaction directly readable as the difference in the $N_{\mathrm{prompt}}$ column.

**Table 7: Per-question cumulative prompt and completion tokens on BrowseComp-Plus**, with $N_{\mathrm{out}}$ held at SELFCOMPACT's value to isolate prompt-side savings. Multiplied through the prices in Table 6, each row reproduces the corresponding cell of Table 4.

| Model | Method | $N_{\mathrm{prompt}}$ | $N_{\mathrm{out}}$ | Cost (Table 4) |
|-------|--------|----------------------|--------------------|----------------|
| **GLM-4.7-Flash** | No Compaction | 11.6M | 8.9k | $0.12 |
| | Fixed-interval | 3.6M | 8.9k | $0.04 |
| | SELFCOMPACT | 3.6M | 8.9k | $0.04 |
| **MiniMax-M2.5** | No Compaction | 5.7M | 16.7k | $0.19 |
| | Fixed-interval | 0.7M | 16.7k | $0.04 |
| | SELFCOMPACT | 1.7M | 16.7k | $0.07 |
| **Mimo-V2-Flash** | No Compaction | 23.8M | 7.0k | $0.24 |
| | Fixed-interval | 13.8M | 7.0k | $0.14 |
| | SELFCOMPACT | 15.8M | 7.0k | $0.16 |

**Table 8: Per-question token consumption for the math benchmarks** (IMO-Answerbench / HMMT Nov 25 / HMMT Feb 26 averaged). Prompt is the cumulative refinement summaries fed back into the next round; output is model generation per round. The Output column matches the [Xk] tag in Table 1; Total is Prompt + Output.

| Method | Prompt (cum.) | Output | Total |
|--------|---------------|--------|-------|
| **Qwen3-4B-Instruct-2507** | | | |
| No Compaction | 1.2k | 16.0k | 17k |
| Fixed-Interval | 7.1k | 44.1k | 51k |
| SELFCOMPACT | 7.9k | 47.9k | 56k |
| **Qwen3-30B-A3B-Instruct-2507** | | | |
| No Compaction | 1.2k | 16.0k | 17k |
| Fixed-Interval | 4.1k | 25.6k | 30k |
| SELFCOMPACT | 4.6k | 28.6k | 33k |
| **Qwen3.5-9B (Thinking Disabled)** | | | |
| No Compaction | 1.2k | 16.0k | 17k |
| Fixed-Interval | 14.4k | 89.0k | 103k |
| SELFCOMPACT | 15.0k | 92.5k | 107k |
| **Qwen3.5-4B (Thinking Disabled)** | | | |
| No Compaction | 1.2k | 16.0k | 17k |
| Fixed-Interval | 10.1k | 63.7k | 74k |
| SELFCOMPACT | 10.7k | 66.6k | 78k |
