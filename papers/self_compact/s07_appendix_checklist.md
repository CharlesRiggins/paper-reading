# Self-Compacting Language Model Agents — Appendices and Checklist

## Appendix A. Math experimental setup

### A.1 Models

The math experiments evaluate four open-weight Qwen models:

| Category | Models | Serving |
|---|---|---|
| Instruction-tuned | Qwen3-4B-Instruct-2507; Qwen3-30B-A3B-Instruct-2507 | vLLM |
| Thinking-disabled | Qwen3.5-4B; Qwen3.5-9B | vLLM |

All checkpoints are loaded from Hugging Face and served via vLLM. The Qwen3-30B-A3B model uses tensor-parallel size 2; all other models run on a single H200 GPU.

### A.2 Benchmarks

The math benchmarks are:

- IMO-Answerbench;
- HMMT November 2025;
- HMMT February 2026.

Final answers are extracted and scored with `math_verify`.

### A.3 Sampling parameters

| Parameter | Value |
|---|---:|
| Temperature | 1.0 |
| Top-p | 0.7 |
| Samples per question | $n=16$ |
| Generation cap per round | 16,384 tokens |
| Per-trajectory round budget | `--max_rounds 12` |
| Summary hard cap | `--max_summary_tokens 512` |

The paper reports mean performance across samples.

### A.4 Scaffold and methods

The authors adopt the ReasoningCache scaffold. At each round, the model is prompted either to solve from scratch or improve on a summary of the previous attempt.

| Method | Implementation detail |
|---|---|
| BASELINE | Single 16k-token budget (`--max_rounds 1`) and no summarization. |
| FIXED-INTERVAL SUMMARY | `--force_tool_calls`; fires the summarizer at the end of every round, i.e. every 16,384 generated tokens, regardless of trajectory state. |
| SELFCOMPACT | Exposes `<tool_call>` as a vLLM stop token so the model can autonomously emit the summarizer; the round ends and summary fires when the stop token appears. |
| SELFCOMPACT w/ RUBRICS | Interposes the rubric probe between rounds; if the rubric fires COMPRESS, the summarizer is invoked; otherwise the model continues from the unmodified prefix. |

### A.5 Math rubric prompt

The math rubric is appended to $(x,y_{1:t})$ at every round boundary and asks for ANSWER / STUCK / HAS-NEXT judgments.

```text
Judge your math-solving state from the conversation above. Answer Q1...Q3 with Y or N and one short
evidence phrase. Answers without evidence must be N. “Non-trivial fact” = a new equation, reduction,
bound, case-elimination, or counterexample, not a paraphrase.

Q1 ANSWER: The latest round states a specific final answer (a \boxed{} expression or “Final Answer:
...”), not merely a partial result. If Y, quote the answer verbatim. If N, state what is still unknown.

Q2 STUCK: Your last 2 rounds added no non-trivial fact — only paraphrases or abandoned attempts. If Y,
name the two rounds and write “no new fact”. If N, name one non-trivial fact from the last 2 rounds.

Q3 HAS-NEXT: You can write the exact next step (case split, substitution, verification, lemma to prove).
If Y, write the step as one imperative sentence. If N, write “NONE”.

Output: exactly 3 lines, no preamble or trailing text.
Q1: Y/N -- <evidence>
Q2: Y/N -- <evidence>
Q3: Y/N -- <evidence>

Fire rule: COMPRESS iff Q1 = Y ∨ (Q2 = Y ∧ Q3 = Y). Branch A (Q1 = Y) fires a lock-in re-prompt
that preserves the boxed answer; Branch B (Q2 = Y ∧ Q3 = Y) fires a reset that preserves the named next
step.
```

### A.6 Math summarizer prompt

```text
Context Refinement Prompt:
Original Prompt: {original_prompt}
Partial Generation: {partial_generation}
Your task is to create a compressed summary for another model to continue solving from.

RULES:
1. If a final answer (e.g., \boxed{}) was found, PRESERVE IT at the end of your summary.
2. Keep key insights, important calculations, and the reasoning path.
3. Remove redundant text, false starts, and unnecessary repetition.
4. If the answer seems wrong or unverified, note that verification is needed.
5. Be concise but preserve all critical mathematical steps.

Output format:
• Key insights and progress made.
• Important intermediate results.
• If found: “Final Answer: [the answer]” or the \boxed{} expression.
• If not solved: what still needs to be done.
```

### A.7 Continuation prompt

```text
You are given a maths problem. You may also be given a summary of a previous attempt to solve it. This
previous attempt may or may not be correct.

### PROBLEM {problem}
### SUMMARY OF PREVIOUS ATTEMPT {summary}
### INSTRUCTIONS
If no summary of a previous attempt is provided, solve the problem from scratch.
If a summary of a previous attempt is provided, your task is to improve upon this attempt. You should rely
on this summary to guide your thinking. Some strategies you could use include:
• Verifying the previous solution.
• Proving the result in a different way.
• Finding alternative problem-solving strategies.
• Continuing from where the previous solution left off, assuming that the previous solution is incomplete.

Reason step-by-step and return your final answer in \boxed{}.
```

## Appendix B. Agentic-search experimental setup

### B.1 Models and serving

The three deployed agents—GLM-4.7-Flash, MiniMax-M2.5, and MiMo-V2-Flash—are accessed through OpenRouter via the AggAgent runtime. The authors use the unmodified `react_agent_selfcheck` scaffold.

### B.2 Benchmarks

The search benchmarks are:

- BrowseComp;
- BrowseComp-Plus;
- DeepSearchQA.

Following prior work, the paper subsamples 150 questions per benchmark.

### B.3 Sampling parameters

All three models use:

| Parameter | Value |
|---|---:|
| Temperature | 1.0 |
| Top-p | 0.95 |
| Max output tokens per call | 10,000 |
| Parallel tool calls | Disabled |
| Per-trajectory cap | 100 LLM calls |

### B.4 Table 5: context windows, 30% triggers, and flags

| Model | Context window | 30% trigger | Model-specific flags |
|---|---:|---:|---|
| GLM-4.7-Flash | 128,000 | 38,400 | `enable_thinking=True`, `clear_thinking=False` |
| MiniMax-M2.5 | 196,608 | 58,982 | `reasoning_split=True` |
| MiMo-V2-Flash | 262,144 | 78,643 | default OpenRouter body |

GLM-4.7-Flash keeps the thinking trace visible to the rubric probe via `enable_thinking=True` and `clear_thinking=False`.

### B.5 Scaffold details

A single trajectory follows the standard ReAct loop. The agent emits a tool call, receives the result, reasons, and either issues another tool call or commits a final answer.

| Benchmark | Tools |
|---|---|
| BrowseComp / DeepSearchQA | `search` backed by Serper API; `visit` fetches and extracts content via crawl4ai. |
| BrowseComp-Plus | `search` and `get_document_bcp` resolve against the benchmark’s released corpus. |

The rubric is appended to a copy of the trajectory at each round boundary so the probe does not pollute the rolling cache.

In the search setup, COMPRESS fires only when four gates pass simultaneously:

| Gate | Condition |
|---|---|
| ROUND | iteration ≥ 3 |
| TOKENS | running prompt length ≥ 40,000 |
| CAP | total summaries so far < 1 |
| PERIOD | ≥ 2 rounds elapsed since the last probe |

A token-percentage backstop forces COMPRESS once the prompt crosses $0.30 \cdot \mathrm{ctx\_window}$ regardless of rubric.

### B.6 Search rubric probe

```text
You are about to decide whether to compress your conversation history into a summary that REPLACES the
full history above. After compression, you continue research from only [system, original_question,
assistant_summary, user_continue]. Compression is irreversible: anything not preserved in the
summary is gone.

Compression is safe ONLY when ALL FOUR of the following hold:
(C1) the trajectory has reached a closed unit (not mid-thought),
(C2) the essential information is reducible to 3–5 cite-able facts without loss,
(C3) something has progressed since the last compression,
(N1) you are NOT currently stuck in a way summarization would mask.

Answer C1, C2, C3, N1 honestly. Each Y answer requires verbatim evidence quoted from the trajectory
above; answers without evidence default to N.

C1 CLOSED-UNIT: The most recent assistant message is a closed unit — a completed tool call whose
result is now visible, or a completed sub-analysis with a clear stopping point. It is NOT mid-sentence
reasoning (“Let me now check...”, “I should next look at...”), and not a half-formulated query. If Y, quote
the closing fragment of the last assistant message. If N, quote the open fragment that shows the trajectory is
mid-thought.

C2 SUMMARIZABLE: You can write 3–5 essential facts (with verbatim citations from the trajectory)
that future-you needs to continue research after compression. Each fact must be a single concrete statement:
a name, date, URL, quoted claim, or resolved sub-question. Answer N if the trajectory’s value is dispersed
across many small inferences (e.g., a list of dead-end queries needed to avoid retries, negative results that
constrain hypothesis space) that would be lost without the dispersal. If Y, list the 3–5 facts numbered, each
with a verbatim citation in quotes, separated by “ | ”. If N, name in one sentence the class of information
that would be lost.

C3 PROGRESS: Since the most recent compression (or since the start of the conversation if none), you
have either obtained a new concrete fact (name, date, URL, or quoted claim) OR refined the sub-question
being pursued. If Y, name the new fact or refined sub-question. If N, state that you are returning the same
state you compressed from.

N1 STUCK: At least 3 of your last 4 search queries returned no new URL or fact (i.e., were duplicates or
returned already-known content). If you have made fewer than 4 searches total, answer N. If Y, name
1 distinct strategy you have NOT yet tried (different tool, different query type, different angle on the
question). If N, name one new URL or fact obtained recently.

Output: exactly 4 lines, no preamble or trailing text.
C1: Y/N -- <evidence>
C2: Y/N -- <if Y: 1. fact "citation" | 2. fact "citation" | 3. fact "citation"; if N: <class of info lost>>
C3: Y/N -- <evidence>
N1: Y/N -- <evidence>

Fire rule: COMPRESS iff C1 = Y ∧ C2 = Y ∧ C3 = Y ∧ N1 = N; otherwise CONTINUE.
```

### B.7 Search summarizer prompt

The search summarizer is the WebResummer variant of Wu et al. (2026b), used unmodified for all three deployed agents. The parsed text preserves the following portion:

```text
You are an expert at analyzing conversation history and extracting relevant information. Your task is to
thoroughly evaluate the conversation history above and the user’s original question to provide a summary
that will REPLACE the full conversation history when you continue.

Task Guidelines
1. Information Analysis:
• Carefully analyze the conversation history to identify truly useful information.
• Focus on information that directly contributes to answering the question.
• Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated in the conversation.
• If information is missing or unclear, do NOT include it in your summary.

2. Summary Requirements:
• Extract only the most relevant information that is explicitly present in the conversation.
• Synthesize information from multiple exchanges when relevant.
• Only include information that is certain and clearly stated in the conversation.
• Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed from the conversation.
```

## Appendix C. Cost analysis of summarization

### C.1 Cost model

For OpenRouter-hosted agents, the paper charges each question as:

$$
\mathrm{COST}(q)=\frac{1}{10^6}\left(p_{\mathrm{cache}}\cdot N_{\mathrm{prompt}} + p_{\mathrm{out}}\cdot N_{\mathrm{out}}\right),
$$

where $N_{\mathrm{prompt}}$ is cumulative prompt tokens and $N_{\mathrm{out}}$ is cumulative completion tokens summed across assistant turns, rubric probes, and summarizer calls.

The authors charge every prompt token at the cache rate $p_{\mathrm{cache}}$ rather than separately tracking cache hits and first-time prefill. This is described as a conservative single-rate approximation: in long many-turn trajectories, first-time prefill tokens are a small fraction of cumulative prompt tokens.

### C.2 Calculation procedure

For each of the 150 sampled questions per benchmark, the paper:

1. loads per-call usage records logged by AggAgent;
2. accumulates $(N_{\mathrm{prompt}}, N_{\mathrm{out}})$;
3. applies the model’s prices from Table 6;
4. reports the mean across the 150 questions.

### C.3 Table 6: OpenRouter prices

Prices are USD per 1,000,000 tokens.

| Model | $p_{\mathrm{in}}$ | $p_{\mathrm{cache}}$ | $p_{\mathrm{out}}$ |
|---|---:|---:|---:|
| GLM-4.7-Flash | 0.07 | 0.01 | 0.40 |
| MiMo-V2-Flash | 0.10 | 0.01 | 0.30 |
| MiniMax-M2.5 | 0.30 | 0.03 | 1.20 |

### C.4 Where compaction saves cost

Under single-rate cache billing, compaction saves cost by shortening every later call’s prompt. If a fire collapses a $T$-token prefix to a $\tilde{t}$-token summary, each later call pays $p_{\mathrm{cache}}\cdot\tilde{t}$ instead of $p_{\mathrm{cache}}\cdot T$, a $T/\tilde{t}$ ratio per call.

Empirically, WebResummer collapses a 50–100k trajectory into $\tilde{t}\approx 1$–3k tokens, giving roughly $20\times$–$80\times$ shrinkage. The rubric probe contributes only an approximately 60-token verdict, and the summarizer contributes $\tilde{t}$ tokens once; both are small relative to the amortized future-call savings.

### C.5 Math cost framing

The Qwen runs are served locally with vLLM, so the paper reports token budgets rather than dollar costs. The `[Xk]` tags in Table 1 are the average sum of generated tokens plus refinement tokens across the $n=16$ samples. Fixed-interval budgets are matched to SELFCOMPACT within ±3k tokens.

## Appendix D. Trajectory token consumption

### D.1 Token lifecycle

A SELFCOMPACT trajectory has the form:

$$
Q \to A_1 \to A_2 \to \cdots \to A_k \to S \to A'_1 \to A'_2 \to \cdots,
$$

where $Q$ is the original question plus system prompt, each $A_t$ is an assistant turn, and $S$ is a summary emitted when the rubric fires COMPRESS.

Each produced token has three billing roles:

| Role | When charged |
|---|---|
| Output | When the model first emits the token. |
| Prefill / uncached input | On the immediately next LLM call, when the token is appended to the prefix and seen by the provider for the first time. |
| Cached input | On every later LLM call until the prefix is reset. |

A single assistant turn $A_t$ in a trajectory of $R$ rounds contributes once to Output, once to Prefill, and $R-t$ times to Cached. A summary $S$ is billed once as output, once as prefill on the first post-compact call, and then as cached input until the next compaction or the end of the trajectory.

### D.2 Table 7: BrowseComp-Plus cumulative tokens

Table 7 holds $N_{\mathrm{out}}$ fixed at SELFCOMPACT’s value to isolate prompt-side savings. Multiplying by Table 6’s prices reproduces Table 4’s BrowseComp-Plus costs.

| Model | Method | $N_{\mathrm{prompt}}$ | $N_{\mathrm{out}}$ | Cost |
|---|---|---:|---:|---:|
| GLM-4.7-Flash | No Compaction | 11.6M | 8.9k | \$0.12 |
| GLM-4.7-Flash | Fixed-interval | 3.6M | 8.9k | \$0.04 |
| GLM-4.7-Flash | SELFCOMPACT | 3.6M | 8.9k | \$0.04 |
| MiniMax-M2.5 | No Compaction | 5.7M | 16.7k | \$0.19 |
| MiniMax-M2.5 | Fixed-interval | 0.7M | 16.7k | \$0.04 |
| MiniMax-M2.5 | SELFCOMPACT | 1.7M | 16.7k | \$0.07 |
| MiMo-V2-Flash | No Compaction | 23.8M | 7.0k | \$0.24 |
| MiMo-V2-Flash | Fixed-interval | 13.8M | 7.0k | \$0.14 |
| MiMo-V2-Flash | SELFCOMPACT | 15.8M | 7.0k | \$0.16 |

### D.3 Table 8: math token consumption

Per-question token consumption averaged over IMO-Answerbench, HMMT Nov 25, and HMMT Feb 26. Prompt is cumulative refinement summaries fed into the next round; output is model generation per round.

| Model | Method | Prompt (cum.) | Output | Total |
|---|---|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | No Compaction | 1.2k | 16.0k | 17k |
| Qwen3-4B-Instruct-2507 | Fixed-Interval | 7.1k | 44.1k | 51k |
| Qwen3-4B-Instruct-2507 | SELFCOMPACT | 7.9k | 47.9k | 56k |
| Qwen3-30B-A3B-Instruct-2507 | No Compaction | 1.2k | 16.0k | 17k |
| Qwen3-30B-A3B-Instruct-2507 | Fixed-Interval | 4.1k | 25.6k | 30k |
| Qwen3-30B-A3B-Instruct-2507 | SELFCOMPACT | 4.6k | 28.6k | 33k |
| Qwen3.5-9B (Thinking Disabled) | No Compaction | 1.2k | 16.0k | 17k |
| Qwen3.5-9B (Thinking Disabled) | Fixed-Interval | 14.4k | 89.0k | 103k |
| Qwen3.5-9B (Thinking Disabled) | SELFCOMPACT | 15.0k | 92.5k | 107k |
| Qwen3.5-4B (Thinking Disabled) | No Compaction | 1.2k | 16.0k | 17k |
| Qwen3.5-4B (Thinking Disabled) | Fixed-Interval | 10.1k | 63.7k | 74k |
| Qwen3.5-4B (Thinking Disabled) | SELFCOMPACT | 10.7k | 66.6k | 78k |

## Appendix E. Qualitative trajectories

The paper walks through four real SELFCOMPACT trajectories on BrowseComp-Plus: three on MiniMax-M2.5 and one on GLM-4.7-Flash. Iteration numbers are scaffold `token_lengths_each_step` indices. At each probe, the model returns C1 closed-unit, C2 summarizable, C3 progress, and N1 stuck, and the fire rule is:

$$
\mathrm{COMPRESS} \iff C1=Y \land C2=Y \land C3=Y \land N1=N.
$$

### E.1 Case A: good fire, correct answer

**Question.** The agent must identify a museum along a river added to the National Register of Historic Places in 1988, named after the last name of the individual who paid just over \$500 for land rights. The gold answer is **Cozens Ranch Museum**.

**What happened.** The agent issued 19 searches and one `get_document` call. Verdicts 1–11 all returned CONTINUE. The first COMPRESS fired at iteration 15 when all four conditions held, collapsing a 39.5k-token prefix to an 810-token summary, a compression ratio of about $49\times$. The model committed the correct answer two iterations later.

| Probe | Verdict | Evidence / interpretation |
|---|---|---|
| iter 11, prefix 28k | C1=Y, C2=N, C3=N, N1=Y ⇒ CONTINUE | Candidate exists, but key facts about land purchase and NRHP/date/river evidence remain dispersed. |
| iter 15, prefix 39.5k | C1=Y, C2=Y, C3=Y, N1=N ⇒ COMPRESS | C2 lists essential facts: “Cozens Ranch Museum,” “added to the National Register of Historic Places in 1988,” “located along the Fraser River,” and land-payment evidence. |

**Take-away.** The rubric did not fire at the first candidate appearance. It waited until dispersed evidence became reducible to cite-able facts. A 30% fixed trigger for MiniMax-M2.5 would have fired later at 58,982 tokens.

### E.2 Case B: rubric correctly suppresses, correct answer

**Question.** A multi-hop band-identification puzzle involving a singer born in June between 1927 and 1930 whose love song was covered by a band formed in 2001–2004. The gold answer is **JadaL**.

**What happened.** The trajectory ran for 99 iterations and repeatedly crossed the 30% threshold. The rubric returned CONTINUE on 25 high-token probes (>40k) because C1=N: the model was mid-search at the points where a threshold policy would have fired. It allowed only three compressions total, all when searches had genuinely closed.

| Probe | Verdict | Rubric reasoning |
|---|---|---|
| iter 18, prefix 52.7k | C1=N, C2=N, C3=N, N1=Y ⇒ CONTINUE | The model is mid-thought: “I need to identify famous singers born in June between 1927–1930. Let me search more specifically.” |
| iter 20, prefix 57.8k | C1=N, C2=N, C3=N, N1=Y ⇒ CONTINUE | Still searching for the singer’s birth date, without reaching a closed unit. |
| iter 38, prefix 47.3k after one earlier fire | C1=N, C2=N ⇒ CONTINUE | Search results have appeared but have not been analyzed into a conclusion. |

**Take-away.** A fixed-interval policy would have fired four to five extra times, each cutting an active multi-search reasoning thread. The C1 closed-unit gate is load-bearing: it permits the prefix to exceed a static threshold when the agent is genuinely mid-search.

### E.3 Case C: failure mode, incorrect answer

**Question.** A description of a Boer War officer with a specific physical appearance, captured at Mafeking, with land in a specific cemetery. The gold answer is **Frederik Christoffel Eloff**.

**What happened.** At iteration 5, with only 14.9k tokens, the model had converged on the wrong candidate: Sarel Johannes Eloff. The rubric fired COMPRESS with all four conditions met. The summary preserved this wrong candidate as essential information, and later searches were conditioned on that identity. Ten summaries fired, and the final answer was another wrong sibling, Stephanus Petrus Erasmus Eloff.

| Probe | Verdict | Consequence |
|---|---|---|
| iter 5, prefix 14.9k | C1=Y, C2=Y, C3=Y, N1=N ⇒ COMPRESS | Prefix 14,883 tokens → summary 833 tokens; summary opens by identifying Sarel Johannes Eloff as matching the description. |

**Take-away.** The rubric verifies structural readiness to compact, not factual correctness. If the model converges early on a plausible wrong identity, the rubric can preserve the wrong candidate and erase alternatives needed for recovery. The authors suggest adding a candidate-aware verifier before firing.

### E.4 Case D: GLM-4.7-Flash, single well-timed fire, correct answer

**Question.** The agent must answer a February 2017 article question about an animal injured after a car accident on a road built by someone married 199 years before the accident. The gold answer is the leopard’s post-accident weight: **37 kg**.

**What happened.** The agent issued 14 searches and four `get_document_bcp` calls across 23 iterations. The rubric was probed at three iterations and fired once, at iteration 12, collapsing a 31,199-token prefix to a 668-token summary, about $47\times$ compression. The model answered correctly shortly after. End-to-end cost was reported as \$0.011 rollout + \$0.007 tool, below the no-compaction per-question average.

| Probe | Verdict | Interpretation |
|---|---|---|
| iter 8, prefix 20.4k | C1=Y, C2=N, C3=N, N1=Y ⇒ CONTINUE | Closed search loop, but no progress facts and recent searches duplicated; compression would freeze a stuck state. |
| iter 10, prefix 25.8k | C1=N, C2=N, C3=N, N1=N ⇒ CONTINUE | The model is mid-thought (“Let me search for French...”). |
| iter 12, prefix 31.2k | C1=Y, C2=Y, C3=Y, N1=N ⇒ COMPRESS | Essential facts include leopard hit by car in Bainskloof Pass, broken back/internal injuries, and vet records reporting 37 kg post-accident weight. |

**Take-away.** Case D mirrors Case A on a different model and rubric configuration. The rubric suppresses two premature probes—one stuck, one mid-thought—and fires below the 30% threshold backstop of $0.30\cdot128{,}000=38{,}400$ tokens.

## NeurIPS Paper Checklist

The original parsed text includes the checklist instructions; they are omitted here as boilerplate. The substantive answers and justifications are retained in compact table form.

| # | Item | Answer | Justification retained from paper |
|---:|---|---|---|
| 1 | Claims | Yes | The abstract and introduction state the SELFCOMPACT scaffold, empirical gains across benchmarks/models at lower cost, and ablations showing the rubric is essential; these are matched by §4.1 and §4.2. |
| 2 | Limitations | Yes | §6 discusses evaluation only on open-weight models, task-specific summarizer/rubric engineering, and the training-free/no-RL scope. |
| 3 | Theory assumptions and proofs | N/A | The paper presents no theorems or formal proofs; Appendix C is token-cost accounting rather than a theoretical result. |
| 4 | Experimental result reproducibility | Yes | Algorithm 1, Appendix A/B model and benchmark settings, prompts, context windows, and Appendix C cost accounting provide reproduction details. |
| 5 | Open access to data and code | No | Code is not released with the submission, but benchmarks are public, models are open-weight or accessible through OpenRouter, and prompts/scaffold details are reproduced. |
| 6 | Experimental setting/details | Yes | The method is training-free; decoding parameters, sample counts, max tokens, context windows, and gating thresholds are listed. |
| 7 | Experiment statistical significance | No | Math reports means over $n=16$ samples and search follows prior work with 150 subsampled questions; error bars are omitted from main tables. |
| 8 | Experiments compute resources | Yes | Math uses H200 GPUs via vLLM; Qwen3-30B-A3B uses TP=2; search runs through OpenRouter with per-question costs and token budgets reported. |
| 9 | Code of ethics | Yes | The work uses released benchmarks and open-weight/hosted models, introduces no human-subject data, and preserves anonymity. |
| 10 | Broader impacts | N/A | SELFCOMPACT is an inference-time scaffold for context compaction and does not introduce a new model, dataset, or direct new capability beyond cost-efficient use of existing systems. |
| 11 | Safeguards | N/A | The paper releases neither pretrained model weights nor scraped data. |
| 12 | Licenses for existing assets | Yes | Models, benchmarks, and tooling are cited at first use and used under released terms without redistribution. |
| 13 | New assets | N/A | No new datasets, model weights, or downloadable assets are released; the new contributions are algorithmic and prompt/scaffold specifications. |
| 14 | Crowdsourcing and human subjects | N/A | The paper involves no crowdsourcing or human-subject research. |
| 15 | IRB approvals | N/A | No human subjects are involved, so no IRB review is required. |
| 16 | Declaration of LLM usage | Yes | LLMs are the central object of study; the same model acts as generator, rubric judge, and summarizer, and the seven evaluated models are named. |
