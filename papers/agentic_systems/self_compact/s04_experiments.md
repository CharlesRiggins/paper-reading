## 4. Experiments

### 4.1 Multi-turn Reasoning in Competition Math

**Setup.** We evaluate 4 models of different sizes in the Qwen family — 2 instruct models `Qwen3-4B-Instruct-2507` and `Qwen3-30B-A3B-Instruct-2507`, and 2 Qwen3.5 [Qwen Team, 2026]: `Qwen3.5-4B` and `Qwen3.5-9B`, on 3 benchmarks: **IMO-Answerbench** [Luong et al., 2025], **HMMT Nov 2025** and **HMMT Feb 2026** [Balunović et al., 2025]. We generate 16 responses for each question in parallel and report the mean across samples. We use vLLM [Kwon et al., 2023] for inference and the `math_verify` package for parsing and evaluation.

We compare SELFCOMPACT against **fixed interval summary**, where summarization is always triggered for every 16k tokens or when the model finishes the generation. Using the same scaffold as ReasoningCache [Wu et al., 2026a], we prompt the model to continue generating based on an existing trajectory or a summary of it. Detailed summarization and continuation prompts can be found in Appendix A.

**Table 1: Performance across competition math benchmarks.** For fixed interval summary, we constrain the avg. number of tokens per question as the same as SELFCOMPACT. Numbers in brackets indicate the average number of tokens used per question.

| Model | Method | IMO-Answerbench | HMMT Nov 25 | HMMT Feb 26 |
|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507** | No Compaction [16k] | 38.9 | 40.8 | 36.5 |
| | Fixed Interval Summary [44k] | 41.4 | 44.0 | 39.2 |
| | SELFCOMPACT [48k] | **45.5** | **47.8** | **42.1** |
| **Qwen3-30B-A3B-Instruct-2507** | No Compaction [16k] | 45.2 | 54.7 | 51.9 |
| | Fixed Interval Summary [26k] | 48.7 | 57.3 | **58.7** |
| | SELFCOMPACT [29k] | **52.1** | **59.4** | 57.6 |
| **Qwen3.5-9B (Thinking Disabled)** | No Compaction [16k] | 25.0 | 38.2 | 34.2 |
| | Fixed Interval Summary [90k] | 33.4 | 42.1 | 44.9 |
| | SELFCOMPACT [93k] | **41.4** | **48.2** | **52.3** |
| **Qwen3.5-4B (Thinking Disabled)** | No Compaction [16k] | 16.9 | 26.4 | 22.5 |
| | Fixed Interval Summary [64k] | 27.4 | 35.5 | 29.1 |
| | SELFCOMPACT [67k] | **34.1** | **37.4** | **30.0** |

**Results.** Table 1 shows the results across three competition math benchmarks and four model configurations. Under matched token budgets, SELFCOMPACT consistently outperforms both the baseline and fixed interval summarization, achieving the **best performance in 11 out of 12 settings**. The gains are most pronounced for the thinking-enabled models: on `Qwen3.5-9B`, SELFCOMPACT improves over the baseline by **+16.4**, **+10.0**, and **+18.1 points** on IMO-Answerbench, HMMT Nov, and HMMT Feb respectively. The one exception is `Qwen3-30B-A3B` on HMMT Feb, where fixed interval summarization edges out SELFCOMPACT by 1.1 points, though the latter still leads on the other two benchmarks.

**Headroom analysis.** We analyze the failure modes of fixed interval summarization for `Qwen3-4B-Instruct-2507` on IMO-Answerbench by tracking answer transitions across the 12 summarization calls. Table 2 reports the number of instances where the model's answer changes from correct to wrong after a summarization and vice versa.

**Table 2: Answer transitions across 12 fixed interval summarization calls** for `Qwen3-4B-Instruct-2507` on IMO-Answerbench.

| Transition | Count |
|---|---|
| Wrong → Correct | 1486 |
| Correct → Wrong | 1009 |

While summarization produces a net gain, **40.4% of all transitions cause degradations**. This motivates a natural question: what if the model could selectively skip summarization when its current answer is already correct? Table 3 shows that an **oracle policy** which suppresses summarization calls whenever the current answer is correct but otherwise follows the same fixed schedule achieves **52.9%**, a **+11.5 improvement** over fixed interval and **+14.0 over baseline**. Notably, this oracle is a strict subset of what a fully adaptive policy could achieve, as it only decides whether to summarize at each fixed interval, not when to summarize.

**Table 3: Oracle analysis on IMO-Answerbench** for `Qwen3-4B-Instruct-2507`.

| Method | Accuracy (%) |
|---|---|
| No Compaction [16k] | 38.9 |
| Fixed Interval Summary [44k] | 41.4 |
| SELFCOMPACT [48k] | 45.5 |
| Oracle (skip if correct) [44k] | **52.9** |

Both our results on Qwen3.5, where adaptive summarization already outperforms fixed interval, and this oracle analysis indicate substantial headroom for adaptive policies that learn when summarization is beneficial.

### 4.2 Agentic Search

**Setup.** We evaluate 3 models of different sizes and families: `GLM-4.7-Flash` (30B total, 3B active; [GLM Team et al., 2025]), `MiniMax-M2.5` (230B total, 10B active; [MiniMax Team, 2026]) and `MiMo-V2-Flash` (309B total, 15B active; [MiMo Team et al., 2026]) on 3 agentic search tasks: **BrowseComp** [Wei et al., 2025], **BrowseComp-Plus** [Chen et al., 2025] and **DeepSearch QA** [Gupta et al., 2026]. We use the exact same scaffold and evaluation settings as Lee et al. [2026]. We defer detailed sampling parameters and prompts to Appendix B.

For cost, we report the per-question USD cost computed as:

$$
\text{COST}(q) = \left(p_{\text{in}} N_{\text{uncached}} + p_{\text{cache}} N_{\text{cached}} + p_{\text{out}} N_{\text{out}}\right) / 10^6,
$$

where the three token totals are summed across every LLM call in the trajectory (assistant turns, rubric probes, and summaries) and each generated token is billed once as output, once as prefill on the next call, then as cached input on every later call until the prefix is reset. Prices come from OpenRouter (Table 6). The full procedure — how $(N_{\text{uncached}}, N_{\text{cached}}, N_{\text{out}})$ are derived from each trajectory's per-call `prompt_tokens` sequence, including the bookkeeping at compaction resets — is given in Appendix C. We compare SELFCOMPACT against the following baseline methods:

- **No Compaction:** No context management is used; the model stops when either max context window is reached, or max number of tool calls are executed.
- **Fixed-interval summary:** Summarization is triggered when 30% of the max context window is consumed, a threshold suggested by Liu et al. [2025], closely resembling the scaffold in Wu et al. [2026a].
- **Delete-all:** All the past trajectory is discarded when 30% of the max context window is consumed.
- **Keep-last-N:** The last 3 turns are saved and the rest is discarded when 30% of the max context window is consumed.

**Main Results.** Table 4 shows the results of the three deployed agents (`GLM-4.7-Flash`, `MiniMax-M2.5`, `MiMo-V2-Flash`) on agentic search. **SELFCOMPACT is the strongest method for all three models**, improving over the no-compaction baseline by **+8.5** (GLM-4.7-Flash), **+9.2** (MiniMax-M2.5), and **+5.3** (MiMo-V2-Flash) absolute points on BrowseComp-Plus.

The accuracy ordering is consistent across all three models: baseline < fixed-interval ≤ SELFCOMPACT. Fixed-interval summarization recovers most of the baseline's lost ground; SELFCOMPACT's autonomous trigger adds up to **+6.3 points** by avoiding poorly-timed compressions — the rubric concentrates that distribution at meaningful checkpoints.

On the cost axis, **SELFCOMPACT is cheaper than the baseline** despite issuing an extra LLM call at each probe. Per-question cost on BrowseComp-Plus drops by **67%** (GLM-4.7-Flash, $0.12 \to 0.04$), **63%** (MiniMax-M2.5, $0.19 \to 0.07$), and **33%** (MiMo-V2-Flash, $0.24 \to 0.16$). The KV-cache reuse described in §3 is what makes the probe affordable: each rubric judgement contributes only its own short generation, since the trajectory's cache is preserved. The bulk of the savings come from the post-summarization regime, where every subsequent token attends to $\tilde{y}$ rather than $y_{1:t}$.

**Table 4: Performance and per-question cost across benchmarks.** "Cost" is USD per question, computed as $(p_{\text{cache}} N_{\text{prompt}} + p_{\text{out}} N_{\text{out}}) / 10^6$ where $N_{\text{prompt}}$ and $N_{\text{out}}$ are the cumulative prompt and completion tokens summed across every LLM call in the trajectory; the full bookkeeping is in Appendix C. We highlight the cost reduction % compared to the "no compaction" baseline for SELFCOMPACT. **Bold** indicates the best value per column within each model block. Following Sun et al. [2025] and Lee et al. [2026], evaluations are done on 150 subsampled questions per benchmark.

| Model | Context Management | BrowseComp Acc. ↑ | BrowseComp Cost ↓ | BrowseComp-Plus Acc. ↑ | BrowseComp-Plus Cost ↓ | DeepSearch QA Acc. ↑ | DeepSearch QA Cost ↓ | Overall Acc. ↑ | Overall Cost ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **GLM-4.7-Flash** | No Compaction | 30.1 | 0.14 | 45.6 | 0.12 | 34.2 | 0.13 | 36.6 | 0.13 |
| | Fixed-interval | 35.4 | 0.06 | 50.0 | 0.04 | 39.1 | 0.05 | 41.5 | 0.05 |
| | Delete-all | 31.6 | 0.03 | 47.3 | 0.02 | 35.8 | 0.03 | 38.2 | 0.03 |
| | Keep-last-N | 33.5 | 0.05 | 51.4 | 0.03 | 38.8 | 0.04 | 41.2 | 0.04 |
| | **SELFCOMPACT (Ours)** | **41.2** | 0.10 (−29%) | **54.1** | **0.04 (−67%)** | **44.0** | 0.07 (−46%) | **46.4** | 0.07 (−46%) |
| **MiniMax-M2.5** | No Compaction | 47.2 | 0.19 | 62.0 | 0.19 | 52.0 | 0.19 | 54.6 | 0.19 |
| | Fixed-interval | 55.3 | 0.07 | 65.9 | 0.04 | 56.7 | 0.06 | 59.3 | 0.06 |
| | Delete-all | 52.7 | 0.05 | 65.0 | 0.03 | 54.9 | 0.04 | 57.5 | 0.04 |
| | Keep-last-N | 54.5 | 0.06 | 67.4 | 0.05 | 57.0 | 0.06 | 59.6 | 0.06 |
| | **SELFCOMPACT (Ours)** | **59.3** | 0.09 (−53%) | **71.2** | 0.07 (−63%) | **61.3** | 0.08 (−58%) | **63.9** | 0.08 (−58%) |
| **MiMo-V2-Flash** | No Compaction | 42.7 | 0.27 | 57.6 | 0.24 | 46.5 | 0.25 | 48.9 | 0.25 |
| | Fixed-interval | 51.6 | 0.14 | 60.2 | 0.14 | 52.3 | 0.14 | 54.7 | 0.14 |
| | Delete-all | 45.5 | 0.08 | 61.2 | 0.11 | 49.7 | 0.09 | 52.1 | 0.09 |
| | Keep-last-N | 49.8 | 0.06 | 63.5 | 0.15 | 53.0 | 0.10 | 55.4 | 0.10 |
| | **SELFCOMPACT (Ours)** | **57.9** | 0.11 (−59%) | **62.9** | 0.16 (−33%) | **56.8** | 0.13 (−48%) | **59.2** | 0.13 (−48%) |

**SELFCOMPACT fires compression earlier.** We plot the number of tokens accumulated before each compaction fires in Figure 2, comparing Fixed-Interval (30% of context) against SELFCOMPACT on BrowseComp-Plus. SELFCOMPACT's rubric-fired summaries skew to the left of the 30% threshold line in all three models, whereas Fixed-Interval pins triggers at the threshold by construction. This left skew indicates that the rubric reaches its "compress now" verdict well before the trajectory consumes 30% of context. This indicates the fixed threshold typically fires too late, retaining stale tokens past the point where the model has already resolved the current sub-question.

**SELFCOMPACT helps on harder problems.** We bin each question by the total output tokens its no-compression baseline trajectory consumed (a model-specific proxy for question difficulty) and split into five equal-size quantiles per model. Within each bin we report accuracy for the baseline, a fixed-budget Threshold (30% of context), and SELFCOMPACT (Figure 3). On easy bins the three policies are within sampling noise, but on the two hardest bins SELFCOMPACT improves over the Threshold baseline by **5–20 pp** consistently across 3 models, indicating that rubric-driven compression helps the most on questions that require deep search and accumulate the largest contexts.

![Figure 2: Distribution of context length when a summary fires in BrowseComp Plus. Top: Fixed-Interval — the policy compresses at a hard 30%-of-max-context budget (red dashed). Bottom: SELFCOMPACT — the rubric fires when the model judges a sub-question resolved. Fixed-Interval pins triggers at the threshold by construction; SELFCOMPACT spreads them across the budget.](images/235ecc6fea7a62fa4ee45d3ff055c05b93e8325b71f5570c70d184e1f42cac64.jpg)

![Figure 3: Accuracy by per-question difficulty on BrowseComp Plus for three models. Difficulty is the total output tokens consumed by the no-compression baseline, split into 5 equal-size quantile bins per model (left = easy, right = hard). Bars compare Baseline (no compression), a fixed-budget Threshold (30% of context), and SELFCOMPACT. SELFCOMPACT matches the others on easy bins and pulls ahead on the hardest bins across all three models.](images/8ca326aec9b01fbaff43052e2521ff8fbabce0810a57f79cdd0bdfc230ecaf38.jpg)
