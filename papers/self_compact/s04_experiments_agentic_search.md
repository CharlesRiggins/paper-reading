# Self-Compacting Language Model Agents — Agentic Search Experiments

## 4.2 Agentic Search

### 4.2.1 Setup

The paper evaluates three deployed agents of different sizes and model families:

| Model | Size description |
|---|---|
| GLM-4.7-Flash | 30B total, 3B active |
| MiniMax-M2.5 | 230B total, 10B active |
| MiMo-V2-Flash | 309B total, 15B active |

The tasks are:

- **BrowseComp**;
- **BrowseComp-Plus**;
- **DeepSearchQA**.

The experiments use the same scaffold and evaluation settings as Lee et al. (2026). Detailed sampling parameters, context windows, and prompts are retained in `s07_appendix_checklist.md`.

### 4.2.2 Cost formula

For OpenRouter-hosted agents, per-question USD cost is computed as:

$$
\mathrm{COST}(q)=\frac{p_{\mathrm{in}}N_{\mathrm{uncached}} + p_{\mathrm{cache}}N_{\mathrm{cached}} + p_{\mathrm{out}}N_{\mathrm{out}}}{10^6},
$$

where the token totals are summed across every LLM call in the trajectory, including assistant turns, rubric probes, and summaries.

The paper explains the token lifecycle:

- each generated token is billed once as output;
- it is billed once as prefill on the next call;
- it is billed as cached input on later calls until the prefix is reset.

Appendix C gives a simplified single-rate cache accounting used for Table 4:

$$
\mathrm{COST}(q)=\frac{p_{\mathrm{cache}}N_{\mathrm{prompt}} + p_{\mathrm{out}}N_{\mathrm{out}}}{10^6}.
$$

### 4.2.3 Baselines

SELFCOMPACT is compared with four context-management regimes:

| Method | Description |
|---|---|
| No Compaction | No context management; model stops when max context window is reached or max tool calls are executed. |
| Fixed-interval summary | Summarization triggers when 30% of the maximum context window is consumed. This follows the threshold suggested by Liu et al. (2025) and resembles prior fixed-schedule scaffolds. |
| Delete-all | All past trajectory is discarded when 30% of the max context window is consumed. |
| Keep-last-N | The last three turns are saved and the rest is discarded when 30% of the max context window is consumed. |
| SELFCOMPACT | Rubric-gated summarization; compress only at closed, useful, non-stuck trajectory states. |

### 4.2.4 Table 4: performance and per-question cost

Accuracy is reported in percent, and cost is USD per question. The cost-reduction percentage in parentheses is relative to the no-compaction baseline for SELFCOMPACT. Evaluations use 150 subsampled questions per benchmark.

| Model | Context Management | BrowseComp Acc ↑ | BrowseComp Cost ↓ | BrowseComp-Plus Acc ↑ | BrowseComp-Plus Cost ↓ | DeepSearchQA Acc ↑ | DeepSearchQA Cost ↓ | Overall Acc ↑ | Overall Cost ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-4.7-Flash | No Compaction | 30.1 | 0.14 | 45.6 | 0.12 | 34.2 | 0.13 | 36.6 | 0.13 |
| GLM-4.7-Flash | Fixed-interval | 35.4 | 0.06 | 50.0 | 0.04 | 39.1 | 0.05 | 41.5 | 0.05 |
| GLM-4.7-Flash | Delete-all | 31.6 | 0.03 | 47.3 | 0.02 | 35.8 | 0.03 | 38.2 | 0.03 |
| GLM-4.7-Flash | Keep-last-N | 33.5 | 0.05 | 51.4 | 0.03 | 38.8 | 0.04 | 41.2 | 0.04 |
| GLM-4.7-Flash | SELFCOMPACT | **41.2** | 0.10 (-29%) | **54.1** | 0.04 (-67%) | **44.0** | 0.07 (-46%) | **46.4** | 0.07 (-46%) |
| MiniMax-M2.5 | No Compaction | 47.2 | 0.19 | 62.0 | 0.19 | 52.0 | 0.19 | 54.6 | 0.19 |
| MiniMax-M2.5 | Fixed-interval | 55.3 | 0.07 | 65.9 | 0.04 | 56.7 | 0.06 | 59.3 | 0.06 |
| MiniMax-M2.5 | Delete-all | 52.7 | 0.05 | 65.0 | 0.03 | 54.9 | 0.04 | 57.5 | 0.04 |
| MiniMax-M2.5 | Keep-last-N | 54.5 | 0.06 | 67.4 | 0.05 | 57.0 | 0.06 | 59.6 | 0.06 |
| MiniMax-M2.5 | SELFCOMPACT | **59.3** | 0.09 (-53%) | **71.2** | 0.07 (-63%) | **61.3** | 0.08 (-58%) | **63.9** | 0.08 (-58%) |
| MiMo-V2-Flash | No Compaction | 42.7 | 0.27 | 57.6 | 0.24 | 46.5 | 0.25 | 48.9 | 0.25 |
| MiMo-V2-Flash | Fixed-interval | 51.6 | 0.14 | 60.2 | 0.14 | 52.3 | 0.14 | 54.7 | 0.14 |
| MiMo-V2-Flash | Delete-all | 45.5 | 0.08 | 61.2 | 0.11 | 49.7 | 0.09 | 52.1 | 0.09 |
| MiMo-V2-Flash | Keep-last-N | 49.8 | 0.06 | **63.5** | 0.15 | 53.0 | 0.10 | 55.4 | 0.10 |
| MiMo-V2-Flash | SELFCOMPACT | **57.9** | 0.11 (-59%) | 62.9 | 0.16 (-33%) | **56.8** | 0.13 (-48%) | **59.2** | 0.13 (-48%) |

### 4.2.5 Main results

SELFCOMPACT is the strongest overall method for all three deployed agents. On BrowseComp-Plus, it improves over no compaction by:

| Model | No Compaction | SELFCOMPACT | Absolute gain |
|---|---:|---:|---:|
| GLM-4.7-Flash | 45.6 | 54.1 | +8.5 |
| MiniMax-M2.5 | 62.0 | 71.2 | +9.2 |
| MiMo-V2-Flash | 57.6 | 62.9 | +5.3 |

The broad accuracy ordering is:

$$
\text{No Compaction} < \text{Fixed-interval} \le \text{SELFCOMPACT}.
$$

Fixed-interval summarization recovers much of the baseline’s lost ground, but SELFCOMPACT’s autonomous trigger adds up to +6.3 points by avoiding poorly timed compressions and concentrating summaries at meaningful checkpoints.

### 4.2.6 Cost results

Despite issuing an extra LLM call at each probe, SELFCOMPACT is cheaper than the no-compaction baseline. On BrowseComp-Plus, per-question cost drops by:

| Model | No Compaction cost | SELFCOMPACT cost | Reduction |
|---|---:|---:|---:|
| GLM-4.7-Flash | 0.12 | 0.04 | 67% |
| MiniMax-M2.5 | 0.19 | 0.07 | 63% |
| MiMo-V2-Flash | 0.24 | 0.16 | 33% |

The paper attributes this to KV-cache reuse and the post-summary regime. Rubric judgments contribute only short generations because the trajectory cache is preserved. Most savings come after summarization, when future calls attend to $\tilde{y}$ rather than the full $y_{1:t}$.

### 4.2.7 SELFCOMPACT fires earlier than fixed thresholds

Figure 2 plots the number of accumulated tokens before each summary fires on BrowseComp-Plus. Fixed-interval compression pins triggers at the 30%-of-context threshold by construction. SELFCOMPACT’s rubric-fired summaries skew to the left of that threshold for all three models.

The interpretation is that the rubric often reaches a “compress now” verdict well before the trajectory consumes 30% of the context. A fixed threshold typically fires too late, retaining stale tokens after the model has already resolved the current sub-question.

### 4.2.8 SELFCOMPACT helps most on harder problems

Figure 3 bins each question by the total output tokens consumed by the no-compression baseline trajectory. This is used as a model-specific proxy for difficulty, split into five equal-size quantile bins.

Within each bin, the paper compares:

- no compaction;
- a fixed-budget threshold at 30% of context;
- SELFCOMPACT.

On easy bins, the three policies are within sampling noise. On the two hardest bins, SELFCOMPACT improves over the threshold baseline by 5–20 percentage points consistently across all three models. The paper interprets this as evidence that rubric-driven compression matters most for tasks requiring deep search and accumulating the largest contexts.
