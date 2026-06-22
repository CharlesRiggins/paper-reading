# Self-Compacting Language Model Agents — Competition Math Experiments

## 4. Experiments

### 4.1 Multi-turn Reasoning in Competition Math

#### 4.1.1 Setup

The paper evaluates four Qwen-family models of different sizes:

| Model group | Models |
|---|---|
| Qwen3 instruct models | Qwen3-4B-Instruct-2507; Qwen3-30B-A3B-Instruct-2507 |
| Qwen3.5 thinking-disabled models | Qwen3.5-4B; Qwen3.5-9B |

The benchmarks are:

- **IMO-Answerbench**;
- **HMMT Nov 2025**;
- **HMMT Feb 2026**.

For each question, the authors generate 16 responses in parallel and report the mean across samples. Inference uses vLLM, and final-answer parsing/evaluation uses the `math_verify` package.

#### 4.1.2 Compared methods

The competition-math comparison focuses on:

| Method | Description |
|---|---|
| No Compaction | Single 16k-token budget; no summarization. |
| Fixed Interval Summary | Summarization always triggers every 16k tokens or when the model finishes generation. The fixed-interval budget is constrained to match SELFCOMPACT’s average token usage. |
| SELFCOMPACT | The model/rubric decides whether to summarize at each probe boundary. |

The setup follows the ReasoningCache scaffold: the model is prompted to continue generating based on either an existing trajectory or a summary of it. Detailed prompts are retained in `s07_appendix_checklist.md`.

#### 4.1.3 Table 1: performance across competition math benchmarks

Numbers in brackets indicate the average number of tokens used per question.

| Model | Method | IMO-Answerbench | HMMT Nov 25 | HMMT Feb 26 |
|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | No Compaction [16k] | 38.9 | 40.8 | 36.5 |
| Qwen3-4B-Instruct-2507 | Fixed Interval Summary [44k] | 41.4 | 44.0 | 39.2 |
| Qwen3-4B-Instruct-2507 | SELFCOMPACT [48k] | **45.5** | **47.8** | **42.1** |
| Qwen3-30B-A3B-Instruct-2507 | No Compaction [16k] | 45.2 | 54.7 | 51.9 |
| Qwen3-30B-A3B-Instruct-2507 | Fixed Interval Summary [26k] | 48.7 | 57.3 | **58.7** |
| Qwen3-30B-A3B-Instruct-2507 | SELFCOMPACT [29k] | **52.1** | **59.4** | 57.6 |
| Qwen3.5-9B (Thinking Disabled) | No Compaction [16k] | 25.0 | 38.2 | 34.2 |
| Qwen3.5-9B (Thinking Disabled) | Fixed Interval Summary [90k] | 33.4 | 42.1 | 44.9 |
| Qwen3.5-9B (Thinking Disabled) | SELFCOMPACT [93k] | **41.4** | **48.2** | **52.3** |
| Qwen3.5-4B (Thinking Disabled) | No Compaction [16k] | 16.9 | 26.4 | 22.5 |
| Qwen3.5-4B (Thinking Disabled) | Fixed Interval Summary [64k] | 27.4 | 35.5 | 29.1 |
| Qwen3.5-4B (Thinking Disabled) | SELFCOMPACT [67k] | **34.1** | **37.4** | **30.0** |

#### 4.1.4 Main result interpretation

Under matched token budgets, SELFCOMPACT consistently outperforms both no compaction and fixed-interval summarization, achieving the best performance in 11 out of 12 settings. The strongest gains are on the Qwen3.5 thinking-disabled models. For Qwen3.5-9B, SELFCOMPACT improves over no compaction by:

| Benchmark | No Compaction | SELFCOMPACT | Absolute gain |
|---|---:|---:|---:|
| IMO-Answerbench | 25.0 | 41.4 | +16.4 |
| HMMT Nov 25 | 38.2 | 48.2 | +10.0 |
| HMMT Feb 26 | 34.2 | 52.3 | +18.1 |

The only exception is Qwen3-30B-A3B on HMMT Feb 26, where fixed-interval summarization scores 58.7 and SELFCOMPACT scores 57.6, a 1.1-point gap. SELFCOMPACT still leads on the other two benchmarks for that model.

### 4.1.5 Headroom analysis: answer transitions under fixed intervals

The paper studies failure modes of fixed-interval summarization for Qwen3-4B-Instruct-2507 on IMO-Answerbench. It tracks answer transitions across 12 fixed-interval summarization calls, recording whether the model’s answer changes from wrong to correct or correct to wrong after summarization.

#### Table 2: answer transitions across 12 fixed-interval summarization calls

| Transition | Count |
|---|---:|
| Wrong → Correct | 1,486 |
| Correct → Wrong | 1,009 |

Fixed summarization yields a net gain, but it also degrades many trajectories: 40.4% of all transitions cause degradations. This is the core empirical evidence that summarization is not uniformly beneficial. The paper asks what would happen if the model could selectively skip summarization when its current answer is already correct.

### 4.1.6 Oracle analysis

The oracle policy suppresses summarization calls whenever the current answer is correct, but otherwise follows the same fixed schedule. It is a strict subset of what a fully adaptive policy could do because it decides only whether to summarize at fixed intervals, not when to summarize.

#### Table 3: oracle analysis on IMO-Answerbench for Qwen3-4B-Instruct-2507

| Method | Accuracy (%) |
|---|---:|
| No Compaction [16k] | 38.9 |
| Fixed Interval Summary [44k] | 41.4 |
| SELFCOMPACT [48k] | 45.5 |
| Oracle (skip if correct) [44k] | **52.9** |

The oracle reaches 52.9%, which is +11.5 points over fixed-interval summary and +14.0 over baseline. This suggests substantial headroom for adaptive policies that can learn or infer when summarization is beneficial.

### 4.1.7 Math-specific interpretation

The math experiments support two claims simultaneously:

1. **Summarization can improve long reasoning**: fixed-interval summary improves over no compaction for every model/benchmark block in Table 1.
2. **Timing matters**: fixed summarization also produces many correct→wrong transitions; SELFCOMPACT improves further by avoiding some poorly timed calls.

The paper therefore treats SELFCOMPACT not as “more summarization,” but as **state-aware summarization**: it fires when a reasoning unit is complete or a final answer should be locked in, and it suppresses when a model still needs the active derivation.
