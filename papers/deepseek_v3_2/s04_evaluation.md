## 4. Evaluation

### 4.1 Main Results

Models evaluated on: **MMLU-Pro**, **GPQA Diamond**, **HLE** (text-only), **LiveCodeBench** (2024.08–2025.04), **Codeforces**, **Aider-Polyglot**, **AIME 2025**, **HMMT Feb 2025**, **HMMT Nov 2025**, **IMOAnswerBench**, **Terminal Bench 2.0**, **SWE-Verified**, **SWE Multilingual**, **BrowseComp**, **BrowseCompZh**, **$\tau^2$-bench**, **MCP-Universe**, **MCP-Mark**, and **Tool-Decathlon**. Temperature set to 1.0, context window to 128K.

| Benchmark (Metric) | Claude-4.5 Sonnet | GPT-5 High | Gemini-3.0 Pro | Kimi-K2 Thinking | MiniMax M2 | **DeepSeek-V3.2 Thinking** |
|---|---|---|---|---|---|---|
| MMLU-Pro (EM) | 88.2 | 87.5 | 90.1 | 84.6 | 82.0 | **85.0** |
| GPQA Diamond (Pass@1) | 83.4 | 85.7 | 91.9 | 84.5 | 77.7 | **82.4** |
| HLE (Pass@1) | 13.7 | 26.3 | 37.7 | 23.9 | 12.5 | **25.1** |
| LiveCodeBench (Pass@1-COT) | 64.0 | 84.5 | 90.7 | 82.6 | 83.0 | **83.3** |
| Codeforces (Rating) | 1480 | 2537 | 2708 | — | — | **2386** |
| AIME 2025 (Pass@1) | 87.0 | 94.6 | 95.0 | 94.5 | 78.3 | **93.1** |
| HMMT Feb 2025 (Pass@1) | 79.2 | 88.3 | 97.5 | 89.4 | — | **92.5** |
| HMMT Nov 2025 (Pass@1) | 81.7 | 89.2 | 93.3 | 89.2 | — | **90.2** |
| IMOAnswerBench (Pass@1) | — | 76.0 | 83.3 | 78.6 | — | **78.3** |
| Terminal Bench 2.0 (Acc) | 42.8 | 35.2 | 54.2 | 35.7 | 30.0 | **46.4** |
| SWE Verified (Resolved) | 77.2 | 74.9 | 76.2 | 71.3 | 69.4 | **73.1** |
| SWE Multilingual (Resolved) | 68.0 | 55.3 | — | 61.1 | 56.5 | **70.2** |
| BrowseComp (Pass@1) | 24.1 | 54.9 | — | 60.2* | 44.0 | **51.4/67.6*** |
| BrowseCompZh (Pass@1) | 42.4 | 63.0 | — | 62.3 | 48.5 | **65.0** |
| $\tau^2$-Bench (Pass@1) | 84.7 | 80.2 | 85.4 | 74.3 | 76.9 | **80.3** |
| MCP-Universe (Success Rate) | 46.5 | 47.9 | 50.7 | 35.6 | 29.4 | **45.9** |
| MCP-Mark (Pass@1) | 33.3 | 50.9 | 43.1 | 20.4 | 24.4 | **38.0** |
| Tool-Decathlon (Pass@1) | 38.6 | 29.0 | 36.4 | 17.6 | 16.0 | **35.2** |

**Key findings:**
- DeepSeek-V3.2 achieves **similar performance with GPT-5-High** on reasoning tasks, but is slightly worse than Gemini-3.0-Pro.
- Compared to K2-Thinking, achieves comparable scores with **substantially fewer output tokens**.
- Performance gains attributed to increased RL training budget, which **already exceeds 10% of pre-training cost**.
- Performance is constrained by a length constraint reward model; removing it yields further improvement (see Speciale).

**Code agent**: Significantly outperforms open-source LLMs on SWE-bench Verified and Terminal Bench 2.0. SWE-bench Verified results are consistent (72–74) across Claude Code, RooCode, and non-thinking mode.

**Search agent**: ~20%+ of test cases exceed the 128K limit. Context management yields the 67.6 score (vs. 51.4 without).

**Tool-use**: Substantially narrows the gap between open and closed models, though still below frontier models. DeepSeek-V3.2 frequently engages in **redundant self-verification**, generating excessively long trajectories that often exceed 128K limits, particularly in MCP-Mark GitHub and Playwright evaluations. The model shows **generalization to out-of-domain agentic scenarios** since environments and toolsets were not encountered during RL training.

### 4.2 Results of DeepSeek-V3.2-Speciale

| Benchmark | GPT-5 High | Gemini-3.0 Pro | Kimi-K2 Thinking | **DS-V3.2 Thinking** | **DS-V3.2 Speciale** |
|---|---|---|---|---|---|
| AIME 2025 (Pass@1) | 94.6 (13k) | 95.0 (15k) | 94.5 (24k) | 93.1 (16k) | **96.0 (23k)** |
| HMMT Feb 2025 (Pass@1) | 88.3 (16k) | 97.5 (16k) | 89.4 (31k) | 92.5 (19k) | **99.2 (27k)** |
| HMMT Nov 2025 (Pass@1) | 89.2 (20k) | 93.3 (15k) | 89.2 (29k) | 90.2 (18k) | **94.4 (25k)** |
| IMOAnswerBench (Pass@1) | 76.0 (31k) | 83.3 (18k) | 78.6 (37k) | 78.3 (27k) | **84.5 (45k)** |
| LiveCodeBench (Pass@1-COT) | 84.5 (13k) | 90.7 (13k) | 82.6 (29k) | 83.3 (16k) | **88.7 (27k)** |
| Codeforces (Rating) | 2537 (29k) | 2708 (22k) | — | 2386 (42k) | **2701 (77k)** |
| GPQA Diamond (Pass@1) | 85.7 (8k) | 91.9 (8k) | 84.5 (12k) | 82.4 (7k) | **85.7 (16k)** |
| HLE (Pass@1) | 26.3 (15k) | 37.7 (15k) | 23.9 (24k) | 25.1 (21k) | **30.6 (35k)** |

DeepSeek-V3.2-Speciale **surpasses Gemini-3.0-Pro** across multiple benchmarks by leveraging increased reasoning tokens. Token efficiency remains significantly inferior to Gemini-3.0-Pro.

#### Competition-Level Achievements

| Competition | P1 | P2 | P3 | P4 | P5 | P6 | Overall | Medal |
|---|---|---|---|---|---|---|---|---|
| IMO 2025 | 7 | 7 | 7 | 7 | 7 | 0 | **35/42** | **Gold** |
| CMO 2025 | 18 | 18 | 9 | 21 | 18 | 18 | **102/126** | **Gold** |
| IOI 2025 | 100 | 82 | 72 | 100 | 55 | 83 | **492/600** | **Gold** |

DeepSeek-V3.2-Speciale ranked **2nd in ICPC WF 2025** and **10th in IOI 2025**, solving 10 out of 12 ICPC World Finals problems.

### 4.3 Synthesis Agentic Tasks

**Are synthetic tasks sufficiently challenging?** A random sample of 50 instances from general synthesized agentic tasks evaluated against frontier models:

| Pass@K | DS-V3.2-Exp | Sonnet-4.5 | Gemini-3.0 Pro | GPT-5-Thinking |
|---|---|---|---|---|
| 1 | 12% | 34% | 51% | **62%** |
| 2 | 18% | 47% | 65% | **75%** |
| 4 | 26% | 62% | 74% | **82%** |

The synthetic data include agentic tasks that are challenging for both DeepSeek-V3.2-Exp and frontier closed-source models.

**Do synthetic tasks generalize?** RL on synthetic data (non-thinking mode, synthetic agentic tasks only) yields substantial improvements over DeepSeek-V3.2-SFT on Tau2Bench, MCP-Mark, and MCP-Universe. In contrast, restricting RL to code and search scenarios does not improve performance on these benchmarks, highlighting the potential of synthetic data.

### 4.4 Context Management of Search Agent

Agentic workflows frequently encounter maximum length limitations that prematurely truncate reasoning. Four strategies when token usage exceeds 80% of the context window:

| Strategy | Description |
|---|---|
| **Summary** | Summarizes overflowed trajectory, re-initiates rollout |
| **Discard-75%** | Discards first 75% tool call history |
| **Discard-all** | Resets context by discarding all previous tool call history |
| **Parallel-fewest-step** | Samples N independent trajectories, selects the one with fewest steps |

Evaluated on BrowseComp with varying compute budgets:

- **Summary** extends average steps to 364, achieving **60.2**, but overall efficiency is relatively low.
- **Discard-all** performs well in both efficiency and scalability, achieving **67.6**, comparable to parallel scaling while using significantly fewer steps.

Test-time compute can be scaled serially (context management) or in parallel, both effectively extending problem-solving capacity. Finding the optimal combination remains a direction for future work.
