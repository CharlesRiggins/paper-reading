# Self-Compacting Language Model Agents

arXiv: — | Anonymous (—, 2026)
Code: —

## Metadata

| Field | Value |
|---|---|
| Title | Self-Compacting Language Model Agents |
| Authors | Anonymous |
| Affiliation | — |
| Year | 2026 |
| arXiv | — |
| Code | — |
| Source PDF | `/Users/charlesriggins/Desktop/paper-reading/papers2review/18449_Self_Compacting_Language.pdf` |
| Parsed text | `/Users/charlesriggins/Desktop/paper-reading/papers2review/_parsed_text/18449_Self_Compacting_Language.md` |

## Files

- `s01_introduction_preliminaries.md` — Reconstructs the paper’s motivation around long reasoning traces, accumulated stale context, and “context rot.” It preserves the BrowseComp motivating example where fixed-interval summarization erases verified facts, while SELFCOMPACT fires after closed reasoning units. It also covers the preliminary framing of long-horizon agents, reactive/periodic compaction, and why content-agnostic token thresholds fail asymmetrically.

- `s02_approach.md` — Presents the full SELFCOMPACT scaffold: the model receives an inline summarization tool and a lightweight rubric that decides whether to fire at probe intervals. It keeps the formal setup $S:(x,y_{1:t})\to\tilde y$, the Algorithm 1 control flow, the COMPRESS/CONTINUE logic, KV-cache reuse, and the cost argument that compaction pays when summaries shrink long prefixes by large factors. This file is the method core and includes the key equations and pseudo-code.

- `s03_experiments_math.md` — Covers competition-math experiments on IMO-Answerbench, HMMT Nov 2025, and HMMT Feb 2026 with four Qwen-family models. It preserves Table 1’s per-model/per-benchmark accuracy numbers and token budgets, plus the fixed-interval transition analysis showing 1,486 wrong→correct and 1,009 correct→wrong transitions. It also includes the oracle skip-if-correct analysis, where accuracy reaches 52.9% on IMO-Answerbench for Qwen3-4B-Instruct-2507.

- `s04_experiments_agentic_search.md` — Covers agentic-search experiments on BrowseComp, BrowseComp-Plus, and DeepSearchQA for GLM-4.7-Flash, MiniMax-M2.5, and MiMo-V2-Flash. It preserves the cost formula, all baselines, the full Table 4 accuracy/cost matrix, and the main observations that SELFCOMPACT improves BrowseComp-Plus by +8.5, +9.2, and +5.3 points over no compaction. It also records the Figure 2/3 findings: SELFCOMPACT fires earlier than the 30% threshold and helps most on hard high-token problems.

- `s05_related_limitations_conclusion.md` — Organizes the related-work discussion into frontier-model context compaction, training-based compaction, training-free compaction, and KV-cache eviction. It also preserves the limitations around open-weight-only evaluation and the absence of reinforcement learning. The conclusion reframes “when to compact” as a scaffold-provided metacognitive capability rather than a behavior that must be built into model weights.

- `s06_references.md` — Provides a standalone reference table extracted from the paper’s bibliography. It includes cited work on context rot, long-horizon search, compaction/folding, KV-cache eviction, benchmark papers, evaluated model reports, and infrastructure such as vLLM and AggAgent. The table is normalized into citation, year, venue/source, and role-in-paper fields for quick lookup.

- `s07_appendix_checklist.md` — Merges the appendices and NeurIPS checklist into one structured file. It retains the math and search experimental setups, exact rubric/summarizer/continuation prompts, model context-window table, OpenRouter price table, token-consumption tables, qualitative trajectory cases A–D, and a compact checklist table with answers and justifications. Boilerplate checklist instructions and submission/reviewer page headers are removed.
