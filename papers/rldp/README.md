# Does Your Large Language Model Have An Intuitive Sense of The Difficulty of A Question?

arXiv: — | Anonymous (—, 2026)
Code: —

## Metadata

- **Title:** Does Your Large Language Model Have An Intuitive Sense of The Difficulty of A Question?
- **Author:** Anonymous
- **Affiliation:** —
- **Year:** 2026
- **Original PDF:** `/Users/charlesriggins/Desktop/paper-reading/papers2review/30251_Does_Your_Large_Language.pdf`
- **Parsed text:** `/Users/charlesriggins/Desktop/paper-reading/papers2review/_parsed_text/30251_Does_Your_Large_Language.md`

## Files

- `s01_introduction.md` — Presents the paper's motivation: adaptive reasoning in LLMs needs reliable difficulty perception to avoid overthinking on easy questions and under-reasoning on hard ones. It preserves the abstract-level claim that hidden representations encode difficulty before explicit reasoning, introduces RLDP and RLDP-AdaSwitch, and records the claimed 1.34×–2.00× token-efficiency gains over rollout-based methods. It also includes the main contribution list and the Figure 1 observation that easy/hard class-center distances vary strongly by layer.

- `s02_background_related_work.md` — Reconstructs the paper's operational difficulty definition for both single-mode and dual-mode LLMs, including the easy/hard labeling rule based on non-thinking and thinking outcomes. It preserves the notation for layer-wise last-token hidden representations $H \in \mathbb{R}^{(L+1)\times d}$ and organizes related work into prompt-based, rollout-based, training-based, and Transformer-layer analysis threads. The file explains why the authors position RLDP as a training-free, rollout-free representation-space method.

- `s03_probing_difficulty_perception.md` — Covers the probing study that asks whether LLMs can perceive question difficulty explicitly via prompts or implicitly through hidden states. It records the paper's conclusion that zero-shot and few-shot prompting are unreliable, while easy and hard questions show layer-dependent separability in high-dimensional hidden representations. It also preserves the center-distance measurement procedure, Figure 2 findings, token-length control argument, low-dimensional projection caveat, and the two takeaways from Section 3.

- `s04_method_rldp_adaswitch.md` — Details the proposed RLDP method and RLDP-AdaSwitch controller. It keeps the LVD layer-selection score, shrinkage-regularized variance, diagonal Mahalanobis distance, layer-wise discriminant, sign-voting global score, threshold rule, and the fast/slow routing workflow. The theoretical analysis is structured around the LRT/LDA interpretation, shrinkage variance floor, and threshold routing as an accuracy-cost operating point.

- `s05_experiments.md` — Provides the full experimental setup, including models, datasets, baselines, metrics, implementation details, and main results for difficulty perception and adaptive reasoning. It preserves the major numeric results from Tables 1 and 2, including E-Acc/H-Acc/Macro-F1 for difficulty perception and Acc/Tokens/TE for adaptive routing across Qwen3-4B, Qwen3-14B, and Llama-3.1-Nemotron-Nano-4B. It also summarizes the ablation findings about reference-pair count, LVD layer selection, distance metrics, cross-dataset/class transfer, noise robustness, probe data efficiency, and top-n layer aggregation.

- `s06_conclusion_discussion.md` — Restates the conclusion that LLM hidden representations contain pre-reasoning difficulty signals and that RLDP can exploit them without training or rollouts. It preserves the stated limitations: RLDP is a pre-reasoning estimate rather than a replacement for deliberate reasoning, and it requires access to internal hidden states, restricting deployment mainly to open-/accessible-weight settings. The file also notes the paper's future direction of extending representation-based difficulty signals to broader training or deployment scenarios.

- `s07_references.md` — Converts the full 50-entry reference list into a structured Markdown table with number, authors, title, source/venue, and year. The references cover overthinking and adaptive reasoning, fast/slow LLM methods, hidden-representation probing, Mahalanobis/LDA foundations, and datasets such as OlympiadBench, AIME, MATH500, LiveCodeBench, and ScienceQA. This file is separated from the main narrative so that citation metadata can be browsed independently.

- `s08_appendix_checklist.md` — Combines the substantive appendices and the NeurIPS checklist while dropping page headers/footers and guideline boilerplate. It preserves the formal appendix statements and proof sketches for LVD/Fisher connection, LRT/LDA equivalence, variance-floor stability, and threshold-routing optimality; it also includes the difficulty-label distributions, implementation prompt templates, algorithmic property comparison, and parameter-analysis tables. The final checklist section records the authors' answers and justifications for claims, limitations, reproducibility, code access, experimental details, statistics, compute, ethics, assets, human subjects, and LLM usage.
