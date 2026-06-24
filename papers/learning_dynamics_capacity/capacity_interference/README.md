# Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention

arXiv: 2605.29548 | Jing Huang et al. (Stanford, Kempner Institute at Harvard, MIT, Anthropic, 2026)
Code: —

## Files

- `s01_introduction.md` — Introduces the central question (why do larger models learn what smaller models cannot?) and proposes a data-centric account. Distinguishes between tasks learnable via data scaling (more data suffices) vs. model scaling (must increase parameters), and previews the synthetic theory + OLMo validation approach.

- `s02_phenomenological_model.md` — Shows from standard neural scaling laws (`α > γ`) that larger models asymptotically outperform smaller models even with infinite data. Provides formal definitions for the two learning regimes and establishes that a portion of the data distribution is inaccessible to small models regardless of data volume.

- `s03_theory.md` — The core theoretical contribution: a multi-task linear regression framework where features are learned in order of utility (`π_k · λ_{k,j}`). Theorem 4 and Corollary 5 show that as width increases, high-frequency task gradients weaken, freeing capacity. Proposition 6 gives the precise critical width `N_r^crit` at which a rare task survives. Matched-frequency injection experiments (Figure 4) confirm the learn-then-forget cycle in small models vs. accumulation in large models.

- `s04_olmo_validation.md` — Validates the theory on real transformers (OLMo 4M→4B) with injected comparison and modular-addition tasks at controlled frequencies. Three levels of evidence: behavioral (accuracy/frequency heatmaps, grokking only in large models), representational (DAS-localized features present only in large models), and gradient (non-task gradients nearly orthogonal to task direction in large models vs. random collisions in small models — `7.58×10⁻⁵` vs. `0.10` cosine similarity difference).

- `s05_discussion.md` — Summarizes the data-centric mechanism (`width ↑ → interference ↓ → rare features survive`), practical implications (data mixture engineering as alternative to blind scaling), reframing of memorization as a stepping stone to abstraction, and limitations (OLMo-scale only, artificial injected tasks, simplified theory, no MoE or overtraining).

- `s06_references.md` — Full 132-entry reference list covering scaling laws, multi-task learning, continual learning, gradient interference, memorization, and mechanistic interpretability.
