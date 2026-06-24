# Learning Dynamics of LLM Finetuning

arXiv: 2407.10490 | Yi Ren et al. (University of British Columbia & Amii, 2024)
Code: https://github.com/Joshua-Ren/Learning_dynamics_LLM

## Files

- `s01_introduction.md` — Abstract and Introduction. Frames the paper's central question: understanding LLM finetuning through a **learning dynamics** lens. Introduces the key idea of decomposing the model's prediction change into three interpretable terms, proposes a unified framework spanning SFT, DPO, and RL-based methods, and previews the **squeezing effect** — where negative gradients on unlikely predictions push probability mass to already-confident tokens, explaining why off-policy DPO degrades even desired outputs.

- `s02_definition_mnist.md` — Section 2: Definition of Learning Dynamics and an MNIST Example. Formalizes the per-step influence decomposition (Proposition 1) where $\Delta \log \pi \approx -\eta \mathcal{A} \mathcal{K} \mathcal{G}$, with $\mathcal{A}$ as the softmax Jacobian, $\mathcal{K}$ as the empirical NTK similarity kernel, and $\mathcal{G}$ as the loss gradient. Demonstrates on MNIST that learning a digit "pulls up" similar-class predictions and that class-pair similarities (e.g., 4↔9) emerge from accumulated influences, validating the framework with a toy example.

- `s03_theory.md` — Section 3: Learning Dynamics of LLM's Finetuning. Extends Proposition 1 to LLMs by handling auto-regressive sequences via causal masking (SFT decomposition), then derives the DPO residual terms $\mathcal{G}_{\text{DPO}\pm} = \beta(1-a)(\pi - \mathbf{y}^{\pm})$. Introduces the **squeezing effect**: when a negative gradient hits an unlikely label, confidence on almost all outputs drops while mass concentrates on the argmax token — a formal explanation for why all response confidences decay during off-policy DPO.

- `s04_experiments.md` — Section 4: Experimental Verifications. Validates the theory on `Pythia` and `Qwen` models with Anthropic-HH and UltraFeedback datasets. SFT experiments reveal bell-shaped curves for similar responses (initially pulled up, then pushed down) and explain cross-question hallucination amplification. DPO experiments confirm the squeezing effect via fast-growing greedy-decoding confidence (−113 → −63). Proposes and validates a simple mitigation ("extend"): train SFT on both chosen and rejected responses before DPO, achieving win rates up to **0.6928** (ChatGPT) and **0.6045** (Claude) over baseline.

- `s05_conclusion.md` — Section 5: Conclusion and Acknowledgements. Summarizes contributions: a step-wise decomposition framework for LLM finetuning, explanation of counter-intuitive phenomena (hallucination amplification, confidence decay), and the squeezing effect as a new mechanism with practical mitigation.

- `s06_references.md` — Full reference list (54 entries) covering NTK theory, LLM alignment, preference optimization, learning dynamics, and related areas from 1998–2025.

- `s07_appendix_a_related_works.md` — Appendix A: Related Works. Covers learning dynamics beyond LLMs (zigzag paths, local elasticity, TracIn, lpNTK), broader LLM finetuning context (repeater phenomena, confidence decay, on-policy vs off-policy), and the distinction between **benign** (well-regulated, as in machine unlearning and PPO) and **harmful** (valley-region) negative gradients.

- `s08_appendix_b_proofs.md` — Appendix B: Proofs and Residual Terms. Full proof of Proposition 1 via first-order Taylor expansion and chain rule. Derives residual terms for SFT, DPO, IPO, SLiC, and SPPO, revealing that SPPO uniquely avoids negative gradients by imposing two positive vectors, eliminating the squeezing effect.

- `s09_appendix_c_entk.md` — Appendix C: The "Relative Stable" eNTK Assumption. Validates the core assumption on both MNIST (direct eNTK computation shows stable ranking of class-pair similarities) and LLMs (via proxy metrics LBK and SignDelta, confirming stable relative influence and pairing effects between similar response types).

- `s10_appendix_d_experiments.md` — Appendix D: More Experimental Details. Defines the 14 response types in the probing dataset across three subspaces ($\mathcal{D}_{\text{IF}}$, $\mathcal{D}_{\text{non-IF}}$, $\mathcal{Y}_{\text{non-hum}}$). Provides additional SFT results showing consistent dynamics across 5 models, format-vs-semantics rephrase analysis, and DPO results confirming longer SFT before DPO worsens the squeezing effect.

- `s11_appendix_e_squeezing.md` — Appendix E: The Squeezing Effect in Detail. Formal analysis via logistic regression with Lemma 1 ($\alpha_i = p_i^{t+1}/p_i^{t}$) and five Claims: (1) $p_y$ guaranteed to decrease, (2) $p_{i^{*}}$ guaranteed to increase, (3A-C) distribution-dependent behavior from uniform to peaky, (4) smaller $p_y^t$ worsens squeezing, (5) larger $|\eta'|$ amplifies all effects. Simple $V=50$ experiment visualizes four regimes: flat, peaky-positive, on-policy negative, and off-policy valley-negative.

- `s12_appendix_f_alignment.md` — Appendix F: A Simple Method to Improve Alignment. The "extend" method: augment SFT with $(\mathbf{x}, \mathbf{y}_u^-)$ pairs, then standard DPO. Full experimental verification on `Qwen1.5-1.8B` with Anthropic-HH: after 4 DPO epochs, the proposed method achieves win rates of **0.6928** (ChatGPT) and **0.6045** (Claude) against baseline, with reduced degenerate responses.
