# Explaining Neural Scaling Laws

arXiv: 2102.06701 | Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, Utkarsh Sharma et al. (Google DeepMind / Johns Hopkins University, 2021)
Code: —

## Files

- `s01_introduction.md` — Introduces the paper's central taxonomy of **four scaling regimes**: variance-limited (universal $\alpha = 1$ exponents from infinite-data / infinite-width limits) and resolution-limited (data-dependent exponents from resolving a smooth data manifold). States the six main contributions, including exact derivations in random feature models, a novel model–dataset duality, and empirical validation on pretrained models. Includes the abstract and Figure 1 overview of all four regimes.

- `s02_four_scaling_regimes.md` — The theoretical core. Proves **Theorem 1** (variance-limited scaling $\mathcal{O}(D^{-1})$ and $\mathcal{O}(P^{-1})$ for smooth losses) and **Theorems 2–3** (resolution-limited bounds $L \sim \mathcal{O}(D^{-1/d})$, $L \sim \mathcal{O}(P^{-1/d})$ on a $d$-dimensional manifold), then refines the latter to the empirical estimate $\alpha \approx 4/d$ via a Taylor expansion. Derives exact scalings in linear random feature models where $\alpha_P = \alpha_D = \alpha_K$ (the kernel spectral decay exponent, with $\alpha_K \propto 1/d$ by Weyl's law), and identifies a duality exchanging feature projections with training-point projections.

- `s03_experiments.md` — Validates all four regimes empirically. Deep teacher-student models confirm the $4/d$ prediction by scanning input dimension. Variance-limited scaling ($\alpha_D = \alpha_W = 1$) is observed universally across architectures, datasets, losses, and batch sizes on real data. Resolution-limited scaling on standard image datasets (WRN, CNN) shows data-dependent exponents, with width affecting the exponent up to a saturating critical width.

- `s04_05_discussion_outlook.md` — Discusses implications and limitations: the asymptotic nature of the theory, breakdown of scaling when the $D \gg P$ or $P \gg D$ hierarchy is lost, and the lack of a precise data-manifold definition. The invariance of exponents to superclassing suggests deep networks learn input-manifold structure (akin to unsupervised learning) rather than task-specific structure. The outlook frames scaling-law theory as physics-inspired theory construction and raises the question of emergent abilities.

- `s06_references.md` — Full 85-entry reference list spanning empirical scaling laws (Hestness, Kaplan, Hoffmann/Chinchilla), kernel methods and NTK theory (Jacot, Lee, Bordelon, Canatar), random matrix theory (Mei, Hastie), and statistical physics approaches to learning curves (Sollich, Opper, Parisi).

- `s07_appendix_a_experimental_setup.md` — Detailed experimental protocols for every panel of Figure 1: WRN-28-10 training on image datasets (78,125 steps, cosine decay), Neural Tangents infinite-width experiments (widths 64–11,585, linearized-model subtraction), teacher-student MSE scaling on $d$-dimensional hypercubes ($d = 2$–$9$), CNN architectures for resolution-limited scaling, and aspect-ratio studies varying WRN depth and width.

- `s08_appendix_b_proof_theorem1.md` — Proves Theorem 1 by cases: finite-degree polynomial loss (Taylor expansion + moment scaling), bounded second derivative (quadratic mean value theorem), and 2-Hölder loss. Includes remarks bounding the loss variance itself via first-order expansion.

- `s09_appendix_c_variance_limited.md` — Rigorous derivation of $\mathcal{O}(D^{-1})$ dataset scaling. Gives intuition for early-time (gradient concentration) and late-time (local-minimum, Hoeffding-bound) regimes, extends to SGD via a batch-moment translation identity, and proves concentration for polynomial losses (Lemma 1 on sample-mean polynomials, Lemma 2 on graph-combinatorial SGD batch averages). Concludes with two non-smooth counterexamples: an unbounded-loss model that breaks scaling at late times ($T \gg D$) and an unbounded-derivative model achieving $L(D) \sim D^{-\alpha}$ with $\alpha \neq 1$.

- `s10_appendix_d_proofs.md` — Proofs of Theorems 2 and 3 using the nearest-neighbor distance scaling $\mathbb{E}[|x - \hat{x}|] = \mathcal{O}(D^{-1/d})$ on a $d$-manifold combined with Lipschitz bounds on $f$, $\mathcal{F}$, and the loss $\ell$.

- `s11_appendix_e_random_features.md` — Full derivation of exact test-loss expressions (S37–S38) for linear random feature models and their asymptotics. Variance-limited scaling follows from expanding $\bar{\mathcal{C}} = \mathcal{C} + \delta\mathcal{C}$ (quartic-tensor corrections at $\mathcal{O}(D^{-1})$). Resolution-limited scaling $L(D) \propto D^{-\alpha_K}$ is derived via hypergeometric-function asymptotics of the spectral sum. Confirms duality holds beyond the asymptotic regime by exchanging feature and dataset projections.

- `s12_appendix_f_learned_features.md` — Tests the theory on pretrained features (EfficientNet-B5 embeddings). Observes all four scaling regimes under low and tuned regularization, with the variance-limited deviation under low regularization attributed to double-descent resonance at $D = P$. Confirms the dataset–feature duality (Figure S6) and compares random vs. learned features, finding teacher-student targets scale better than real targets, with task-label coefficients $\bar{\omega}_i$ approximately constant only for well-performing features.
