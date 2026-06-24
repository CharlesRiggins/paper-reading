## Appendix A. Additional Theoretical Analysis and Proofs

This appendix provides formal statements and proof sketches for the theoretical rationales used in Section 4.3. The results align with Eq. (2)–(8) and focus on: (i) the class-conditional LRT/LDA interpretation of the plug-in discriminant $f_l$; (ii) the variance floor induced by shrinkage and $\epsilon$; and (iii) the oracle optimality and Pareto efficiency of threshold routing.

### A.1 Setup and Notation for Proofs

Fix a layer $l$ and denote the last-token representation by $h\in\mathbb{R}^d$. The class-wise reference means are $\mu_l^+$ for easy and $\mu_l^-$ for hard. The pooled diagonal variance vector is $\sigma_l^2\in\mathbb{R}_{\ge0}^d$ as in Eq. (2), and the shrinkage-regularized variance vector $\tilde{\sigma}_l^2$ is defined by Eq. (3).

The diagonal Mahalanobis distance is

$$
D_l(h\parallel \mu)=\sum_{j=1}^{d}\frac{(h_j-\mu_j)^2}{\tilde{\sigma}_{l,j}^2+\epsilon},\qquad \epsilon>0,
\tag{5}
$$

with layer-wise plug-in discriminant

$$
f_l(h)=D_l(h\parallel \mu_l^-)-D_l(h\parallel \mu_l^+).
\tag{6}
$$

The global score aggregates per-layer polarities:

$$
F(H_t)=\sum_{l\in\mathcal{L}}\mathrm{sign}(f_l(H_{t,l,:})),
\tag{7}
$$

and the default decision is easy if $F(H_t)>0$ and hard otherwise, with ties routed to hard/slow.

### A.2 Auxiliary Algebraic Lemmas

**Quadratic expansion under a shared diagonal metric.** Let $w\in\mathbb{R}_{>0}^d$ and define

$$
\lVert x\rVert_w^2\triangleq\sum_{j=1}^{d}w_jx_j^2.
$$

For any $\mu^+,\mu^-\in\mathbb{R}^d$ and $h\in\mathbb{R}^d$,

$$
\lVert h-\mu^-\rVert_w^2-\lVert h-\mu^+\rVert_w^2
=2\langle w\odot(\mu^+-\mu^-),h\rangle+\lVert\mu^-\rVert_w^2-\lVert\mu^+\rVert_w^2.
$$

The proof expands both quadratic terms and cancels the shared $\sum_j w_jh_j^2$ component.

**Log-likelihood ratio of Gaussians with shared diagonal covariance.** Assume

$$
p(h\mid y)=\mathcal{N}(\mu_y,\Sigma),\qquad y\in\{+,-\},
$$

with shared diagonal covariance $\Sigma=\mathrm{diag}(v_1,\ldots,v_d)$ and $v_j>0$. Then

$$
\mathrm{LLR}(h)=\log p(h\mid +)-\log p(h\mid -)
=-\frac{1}{2}\left(\lVert h-\mu^+\rVert_w^2-\lVert h-\mu^-\rVert_w^2\right)+c_0,
$$

where $w_j=v_j^{-1}$ and $c_0$ is independent of $h$.

### A.3 Layer Separability Proxy (LVD) and Fisher/LDA Connection

For layer $l$, define $\Delta\mu_l=\mu_l^+-\mu_l^-\in\mathbb{R}^d$. The diagonal covariance proxy used by Eq. (5) is

$$
\tilde{\Sigma}_l=\mathrm{diag}(\tilde{\sigma}_{l,1}^2+\epsilon,\ldots,\tilde{\sigma}_{l,d}^2+\epsilon),\qquad \epsilon>0.
$$

The classical two-class Fisher criterion at layer $l$ is

$$
J_l(w)=\frac{(w^\top\Delta\mu_l)^2}{w^\top\Sigma_lw},\qquad w\in\mathbb{R}^d\setminus\{0\}.
$$

Under the diagonal-covariance approximation $\Sigma_l\approx\tilde{\Sigma}_l$, the maximizer satisfies

$$
w_l^\star\propto\tilde{\Sigma}_l^{-1}\Delta\mu_l,
$$

and the achieved separability is proportional to

$$
\Delta\mu_l^\top\tilde{\Sigma}_l^{-1}\Delta\mu_l
=\sum_{j=1}^{d}\frac{(\Delta\mu_{l,j})^2}{\tilde{\sigma}_{l,j}^2+\epsilon}.
$$

The LVD score

$$
I(l)=\frac{\lVert\mu_l^+-\mu_l^-\rVert_2}{\sqrt{\lVert\tilde{\sigma}_l^2\rVert_2}}
\tag{4}
$$

captures the same qualitative dependence: larger mean gaps and smaller within-class dispersion yield larger separability. RLDP uses $I(l)$ as an inexpensive layer-ranking proxy.

### A.4 Class-conditional LRT/LDA View of the Plug-in Discriminant

**Proposition 2.** Fix layer $l$ and assume a shared diagonal-covariance Gaussian model:

$$
p(h\mid\text{easy})=\mathcal{N}(\mu_l^+,\Sigma_l),\qquad
p(h\mid\text{hard})=\mathcal{N}(\mu_l^-,\Sigma_l),
$$

where $\Sigma_l=\mathrm{diag}(v_{l,1},\ldots,v_{l,d})$ and $v_{l,j}>0$. If the Mahalanobis metric used in Eq. (5) is proportional to $\Sigma_l^{-1}$,

$$
\frac{1}{\tilde{\sigma}_{l,j}^2+\epsilon}=\kappa v_{l,j}^{-1}
$$

for some $\kappa>0$, then there exist $\alpha>0$ and a constant $c$ independent of $h$ such that

$$
\mathrm{LLR}(h)=\alpha f_l(h)+c.
$$

Consequently, every LLR threshold decision has an equivalent threshold decision in terms of $f_l(h)$.

**Proof sketch.** Using the Gaussian LLR expression and the quadratic expansion lemma, $\mathrm{LLR}(h)$ is a weighted quadratic difference between distances to the two class means. Under metric proportionality, $f_l(h)$ is the same distance difference up to scale. Substitution yields $\mathrm{LLR}(h)=\frac{1}{2\kappa}f_l(h)+c_0$.

**Connection to diagonal LDA.** Under the same shared-covariance assumption, the Bayes-optimal discriminant is linear in $h$ and equivalent to Fisher/LDA with normal vector $\Sigma_l^{-1}(\mu_l^+-\mu_l^-)$. Since $f_l(h)$ expands to a linear form plus an offset, Eq. (6)–(8) has an LDA-style interpretation.

### A.5 Variance Floor Induced by Shrinkage and $\epsilon$

**Proposition 3.** Let $\sigma_l^2\in\mathbb{R}_{\ge0}^d$ be the pooled variance vector. Define

$$
\tilde{\sigma}_l^2=(1-\lambda)\sigma_l^2+\lambda\bar{\sigma}_l^2\mathbf{1}_d,
\qquad
\bar{\sigma}_l^2=\frac{1}{d}\sum_{j=1}^{d}\sigma_{l,j}^2,
\qquad \lambda\in[0,1].
$$

Using $\tilde{\sigma}_{l,j}^2+\epsilon$ in Eq. (5) with $\epsilon>0$ implies:

1. **Absolute floor:** $\tilde{\sigma}_{l,j}^2+\epsilon\ge\epsilon$ for all $j$.
2. **Shrinkage floor when $\lambda>0$:** $\tilde{\sigma}_{l,j}^2+\epsilon\ge\lambda\bar{\sigma}_l^2+\epsilon$ for all $j$.
3. **Bounded precision:** $(\tilde{\sigma}_{l,j}^2+\epsilon)^{-1}\le\epsilon^{-1}$ for all $j$.

This matters because small reference sets can produce near-zero variances. Without $\epsilon$, the Mahalanobis precision could become arbitrarily large and a few coordinates could dominate $D_l$ and $f_l$. Shrinkage and $\epsilon$ stabilize plug-in diagonal precision, consistent with covariance shrinkage and regularized discriminant analysis [20, 11].

### A.6 Oracle Optimality and Pareto Efficiency of Threshold Routing

Eq. (8) applies a global threshold to $F$ and routes to slow mode when an instance is predicted hard. The threshold controls the slow-trigger frequency: lowering it increases slow reasoning, raising cost and possibly accuracy; raising it reduces slow reasoning and cost.

In an oracle setting, let $A_f(x)$ and $A_s(x)$ be conditional correctness probabilities under fast and slow modes. Define the marginal gain

$$
\Delta(x)=A_s(x)-A_f(x).
$$

Assume the additional slow-mode cost is constant and a routing policy is $r:X\to\{0,1\}$, where $r(x)=1$ triggers slow mode.

**Proposition 4.** Given trigger budget $\rho\in[0,1]$, the problem

$$
\max_{r:X\to\{0,1\}}\mathbb{E}[A_f(x)+r(x)\Delta(x)]
\quad\text{s.t.}\quad
\mathbb{E}[r(x)]\le\rho
$$

has an optimal threshold policy

$$
r^*(x)=\mathbf{1}\{\Delta(x)>\gamma\}
$$

for some $\gamma\ge0$, with tie-breaking if needed. The proof is an exchange argument: if a lower-gain point is triggered while a higher-gain point is not, swapping them improves the objective without changing budget.

**Proposition 5.** Sweeping the trigger budget $\rho$, equivalently sweeping the threshold $\gamma$, traces a Pareto-efficient accuracy–cost frontier. No feasible policy can achieve strictly higher accuracy at no higher cost, or strictly lower cost at no lower accuracy, because each threshold policy is accuracy-optimal for its trigger rate.

In practice, the true $\Delta(x)$ is unknown. RLDP constructs an inexpensive representation-based score $F(H_t)$ as a proxy ordering signal for triggering. The oracle analysis justifies why a single global threshold is a principled and compute-controllable routing mechanism when the proxy is reasonably monotone.

---

## Appendix B. Difficulty Labels and Representation Analyses

### B.1 Difficulty Definition and Discussion

For standard single-mode LLMs, the paper adopts a result-oriented criterion: correctly answered instances are easy and incorrectly answered instances are hard. For dual-mode LLMs, difficulty is characterized by required reasoning depth. A problem is easy if both non-thinking and thinking modes solve it; it is hard if non-thinking fails but thinking succeeds.

Instances unsolved by both modes are excluded. The reason is that they do not meaningfully supervise the distinction between reasoning modes: depending on the external objective, one might choose non-thinking for efficiency or thinking for a small chance of improvement.

#### Table 3. Difficulty distribution across mathematical reasoning datasets

| Model | Olympiad Easy/Hard | AIME Easy/Hard | MATH500 Easy/Hard |
|---|---:|---:|---:|
| Qwen3-4B | 294 / 162 | 315 / 448 | 378 / 71 |
| Qwen3-14B | 335 / 145 | 400 / 439 | 410 / 45 |
| Llama-3.1-Nemotron-Nano-4B | 164 / 310 | 130 / 650 | 257 / 197 |
| DeepSeek-R1-Distill-Qwen-7B | 432 / 242 | 639 / 294 | 445 / 55 |
| Qwen2.5-7B-Instruct | 246 / 428 | 199 / 734 | 352 / 148 |
| Llama-3.1-8B-Instruct | 124 / 550 | 129 / 804 | 258 / 242 |

#### Table 4. Difficulty distribution for code generation and QA datasets

| Model | LiveCodeBench Easy/Hard | ScienceQA Easy/Hard |
|---|---:|---:|
| Qwen3-4B | 345 / 290 | 1810 / 242 |
| Qwen3-14B | 343 / 150 | 1935 / 146 |
| Llama-3.1-Nemotron-Nano-4B | 174 / 362 | 607 / 848 |

All label distributions are obtained using Pass@1 evaluation with temperature 0.6 and top-p 0.95.

### B.2–B.7 Layer-wise Distance Analyses Across Models

The appendix repeats the layer-wise center-distance analysis for multiple models:

- **Qwen3-4B:** Figure 4 reports normalized Euclidean and cosine distances on Olympiad, AIME, and MATH500; Figure 5 reports normalized Euclidean distances on LiveCodeBench and ScienceQA.
- **Qwen3-14B:** Figure 6 reports normalized Euclidean and cosine distances on the math datasets; Figure 7 covers LiveCodeBench and ScienceQA.
- **Llama-3.1-Nemotron-Nano-4B-v1.1:** Figure 8 covers math datasets; Figure 9 covers LiveCodeBench and ScienceQA.
- **DeepSeek-R1-Distill-Qwen-7B:** Figure 10 reports math dataset distances.
- **Qwen2.5-7B-Instruct:** Figure 11 reports math dataset distances.
- **Llama-3.1-8B-Instruct:** Figure 12 reports math dataset distances.

Across these analyses, the central observation remains that easy and hard problems produce layer-dependent separability in hidden representation space.

### B.8 Robustness to Dataset Variations and Token Length

The paper examines two confounders: dataset identity and token length. Across Olympiad, AIME, and MATH500, layer-wise distance trajectories show consistent qualitative patterns despite differences in domains and difficulty distributions, reducing the likelihood that the effect is merely dataset-specific.

For token length, the authors compare token-count distributions between easy and hard subsets. MATH500 shows larger length differences, while AIME and Olympiad show substantial overlap. A strict length-matched analysis selecting instances with 55–65 tokens still preserves the layer-wise separability pattern. Therefore, the separation cannot be explained solely by token length.

### B.9 Low-dimensional Representation

The paper projects layer-wise representations into low-dimensional spaces using UMAP, PCA, and t-SNE. In these projections, easy and hard examples are largely entangled, and low-dimensional center distances fluctuate irregularly across layers. This suggests that effective difficulty perception relies on the high-dimensional structure of Transformer hidden states rather than low-dimensional visualization geometry.

---

## Appendix C. Implementation Details

### C.1 Baselines

#### C.1.1 Prompt

The prompt baseline asks the model to classify difficulty directly. The prompt template is:

> Your task is to analyze a given problem and assess its difficulty level (Easy or Hard).  
> Now classify this question: `{question}`  
> Do not solve the problem. Just assess the problem difficulty level.  
> Your final answer should be in this form: `The final answer is boxed{XX}`, where XX is either `Easy` or `Hard`.

Experiments use Pass@1 with temperature 0.6.

#### C.1.2 Few-Shot

The few-shot baseline provides easy and hard examples before prediction. It uses 10 example pairs and Pass@1 with temperature 0.6.

> Your task is to analyze a given problem and assess its difficulty level (Easy or Hard).  
> Here are two sets of examples:  
> Easy examples: `{q1}`  
> Hard examples: `{q2}`  
> Now classify this question: `{question}`  
> Please use the examples above and your own judgment.  
> Do not solve the problem. Just assess the problem difficulty level.  
> Your final answer should be in this form: `The final answer is boxed{XX}`, where XX is either `Easy` or `Hard`.

#### C.1.3 AG

AG [22] perceives difficulty using output consistency. The experiments follow the authors' implementation, with temperature 0.8 and CoT reasoning using $k=10$ sampled outputs. AG is not applicable to LiveCodeBench because that benchmark requires code generation at test time.

#### C.1.4 LLMs-Ranking

LLMs-Ranking [42] compares randomly grouped problems and aggregates pairwise comparisons. The configuration uses batch size $B=8$, $R=5$ random-grouping rounds, temperature 0.6, pre-sample size $p=4$, and decision window $k=32$. It is also not applicable to LiveCodeBench.

#### C.1.5 HaluSearch-Gen

HaluSearch-Gen [6] fine-tunes an LLM to perceive difficulty. The paper splits 70% of each dataset for training, uses GPT-4o-mini to generate reward data, and applies SFT. The reward-generation prompt asks for a difficulty score from 1 to 5, using the question, correct answer, and generated answer, and requests only the numeric score.

#### C.1.6 Probe

Probe [30, 21] trains a representation-based linear probe. The paper splits 70% of each dataset as training data. It uses last-token hidden representations at layer 21 for Qwen3-4B, layer 40 for Qwen3-14B, and layer 20 for Llama-3.1-Nemotron-Nano-4B-v1.1.

#### C.1.7 Comparison of Difficulty-Perception Methods

| Property | Zero-shot | Few-shot | AG | LLMs-Ranking | HaluSearch-Gen | Probe | Ours |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training-free | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| Rollout-free | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Representation-based | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

The appendix stresses that rollout-based approaches require repeated model sampling and prompt-based approaches require output generation. HaluSearch-Gen and Probe require supervised training. RLDP requires neither training nor output generation and uses only a single prefill forward pass.

### C.2 Implementation Details of Fast and Slow Reasoning Modes

For single-mode LLMs, fast mode is standard CoT generation with a single sample. Slow mode is self-consistency CoT, sampling $k=10$ reasoning paths and selecting the final answer by majority vote.

For dual-mode LLMs, the paper uses built-in control tokens. In Qwen3 models, fast mode inserts `/no_think` to encourage concise direct responses, while slow mode uses `/think` to induce more deliberative reasoning.

All experiments use Pass@1 evaluation with decoding temperature 0.6 and top-p 0.95.

### C.3 Evaluation Metrics

Difficulty perception reports easy accuracy, hard accuracy, and Macro-F1 [38]. Overall reasoning efficiency reports final answer accuracy, total token cost including discrimination and reasoning, and token efficiency:

$$
\mathrm{TE}=\frac{\mathrm{Acc}}{\sqrt{\mathrm{Cost}}}.
$$

### C.4 Parameter Analysis

#### C.4.1 Number of Reference Pairs

The paper varies reference pairs from 1 to 10 over 10 random seeds. Figures 16–18 show that performance improves as reference count increases and then stabilizes.

#### C.4.2 Layer-wise Perception Analysis

Layer-wise experiments use 10 reference pairs and 10 random seeds. Figures 19–21 compare F1 and LVD for Qwen3-4B, Qwen3-14B, and Llama-3.1-Nemotron-Nano-4B. F1 trends track LVD profiles. Table 9 shows that Layer 0 collapses to all-hard predictions.

| Model | Layer | Olympiad E/H/F1 | AIME E/H/F1 | MATH500 E/H/F1 |
|---|---|---:|---:|---:|
| Qwen3-4B | Layer 0 | 0.00 / 100.00 / 25.85 | 0.00 / 100.00 / 37.09 | 0.00 / 100.00 / 12.45 |
| Qwen3-14B | Layer 0 | 0.00 / 100.00 / 22.69 | 0.00 / 100.00 / 34.38 | 0.00 / 100.00 / 7.45 |
| Llama-3.1-NN-4B-v1.1 | Layer 0 | 0.00 / 100.00 / 39.79 | 0.00 / 100.00 / 45.71 | 0.00 / 100.00 / 30.11 |

#### C.4.3–C.4.7 Additional Parameter Analyses

The appendix includes detailed results for distance metrics, cross-dataset and cross-class transfer, reference-set noise robustness, Probe under limited data, and top-n discriminative layer aggregation. These results are summarized in Section 5.4 and support four conclusions: Mahalanobis distance is the most stable, RLDP transfers across reference sources, moderate reference noise does not break the method, and top-1 LVD layer selection is simpler and more robust than aggregating many layers.

---

## NeurIPS Paper Checklist

The original checklist includes extensive conference guidelines. This structured version retains the authors' answers and justifications.

| # | Topic | Answer | Justification |
|---|---|---|---|
| 1 | Claims | Yes | The abstract and Section 1 clearly state the contributions and scope. |
| 2 | Limitations | Yes | Limitations are discussed in Section 6 (Conclusion and Discussion). |
| 3 | Theory assumptions and proofs | Yes | Appendix A provides detailed theoretical analysis, assumptions, and structured proofs. |
| 4 | Experimental result reproducibility | Yes | Section 5.1, Appendix C, and Section 4 provide setups and method details needed for reproduction. |
| 5 | Open access to data and code | No | The authors state this is temporary and that they are committed to releasing code upon acceptance. |
| 6 | Experimental setting/details | Yes | Section 5.1 and Appendix C describe model settings, training settings, evaluation settings, and baselines. |
| 7 | Experiment statistical significance | Yes | The authors provide mean and standard deviation over 10 runs for RLDP experiments in Table 10. |
| 8 | Experiments compute resources | Yes | The authors state detailed information is provided in Appendix C. |
| 9 | Code of ethics | Yes | The authors state that the research complies with the NeurIPS Code of Ethics. |
| 10 | Broader impacts | N/A | The authors state that the paper focuses on LLM difficulty perception and resource allocation and is not directly related to societal impacts. |
| 11 | Safeguards | Yes | The authors state that the paper is not relevant to high-risk data/model release issues. |
| 12 | Licenses for existing assets | Yes | The authors state that citations and URLs are provided throughout the paper. |
| 13 | New assets | N/A | The paper does not release new assets. |
| 14 | Crowdsourcing and research with human subjects | N/A | The paper does not involve crowdsourcing or human-subject research. |
| 15 | IRB approvals or equivalent | N/A | The paper does not involve crowdsourcing or human-subject research. |
| 16 | Declaration of LLM usage | Yes | The authors state that LLM usage is described in Appendix C, Section 3, and Section 4. |
