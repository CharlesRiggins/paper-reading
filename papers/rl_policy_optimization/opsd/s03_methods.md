## 3. Methods

> **Figure 2 — Prompt example for student and teacher policies.** Both policies share the same parameters $\theta$ but differ in conditioning context. The teacher receives the ground-truth solution $y^\star$ as privileged information before generation. To ensure a natural transition before evaluating the student's rollout, the teacher is prompted to rationalize and generate its own solution. Note that the teacher does **not** generate tokens — rationalization is done implicitly through one forward pass (prefilling).

### 3.1 Learning from Verifiable Reasoning Dataset

The authors consider a dataset of problem-solution pairs

$$
\mathcal{S}=\{(x_{i},y_{i}^{\star})\}_{i=1}^{N},
$$

where each $x_i$ denotes a problem and $y_i^\star$ is the corresponding reference solution, which may include chain-of-thought reasoning. (For brevity, the sample index $i$ is omitted, using $(x, y^\star)$ for a generic sample.)

Learning signals can be extracted from this dataset in different ways, each with drawbacks:

- **Standard SFT** on $\mathcal{S}$ can be viewed as off-policy distillation/imitation learning using expert trajectories, but it suffers from distribution mismatch between training and inference.
- **RLVR** (e.g., GRPO) addresses this by optimizing on-policy samples and assigning binary rewards by comparing generated answers against $y^\star$. However, RLVR is computationally expensive and the reward signal is sparse, providing the same feedback across all tokens regardless of where errors occur.
- A **process reward model (PRM)** can provide dense, token-level feedback during RL. However, acquiring labels for PRM training is prohibitively expensive and difficult to scale [15, 46].
- **On-policy distillation** [1, 37, 18] addresses distribution shift by training on the student's own samples, but requires a separate, often larger, teacher model.

The authors instead seek a training signal that is **dense, on-policy, and requires no external teachers or reward models** — motivating OPSD.

### 3.2 On-Policy Self-Distillation

**Motivation: Learning by understanding solutions.** The framework is inspired by how students learn: when struggling with a problem, rather than extended trial-and-error, a student can examine the solution, understand the reasoning, and internalize the approach. Similarly, if a model has access to the correct answer or reasoning $y^\star$ and is sufficiently capable, it can rationalize the reasoning steps and teach itself — analogous to a student reviewing a solution and retracing why it works. The ground-truth solution $y^\star$ is exploited directly as **privileged information** during training, enabling the model to serve as its own teacher without external reward or teacher models.

**Teacher and student policies.** Two conditional distributions are instantiated from the same language model $p_\theta$ by varying the conditioning context. The **teacher policy** conditions on privileged information — both the problem $x$ and the reference solution $y^\star$:

$$
p_{T}(\cdot\mid x,y^{\star})\;\triangleq\;p_{\theta}(\cdot\mid x,y^{\star}).
$$

The **student policy** observes only the problem statement, matching the inference-time condition:

$$
p_{S}(\cdot\mid x)\;\triangleq\;p_{\theta}(\cdot\mid x).
$$

Both policies share the same parameters $\theta$ but differ only in conditioning context. To encourage the teacher to naturally evaluate the student's generation, a prompt is added asking the teacher to generate a new solution after seeing the reference solution (Figure 2). However, the teacher does not generate tokens; it only rationalizes implicitly through prefilling.

**On-policy sampling from the student.** Given a problem $x$, the student generates an on-policy response

$$
\hat{y}=(\hat{y}_{1},\ldots,\hat{y}_{|\hat{y}|})\sim p_{S}(\cdot\mid x).
$$

Both policies then evaluate this student-generated trajectory. At each position $n$, they induce next-token distributions over $y_n \in \mathcal{V}$ conditioned on the same student prefix:

$$
p_{S}\!\left(y_{n}\mid x,\hat{y}_{<n}\right),\qquad p_{T}\!\left(y_{n}\mid x,y^{\star},\hat{y}_{<n}\right),
$$

where $\hat{y}_{<n}\triangleq(\hat{y}_{1},\ldots,\hat{y}_{n-1})$.

**Training objective: Full-vocabulary logit distillation.** A full-vocabulary divergence objective matches the teacher and student next-token distributions at each position. Given a student-generated sequence $\hat{y}$, define the trajectory-averaged, token-wise divergence:

$$
\begin{split}D\bigl(p_{T}\,\|\,p_{S}\bigr)(\hat{y}\mid x)&\triangleq\frac{1}{|\hat{y}|}\sum_{n=1}^{|\hat{y}|}D\biggl(p_{T}\!\left(\cdot\mid x,y^{\star},\hat{y}_{<n}\right)\\ &\qquad\big\|\;p_{S}\!\left(\cdot\mid x,\hat{y}_{<n}\right)\biggr),\end{split} \tag{6}
$$

where $D$ can be any divergence measure, such as the generalized **Jensen-Shannon divergence** $\operatorname{JSD}_\beta$, defined for a weight $\beta \in [0,1]$ as:

$$
\operatorname{JSD}_{\beta}(p_{T}\|p_{S})=\beta D_{KL}(p_{T}\|m)+(1-\beta)D_{KL}(p_{S}\|m) \tag{7}
$$

where $m=\beta p_{T}+(1-\beta)p_{S}$ is the interpolated mixture distribution. This full-vocabulary formulation provides dense, token-level feedback: the teacher, informed by $y^\star$, exposes the student to the entire distribution over plausible next tokens and guides it toward reasoning paths that lead to the correct answer.

The expected divergence is minimized over on-policy student samples:

$$
\mathcal{L}(\theta)=\mathbb{E}_{(x,y^{\star})\sim\mathcal{S}}\left[\mathbb{E}_{\hat{y}\sim p_{S}(\cdot\mid x)}\left[D\bigl(p_{T}\,\|\,p_{S}\bigr)(\hat{y}\mid x)\right]\right]. \tag{8}
$$

Gradients are backpropagated **only through the student policy** $p_S$, while the teacher $p_T$ acts as a fixed full-distribution target conditioned on privileged information $(x, y^\star)$.

**Per-Token Pointwise Divergence Clipping.** In experiments, token-level divergence is highly skewed across vocabulary entries: a small subset of **stylistic tokens** exhibits much higher divergence than mathematically meaningful tokens (see Table 5). This imbalance causes the training signal to be dominated by stylistic patterns. To address this, pointwise clipping is applied to the vocabulary-level divergence contributions. Let $D_f(p_T \| p_S)$ denote an $f$-divergence. At each token position $n$ and vocabulary entry $v$, define:

$$
\ell_{n,v}^{(f)}=p_{T}(v\mid\cdot)\;f\!\left(\frac{p_{S}(v\mid\cdot)}{p_{T}(v\mid\cdot)}\right).
$$

The **clipped divergence** is then computed as:

$$
D_{\mathrm{clip}}^{(f)}(p_{T}\|p_{S})=\frac{1}{|\hat{y}|}\sum_{n=1}^{|\hat{y}|}\sum_{v\in\mathcal{V}}\min(\ell_{n,v}^{(f)},\tau).
$$

**Alternative objective: Sampled-token distillation through policy gradient.** Following recent on-policy distillation methods [18], an alternative forms a **sampled-token reward signal** (a reverse-KL signal on sampled actions) and optimizes with policy gradient. For each position $n$ in a sampled sequence $\hat{y}$, define the advantage term:

$$
A_{n}(x,\hat{y})=\log p_{T}\!\left(\hat{y}_{n}\mid x,y^{\star},\hat{y}_{<n}\right)-\log p_{S}\!\left(\hat{y}_{n}\mid x,\hat{y}_{<n}\right),
$$

and optimize the policy-gradient-style objective:

$$
\begin{split}\mathcal{L}(\theta)&=-\mathbb{E}_{(x,y^{\star})\sim\mathcal{S}}\biggl[\mathbb{E}_{\hat{y}\sim p_{S}(\cdot\mid x)}\biggl[\frac{1}{|\hat{y}|}\sum_{n=1}^{|\hat{y}|}A_{n}(x,\hat{y})\\ &\qquad\times\log p_{S}\!\left(\hat{y}_{n}\mid x,\hat{y}_{<n}\right)\biggr]\biggr].\end{split} \tag{9}
$$

$A_n(x, \hat{y})$ is treated as a **constant with respect to** $\theta$ (gradients do not flow through the advantage), so that gradients take the usual policy-gradient form $A_n \nabla_\theta \log p_S$. Compared to the full-vocabulary divergence objective, this on-policy shaping objective operates only on sampled tokens, using the teacher's log-probabilities to provide dense, trajectory-level shaping signals without explicitly matching the full distribution at each step.

**OPSD as dense-reward policy gradient and comparison to STaR.** The objective in Equation 9 can be seen as policy gradient with **dense, token-level rewards**. Appendix D formalizes this and contrasts with **STaR** [45], a closely related method that also uses the same model to generate reasoning traces, then performs rejection sampling followed by SFT on correct traces. STaR can be viewed as policy gradient with a **sequence-level binary reward** that assigns identical credit to all tokens and vanishes when samples are incorrect. In contrast, **OPSD provides feedback at every token position regardless of final-answer correctness.**

> **Algorithm 1 — On-Policy Self-Distillation (OPSD).** For each problem-solution pair $(x, y^\star)$: (1) sample an on-policy student rollout $\hat{y}\sim p_S(\cdot\mid x)$; (2) evaluate both the student $p_S(\cdot\mid x,\hat{y}_{<n})$ and the privileged teacher $p_T(\cdot\mid x,y^\star,\hat{y}_{<n})$ at each position $n$ in a single forward pass each; (3) compute the per-token (clipped) divergence between them; (4) backpropagate only through the student. The teacher is fixed to the initial policy.
