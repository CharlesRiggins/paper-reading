## Appendix A — Limitations and Future Directions

Due to computational constraints, experiments are limited to models up to **8B parameters**. It remains an open question whether the trend continues at scales beyond 8B.

Several promising directions are identified:

- **Leveraging correctness verification.** The current framework does not explicitly leverage correctness verification of generated answers; incorporating such signals could provide additional learning objectives beyond distribution matching.
- **Curriculum learning by difficulty.** Problem difficulty plays a crucial role in self-distillation: if reasoning problems exceed the model's comprehension threshold, the teacher policy cannot provide meaningful supervision even with access to ground-truth solutions. This suggests **curriculum strategies** — gradually increasing problem difficulty as the model improves. Exploring adaptive curricula that maintain problems at the **frontier of model capabilities** is an important direction for scaling OPSD to harder reasoning tasks.

---

## Appendix B — Experimental Details

> **Table 5 — Per-token KL divergence by token category across generation styles.** Mean per-token KL divergence broken down by token category (math / style / other; see Appendix C), averaged over 10 problems. "Thinking Mode off/on" indicates whether the student or teacher prompt format enables thinking mode. The finding: when the **student's thinking mode is off and the teacher's thinking mode is on**, the KL signal on math-related tokens is the highest — and this setup is chosen for the main experiments.

Training and evaluation configurations for SFT, GRPO, and OPSD are provided in the paper's Tables 6, 7, and 8 (presented as figures). Key points:

- The main OPSD experiments adopt the **Thinking-Mode-off student / Thinking-Mode-on teacher** configuration.
- The clipping parameter $\tau$ was **not tuned**; the authors note that optimizing it may yield further performance gains within the same 100-step budget for larger models.
- All experiments used **8× A100 or H100 GPUs** with gradient checkpointing and **Flash Attention 2** for memory efficiency.
- Optimizer: **AdamW** [17]; precision: **bfloat16** for all training runs.
- For OPSD, unless otherwise stated, **full-vocabulary logit distillation** was used.
- Released training code: `https://github.com/siyan-zhao/OPSD`.

> **Tables 6, 7, 8 (figure-only in source):** Table 6 — training configuration for GRPO and OPSD; Table 7 — training configuration for SFT; Table 8 — evaluation parameters (incl. temperature $1.0$, max generation length $38$k for the main Table 2 evaluation).

---

## Appendix C — Token Category Definitions

Tokens are categorized into **style** and **math** groups using predefined keyword lists, used to analyze the per-token KL divergence between stylistic tokens and mathematical knowledge tokens (Section 4.3.1–4.3.2).

**Style Tokens:** maybe, perhaps, probably, possibly, let, okay, ok, alright, hmm, wait, because, since, so, thus, hence, therefore, but, however, although, though, yet, or, alternatively, instead, otherwise, actually, really, just, simply, basically, very, quite, pretty, rather, fairly, now, then, next, first, second, finally, try, see, check, note, recall, think, idea, strategy, approach, method, way, would, could, should, might, can, huge, large, big, small, tiny, interesting, tricky, complex, simple.

**Math Tokens:** exponential, exponent, power, powers, base, logarithm, logarithms, log, ln, compare, comparing, comparison, less, equal, larger, smaller, greater, factor, factors, prime, divisible, equation, expression, formula, inequality, rational, irrational, real, integer, coefficient, variable, constant, sum, product, difference, quotient, fraction, denominator, numerator, root, square, cube, nth, maximum, minimum, optimize, bound.

---

## Appendix D — Policy-Gradient Interpretation of OPSD and Comparison to STaR

The OPSD objective in Equation 9 can be interpreted as a **policy-gradient update with a dense, token-level reward** signal derived from privileged information. This appendix shows: (1) OPSD is a dense-reward policy gradient, and (2) STaR's learning signal is sequence-level while OPSD is token-level.

### D.1 STaR as Sequence-Level Policy-Gradient

STaR [45] can be viewed as an approximation to an RL-style policy gradient. The language model $p_\theta$ induces a joint distribution over rationale $r$ and answer $y$:

$$
p_{\theta}(r,y\mid x)=p_{\theta}(r\mid x)\,p_{\theta}(y\mid x,r),
$$

where the model first samples a latent rationale $r$ before predicting the final answer $y$. Given an indicator reward $R(y)=\mathbf{1}(y=y^{\star})$, the expected return across the dataset $\mathcal{S}=\{(x_i, y_i^\star)\}_{i=1}^N$ is

$$
J_{\text{STaR}}(\theta)=\sum_{i=1}^{N}\mathbb{E}_{(r,y)\sim p_{\theta}(\cdot\mid x_{i})}\big[\mathbf{1}(y=y_{i}^{\star})\big]. \tag{10}
$$

Applying the log-derivative trick yields a policy gradient:

$$
\nabla_{\theta}J_{\text{STaR}}(\theta)=\sum_{i=1}^{N}\mathbb{E}_{(r,y)\sim p_{\theta}(\cdot\mid x_{i})}\Big[\mathbf{1}(y=y_{i}^{\star})\,\nabla_{\theta}\log p_{\theta}(r,y\mid x_{i})\Big]. \tag{11}
$$

The indicator function **discards the gradient for all sampled rationales that do not lead to the correct answer** $y_i^\star$ — this corresponds to the filtering step in STaR. One limitation: STaR's reward is **sequence-level** — the binary indicator $\mathbf{1}(y=y^\star)$ provides the same signal to all tokens in a trajectory, offering no intermediate credit assignment, and when all sampled trajectories are incorrect, the learning signal vanishes.

### D.2 OPSD as Dense-Reward Policy Gradient

The sampled-token objective in Equation 9 can also be viewed as a policy-gradient method, but with a **token-level reward**. Fix a training pair $(x, y^\star)$ and let the student generate a trajectory $\hat{y}\sim p_S(\cdot\mid x)$. At each position $n$, define the per-token reward:

$$
r_{n}(x,\hat{y})\triangleq\log p_{T}(\hat{y}_{n}\mid x,y^{\star},\hat{y}_{<n})-\log p_{S}(\hat{y}_{n}\mid x,\hat{y}_{<n}).
$$

This reward measures how much the privileged teacher prefers the sampled token $\hat{y}_n$ relative to the student. The reward $r_n$ (equivalently, the advantage $A_n$) is treated as a **constant with respect to** $\theta$ (stop-gradient through both $p_T$ and $p_S$). Under this treatment, the gradient of Equation 9 takes the standard policy-gradient form:

$$
\nabla_{\theta}\mathcal{L}(\theta)=-\mathbb{E}_{(x,y^{\star})\sim\mathcal{S}}\left[\mathbb{E}_{\hat{y}\sim p_{S}(\cdot\mid x)}\left[\frac{1}{|\hat{y}|}\sum_{n=1}^{|\hat{y}|}r_{n}(x,\hat{y})\,\nabla_{\theta}\log p_{S}(\hat{y}_{n}\mid x,\hat{y}_{<n})\right]\right],
$$

which corresponds to maximizing the expected per-token reward along on-policy student rollouts:

$$
J_{OPSD}(\theta)=\mathbb{E}_{(x,y^{\star})\sim\mathcal{S}}\left[\mathbb{E}_{\hat{y}\sim p_{S}(\cdot\mid x)}\left[\frac{1}{|\hat{y}|}\sum_{n=1}^{|\hat{y}|}r_{n}(x,\hat{y})\right]\right].
$$

This reward is **dense**: it provides a learning signal at every token position, regardless of whether the final answer is correct.

### Comparison

Both STaR and OPSD can be understood as policy-gradient methods, but their reward structures differ fundamentally. **STaR** uses a sequence-level indicator $\mathbf{1}(y=y^\star)$ that assigns the same signal to all tokens; when all sampled trajectories are incorrect, the learning signal vanishes entirely. In contrast, **OPSD** provides a token-level reward $r_n$ at every position, enabling fine-grained credit assignment even when the final answer is wrong.
