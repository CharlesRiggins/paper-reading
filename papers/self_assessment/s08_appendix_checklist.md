## Appendix A. Related Work

### A.1 Uncertainty estimation and calibration in language models

A large body of work estimates whether generated answers are reliable. Classical calibration asks whether predicted probabilities match empirical correctness, with temperature scaling as a widely used post-hoc method. For language models, uncertainty is harder because outputs are free-form sequences rather than fixed classes.

Recent surveys and benchmarks organize LLM uncertainty into likelihood-based, sampling-based, semantic, verbalized, and hybrid methods. Semantic uncertainty methods aggregate generations by meaning rather than surface string, while other work studies black-box uncertainty, rank calibration, and subjective uncertainty in natural language generation. This paper is complementary: instead of only estimating uncertainty after generation, it trains the model to expose self-assessment explicitly within the response itself.

### A.2 Verbalized confidence and model self-knowledge

Prior work asks whether language models know when they are likely to be correct. Models can be trained to express uncertainty in words, and P(True)-style prompting can query a model’s estimate of whether its answer is correct. However, verbalized confidence scores are often prompt-sensitive and may remain overconfident without additional training.

This paper builds on verbalized confidence but differs in two ways. First, confidence is trained with a proper-scoring-rule-style reward rather than elicited only by prompting. Second, the paper analyzes how calibration changes hidden confidence structure.

### A.3 Reward-based post-training for calibrated behavior

Post-training with reinforcement learning is a standard way to shape LLM behavior. Calibration-aware rewards and confidence-aware RL objectives can encourage lower confidence on incorrect answers and higher confidence on correct answers.

The paper’s trajectory-reweighting analysis gives a local mathematical account: under a small policy-improvement step, the reward tilts probability mass away from overconfident wrong trajectories and toward confident correct trajectories. This frames reward shaping as support-preserving redistribution of existing reasoning trajectories, not merely post-hoc rescaling.

### A.4 Selective prediction, abstention, and retrieval decisions

Uncertainty estimates are useful only when connected to downstream decisions such as abstention, tool use, or retrieval. Adaptive-RAG, FLARE, DRAGIN, ADARAGUE, SEAKR, Probing-RAG, and other retrieval systems use complexity, confidence, or uncertainty features to decide when retrieval is worthwhile.

This paper differs in where the signal comes from: instead of relying only on external uncertainty features or passive detectors, it trains the generator itself to emit a confidence score or reasoning-time `<uncertain>` marker that can serve as a trigger.

### A.5 Learned tokens, control markers, and intervention

A related line studies whether special tokens or learned markers can package complex behaviors into compact controllable symbols. Gist tokens compress prompts into learned handles; neologism learning studies new controllable token meanings; Self-RAG trains retrieval and critique tokens; backtracking tokens mark unsafe or undesirable generation paths.

The paper’s `<uncertain>` marker is closest to learned control-token methods, but its specific role is to expose a high-risk reasoning state before the final answer. It is step-level, binary, and intervention-oriented, unlike final-answer scalar confidence.

### A.6 Internal states and mechanistic evidence

Several works suggest hidden states contain information about truthfulness, correctness, or hallucination risk not always faithfully expressed in output. This paper follows that motivation but compares two trained self-assessment signals. It finds that verbalized confidence largely preserves representation geometry while sharpening a confidence-related structure, whereas `<uncertain>` produces broader late-layer changes around emission.

## Appendix B. Proofs for Trajectory-Reweighting Analysis

The appendix gives short proofs for the theoretical claims in Section 3. It uses the tilted-distribution idealization

$$
\pi_{\theta'}(z \mid x) \propto \pi_{\theta}(z \mid x)\exp(\eta r(z;x)),
\tag{4}
$$

as a first-order analytical model of one-step uncertainty-aware policy improvement under GRPO.

For fixed input $x$, define the partition function

$$
Z(x) = \sum_z \pi_\theta(z \mid x) \exp(\eta r(z;x)).
\tag{5}
$$

Then

$$
\pi_{\theta'}(z \mid x)=\frac{\pi_\theta(z \mid x)\exp(\eta r(z;x))}{Z(x)}.
\tag{6}
$$

### B.1 One-step relative improvement

For any two trajectories $z_1,z_2$ for the same input $x$,

$$
\log \frac{\pi_{\theta'}(z_1\mid x)}{\pi_{\theta'}(z_2\mid x)}
=
\log \frac{\pi_{\theta}(z_1\mid x)}{\pi_{\theta}(z_2\mid x)}
+\eta(r(z_1;x)-r(z_2;x)).
\tag{7}
$$

Thus a single improvement step increases the relative likelihood of higher-reward trajectories and decreases that of lower-reward trajectories.

### B.2 Selective suppression of overconfident errors

For wrong trajectories, the main reward is

$$
r(z;x)=-p(z)^2.
\tag{14}
$$

This is strictly decreasing in confidence $p(z)$ over $[0,1]$. Therefore, among two wrong trajectories with $p(z_1)>p(z_2)$, the higher-confidence wrong trajectory receives lower reward and is downweighted more strongly. For correct trajectories, the reward

$$
r(z;x)=2p(z)-p(z)^2
\tag{19}
$$

is increasing in $p(z)$ over $[0,1]$, so higher-confidence correct trajectories are relatively amplified.

### B.3 Answer improvement without new knowledge

Define the confidence-weighted score of answer $y$ as

$$
S_\theta(y\mid x)=\sum_{z:g(z)=y}\pi_\theta(z\mid x)p(z),
\tag{20}
$$

and define the answer margin

$$
\Gamma_\theta(x)=S_\theta(y^\star\mid x)-\max_{y\ne y^\star}S_\theta(y\mid x),
\tag{21}
$$

where $y^\star$ is the correct answer. If the update makes $\Gamma_{\theta'}(x)>0$ while $\Gamma_\theta(x)\le 0$, the model’s prediction flips from incorrect to correct without introducing any new reasoning trajectory.

### B.4 Support-preserving answer reweighting theorem

The theorem states that if every trajectory producing the correct answer has reward at least $a$ and every trajectory producing a competing wrong answer $\bar y$ has reward at most $b$, with $a>b$, then

$$
\frac{M_{\theta'}(y^\star\mid x)}{M_{\theta'}(\bar y\mid x)}
\ge
\exp(\eta(a-b))
\frac{M_\theta(y^\star\mid x)}{M_\theta(\bar y\mid x)}.
\tag{31}
$$

Moreover,

$$
\operatorname{supp}\pi_{\theta'}(\cdot\mid x)=\operatorname{supp}\pi_\theta(\cdot\mid x).
\tag{32}
$$

The interpretation is that the update cannot create a correct trajectory absent from the original support; it can only increase the relative mass of correct trajectories already present but underweighted.

For the Brier-style verbal-confidence reward,

$$
r(z;x)=
\begin{cases}
2p(z)-p(z)^2, & g(z)=y^\star,\\
-p(z)^2, & g(z)\ne y^\star.
\end{cases}
\tag{49}
$$

If correct-answer trajectories satisfy $p(z)\ge \alpha$, then wrong trajectories are exponentially downweighted relative to correct trajectories by a factor depending on $\eta(2\alpha-\alpha^2)$.

## Appendix C. Additional Mechanistic Evidence

### C.1 Expanded token-level divergence analysis

The main text emphasizes per-token localization. Appendix C adds a total-KL-mass view, while warning that long reasoning spans dominate raw mass simply because they occupy many more positions than uncertainty tokens.

**Figure 7, cleaned caption:** KL mass fractions by token type. Under verbalized confidence, most total KL mass lies in reasoning tokens even though confidence digits are enriched on a per-token basis. Under `<uncertain>`, the marker token is enriched, but total KL mass remains dominated by reasoning and nearby context tokens.

The point is that uncertainty is computed across the reasoning trace and becomes especially visible only at a small number of output positions. Under verbalized confidence, the `Confidence:` label is essentially inert; training alters the scalar value rather than the template. Under `<uncertain>`, elevated KL in nearby pre/post windows suggests a local uncertainty-related computation regime around emission.

### C.2 Hidden-state patching

Activation patching argues against treating the uncertainty token itself as the entire causal mechanism. For verbalized confidence, patching hidden states at confidence-digit positions produces little disruption, while patching random reasoning positions produces larger changes. For `<uncertain>`, patching the marker position matters, but random reasoning positions are still more disruptive on average.

**Figure 8, cleaned caption:** Hidden-state patching at the signal position versus reasoning positions. In both methods, patching the reasoning trace is at least as disruptive as patching the signal token itself, consistent with uncertainty being assembled across the trajectory and exposed at the designated output position.

### C.3 Parameter-space drift and embedding repositioning

Both calibrated models place most parameter drift in attention value/output projections and MLP projections, with minimal change in normalization layers. The overall update magnitudes are comparable.

**Figure 9, cleaned caption:** Relative Frobenius weight drift across layers and module types. Both calibrated models show similar update structure in parameter space, with the largest changes in `v_proj`, `o_proj`, and MLP projection layers.

This makes the difference in representation geometry especially meaningful: similar magnitudes of weight drift yield different geometric consequences. Verbalized confidence can be realized through a geometry-preserving readout adjustment, whereas the explicit marker encourages deeper reorganization of the computation that produces the marker.

The embedding-drift analysis also supports this distinction. Under `<uncertain>`, token embeddings corresponding to components of `<uncertain>` drift more than a random-token baseline; under verbalized confidence, those component tokens drift less than baseline on average.

### C.4 Mechanism-to-utility linkage

For verbalized confidence, localization-related features predict per-example confidence shifts with cross-validated $R^2=0.51$. Examples in the top localization quartile show confidence shifts 86% larger than those in the bottom quartile (0.207 vs. 0.111). This supports the claim that verbalized-confidence training learns when to engage a stronger confidence adjustment based on information accumulated in the reasoning trace.

For `<uncertain>`, the same continuous utility proxy is weak and unstable. The paper argues that the marker’s operative mechanism is better framed as a binary emission decision than as graded variation within already-emitting examples.

### C.5 Summary

The additional mechanism analyses reinforce three points:

1. Localization is real but should be understood per-token rather than by raw KL mass.
2. Localization does not mean the uncertainty token itself is the whole mechanism; supporting computation is distributed across the reasoning trajectory.
3. Verbalized confidence and `<uncertain>` differ in how deeply they rewrite the computation that supports uncertainty.

## Appendix D. Supplementary Quantitative Results

### D.1 Detailed results for verbalized confidence

**Table 5. Calibration metrics for Llama-3-8B before and after calibration training**

| Metric | Llama-3-8B (base) | Llama-3-8B (calibrated) |
|---|---:|---:|
| Accuracy ↑ | 0.345 | 0.358 |
| Avg. verbalized confidence | 0.869 | 0.403 |
| Overconfidence gap (conf − acc) ↓ | +0.523 | +0.045 |
| ECE ↓ | 0.383 | 0.049 |
| Brier score ↓ | 0.504 | 0.166 |
| NLL (confidence) ↓ | 4.987 | 0.498 |
| Parse rate ↑ | 0.996 | 1.000 |

**Table 6. Dataset-level verbalized-confidence summary**

| Dataset | n | Accuracy Base | Accuracy Cal. | ECE Base | ECE Cal. | AUSC Base | AUSC Cal. | Overconf. Base | Overconf. Cal. | Conf. on wrong Base | Conf. on wrong Cal. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMultihopQA | 500 | 22.4 | 28.0 | 0.407 | 0.164 | 0.267 | 0.458 | 84.8 | 13.3 | 0.823 | 0.343 |
| HotpotQA | 500 | 35.7 | 35.6 | 0.459 | 0.085 | 0.477 | 0.536 | 92.2 | 4.0 | 0.854 | 0.293 |
| MuSiQue | 500 | 12.1 | 14.4 | 0.543 | 0.089 | 0.126 | 0.224 | 84.6 | 0.0 | 0.809 | 0.207 |
| NQ | 500 | 46.5 | 48.2 | 0.367 | 0.033 | 0.560 | 0.625 | 96.3 | 0.4 | 0.888 | 0.426 |
| TriviaQA | 500 | 63.9 | 68.4 | 0.242 | 0.211 | 0.749 | 0.838 | 94.4 | 0.6 | 0.866 | 0.329 |
| Aggregate | 2500 | 35.4 | 38.0 | 0.408 | 0.119 | 0.430 | 0.526 | 89.9 | 3.5 | 0.869 | 0.360 |

Table 7 further breaks down error confidence bands, per-dataset conversion, confidence separation, question-level consistency, and residual calibration by confidence bin. Key numbers include wrong-answer confidence mean dropping from 0.837 to 0.306, median dropping from 0.900 to 0.200, and high-confidence wrong errors ($c>0.7$) dropping from 1482 cases (88.6%) to 63 cases (3.9%).

### D.2 Additional results for reasoning-time signaling

**Table 8. Base model vs. calibrated model on factual reasoning datasets**

| Dataset | Accuracy Base | Accuracy Cal. | Δ | Answer Line Base | Answer Line Cal. | Emit Rate Base | Emit Rate Cal. | W+E Base / Cal. | C+E Base / Cal. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMultihopQA | 12.4 | 25.6 | +13.2 | 51.4 | 100.0 | 43.2 | 59.8 | 43.2 / 55.4 | 0.0 / 4.4 |
| HotpotQA | 22.0 | 27.8 | +5.8 | 62.8 | 99.6 | 36.6 | 70.8 | 35.8 / 59.0 | 0.8 / 11.8 |
| MuSiQue | 4.2 | 6.6 | +2.4 | 50.8 | 100.0 | 49.8 | 94.6 | 49.8 / 89.0 | 0.0 / 5.6 |
| NQ | 21.6 | 40.8 | +19.2 | 77.0 | 100.0 | 23.6 | 56.6 | 23.0 / 39.6 | 0.6 / 17.0 |
| TriviaQA | 36.2 | 56.0 | +19.8 | 72.0 | 100.0 | 31.8 | 44.2 | 30.2 / 31.6 | 1.6 / 12.6 |
| Macro Avg. | 17.67 | 28.53 | +10.86 | 58.90 | 99.93 | 38.53 | 68.87 | 37.97 / 58.70 | 0.57 / 10.17 |

**Table 9. Probe diagnostics for the `<uncertain>` marker**

Layer sweep on emitted dev examples:

| Layer | Dev AUROC | Dev AUPRC | Trigger Precision | Trigger Recall | Trigger F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6136 | 0.8365 | 0.8008 | 0.8216 | 0.8111 |
| 8 | 0.7291 | 0.8951 | 0.8040 | 0.8798 | 0.8402 |
| 16 | 0.7382 | 0.8940 | 0.8324 | 0.8657 | 0.8487 |
| 24 | 0.6915 | 0.8662 | 0.8171 | 0.8597 | 0.8379 |
| Final | 0.7371 | 0.8926 | 0.8190 | 0.8798 | 0.8483 |

Emitted subset:

| Statistic | Base | Cal. |
|---|---:|---:|
| Train emit cases | 1233 | 6334 |
| Dev emit cases | 133 | 649 |
| Wrong@emit (dev) | 0.9774 | 0.7689 |
| Correct@emit (dev) | 0.0226 | 0.2311 |
| Total wrong (dev) | 848 | 653 |

### D.3 Reward ablations and cross-family transfer

**Table 10. Reward ablations**

Verbalized confidence reward ablation:

| Setting | Reward / format | Acc. | Brier | ECE | Overconf. | Gap |
|---|---|---:|---:|---:|---:|---:|
| Base prompt | none, decimal | 24.5 | -0.337 | 0.647 | 89.4 | +0.111 |
| Final design | Brier, decimal | 27.4 | +0.110 | 0.133 | 3.0 | +0.330 |
| Format variant | Brier, integer-style | 24.6 | +0.043 | 0.234 | 25.5 | +0.611 |

Local marker reward ablation:

| Setting | Reward order {C¬E, CE, WE, W¬E} | Acc. | Emit | Rec.-err. | Sep. | Len. |
|---|---|---:|---:|---:|---:|---:|
| Base prompt | none | 29.7 | 30.0 | 39.3 | +31.2 | 568 |
| Early asymmetric | {+5, +1, 0, -1} | 48.7 | 40.0 | 61.2 | +45.2 | 412 |
| Final design | {+5, +3.5, 0, -2} + spam penalty | 46.6 | 57.5 | 76.0 | +40.3 | 491 |

**Table 11. Cross-family robustness results**

Verbalized confidence:

| Family | Recipe | Acc. | Brier | ECE | Overconf. | Gap | Conf. support |
|---|---|---:|---:|---:|---:|---:|---|
| Llama-3-8B | Brier GRPO | 27.4 | +0.110 | 0.133 | 3.0 | +0.330 | 7 values |
| Qwen2.5-7B | Brier GRPO | 21.1 | +0.039 | 0.039 | 7.2 | +0.636 | 8 values |

`<uncertain>` marker:

| Family | Recipe | Acc. | Emit | Rec.-err. | Sep. | Mean len. | CNE / CE / WE / WNE |
|---|---|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | marker GRPO | 47.1 | 57.7 | 72.9 | +32.7 | 478 | 27.9 / 19.2 / 38.5 / 14.4 |
| Qwen2.5-7B | marker GRPO | 41.7 | 55.6 | 72.9 | +40.8 | 228 | 28.6 / 13.1 / 42.5 / 15.8 |

The intended claim is qualitative transfer: the same objectives induce useful self-assessment behavior in both model families, although learned equilibria differ.

## Appendix E. Experimental Setup

The appendix records experimental configuration needed to reproduce the main `<uncertain>`-marker training runs. Code, configurations, and checkpoints are stated to be released after de-anonymization.

### E.1 Main training environment

Training uses:

- `verl/HybridFlow`,
- vLLM v0.8.5,
- Hugging Face Transformers,
- bfloat16 precision,
- FSDP2,
- 2 NVIDIA H100 80GB GPUs.

Main experiments use `meta-llama/Llama-3.1-8B-Instruct`; cross-family robustness uses `Qwen/Qwen2.5-7B-Instruct`.

### E.2 Data and reward

The `<uncertain>`-marker training data contains 11,000 training prompts and a 1,100-example held-out dev split derived from TriviaQA-style factual prompts with during-reasoning self-assessment traces. The same prompt set is used for supervised warm start and GRPO post-training.

The rule-based reward combines final-answer correctness with whether the decoded response contains `<uncertain>`:

$$
r(y,y^\star)=
\begin{cases}
+5.0 & \text{if correct and no emit},\\
+3.5 & \text{if correct and emit},\\
0.0 & \text{if wrong and emit},\\
-2.0 & \text{if wrong and no emit}.
\end{cases}
\tag{52}
$$

A spam penalty of -0.5 per extra emission, capped at -2.0, is applied when a response contains more than two `<uncertain>` emissions.

### E.3 Supervised warm-start hyperparameters

The supervised warm-start stage uses:

- AdamW,
- learning rate $1.0\times 10^{-5}$,
- linear warmup ratio 0.05,
- gradient clipping 1.0,
- 2 epochs,
- global batch size 256,
- micro-batch size 4 per GPU,
- maximum sequence length 2048,
- left truncation,
- seed 42,
- bf16 precision,
- FSDP2,
- masked cross-entropy on assistant turns only,
- checkpoints every 42 steps.

### E.4 GRPO hyperparameters

GRPO is applied after warm start with:

- AdamW actor learning rate $1.0\times 10^{-6}$,
- KL coefficient $\beta=0.01$,
- token-level k1 KL,
- global prompt batch size 256,
- PPO mini-batch size 64,
- PPO micro-batch size 16 per GPU,
- maximum prompt length 1024,
- maximum response length 512,
- vLLM rollout with tensor-parallel size 2,
- rollout temperature 1.0,
- top-p 0.95,
- rollout group size $n=1$,
- reference log-prob micro-batch size 32 per GPU,
- gradient checkpointing and padding removal,
- 5 epochs, approximately 43 steps per epoch and 215 total steps,
- validation every 5 steps,
- checkpoints every 50 steps,
- no separate critic; `algorithm.adv_estimator=grpo`.

All `<uncertain>` evaluation uses greedy vLLM inference on the 1,100-example dev split with temperature 0.0, tensor-parallel size 1, GPU memory utilization 0.85, maximum model length 4096, maximum response length 512, and literal `<uncertain>` emissions preserved.

### E.5 Baseline implementation details

For verbalized confidence, Panel A of Table 3 uses a 2WikiMultihopQA evaluation set ($n=500$). The model always emits an answer and decimal confidence $p\in[0,1]$. Metrics are accuracy, Brier reward, ECE, and overconfident wrong rate.

For `<uncertain>`, Panel B uses the 1,100-example counterfactual evaluation set. Each method produces a binary trigger analogous to uncertainty emission. Metrics are trigger rate, precision, recall, untouched-set accuracy, and wrong rate within triggered examples.

Baselines include direct base confidence, P(True), global/adaptive temperature scaling, SFT confidence variants, emit heuristic, hidden-state probe, output classifier, Self-RAG, FLARE, and ADARAGUE. All baseline generations use greedy decoding with vLLM, and all methods except Self-RAG share the same Llama-3.1-8B-Instruct base model.

## Appendix F. Epistemic Error and Aleatoric Error Examples

### F.1 Judge prompt for error-type classification

The LLM judge classifies an incorrect response as **EPISTEMIC** or **ALEATORIC** based on reasoning content, explicitly ignoring the final confidence number.

Definitions:

- **EPISTEMIC:** the model is confident and assertive, with no hedging or acknowledgement that it might be wrong. It “thinks it knows” even though it is wrong.
- **ALEATORIC:** the model expresses uncertainty, hedges, admits lack of information, or signals it is guessing. It “knows it doesn’t know.”

The judge receives the question, model response, and gold answer, and must output:

```text
Classification: EPISTEMIC | ALEATORIC
Reasoning: <one sentence explaining the key signal in the response text>
```

### F.2 Qualitative examples of epistemic and aleatoric errors

The appendix shows examples such as:

- A baseline epistemic hallucination on HotpotQA: for “Glad to Be Unhappy,” the model predicts Randy Newman with confidence 0.9, while the gold answer is Richard Charles Rodgers.
- A residual calibrated epistemic error on 2WikiMultihopQA: the model misattributes a film’s country and confidently concludes “No” with confidence 0.9.
- A calibrated aleatoric error on Natural Questions: the model gives a wrong answer but says it is not sure whether Spanish Town was ever Jamaica’s capital and emits confidence 0.05.

The examples illustrate that calibration training does not eliminate all confident errors, but shifts many wrong responses toward explicit uncertainty.

### F.3 Four-way examples for the `<uncertain>` marker

The marker is binary and local to the reasoning trace. The appendix shows a four-way decision table:

| Case | Meaning |
|---|---|
| Correct + no marker | Desired no-intervention case; the model answers correctly without unnecessary uncertainty. |
| Wrong + marker | Intended failure mode; the answer is wrong, but the model flags an answer-critical uncertainty that a controller can use. |
| Correct + marker | False-positive mode; the answer is correct, but the marker fires on ambiguity or peripheral uncertainty. |
| Wrong + no marker | Residual silent-failure mode; the model commits to a wrong answer without recognizing risk. |

Examples include Tellurium as a correct/no-marker case, a wrong composer-country case with marker emitted, a correct 1966 World Cup answer with marker emitted on phrasing ambiguity, and a show-jumping refusal question where the model wrongly answers 4 faults with no marker.

## Appendix G. Broader Impact Statement

The work aims to reduce confident hallucinations by training LLMs to expose self-assessment within their own responses. Such signals may help downstream systems trigger retrieval, verification, abstention, or human review before acting on unreliable outputs.

However, explicit self-assessment can also create over-trust. A high confidence score is not a correctness guarantee, and the absence of `<uncertain>` does not imply safety. The methods should be used as control cues rather than standalone certificates of correctness. High-stakes deployment would require domain-specific calibration, external verification, and human oversight.

## NeurIPS Paper Checklist Summary

| # | Topic | Answer | Notes |
|---:|---|---|---|
| 1 | Claims | Yes | Main contributions and scope are stated in abstract and introduction. |
| 2 | Limitations | Yes | Main body discusses limitations, especially factual QA/adaptive retrieval scope and single-marker design. |
| 3 | Theory assumptions and proofs | Yes | The paper points to appendix proofs for trajectory-reweighting analysis. |
| 4 | Experimental reproducibility | Yes | Hyperparameters and training environment are disclosed; framework built on open-source HybridFlow. |
| 5 | Open access to data and code | No | No public repository during anonymous review; release promised after de-anonymization. |
| 6 | Experimental setting/details | Yes | Appendix E specifies data splits, training, evaluation, and baseline details. |
| 7 | Statistical significance | No | Reports point estimates and dataset sizes, but no formal error bars/significance tests; most metrics are single-run. |
| 8 | Compute resources | Yes | Uses 2 H100 80GB GPUs for post-training. |
| 9 | Code of ethics | Yes | Authors state conformance with NeurIPS code of ethics. |
| 10 | Broader impacts | Yes | Discusses reliability benefits and over-trust risks. |
| 11 | Safeguards | N/A | Paper states no high-risk release safeguards are required. |
| 12 | Licenses for existing assets | Yes | Mentions training framework and base models. |
| 13 | New assets | N/A | Paper does not release new assets at submission. |
| 14 | Crowdsourcing/human subjects | N/A | No crowdsourcing or human subjects. |
| 15 | IRB approvals | N/A | No human-subject research. |
| 16 | Declaration of LLM usage | Yes | LLMs are the main objects of study; an LLM judge is used for epistemic/aleatoric classification. |