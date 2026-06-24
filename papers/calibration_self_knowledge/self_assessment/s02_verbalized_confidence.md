## 3. Verbalized Confidence

### 3.1 Objective and hypothesis

The first design choice exposes self-assessment **after reasoning**. The model produces a scalar estimate of whether its final answer is correct. The goal is not merely to improve calibration metrics, but to understand how such a signal can be learned without degrading the underlying reasoning process.

The central hypothesis is that the pretrained model already contains a weak confidence-related structure in its hidden trajectory but does not faithfully express that structure at the output level. Calibration training should sharpen this existing structure rather than replace the reasoning policy. Concretely, given a reasoning trajectory with hidden states $h_{1:T}$, the model learns to map this trajectory-level evidence to a confidence $c$ that better approximates the probability that the final answer is correct.

### 3.2 Confidence-aware reward

The model is trained with a simple confidence-aware reward. Let $p$ be the verbalized confidence associated with the final answer. The reward is

$$
r(x,y,p) = 2p - p^2 \quad \text{if the final answer is correct},
$$

and

$$
r(x,y,p) = -p^2 \quad \text{otherwise}.
$$

This directly rewards justified confidence and penalizes overconfident errors. The reward is applied only after the full reasoning trajectory has been completed. The key intuition is that GRPO should suppress confident wrong trajectories and amplify confident correct ones, improving self-assessment without a separate supervised confidence label.

### 3.3 Local trajectory-reweighting view

Under a small GRPO-style update, the post-update policy can be approximated as

$$
\pi_{\theta'}(z \mid x) \propto \pi_{\theta}(z \mid x) \exp(\eta r(z;x)),
\tag{1}
$$

where $z$ is a complete reasoning trajectory for input $x$ and $\eta > 0$ is an effective step size. For any two trajectories $z_1, z_2$ for the same question,

$$
\log \frac{\pi_{\theta'}(z_1 \mid x)}{\pi_{\theta'}(z_2 \mid x)}
=
\log \frac{\pi_{\theta}(z_1 \mid x)}{\pi_{\theta}(z_2 \mid x)}
+ \eta \left(r(z_1;x) - r(z_2;x)\right).
\tag{2}
$$

Thus higher-reward trajectories gain relative probability mass. Higher-confidence errors are downweighted more strongly, while higher-confidence correct answers are amplified. Importantly, the update redistributes mass among existing trajectories rather than requiring entirely new reasoning support. Appendix B formalizes this support-preserving reweighting.

### 3.4 Calibration results

On the calibration evaluation, training preserves response quality while sharply improving self-assessment quality:

| Metric | Base | Calibrated |
|---|---:|---:|
| Accuracy | 0.345 | 0.358 |
| ECE | 0.383 | 0.049 |
| Brier score | 0.504 | 0.166 |
| NLL | 4.987 | 0.498 |
| Overconfidence gap | +0.523 | +0.045 |

The baseline is dominated by confidently wrong predictions, whereas the calibrated model assigns substantially lower confidence to incorrect answers. The paper emphasizes that calibration does not simply rescale confidence; it suppresses overconfident error without degrading reasoning accuracy.

Training dynamics are reported in Appendix D.1, including reward curves and reliability diagrams. Reward-format ablations and Qwen cross-family results are reported in Appendix D.3.

### 3.5 Mechanism evidence: logit lens and PCA

The paper complements the headline calibration result with two views of the confidence-token hidden state:

1. A **logit lens** aggregating predicted digits into LOW, MID, and HIGH confidence bins.
2. A **PCA projection** of final-layer activations.

**Figure 2, cleaned caption:** Logit-lens analysis of the confidence-token hidden state. Calibration sharpens late-layer confidence routing and yields a cleaner final-layer confidence structure.

**Figure 3, cleaned caption:** PCA analysis of the confidence-token hidden state, colored by verbalized confidence.

Both views tell the same story. In the base model, correct and wrong answers both end with dominant mass in the HIGH bin, and the PCA geometry is broad and diffuse. After calibration, low-confidence errors are redirected away from HIGH toward LOW; correct answers become more conservative rather than saturating the maximum digit; and PCA geometry becomes smoother and more ordered along a low-to-high confidence axis.

These observations support the late-stage mapping hypothesis: the underlying signal already exists in the pretrained model, and training makes its translation into verbalized confidence more selective and cleanly separated.

### 3.6 Error analysis: epistemic vs. aleatoric errors

The paper analyzes how calibration changes the type of errors the model makes. It defines:

- **Epistemic errors:** wrong answers with confidence above 0.5.
- **Aleatoric errors:** wrong answers with confidence at most 0.5.
- **Strict epistemic errors:** wrong answers with confidence above 0.7, isolating strongly overconfident hallucinations.

Incorrect responses are also classified by an LLM judge that reads the reasoning text and explicitly ignores the final confidence value.

**Table 1. Aggregate error decomposition**

| Error type | Base | Calibrated |
|---|---:|---:|
| Epistemic | 92.4% | 34.9% |
| Aleatoric | 7.6% | 65.1% |
| Strict epistemic | 88.6% | 3.9% |

The sharpest qualitative shift is that the baseline’s errors are almost all epistemic and most are strongly overconfident, whereas after calibration the majority of errors become low-confidence errors. The strict epistemic rate drops by more than an order of magnitude.

The same conversion holds across datasets, though its strength varies. The largest reductions occur on MuSiQue and HotpotQA, while Natural Questions remains harder: strongly overconfident errors nearly disappear, but some mistakes remain in the moderate-confidence range. Appendix D.1 further shows that calibration increases confidence separation between correct and incorrect answers rather than simply shifting all scores downward.

### 3.7 Takeaway

The main behavioral conclusion of verbalized confidence is that the model changes from being **confidently wrong** to being **uncertain when wrong**. The mechanism evidence suggests this happens by sharpening a confidence-related structure already present in the pretrained model, not by rewriting the whole reasoning process.