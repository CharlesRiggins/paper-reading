# NeurIPS 2026 Review Form

---

## Summary

The paper starts from a neat observation: before any reasoning happens, LLM hidden representations already encode signals about problem difficulty. Building on this, the authors propose RLDP, a training-free and rollout-free method that judges whether a question is easy or hard by comparing its hidden states (from a single prefill forward pass) against a small set of easy/hard reference problems using shrinkage-regularized diagonal Mahalanobis distance and sign-based layer voting. On top of RLDP they add AdaSwitch, a lightweight controller that routes easy problems to a fast mode and hard problems to a slow mode. Experiments across math, code, and QA datasets on three models show RLDP gives balanced difficulty perception and achieves 1.34x-2x token efficiency over rollout-based methods.

---

## Strengths And Weaknesses

The core insight here is genuinely nice and clean. Moving difficulty perception out of the generation space and into the representation space is a well-motivated idea, and the fact that a single prefill pass with 10 reference pairs is enough to get usable signals is a strong practical result. The theoretical connection to LDA/LRT gives the method some classical grounding rather than being pure empirical hacking, and the experiments are fairly comprehensive — 5 datasets, 3 models, 6 baselines, plus solid ablations covering noise robustness, cross-dataset transfer, and a low-resource comparison against Probe.

That said, several issues hold me back from a higher score.

First, the difficulty definition is somewhat circular and model-relative. Easy and hard are defined by whether the model itself gets the answer right under different modes, and — more worryingly — problems that both modes fail are excluded entirely. This means the "truly hard" questions that would most stress-test the method are thrown out before evaluation even begins. We don't know how large this excluded set is or how the difficulty distribution shifts because of it, which makes the reported numbers hard to interpret in an absolute sense.

Second, the method only produces a binary easy/hard label. Real-world adaptive reasoning would benefit from a continuous difficulty score that can control the degree of reasoning effort, not just a coin flip between two modes. The paper acknowledges this as a future direction but doesn't engage with how much information is lost by collapsing to binary.

Third, the method requires white-box access to intermediate hidden states, which rules out all closed-weight and API-only models. This is a significant practical limitation — the models where adaptive reasoning efficiency matters most (frontier proprietary models with expensive inference) are exactly the ones where RLDP cannot be deployed.

Fourth, some of the baseline comparisons feel uneven. LLMs-Ranking collapses to predicting nearly all-hard (0/100 splits) on several datasets, and in the adaptive reasoning table the token efficiency advantage over Probe — which also uses representations and avoids generation — is marginal (1.85 vs 1.84 average TE on Qwen3-4B). The headline efficiency gains are largely over rollout-based methods that are inherently expensive, so the "1.34x-2x" framing, while technically correct, somewhat oversells the method's edge over the most relevant comparison point.

---

## Quality

3

---

## Clarity

3

---

## Significance

2

---

## Originality

3

---

## Questions

1. What fraction of problems are excluded under the dual-fail rule (both thinking and non-thinking fail)? How does the difficulty distribution look before and after exclusion? This would help assess whether the reported numbers generalize to the full problem distribution or only to a filtered subset.

2. How sensitive is the LVD layer selection to the choice of reference pairs? If you draw a different set of 10 reference pairs, does the selected layer change? This is important because the entire method hinges on picking the right layer, and the paper reports results over 10 seeds but doesn't explicitly analyze layer-selection stability.

3. Have you compared RLDP against a simple logit-based confidence baseline — e.g., using the model's own answer token probability or entropy as a difficulty proxy? If a cheap logit signal works comparably, the motivation for digging into hidden states weakens considerably.

4. How much does the diagonal covariance assumption matter in practice? The LDA/LRT theory assumes shared diagonal Gaussians, but Transformer hidden states are known to be highly anisotropic and non-Gaussian. Have you checked whether off-diagonal covariance terms or non-Gaussian structure meaningfully change the discriminant?

5. Could you report results with a continuous difficulty score (e.g., using the raw F(H_t) value) instead of the binary threshold? Even a simple 3-way routing (easy/medium/hard) would demonstrate whether the binary formulation is a real bottleneck.

---

## Limitations

The authors acknowledge two limitations: RLDP is a pre-reasoning estimate and not a replacement for full reasoning, and the method requires access to internal hidden states. These are fair. However, they do not discuss the circularity of the difficulty definition or the potential selection bias from excluding dual-fail problems, which I think should be added. The restriction to binary difficulty is also framed as a minor future direction rather than a meaningful limitation of the current approach.

---

## Rating

4

---

## Confidence

3

---

## Ethical Concerns

- [x] NO or VERY MINOR ethics concerns only

---

## Paper Formatting Concerns

None.

---

## Code Of Conduct Acknowledgement

- [x] Yes

---

## Responsible Reviewing Acknowledgement

- [x] Yes
