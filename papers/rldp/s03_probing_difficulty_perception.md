## 3. Probing LLM's Perception of Problem Difficulty

This section examines whether LLMs can perceive problem difficulty and where the difficulty signal is located. The paper evaluates the capability from two complementary perspectives: (i) explicit difficulty judgments elicited through prompting, and (ii) implicit difficulty signals reflected in layer-wise Transformer hidden representations.

### 3.1 Can LLMs Directly Judge Problem Difficulty?

The authors first assess explicit difficulty-discrimination ability. They design a targeted prompt that asks the model to classify each input problem as easy or hard, then test it across multiple LLMs. These results appear in the **ZERO-SHOT** row of Table 1.

The zero-shot outcomes show a consistent failure pattern. For most models and datasets, predictions collapse to a near-constant label, which indicates that the models do not produce calibrated difficulty judgments when directly prompted in a zero-shot setting.

The paper then asks whether in-context exemplars can elicit a more stable notion of difficulty. It constructs a few-shot prompt that provides several easy and hard examples before querying a new instance. As shown in the **FEW-SHOT** row of Table 1, few-shot prompting improves discrimination on some datasets, but the gains are limited and inconsistent across models and benchmarks.

The authors suggest a possible explanation: in-context learning may capture superficial input-output regularities rather than content-level mappings from inputs to latent properties [31, 12]. Therefore, prompt-based explicit difficulty judgments remain unreliable as a control signal for adaptive reasoning. Prompt and generation details are provided in Appendix C.1.

> **Takeaway 1.** LLMs do not yield reliable difficulty perception with explicit prompting. Zero-shot prompting yields highly skewed predictions with poor calibration, and few-shot demonstrations provide only limited and inconsistent gains.

### 3.2 Are Easy and Hard Problems Separable in the Hidden Representation Space?

Inspired by the fact that humans can often form a coarse sense of difficulty at first glance, the paper hypothesizes that LLMs may exhibit a similar intuitive signal encoded in hidden representations.

#### 3.2.1 Extracting Layer-wise Representations

For each problem, the paper obtains hidden representations $H$ by extracting the hidden representation of the last token from every Transformer layer, as defined in Eq. (1). The set of hidden representations for all easy problems is denoted by $H^+$, and the set for all hard problems is denoted by $H^-$.

#### 3.2.2 Measuring Layer-wise Separability

To assess whether easy and hard problems diverge internally, the authors compute class-wise mean representations at each layer. For layer $l$, the layer-wise means are denoted by $\mu_l^+$ and $\mu_l^-$ for easy and hard examples respectively.

The means are jointly z-score normalized to obtain $\hat{\mu}_l^+$ and $\hat{\mu}_l^-$. Their separation is quantified by the Euclidean distance

$$
\mathrm{dist}_l = \left\| \hat{\mu}_l^+ - \hat{\mu}_l^- \right\|_2,
$$

which measures how distinct the easy and hard representations are at layer $l$. The authors also compute cosine distance between $\mu_l^+$ and $\mu_l^-$ as a complementary metric.

#### 3.2.3 Main Observations from Figure 2

Figure 2 reports layer-wise center-distance analysis for Qwen3-4B across different metrics and settings.

- **Figure 2a:** normalized Euclidean distance on Olympiad, AIME, and MATH500.
- **Figure 2b:** cosine distance on the same datasets.
- **Figure 2c:** token-length controlled normalized Euclidean distance, restricted to 55–65 tokens.

For both distance metrics, separability is weak in early layers, increases with depth, peaks in middle layers, and then tapers in upper layers. The token-length-controlled analysis preserves the same layer-wise separability pattern, which rules out the simple explanation that hard problems merely have longer questions. Additional details appear in Appendix B.8.

The paper observes similar layer-dependent separability patterns across multiple LLMs, although exact trends vary. This suggests that difficulty-related information is broadly reflected in hidden representations. At the same time, low-dimensional projections fail to preserve the signal: class distributions become entangled and center distances fluctuate irregularly across layers, as discussed in Appendix B.9. This supports the claim that difficulty perception relies on high-dimensional hidden-state structure.

> **Takeaway 2.** Difficulty is encoded implicitly in hidden-layer representations of LLMs. Easy and hard question instances exhibit consistent, layer-dependent separability in hidden representations across different datasets and LLMs.
