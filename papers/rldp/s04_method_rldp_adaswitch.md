## 4. Leveraging LLM's Representation Intuition: No Fine-tuning, No Extra Rollout

The observations in Section 3 motivate **Referential Latent Difficulty Perception (RLDP)**, which perceives problem difficulty directly from layer-wise hidden representations. RLDP requires neither fine-tuning nor additional rollouts or answer generation. The paper then builds **RLDP-AdaSwitch**, a lightweight controller that dynamically allocates reasoning effort and explicitly trades computation for accuracy.

### 4.1 RLDP: Referential Latent Difficulty Perception

RLDP is based on the finding that difficulty information is encoded in a layer-dependent manner. It infers difficulty before decoding by contrasting a target problem's layer-wise hidden representations with those from a small set of easy and hard reference problems.

#### 4.1.1 Layer-wise Variance-Normalized Discriminability

Not all layers contribute equally to difficulty discrimination. To identify layers that reliably encode difficulty-related information, the paper introduces **Layer-wise Variance-Normalized Discriminability (LVD)**.

Let the hidden representations of easy and hard reference problems be $H^+$ and $H^-$. Let the hidden representation of a target problem be

$$
H_t \in \mathbb{R}^{(L+1)\times d}.
$$

For each layer $l$, the class-wise mean representations of easy and hard reference problems are $\mu_l^+$ and $\mu_l^-$. These are plug-in estimates computed from the reference set.

Mean separation alone can be misleading because Transformer representation spaces may have anisotropic scaling. RLDP therefore estimates a pooled variance:

$$
\sigma_l^2 = \frac{1}{2}\left(\mathrm{Var}(H^+_{:,l,:}) + \mathrm{Var}(H^-_{:,l,:})\right).
\tag{2}
$$

This variance estimate is stabilized using shrinkage regularization:

$$
\tilde{\sigma}_l^2 = (1-\lambda)\sigma_l^2 + \lambda \bar{\sigma}_l^2 \mathbf{1}_d,
\tag{3}
$$

where $\bar{\sigma}_l^2$ is the mean variance across feature dimensions, $\mathbf{1}_d \in \mathbb{R}^d$ is an all-ones vector, and $\lambda \in [0,1]$ is a shrinkage coefficient.

The LVD score for layer $l$ is defined as

$$
I(l) = \frac{\left\|\mu_l^+ - \mu_l^-\right\|_2}{\sqrt{\left\|\tilde{\sigma}_l^2\right\|_2}}.
\tag{4}
$$

The LVD score measures class separation normalized by pooled intra-class variance. Layers with higher scores are considered more difficulty-discriminative and are included in the discriminative layer set $\mathcal{L}$.

#### 4.1.2 Difficulty Perception via Mahalanobis Discrimination

Given the selected layer set $\mathcal{L}$, RLDP evaluates a target problem by measuring its deviation from easy and hard reference distributions.

Let

$$
h \triangleq H_{t,l,:} \in \mathbb{R}^d
$$

be the target problem representation at layer $l$. RLDP computes a shrinkage-regularized diagonal Mahalanobis distance [8] to each class mean:

$$
D_l(h \parallel \mu_l) = \sum_{j=1}^{d}\frac{(h_j - \mu_{l,j})^2}{\tilde{\sigma}_{l,j}^2 + \epsilon}.
\tag{5}
$$

The layer-wise discriminant is then

$$
f_l(h) = D_l(h \parallel \mu_l^-) - D_l(h \parallel \mu_l^+),
\tag{6}
$$

where positive values indicate that the target representation is closer to easy references than to hard references.

To aggregate evidence across the selected layers, RLDP uses sign-based voting:

$$
F(H_t) = \sum_{l\in\mathcal{L}} \mathrm{sign}\left(f_l(H_{t,l,:})\right).
\tag{7}
$$

This preserves the direction of each layer's judgment while suppressing scale-induced instability. The final label is assigned by thresholding:

$$
\mathrm{label}(H_t) =
\begin{cases}
\text{easy}, & F(H_t) > \tau,\\
\text{hard}, & F(H_t) \le \tau.
\end{cases}
\tag{8}
$$

The default global threshold is $\tau=0$, and ties are assigned to the hard class. In summary, RLDP selects discriminative layers using LVD, classifies problems via the difference of shrinkage-regularized diagonal Mahalanobis distances to easy and hard reference representations, and aggregates layer-wise evidence through sign voting.

The paper emphasizes that Mahalanobis distance accounts for anisotropic feature scaling in Transformer hidden spaces and yields more stable difficulty perception than common alternatives. This claim is empirically validated in Section 5.4.

### 4.2 RLDP-AdaSwitch: Adaptive Fast-slow Thinking

RLDP-AdaSwitch uses the perceived difficulty signal as an internal control variable to allocate reasoning effort. It is plug-and-play and can be incorporated into existing LLMs with minimal integration.

The controller has two components:

1. **Difficulty perception module:** RLDP predicts whether the problem is easy or hard from hidden representations.
2. **Mode-selection module:** the system routes the problem to different reasoning modes based on the predicted difficulty.

The workflow is as follows:

1. Given a problem, the model first performs a single **prefill pass** without decoding to obtain hidden representations.
2. RLDP predicts the problem difficulty from those representations.
3. If the problem is predicted easy, the model uses the **fast mode**, such as CoT or `/no_think` mode.
4. If the problem is predicted hard, the model uses the **slow mode**, such as self-consistency CoT or `/think` mode.

Through this design, RLDP-AdaSwitch self-assesses difficulty and allocates reasoning effort without modifying model parameters and without adding output-generation overhead for perception.

> **Figure 3 (described).** RLDP-AdaSwitch first acquires target representations from the Transformer stack, compares them with offline reference statistics from easy and hard problems, uses LVD-based layer selection and Mahalanobis layer-wise discriminants, aggregates signs into a global difficulty score, and routes the problem to fast or slow mode.

### 4.3 Theoretical Analysis

The paper provides theoretical rationales for the design of RLDP. The main discriminant $f_l$ is defined by the difference of shrinkage-regularized diagonal Mahalanobis distances in Eq. (5)–(6), then the method aggregates the polarity of $f_l$ over a selected layer set $\mathcal{L}$ into $F$ in Eq. (7) and applies the threshold decision in Eq. (8). The LVD score in Eq. (4) is used as a lightweight scale-normalized proxy for ranking layers under a diagonal-covariance separability intuition.

#### 4.3.1 Class-conditional LRT/LDA View of $f_l$

Under an idealized shared diagonal-covariance Gaussian model at layer $l$,

$$
p(h\mid \text{easy}) = \mathcal{N}(\mu_l^+, \Sigma_l), \qquad
p(h\mid \text{hard}) = \mathcal{N}(\mu_l^-, \Sigma_l),
$$

with diagonal $\Sigma_l$, if the diagonal metric in Eq. (5) is aligned up to a constant scale with $\Sigma_l^{-1}$, then the class-conditional log-likelihood ratio

$$
\mathrm{LLR}(h)=\log p(h\mid \text{easy}) - \log p(h\mid \text{hard})
$$

is an affine transform of $f_l(h)$. Therefore, $f_l$ induces the same ordering, and the same threshold decisions, as the Bayes-optimal LRT/LDA rule under this approximation.

#### 4.3.2 Variance Floor from Shrinkage and $\epsilon$

Let $\tilde{\sigma}_l^2$ be the shrinkage variance estimate in Eq. (3), and use $\tilde{\sigma}_{l,j}^2+\epsilon$ in Eq. (5) with $\epsilon>0$. Then for every coordinate $j$,

$$
\tilde{\sigma}_{l,j}^2 + \epsilon \ge \epsilon,
$$

and consequently

$$
(\tilde{\sigma}_{l,j}^2 + \epsilon)^{-1} \le \epsilon^{-1}.
$$

Thus, per-dimension precision is uniformly bounded. This prevents small-sample plug-in variances from producing extreme Mahalanobis weights and prevents the discriminant $f_l$ from being dominated by a few near-zero-variance coordinates.

#### 4.3.3 Threshold as an Accuracy–Cost Control Knob

The global threshold in Eq. (8) controls how often the system invokes slow reasoning. Lowering the threshold increases slow-mode triggers, typically raising cost and potentially accuracy; raising it does the opposite. The appendix gives a more formal oracle argument: if a score correctly orders instances by the marginal benefit of slow reasoning, then threshold routing is optimal under a budget or trigger-rate constraint and traces a Pareto-efficient accuracy–cost frontier.
