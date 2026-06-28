## 3 From Spikes to Sinks

We adopt the Llama-style architecture (Touvron et al., 2023a, b; Grattafiori et al., 2024) as the reference design throughout this work.

### 3.1 The Emergence of Massive Activations

Massive activations exhibit several well-documented regularities (Bondarenko et al., 2023; Xiao et al., 2023; Nrusimha et al., 2024; Sun et al., 2024; Yu et al., 2024a):

1. (i) They appear only in intermediate layers.
2. (ii) They appear only in a small number of channels.
3. (iii) The affected channels consistently spike together.
4. (iv) The spikes maintain almost fixed inter-channel ratios.
5. (v) They appear only in a small number of tokens.

In this subsection, we trace the emergence of massive activations systematically, showing how each of these properties arises. As we will see in the next subsection, these properties play a foundational role in enabling attention sinks.

#### 3.1.1 The Life Cycle of Massive Activations

> **Figure 1:** Top-3 channel magnitudes across depth in `Llama-2-7B` and `Qwen3-8B` (post-residuals vs. block outputs). In both models, early blocks inject massive activations that persist through most of the network before being neutralized by late blocks.

To characterize how massive activations vary with depth, we track the top-3 channel magnitudes of post-residual hidden representations (Figure 1, top panels), following Sun et al. (2024). The magnitudes follow a "rise–plateau–fall" trajectory: a sharp increase in early blocks, a long plateau through intermediate blocks, and an abrupt return to typical magnitudes near the end. This suggests a three-stage life cycle: (1) early blocks inject extreme values into the hidden representations; (2) intermediate blocks propagate these values via the residual connection; and (3) late blocks neutralize them by injecting extreme values with opposite sign. We describe each stage in turn.

##### Step-up blocks.

By examining the individual block outputs (Figure 1, bottom panels), we find that massive activations are reliably introduced by one or two early blocks, which we term **step-up blocks**. Prior to these blocks, spike tokens have magnitudes comparable to standard tokens. The step-up blocks produce extreme values in the spike channels, which are then added to the hidden representation via the residual connection, creating the massive activations.

##### Residual accumulation.

In pre-norm Transformers, the hidden representation at depth $i$ can be expressed by unrolling the recurrence in Equation 4:

$$
\mathbf{H}_{i+1}=\mathbf{H}_{1}+\sum_{j=1}^{i}\mathcal{F}_{j}(\operatorname{RMSNorm}(\mathbf{H}_{j})).
$$

*(Equation 14)*

Because the residual stream is additive, extreme values injected by any block $\mathcal{F}_{j}$ persist through all subsequent blocks unless explicitly counteracted. Empirically, intermediate block contributions to spike channels are typically two to three orders of magnitude smaller than the massive activations themselves. As a result, the massive activations introduced by step-up blocks dominate the residual stream until a later block cancels them.

##### Step-down blocks.

As shown in Figure 1, massive activations consistently disappear near the end of the network. Symmetrically to step-up blocks, we identify one or a few late blocks, termed **step-down blocks**, whose outputs match the massive activations in magnitude but carry the opposite sign in the corresponding channels. By contributing an additive inverse via the residual connection, these blocks neutralize the massive activations, returning the hidden representation to a standard range.

Table 1 summarizes the step-up and step-down block indices across models. The consistent positioning of step-up blocks near the beginning and step-down blocks near the end directly accounts for Property (i): massive activations are confined to intermediate layers because they are injected early and systematically neutralized before the final output.

> **Table 1:** Step-up and step-down block indices across models. Step-up blocks appear near the beginning of the network and step-down blocks near the end, confining massive activations to intermediate layers. Odd and even indices denote attention and feed-forward blocks, respectively.

| Model | # Blocks | Step-Up | Step-Down |
| --- | --- | --- | --- |
| `Llama-2-7B` | 64 | 4 | 62 |
| `Llama-2-13B` | 80 | 8 | 78, 79 |
| `Llama-3-8B` | 64 | 4 | 64 |
| `Qwen2.5-7B` | 56 | 8, 10 | 54, 55 |
| `Qwen2.5-14B` | 96 | 10 | 90, 92, 94, 95 |
| `Qwen3-8B` | 72 | 14 | 70, 72 |
| `Qwen3-14B` | 80 | 14 | 79 |

> **Figure 2:** Input-output characteristics of $\operatorname{SiLU}$ in step-up and step-down blocks of `Llama-2-7B`. Based on $1024$ randomly sampled sentences from the `C4` dataset (Raffel et al., 2020), we plot the cosine similarity and norm ratio for each token. Points are colored by the maximum magnitude of the block output. For spike tokens (red points), both direction and norm remain largely unchanged, indicating that the $\operatorname{SiLU}$ gate operates in a near-identity regime.

#### 3.1.2 Feed-Forward Block as Directional Quadratic Amplifier

While both attention and feed-forward blocks possess the theoretical capacity to produce large outputs, our analysis reveals that the SwiGLU-based feed-forward block is the primary source of massive activations, functioning as a **directional quadratic amplifier**. We characterize the mechanism by which extreme activations arise for a small subset of tokens in `Llama-2-7B`. Other models share the same high-level mechanism, but differ in more precise details that instantiate it; we defer those results to Appendix C.

##### Near-identity gating regime.

Let $\mathbf{h}^{(s)}\in\mathbb{R}^{d_{\text{model}}}$ denote the normalized input of a spike token to a step-up or step-down feed-forward block. We empirically observe that the $\operatorname{SiLU}$ nonlinearity operates in a near-identity regime ($\operatorname{SiLU}(x)\approx x$), as shown in Figure 2. Under this approximation, the feed-forward transformation reduces to:

$$
\mathcal{F}_{\text{ffn}}(\tilde{\mathbf{h}}^{(s)})\approx\mathbf{W}_{\text{down}}\cdot\left((\mathbf{W}_{\text{gate}}\tilde{\mathbf{h}}^{(s)})\odot(\mathbf{W}_{\text{up}}\tilde{\mathbf{h}}^{(s)})\right).
$$

*(Equation 15)*

##### High-gain quadratic structure.

Let $W_{\text{gate}}^{(i)}$ and $W_{\text{up}}^{(i)}$ denote the $i$-th rows of the respective weight matrices, and let $W_{\text{down}}^{(k,i)}$ denote the $(k,i)$-th entry of $W_{\text{down}}$. Each output coordinate $k$ then admits the quadratic form (derived in detail in Theorem B.2):

$$
\mathcal{F}_{\text{ffn}}(\tilde{\mathbf{h}}^{(s)})_{k}\approx\tilde{\mathbf{h}}^{(s)\top}\mathbf{U}_{k}\tilde{\mathbf{h}}^{(s)}=\tilde{\mathbf{h}}^{(s)\top}\mathbf{S}_{k}\tilde{\mathbf{h}}^{(s)},
$$

*(Equation 16)*

where

$$
\mathbf{U}_{k}=\sum_{i=1}^{d_{\text{ffn}}}\mathbf{W}_{\text{down}}^{(k,i)}\,\mathbf{W}_{\text{gate}}^{(i)}\mathbf{W}_{\text{up}}^{(i)\top},\quad
\mathbf{S}_{k}=\tfrac{1}{2}(\mathbf{U}_{k}+\mathbf{U}_{k}^{\top}).
$$

*(Equation 17)*

Figure 3 shows the Frobenius norms $\|\mathbf{U}_{k}\|_{F}$ across all output coordinates and feed-forward blocks of `Llama-2-7B`. Spike channels correspond precisely to coordinates with exceptionally large $\|\mathbf{U}_{k}\|_{F}$, and these high-norm coordinates appear exclusively in step-up and step-down blocks. Inspection of the weight matrices reveals that, for high-gain channels $k$, $W_{\text{down}}$ contains anomalously large entries $W_{\text{down}}^{(k,i)}$ for certain intermediate dimensions $i$, and the corresponding rows $W_{\text{gate}}^{(i)}$ and $W_{\text{up}}^{(i)}$ are highly collinear, consistent with prior observations from Yu et al. (2024a).

> **Figure 3:** Frobenius norms $\|\mathbf{U}_{k}\|_{F}$ for the quadratic forms in `Llama-2-7B`. Spike channels align with $\mathbf{U}_{k}$ matrices that have substantially larger norms than typical channels. These high-norm coordinates appear exclusively in step-up and step-down blocks.

##### Rank-one dominance.

Figure 4 compares the eigenvalue spectra of $\mathbf{S}_{k}$ for spike versus non-spike channels. For spike channels, $\mathbf{S}_{k}$ is dominated by a single eigenvalue $\lambda_{\star}$ whose magnitude far exceeds the rest of the spectrum. Let $\mathbf{s}_{\star}$ denote the corresponding unit eigenvector. In such cases, the feed-forward block then acts as a directional quadratic amplifier for these channels:

$$
\mathcal{F}_{\text{ffn}}(\tilde{\mathbf{h}}^{(s)})_{k}\approx\tilde{\mathbf{h}}^{(s)\top}\mathbf{S}_{k}\tilde{\mathbf{h}}^{(s)}\approx\lambda_{\star}(\mathbf{s}_{\star}^{\top}\tilde{\mathbf{h}}^{(s)})^{2}=\lambda_{\star}\sqrt{d_{\text{model}}}\cos(\mathbf{s}_{\star},\tilde{\mathbf{h}}^{(s)}).
$$

*(Equation 19)*

When the input $\mathbf{h}^{(s)}$ aligns with the spike direction $\mathbf{s}_{\star}$, the squared projection is amplified by $\lambda_{\star}$, producing massive activations. Crucially, inspection of the spike directions across all spike channels reveals that their $\mathbf{S}_{k}$ matrices share nearly the same principal eigenvector $\mathbf{s}_{\star}$. Consequently, when an input aligns with this common spike direction, all spike channels are activated simultaneously.

> **Figure 4:** Eigenvalue spectra of $\mathbf{S}_{k}$ for spike vs. non-spike channels in `Llama-2-7B`. Spike channels exhibit a single dominant eigenvalue $\lambda_{\star}$ that is orders of magnitude larger than the remainder of the spectrum; non-spike channels show no such outlier.

This analysis accounts for Property (ii): the scarcity of high-gain quadratic forms explains why massive activations are confined to a small subset of channels. Furthermore, the existence of a shared spike direction across these channels underpins Properties (iii), (iv) and (v); specifically, it accounts for the synchronized triggering of affected channels and their invariant activation magnitude ratios, which are governed by the leading eigenvalues of $\mathbf{S}_{k}$. Finally, because these spike directions are restricted to a highly localized region of the high-dimensional space $\mathbb{R}^{d_{\text{model}}}$, extreme activations occur only for tokens whose representations closely align with $\mathbf{s}_{\star}$. For the vast majority of tokens, the projection onto this direction is negligible.

#### 3.1.3 What Makes a Token a Spike Token

While the feed-forward block provides the capacity for amplification, it requires the $\mathbf{h}^{(s)}$ to align with the trigger direction $\mathbf{s}_{\star}$ in order to generate massive activations. Prior work (Sun et al., 2024) establishes that spike tokens are almost exclusively the first tokens or delimiter tokens; we now study why these tokens consistently achieve such alignments.

##### First tokens.

The initial position serves as the most consistent catalyst for massive activations. A vocabulary-wide probe (Table 2) reveals that over $98\%$ of vocabulary items manifest as spike tokens when placed at position $0$, but rarely do so at subsequent indices. This disparity confirms that the phenomenon is driven by architectural position rather than token semantics. The few exceptions are primarily rare characters from low-resource scripts; we find that their embeddings are close to the initialization values, likely due to infrequent gradient updates during pre-training.

> **Table 2:** Ubiquity of initial spikes across diverse LLMs. For nearly all evaluated models, positional occupancy at the initial position induces massive activations in intermediate layers, independent of the token's semantic identity.

| Model | # Vocab | # Spike Token | Ratio |
| --- | --- | --- | --- |
| `Llama-2-7B` | 32,000 | 31,887 | **99.65%** |
| `Llama-2-13B` | 32,000 | 31,889 | **99.65%** |
| `Llama-3-8B` | 128,256 | 127,956 | **99.77%** |
| `Qwen2.5-7B` | 152,064 | 149,587 | 98.40% |
| `Qwen2.5-14B` | 152,064 | 149,645 | 98.40% |
| `Qwen3-8B` | 151,936 | 151,830 | **99.93%** |
| `Qwen3-14B` | 151,936 | 151,824 | **99.93%** |

The behavior of the initial position arises because the attention block collapses to a simple linear map. Since the first token only attends to itself, its output reduces to:

$$
\mathcal{F}_{\text{attn}}(\mathbf{h}^{(1)})=\sum_{i=1}^{N_{\text{head}}}\mathbf{W}_{O}^{(i)\top}\mathbf{W}_{V}^{(i)\top}\mathbf{h}^{(1)}\equiv\mathbf{W}_{VO}^{\top}\,\mathbf{h}^{(1)},
$$

*(Equation 22)*

where $W_{VO}\coloneqq\sum_{i=1}^{N_{\text{head}}}W_{V}^{(i)}\,W_{O}^{(i)}$. In this regime, the attention block applies a static linear transformation that is identical across all prompts, consistently steering the first tokens' representations toward the trigger direction $\mathbf{s}_{\star}$ and thereby inducing the massive activations observed in intermediate layers.

##### Delimiter Tokens.

Tokens such as periods and newlines follow a mechanistic trajectory similar to first-token sinks. In the early attention blocks, these tokens exhibit significantly elevated post-$\operatorname{RMSNorm}$ magnitudes, stemming from the near-collinearity of their embeddings with the learned scaling parameters of $\operatorname{RMSNorm}$. This magnitude surge induces attention heads to allocate disproportionate weight to the token itself, regardless of the preceding context. So delimiter tokens emulate the isolated environment of the first token across multiple heads. This self-sinking behavior allows static linear transformations to project their latent states toward the same high-gain manifold as the first token. Once aligned with $\mathbf{s}_{\star}$, these representations undergo directional quadratic amplification.

In summary, a token transitions into a spike token when it demonstrates a strong self-sinking bias in early layers, establishing the stable linear trajectory required to activate the directional quadratic amplifier.

### 3.2 The Emergence of Attention Sinks

Having traced the generation and propagation of massive activations, we now characterize how these spike tokens induce the attention sink phenomenon. Specifically, we demonstrate that normalization transforms spike tokens into sparse, bounded, and nearly constant input vectors, enabling the formation of attention sinks.

#### 3.2.1 Normalization Transforms Spike Tokens

In pre-norm Transformer architectures, each attention block operates on normalized hidden representations. Let $\mathbf{h}^{(s)}$ denote the hidden representation of a spike token and let $\tilde{\mathbf{h}}^{(s)}$ denote the output of $\operatorname{RMSNorm}(\mathbf{h}^{(s)})$. The transformation imparts three properties central to attention sink formation.

##### Bounded Range.

Normalization suppresses the extreme magnitudes of spikes, mapping the representation to a bounded range (proof deferred to Theorem B.3):

$$
|\tilde{\mathbf{h}}^{(s)}_{i}|\leq\sqrt{d_{\text{model}}},\quad\forall\,i\in\{1,\dots,d_{\text{model}}\}.
$$

*(Equation 23)*

Hence, even if the pre-norm input contains values on the order of thousands, the block output $\tilde{\mathbf{h}}^{(s)}$ remains moderate and numerically stable.

##### Sparsification.

Because the norm $\|\mathbf{h}^{(s)}\|$ is dominated by a few outlier coordinates, the normalization process effectively suppresses non-spike channels. Consequently, the normalized state $\tilde{\mathbf{h}}^{(s)}$ can be approximated as:

$$
\tilde{\mathbf{h}}^{(s)}\approx\sum_{i\in\mathcal{C}}\tilde{\mathbf{h}}^{(s)}_{i}\mathbf{e}_{i},
$$

*(Equation 24)*

where $\mathcal{C}$ denotes the set of spike channel indices and $\mathbf{e}_{i}$ represents the $i$-th standard basis vector. This transformation yields a sparse, approximately multi-hot representation that is concentrated within a low-dimensional subspace of the original embedding space.

> **Figure 5:** Cosine similarity among spike tokens before and after the step-up block in `Llama-2-7B`. Pre-step-up representations vary across spike tokens, but post-step-up representations collapse to nearly identical directions, empirically validating the near-constant approximation.

##### Near-constant vector.

Spike channels maintain nearly fixed magnitude ratios across spike tokens (Property (iv)), so the normalized values $\tilde{\mathbf{h}}^{(s)}_{i}$ for $i\in\mathcal{C}$ are approximately token-invariant. Consequently, for any spike tokens $a$ and $b$:

$$
\operatorname{RMSNorm}(\mathbf{h}^{(a)})\approx\operatorname{RMSNorm}(\mathbf{h}^{(b)}),
$$

*(Equation 25)*

even when $\mathbf{h}^{(a)}$ and $\mathbf{h}^{(b)}$ differ substantially in their non-spike channels. Normalization thus collapses distinct representations into a near-constant sparse vector, largely erasing token-specific variation. This collapse is empirically demonstrated in Figure 5, where spike tokens following the step-up blocks exhibit cosine similarities approaching $1.0$.

> **Figure 6:** t-SNE visualization of query and key vectors for a representative sink head (left) and non-sink head (right). In the sink head, sink keys $\mathbf{k}^{(s)}$ lie closer to $\mathbf{q}^{(n)}$ than non-sink keys $\mathbf{k}^{(n)}$, creating large logit gaps. In the non-sink head, $\mathbf{k}^{(s)}$ and $\mathbf{k}^{(n)}$ are approximately equidistant from $\mathbf{q}^{(n)}$, preventing the formation of a privileged sink position.

#### 3.2.2 Geometric Alignment Creates Sinks

Spike tokens produce sparse normalized representations, which severely restrict the dimensionality of their resulting attention projections. For a given head, the key vector $\mathbf{k}^{(s)}$ of a sink token is given by:

$$
\mathbf{k}^{(s)}=\mathbf{W}_{K}^{\top}\,\tilde{\mathbf{h}}^{(s)}\approx\sum_{i\in\mathcal{C}}\tilde{\mathbf{h}}^{(s)}_{i}\,\mathbf{W}_{K}^{\top}\,\mathbf{e}_{i},
$$

*(Equation 26)*

where $\mathbf{W}_{K}^{\top}\mathbf{e}_{i}$ corresponds to the $i$-th row of the weight matrix. Consequently, the keys $\mathbf{k}^{(s)}$ are confined to the span of only a few rows. In practice, we find this subspace typically collapses to only one or two dimensions—a significant reduction compared to the full head dimension $d_{\text{head}}$.

Although empirical analysis shows that non-sink queries $\mathbf{q}^{(n)}$ and keys $\mathbf{k}^{(n)}$ also reside in a constrained subspace, their manifold is significantly more expansive than that of the spike tokens. We posit that the emergence of an attention sink is determined not by the absolute volume of these subspaces, but by their **relative geometric alignment**:

- **Sink Heads:** The $\mathbf{q}^{(n)}$ subspace is positioned closer to the fixed $\mathbf{k}^{(s)}$ than to the $\mathbf{k}^{(n)}$ subspace. This alignment produces large, consistent logit gaps in favor of the sink token across diverse inputs.
- **Non-Sink Heads:** The $\mathbf{q}^{(n)}$ subspace is more closely aligned with its non-sink keys $\mathbf{k}^{(n)}$, resulting in attention patterns that distribute mass according to token semantics rather than a fixed default position.

As visualized via t-SNE (Maaten and Hinton, 2008) in Figure 6, the difference between sink and non-sink heads lies in this subspace alignment. In sink heads, the model exploits the near-constant nature of spike keys to create a stable default position for attention mass, effectively offloading excess attention weight to a token whose representation has been neutralized by the normalization function.

Attention sinks arise from two properties of spike tokens after normalization: **sparsity** and **near-constancy**. Sparsity restricts sink keys to a low-dimensional subspace (often one or two dimensions) of the row space of $W_{K}$. Near-constancy keeps those keys nearly invariant across prompts. Together, these properties allow the model to reliably separate sink keys from non-sink keys into distinct subspaces, and this separation manifests as the logit gaps characteristic of attention sinks.

### 3.3 Summary of Findings

This section links massive activations and attention sinks through an architecture-driven pathway in pre-norm Transformers. Massive activations originate from a small number of early step-up feed-forward blocks. In these blocks, SwiGLU behaves as a directional quadratic amplifier: rare high-gain quadratic forms share a common trigger direction, and when a token aligns with it, the token becomes a token carrying massive activations. Because the residual stream is additive, these outliers persist across intermediate layers.

Normalization then maps spike-token representations to inputs that are sparse and nearly constant. As a result, diverse spike tokens collapse to the same vector, making their keys low-dimensional and nearly invariant across prompts. The learned key projection $W_{K}$ consequently maps spike keys and non-spike keys into distinct subspaces. Attention sinks then emerge in heads whose query subspace aligns more strongly with the fixed sink-key subspace than with the non-sink-key subspace, creating the persistent logit gaps that define attention sinks. This completes the account of how massive activations and attention sinks co-emerge.
