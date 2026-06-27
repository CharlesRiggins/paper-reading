## 3. Method

### Notation

Let $N$ be the number of transformer layers, $L$ the sequence length, and $k$ the number of selected tokens per query. At layer $\ell$, the lightning indexer produces a score vector $\mathbf{I}^{(\ell)}_t \in \mathbb{R}^{L}$ for query position $t$, and the top-$k$ index set is

$$
\mathcal{T}^{(\ell)}_t = \mathrm{Top\text{-}k}\left(\mathbf{I}^{(\ell)}_t\right).
$$

Throughout the paper, $k=2048$. Let $\mathbf{p}^{(\ell)}_t$ denote the aggregated attention distribution at layer $\ell$ for position $t$, obtained by averaging softmax attention weights across heads. Let

$$
\mathbf{q}^{(\ell)}_t = \mathrm{Softmax}\left(\mathbf{I}^{(\ell)}_t\right)
$$

be the indexer's output distribution.

### Overview

IndexCache modifies DSA by assigning each of the $N$ layers one of two roles, encoded as a binary pattern string $\mathbf{c}=c_1c_2\cdots c_N$ with $c_\ell \in \{\texttt{F},\texttt{S}\}$.

- `F` (**Full**) layers retain their own indexer, compute fresh $\mathcal{T}^{(\ell)}_t$, and perform sparse core attention on the selected subset, exactly as in standard DSA.
- `S` (**Shared**) layers have no indexer. They inherit the index set from the nearest preceding `F` layer:

$$
\mathcal{T}^{(\ell)}_t \leftarrow \mathcal{T}^{(f(\ell))}_t,\qquad
f(\ell)=\max\{j<\ell: c_j=\texttt{F}\}.
$$

The first layer is always `F`. At inference, an `S` layer skips the indexer forward pass and reuses the cached index tensor from its `F` predecessor. The only inference-loop change is a conditional branch that either runs the indexer or copies the cached indices. If most layers can safely share indices, IndexCache removes a large fraction of the $O(NL^2)$ total indexer cost while leaving the $O(NLk)$ sparse core attention unchanged.

### 3.1 Training-Free IndexCache

Given a pretrained DSA model, training-free IndexCache seeks a pattern $\mathbf{c}$ that maximizes the number of Shared layers while minimizing quality loss. The obvious solution is **uniform interleaving**: retain every $r$-th layer's indexer and skip the rest, such as `FSSSFSSS...` for $r=4$. The paper shows that this is suboptimal because indexer importance varies significantly by layer. Some early and transitional layers are much more sensitive to indexer removal than others, so uniform interleaving may remove a critical indexer while retaining a redundant one.

#### 3.1.1 Why Uniform Interleaving Is Suboptimal

Uniform interleaving ignores the functional structure of the network. If a critical layer is forced to inherit an index set that misses even a small number of important tokens, the hidden-state perturbation may propagate through all subsequent layers and degrade final output quality. Conversely, some layers can safely share indices with little effect. The correct pattern is therefore not determined purely by spacing; it must be selected using a model-level signal.

#### 3.1.2 Layer Selection Algorithm

The proposed greedy search starts from the all-F baseline and incrementally converts layers to `S`. It uses LM loss on a fixed calibration set as a proxy for downstream quality. Candidate patterns are evaluated on exactly the same cached mini-batches, so loss differences reflect only the pattern change rather than data variance.

Algorithmically:

1. Initialize $\mathbf{c}\leftarrow \texttt{F}^N$.
2. Set the removable candidate set to $\mathcal{R}=\{2,3,\ldots,N\}$ because layer 1 must remain `F`.
3. For each step up to target $K$ Shared layers, evaluate every currently-F candidate layer $\ell\in\mathcal{R}$ by tentatively setting $c_\ell\to\texttt{S}$.
4. Choose

$$
\ell^* = \arg\min_{\ell\in\mathcal{R}} \mathrm{EvalLoss}\left(M,\mathcal{D},\mathbf{c}|_{c_\ell\to\texttt{S}}\right),
$$

then commit $c_{\ell^*}\leftarrow\texttt{S}$ and remove $\ell^*$ from $\mathcal{R}$.
5. Return the final pattern $\mathbf{c}$.

A full search from all-F to all-S performs $N(N-1)/2$ forward passes. With pipeline parallelism across $P$ stages, the search can be accelerated by splitting layers into $P$ blocks, fixing each block's first layer as `F`, and searching blocks sequentially within each step. Up to $P$ layers can then be placed per step, reducing forward passes by roughly $P\times$.

The greedy solution has three useful empirical properties. First, it outperforms uniform interleaving at the same retention ratio. Second, the per-step LM validation-loss curve separates "easy" layers from "critical" layers, suggesting a natural ordering of indexer importance. Third, results are stable across calibration sets, implying that the importance ranking is an intrinsic model property rather than a data artifact; LM loss also correlates with downstream performance.

### 3.2 Training-Aware IndexCache with Multi-Layer Distillation

Training-free IndexCache requires no weight updates, but each original DSA indexer was trained to serve only its own layer. If the model can be trained or continued-pretrained, IndexCache can explicitly train each retained indexer to serve multiple layers.

#### From single-layer to multi-layer distillation

In standard DSA training, each layer-$\ell$ indexer is distilled against its own aggregated attention distribution $\mathbf{p}^{(\ell)}_t$ through

$$
\mathcal{L}^{\mathrm{I}} = \sum_t D_{\mathrm{KL}}\left(\mathbf{p}^{(\ell)}_t \middle\| \mathbf{q}^{(\ell)}_t\right).
$$

IndexCache generalizes this to a multi-layer objective. Suppose layer $\ell$ is a retained `F` layer, and layers $\ell+1,\ldots,\ell+m$ are subsequent `S` layers that will reuse its index set. The multi-layer distillation loss is

$$
\mathcal{L}^{\mathrm{I}}_{\mathrm{multi}}
= \sum_{j=0}^{m}\frac{1}{m+1}\sum_t
D_{\mathrm{KL}}\left(\mathbf{p}^{(\ell+j)}_t \middle\| \mathbf{q}^{(\ell)}_t\right).
$$

This trains the retained indexer to predict a top-$k$ set jointly useful for all layers it serves rather than overfitting to its own layer.

#### Gradient equivalence to averaged-target distillation

The paper proves that the multi-layer loss has the same gradient as distilling against a single averaged target distribution. Define

$$
\bar{\mathbf{p}}_t = \sum_{j=0}^{m}\frac{1}{m+1}\mathbf{p}^{(\ell+j)}_t
$$

and

$$
\mathcal{L}^{\mathrm{I}}_{\mathrm{avg}} = \sum_t D_{\mathrm{KL}}\left(\bar{\mathbf{p}}_t \middle\| \mathbf{q}^{(\ell)}_t\right).
$$

Then

$$
\nabla_\theta \mathcal{L}^{\mathrm{I}}_{\mathrm{multi}} = \nabla_\theta \mathcal{L}^{\mathrm{I}}_{\mathrm{avg}}.
$$

The proof uses the fact that $\mathbf{q}^{(\ell)}_t$ is the only parameter-dependent term inside $D_{\mathrm{KL}}(\mathbf{p}\|\mathbf{q}^{(\ell)}_t)$, so the entropy of $\mathbf{p}$ vanishes under differentiation:

$$
\nabla_\theta D_{\mathrm{KL}}\left(\mathbf{p}\middle\|\mathbf{q}^{(\ell)}_t\right)
= -\nabla_\theta \sum_s \mathbf{p}(s)\log \mathbf{q}^{(\ell)}_t(s).
$$

Applying this to the multi-layer sum gives the same gradient as the KL loss to $\bar{\mathbf{p}}_t$.

#### Interpretation and training

Multi-layer distillation is therefore not only a heuristic regularizer. It is exactly equivalent, in gradient, to distilling the indexer toward the **centroid** of the attention distributions of the served layers. The retained indexer learns a consensus top-$k$ distribution that covers important tokens across all layers in its sharing block.

The paper implements $\mathcal{L}^{\mathrm{I}}_{\mathrm{multi}}$ rather than explicitly constructing $\mathcal{L}^{\mathrm{I}}_{\mathrm{avg}}$ because it is more efficient: subsequent `S` layers only need the current layer's predicted $\mathbf{q}^{(\ell)}$, while averaged-target training would require passing both $\mathbf{q}^{(\ell)}$ and $\mathbf{p}^{(\ell)}$ with extra memory and runtime overhead.

Training follows standard DSA's two-stage procedure. During dense warm-up, the indexer in each `F` layer is trained using $\mathcal{L}^{\mathrm{I}}_{\mathrm{multi}}$ while other parameters are frozen. During sparse training, the indexer continues using the same multi-layer loss computed over selected top-$k$ tokens, and the LM loss trains the remaining model parameters.
