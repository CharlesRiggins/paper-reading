## 2 Preliminaries

### 2.1 Next-Token Prediction

Next-token prediction is a self-supervised learning objective that leverages the sequential structure of natural language. By treating token order as a natural supervisory signal, models can be trained on vast unlabeled corpora. Formally, let $x\coloneqq(x_{1},\ldots,x_{T})$ be a sequence of $T$ tokens, where each token $x_{i}$ takes values in a finite vocabulary $V$. A language model parameterized by $\theta$ defines a joint distribution:

$$
\mathbb{P}_{\theta}(\mathbf{x})=\mathbb{P}_{\theta}(x_{1},\ldots,x_{T}).
$$

*(Equation 1)*

Direct modeling of this joint distribution is computationally intractable due to the exponential growth of the sample space. Autoregressive models address this by factorizing the joint distribution into a product of conditional probabilities:

$$
\mathbb{P}_{\theta}(x_{1},\ldots,x_{T})=\prod_{i}\mathbb{P}_{\theta}(x_{i}\mid\mathbf{x}_{<i}),
$$

*(Equation 2)*

where $x_{<i}\coloneqq(x_{1},\dots,x_{i-1})$ represents the prefix (context) preceding index $i$.

In decoder-only Transformers, each conditional is computed by mapping the prefix $x_{<i}$ to a distribution over $V$. During training, all conditionals are produced in parallel by supplying the ground-truth prefix at every position via teacher forcing (Williams and Zipser, 1989). Given a training corpus $\mathcal{D}$, parameters $\theta$ are learned by minimizing the expected negative log-likelihood:

$$
\mathcal{L}(\theta)\coloneqq-\mathbb{E}_{\mathbf{x}\sim\mathcal{D}}\left[\sum_{i}\log\mathbb{P}_{\theta}(x_{i}\mid\mathbf{x}_{<i})\right].
$$

*(Equation 3)*

This objective reduces language modeling to a sequence of conditional classification problems over $V$, with the conditioning context growing with $i$.

### 2.2 Transformer Architecture

We adopt the Llama-style architecture (Touvron et al., 2023a, b; Grattafiori et al., 2024) as the reference design throughout this work.

##### Token embedding.

A natural language sentence is first decomposed into a sequence of discrete tokens by a tokenizer, then mapped to continuous vectors via an embedding table. Specifically, each token is mapped to a $d_{\text{model}}$-dimensional vector. For a sequence of $T$ tokens, we denote the resulting hidden representation by $H_{1}\in\mathbb{R}^{T\times d_{\text{model}}}$.

##### Transformer layers.

Starting from $H_{1}$, a stack of $L$ Transformer layers transforms the hidden representation while preserving its dimensionality. Each layer consists of two blocks—an attention block and a feed-forward block—yielding $2L$ blocks in total.

Let $\mathbf{H}_{i}\in\mathbb{R}^{T\times d_{\text{model}}}$ denote the input to block $i$, and let $\mathcal{F}_{i}(\cdot)$ denote its transformation. Every block employs a residual connection with pre-norm configuration:

$$
\mathbf{H}_{i+1}=\mathbf{H}_{i}+\mathcal{F}_{i}(\operatorname{RMSNorm}(\mathbf{H}_{i})),
$$

*(Equation 4)*

where $\mathcal{F}_{i}$ is the attention block when $i$ is odd and the feed-forward block when $i$ is even. The function $\operatorname{RMSNorm}(\cdot)$ (Zhang and Sennrich, 2019) is applied row-wise:

$$
\operatorname{RMSNorm}(\mathbf{h})\coloneqq\sqrt{d_{\text{model}}}\frac{\mathbf{h}}{\|\mathbf{h}\|},
$$

*(Equation 5)*

where $\mathbf{h}\in\mathbb{R}^{d_{\text{model}}}$ is a single row of $\mathbf{H}_{i}$. We omit the learnable scale parameter from the $\operatorname{RMSNorm}$ formulation here, since every $\operatorname{RMSNorm}$ is immediately followed by a linear layer and the scale parameter can be absorbed into the subsequent weight matrix during the forward pass.

##### Attention block.

For each head $i$, the query, key, and value projections are parameterized by $W_{Q}^{(i)},W_{K}^{(i)},W_{V}^{(i)}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{head}}}$:

$$
\mathbf{Q}^{(i)}\coloneqq\tilde{\mathbf{H}}\mathbf{W}_{Q}^{(i)},\quad
\mathbf{K}^{(i)}\coloneqq\tilde{\mathbf{H}}\mathbf{W}_{K}^{(i)},\quad
\mathbf{V}^{(i)}\coloneqq\tilde{\mathbf{H}}\mathbf{W}_{V}^{(i)},
$$

$$
\mathbf{A}^{(i)}\coloneqq\operatorname{softmax}\!\left(\frac{\mathbf{Q}^{(i)}\mathbf{K}^{(i)\top}}{\sqrt{d_{\text{head}}}}+\mathbf{M}_{\text{causal}}\right),\quad
\mathbf{O}^{(i)}\coloneqq\mathbf{A}^{(i)}\mathbf{V}^{(i)},
$$

*(Equation 6)*

where $\operatorname{softmax}$ is applied row-wise to ensure each row of $\mathbf{A}^{(i)}$ forms a valid probability distribution. The causal mask $\mathbf{M}_{\text{causal}}\in\mathbb{R}^{T\times T}$ enforces the autoregressive property: its entries are $0$ on and below the diagonal and $-\infty$ above, preventing each position from attending to future tokens. For simplicity, we omit positional encoding from our description. In practice, Llama applies Rotary Position Embeddings (Su et al., 2024) to the $\mathbf{Q}^{(i)}$ and $\mathbf{K}^{(i)}$ before computing $\mathbf{A}^{(i)}$.

The multi-head outputs are concatenated and projected through $W_{O}\in\mathbb{R}^{(N_{\text{head}}\cdot d_{\text{head}})\times d_{\text{model}}}$:

$$
\mathcal{F}_{\text{attn}}(\tilde{\mathbf{H}})\coloneqq\operatorname{Concat}\!\left(\mathbf{O}^{(1)},\dots,\mathbf{O}^{(N_{\text{head}})}\right)\mathbf{W}_{O}.
$$

*(Equation 11)*

##### Feed-forward block.

While the attention block facilitates information exchange across token positions, the feed-forward block operates independently on each position. Modern LLMs typically employ the SwiGLU activation function (Shazeer, 2020). For an input vector $\mathbf{h}\in\mathbb{R}^{d_{\text{model}}}$ (a row of $\mathbf{H}$), the feed-forward transformation is defined as:

$$
\mathcal{F}_{\text{ffn}}(\tilde{\mathbf{h}})\coloneqq\mathbf{W}_{\text{down}}\cdot\left(\operatorname{SiLU}(\mathbf{W}_{\text{gate}}\tilde{\mathbf{h}})\odot(\mathbf{W}_{\text{up}}\tilde{\mathbf{h}})\right),
$$

*(Equation 12)*

where $\odot$ denotes the element-wise (Hadamard) product. The weight matrices are the gate-projection $W_{\text{gate}}\in\mathbb{R}^{d_{\text{ffn}}\times d_{\text{model}}}$, the up-projection $W_{\text{up}}\in\mathbb{R}^{d_{\text{ffn}}\times d_{\text{model}}}$, and the down-projection $W_{\text{down}}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ffn}}}$. Here $d_{\text{ffn}}$ denotes the intermediate dimension, which is typically three or four times larger than $d_{\text{model}}$.

##### Prediction head.

After all $2L$ blocks, the final hidden representation passes through a $\operatorname{RMSNorm}$ layer and a linear projection to produce logits for next-token prediction:

$$
\mathbf{Y}\coloneqq\operatorname{RMSNorm}(\mathbf{H}_{2L+1})\,\mathbf{W}_{\text{head}},
$$

*(Equation 13)*

where $\mathbf{H}_{2L+1}$ is the output of the last residual block, $W_{\text{head}}\in\mathbb{R}^{d_{\text{model}}\times|V|}$ is the projection head, and $\mathbf{Y}\in\mathbb{R}^{T\times|V|}$ is the matrix of output logits.
