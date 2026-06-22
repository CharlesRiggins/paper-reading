# 2. Block-attention

## 2.1 Main Idea

Let the input sequence be $\mathcal{S} = \{s_0, s_1, \dots, s_n\}$, with corresponding KV states $\mathcal{K} = \{k_0, k_1, \dots, k_n\}$ and $\mathcal{V} = \{v_0, v_1, \dots, v_n\}$. For autoregressive models, if a text block $\{s_i, \dots, s_j\}$ changes to $\{s'_i, \dots, s'_m\}$, the KV states for the entire sequence must be recomputed. Block-attention's design goal is to recompute only the changed block, obtaining results equivalent to full re-encoding.

As shown in Figure 1, Block-attention splits the input sequence into independent blocks. Each block performs self-attention independently, while only the last block can attend to all preceding blocks. When a block $b_i$ is updated to $b'_i$, only that block's KV states and the last block's KV states need recomputation.

Implementing a Block-attention LLM requires solving three problems: **block segmentation**, **position re-encoding**, and **model adaptation**.

## 2.2 Block Segmentation

The basic principle: divide semantically independent parts into different blocks. In RAG, retrieved documents are mutually independent and naturally form separate blocks; the user query becomes the last block. This principle also applies to other scenarios:
- **Code generation**: each function as a block
- **Multi-turn dialogue**: each turn as a block
- **ICL**: each exemplar as a block

This paper focuses on the RAG scenario.

## 2.3 Position Re-encoding

The same document may appear at different positions in different prompts; its KV states must be re-encoded with updated position information. Using **Rotary Position Encoding (RoPE)** as an example, moving a block $b = \{s_i, \dots, s_j\}$ to $b' = \{s_{i_\Delta}, \dots, s_{j_\Delta}\}$ requires three steps:

1. Compute the original position encoding for token $s_i$: $f(x_i, i)$

2. Rotate counterclockwise by $i\theta$ degrees to zero out its position encoding:

$$f(x_{i0}, 0) = f(x_i, i) \begin{bmatrix} \cos(-i\theta) & -\sin(-i\theta) \\ \sin(-i\theta) & \cos(-i\theta) \end{bmatrix}$$

3. Rotate clockwise by $(i_\Delta)\theta$ degrees to obtain the new position encoding:

$$f(x_{i_\Delta}, i_\Delta) = f(x_i, i) \begin{bmatrix} \cos((i_\Delta - i)\theta) & -\sin((i_\Delta - i)\theta) \\ \sin((i_\Delta - i)\theta) & \cos((i_\Delta - i)\theta) \end{bmatrix}$$

The same applies to other tokens. In practice, zero out the position encoding of each block's first token, then directly derive new encoding vectors from the new position indices.

## 2.4 Block Fine-tuning

Since pretraining uses full attention, direct switching causes train-inference mismatch. Block fine-tuning simply replaces the traditional lower-triangular attention mask with the **block attention mask** shown in Figure 1 (right). This ensures training-inference consistency.

## 2.5 Inference

During inference, the Block-attention model retrieves the KV states of the first $k-1$ blocks from the KV cache, re-encodes their positions using the position re-encoding formula based on their positions in the input sequence, then computes the KV states and output of the last block. In RAG, the last block is the user query. Figure 2 illustrates the inference pipeline.
