## 2. Background and Related Work

### 2.1 Difficulty Definition

Difficulty perception requires an operational definition of difficulty that matches the model setting. For single-mode reasoning LLMs, a common approach is to define difficulty by whether a problem can be correctly solved [26, 4]. Under this view, instances answered correctly are considered easy and those answered incorrectly are considered hard.

For dual-mode models, such a binary correctness-only definition is insufficient. The paper follows prior work [16, 25] and incorporates mode-specific outcomes. An instance is labeled **easy** if it is correctly solved by both the non-thinking and thinking modes, because additional deliberation is unnecessary. An instance is labeled **hard** if it fails under non-thinking mode but succeeds under thinking mode, because extended reasoning is required. Appendix B.1 provides further discussion of this definition and why it is model-dependent.

This definition makes difficulty relative to a specific model's capability. The same problem may be easy for a stronger model and hard for a weaker one, which aligns with the goal of adaptive reasoning: difficulty should reflect how much effort a given model needs rather than an absolute external notion.

### 2.2 Notations

The paper considers an LLM with $L$ Transformer layers and hidden dimension $d$. For an input $x$, a single forward pass produces a sequence of hidden representations across layers. The hidden representation of the last token at layer $l \in \{0, 1, \ldots, L\}$ is denoted by

$$
h^{(l)} \in \mathbb{R}^{d},
$$

where $l=0$ corresponds to the embedding layer. Stacking the $d$-dimensional vectors from all layers yields the layer-wise representation matrix:

$$
H = [h^{(0)}, h^{(1)}, \ldots, h^{(L)}] \in \mathbb{R}^{(L+1) \times d}.
\tag{1}
$$

Here, $H$ is the core object used by RLDP: it captures how the model internally represents the input problem before any explicit decoding-based reasoning is performed.

### 2.3 Problem Difficulty Perception

Difficulty perception is a fundamental requirement for improving LLM reasoning efficiency. Existing approaches can be grouped into three categories.

1. **Prompt-based methods.** These estimate difficulty by directly querying the LLM [33, 34]. They are simple to apply but highly sensitive to prompt design and often unstable.
2. **Rollout-based methods.** These infer difficulty from output-level signals during inference, such as answer consistency, entropy, or confidence [26, 22, 28, 42]. They can work in some cases but require repeated sampling, leading to substantial additional decoding tokens.
3. **Training-based methods.** These fine-tune LLMs [6] or train auxiliary classifiers [30, 21] as difficulty judges. They introduce additional training overhead and depend heavily on high-quality training data.

In contrast, RLDP is designed as a **training-free difficulty perception method** that operates purely on hidden representations obtained from a single forward pass. It does not rely on explicit model outputs, repeated rollouts, or supervised judge training.

### 2.4 Analyzing Transformer Layers in LLMs

Many recent studies analyze Transformer layers to understand how LLMs work [48, 1]. Some use interventions on hidden layers [40, 13] to study residual flow [7, 18], while others use metrics such as matrix entropy [37] to measure representations in middle layers. Most of this prior work focuses on inter-layer differences.

This paper instead examines how **easy and hard instances induce distinct layer-dependent representation patterns**. Rather than asking only how information changes from layer to layer, it asks whether difficulty-related information separates classes in the hidden representation space and whether that signal can be used for adaptive inference.
