## 3. Preliminaries: Language Modeling and Memory Editing

MEMIT aims to modify factual associations stored in an autoregressive LLM. Such a model generates text by sampling from a conditional token distribution parameterized by a $D$-layer transformer decoder $G$:

$$
\mathbb{P}[x_{[t]} \mid x_{[1]}, \dots, x_{[E]}] \triangleq G([x_{[1]}, \dots, x_{[E]}]) = \mathrm{softmax}\left(W_y h_{[E]}^D\right). \tag{1}
$$

Here $h_{[E]}^D$ is the transformer hidden state at the final layer $D$ and ending token $E$. The state is computed recursively as:

$$
h_{[t]}^l(x) = h_{[t]}^{l-1}(x) + a_{[t]}^l(x) + m_{[t]}^l(x), \tag{2}
$$

$$
a^l = \mathrm{attn}^l\left(h_{[1]}^{l-1}, h_{[2]}^{l-1}, \dots, h_{[t]}^{l-1}\right), \tag{3}
$$

$$
m_{[t]}^l = W_{out}^l \sigma\left(W_{in}^l \gamma\left(h_{[t]}^{l-1}\right)\right). \tag{4}
$$

$h_{[t]}^0(x)$ is the embedding of token $x_{[t]}$, and $\gamma$ is layer normalization. Attention and MLPs are written in parallel, following `GPT-Neo` / `GPT-J` style transformer implementations.

The paper studies facts as triples $(s, r, o)$: subject $s$, relation $r$, and object $o$. For example, $(s=\text{Michael Jordan}, r=\text{plays sport}, o=\text{basketball})$. A generator $G$ recalls a memory for $(s_i,r_i,*)$ when a natural-language prompt $p_i=p(s_i,r_i)$, such as “Michael Jordan plays the sport of,” causes the model to predict the next token(s) corresponding to $o_i$.

The list of edit requests is defined as:

$$
\mathcal{E}=\{(s_i,r_i,o_i)\mid i\}\ \text{s.t.}\ \nexists i,j.\ (s_i=s_j)\wedge(r_i=r_j)\wedge(o_i\neq o_j). \tag{5}
$$

This logical constraint rules out conflicting requests: one may edit “Michael Jordan plays” to “baseball,” but then must exclude a simultaneous edit assigning the same subject-relation pair to another sport.

A memory edit is not judged only by whether the edited prompt succeeds. The paper evaluates four desiderata:

- **Efficacy**: the edited statement receives higher probability than the original prediction; e.g., “Michael Jordan plays the sport of baseball.”
- **Generalization**: paraphrases such as “What is Michael Jordan’s sport?” also elicit the edited answer.
- **Specificity**: related but unedited subjects such as Kobe Bryant or Magic Johnson should not be changed.
- **Fluency**: generation should remain coherent; degenerate outputs like repeated “baseball baseball baseball” are failures.

The challenge is to satisfy these desiderata at the scale of thousands of edits, not merely for one or a few facts.
