## Appendix B Theorems and Derivations

### Theorem B.1 (Attention output as a sum over heads)

Let $\mathbf{O}^{(h)}\coloneqq\mathbf{A}^{(h)}\mathbf{V}^{(h)}\in\mathbb{R}^{T\times d_{\text{head}}}$ with $\mathbf{V}^{(h)}\coloneqq\tilde{\mathbf{H}}\mathbf{W}_{V}^{(h)}$, and let $W_{O}\in\mathbb{R}^{(N_{\text{head}}\cdot d_{\text{head}})\times d_{\text{model}}}$ be partitioned head-wise as

$$
\mathbf{W}_{O}=\operatorname{Concat}\!\left({\mathbf{W}_{O}^{(1)}},\dots,{\mathbf{W}_{O}^{(N_{\text{head}})}}\right)^{\top},
$$

*(Equation 29)*

where $W_{O}^{(h)}\in\mathbb{R}^{d_{\text{head}}\times d_{\text{model}}}$. Then the attention block output

$$
\operatorname{Concat}\!\left(\mathbf{O}^{(1)},\dots,\mathbf{O}^{(N_{\text{head}})}\right)\cdot\mathbf{W}_{O}
$$

*(Equation 30)*

can be written as

$$
\sum_{h=1}^{N_{\text{head}}}\mathbf{A}^{(h)}\tilde{\mathbf{H}}\mathbf{W}_{V}^{(h)}\mathbf{W}_{O}^{(h)}.
$$

*(Equation 31)*

**Proof.** By definition of each head,

$$
\mathbf{O}^{(h)}=\mathbf{A}^{(h)}\mathbf{V}^{(h)}=\mathbf{A}^{(h)}\,\tilde{\mathbf{H}}\mathbf{W}_{V}^{(h)}.
$$

*(Equation 32)*

Using the head-wise block partition of $W_{O}$, multiplying the concatenation by $W_{O}$ decomposes additively:

$$
\operatorname{Concat}(\mathbf{O}^{(1)},\dots,\mathbf{O}^{(N_{\text{head}})})\cdot\mathbf{W}_{O}=\sum_{h=1}^{N_{\text{head}}}\mathbf{O}^{(h)}\mathbf{W}_{O}^{(h)}.
$$

*(Equation 33)*

Substituting $\mathbf{O}^{(h)}=\mathbf{A}^{(h)}\tilde{\mathbf{H}}\mathbf{W}_{V}^{(h)}$ into the above equation yields

$$
\operatorname{Concat}\left(\mathbf{O}^{(1)},\dots,\mathbf{O}^{(N_{\text{head}})}\right)\mathbf{W}_{O}=\sum_{h=1}^{N_{\text{head}}}\mathbf{A}^{(h)}\,\tilde{\mathbf{H}}\mathbf{W}_{V}^{(h)}\,\mathbf{W}_{O}^{(h)}.
$$

*(Equation 34)*

∎

### Theorem B.2 (Quadratic-form approximation of a SiLU feed-forward coordinate)

Under the approximation of the SiLU function (Equation 15) for spike tokens with input representations $\tilde{\mathbf{h}}$, the output of the feed-forward block $\mathbf{y}$ has the quadratic-form approximation for coordinate $k$:

$$
\mathbf{y}_{k}=\tilde{\mathbf{h}}^{\top}\mathbf{U}_{k}\tilde{\mathbf{h}},
$$

*(Equation 35)*

where

$$
\mathbf{U}_{k}\coloneqq\sum_{i=1}^{d_{\text{ffn}}}\mathbf{W}_{\text{down}}^{(k,i)}\;\mathbf{W}_{\text{gate}}^{(i)}\mathbf{W}_{\text{up}}^{(i)\top}.
$$

*(Equation 36)*

**Proof.** Let $\mathbf{z}\in\mathbb{R}^{d_{\text{ffn}}}$ denote the intermediate hidden state

$$
\mathbf{z}\coloneqq\mathbf{W}_{\text{gate}}\tilde{\mathbf{h}}\odot\mathbf{W}_{\text{up}}\tilde{\mathbf{h}}.
$$

*(Equation 37)*

The $i$-th element of $\mathbf{z}$, i.e. $z_{i}$, can be represented as

$$
\mathbf{z}_{i}=({\mathbf{W}_{\text{gate}}^{(i)\top}}\tilde{\mathbf{h}})\cdot(\mathbf{W}_{\text{up}}^{(i)\top}\tilde{\mathbf{h}})=\tilde{\mathbf{h}}^{\top}\mathbf{W}_{\text{gate}}^{(i)}\mathbf{W}_{\text{up}}^{(i)\top}\tilde{\mathbf{h}},
$$

*(Equation 38)*

where $W_{\text{gate}}^{(i)}$ and $W_{\text{up}}^{(i)}$ are the $i$-th rows of the weight matrices. The term $W_{\text{gate}}^{(i)}{W_{\text{up}}^{(i)}}^{\top}$ is the outer product of two vectors, forming a rank-1 matrix $\mathbf{V}_{i}$. We can now rewrite the hidden element as the quadratic form

$$
\mathbf{z}_{i}=\tilde{\mathbf{h}}^{\top}\mathbf{V}_{i}\tilde{\mathbf{h}}.
$$

*(Equation 39)*

Next, consider the $k$-th element of the output vector $\mathbf{y}$, denoted $y_{k}$. By the linearity of the final projection ($W_{\text{down}}$):

$$
\mathbf{y}_{k}=\sum_{i}\mathbf{W}_{\text{down}}^{(k,i)}\mathbf{z}_{i}=\sum_{i}\mathbf{W}_{\text{down}}^{(k,i)}\cdot\left(\tilde{\mathbf{h}}^{\top}\mathbf{V}_{i}\tilde{\mathbf{h}}\right)=\tilde{\mathbf{h}}^{\top}\left(\sum_{i}\mathbf{W}_{\text{down}}^{(k,i)}\mathbf{V}_{i}\right)\tilde{\mathbf{h}}.
$$

*(Equation 40)*

Therefore, we conclude that the output element is in quadratic form

$$
\mathbf{y}_{k}=\tilde{\mathbf{h}}^{\top}\mathbf{U}_{k}\tilde{\mathbf{h}},
$$

*(Equation 41)*

where the matrix $\mathbf{U}_{k}$ is defined as the weighted sum of the rank-1 components

$$
\mathbf{U}_{k}=\sum_{i}\mathbf{W}_{\text{down}}^{(k,i)}\mathbf{V}_{i}=\sum_{i}\mathbf{W}_{\text{down}}^{(k,i)}\cdot\left(\mathbf{W}_{\text{gate}}^{(i)}\mathbf{W}_{\text{up}}^{(i)\top}\right).
$$

*(Equation 42)*

This derivation establishes the form given in Equation 17 and concludes the proof. ∎

### Theorem B.3 (Coordinate bound under RMS normalization)

Let $\mathbf{h}\in\mathbb{R}^{d_{\text{model}}}$ be any non-zero vector and define its RMS-normalized version

$$
\tilde{\mathbf{h}}\coloneqq\operatorname{RMSNorm}(\mathbf{h})=\sqrt{d_{\text{model}}}\,\frac{\mathbf{h}}{\|\mathbf{h}\|_{2}}.
$$

*(Equation 43)*

Then every coordinate of $\tilde{\mathbf{h}}$ is bounded in magnitude by $\sqrt{d_{\text{model}}}$:

$$
\big|\tilde{\mathbf{h}}_{i}\big|\leq\sqrt{d_{\text{model}}},\quad\forall i\in\{1,\dots,d_{\text{model}}\}.
$$

*(Equation 44)*

**Proof.** For any coordinate $i\in\{1,\dots,d_{\text{model}}\}$, by definition,

$$
\big|\tilde{\mathbf{h}}_{i}\big|=\sqrt{d_{\text{model}}}\,\frac{|\mathbf{h}_{i}|}{\|\mathbf{h}\|_{2}}.
$$

*(Equation 45)*

Since

$$
\|\mathbf{h}\|_{2}^{2}=\sum_{j=1}^{d_{\text{model}}}\mathbf{h}_{j}^{2}\geq\mathbf{h}_{i}^{2},
$$

*(Equation 46)*

we have $\|\mathbf{h}\|_{2}\geq|\mathbf{h}_{i}|$. Therefore,

$$
\frac{|\mathbf{h}_{i}|}{\|\mathbf{h}\|_{2}}\leq 1\quad\Longrightarrow\quad\big|\tilde{\mathbf{h}}_{i}\big|\leq\sqrt{d_{\text{model}}}.
$$

*(Equation 47)*

∎
