## 4. Method

MEMIT inserts memories by updating transformer mechanisms identified through causal mediation analysis. In `GPT-2 XL`, prior ROME work found a sequence of critical MLP layers that mediate factual association recall at the last subject token $S$. MEMIT applies the same interpretability view to larger models and operates in two phases: first calculating vector associations the critical layers should remember, then storing portions of those desired memories in every layer $l\in\mathcal{R}$.

Throughout the paper, the focus is on states representing the **last subject token** $S$ of prompt $p_i$, so $h_i^l=h_{[S]}^l(p_i)$. Similarly, $m_i^l$ and $a_i^l$ denote the MLP and attention outputs at token $S$ for prompt $p_i$.

### 4.1. Identifying the Critical Path of MLP Layers

The authors apply causal tracing to `GPT-J` (6B) and measure the average indirect causal effect of each $h_i^l$ on factual prompts $p_i$, with either the attention or MLP modules for token $S$ disabled. The results show a concentration of mediating states and, crucially, a mediating role for a **range of MLP modules**. This appears as a large gap between the causal effect of single states and the causal effects when MLPs are severed; the gap diminishes after layer 8.

Unlike ROME, which uses this test to identify a single edit layer, MEMIT selects an entire range of critical MLP layers. For `GPT-J`, the selected range is:

$$
\mathcal{R}=\{3,4,5,6,7,8\}.
$$

Given that a range of MLPs jointly mediate factual recall, the paper asks what role one MLP plays in storing a memory. Since transformer residual streams are read from and written to by all attention and MLP modules, unrolling Eqn. 2 gives:

$$
h_i^L=h_i^0+\sum_{l=1}^{L}a_i^l+\sum_{l=1}^{L}m_i^l. \tag{6}
$$

Thus each MLP adds a contribution to the memory at $h_i^L$, which later attention modules read to produce the output. MEMIT therefore spreads desired memory changes across all critical layers $m_i^l$ for $l\in\mathcal{R}$.

### 4.2. Batch Update for a Single Linear Associative Memory

In one layer $l$, MEMIT must store a large batch of $u\gg1$ memories while preserving previously stored memories. Let $W_0\triangleq W_{out}^l$. Following the classical view of a linear layer as an associative memory, $W_0$ maps input keys $k_i$ to memory values $m_i$ by minimizing squared error:

$$
W_0\triangleq\underset{\hat{W}}{\operatorname{argmin}}\sum_{i=1}^{n}\left\|\hat{W}k_i-m_i\right\|^2. \tag{7}
$$

Stacking keys and memories into matrices $K_0=[k_1\mid k_2\mid\dots\mid k_n]$ and $M_0=[m_1\mid m_2\mid\dots\mid m_n]$, the normal equation is:

$$
W_0K_0K_0^T=M_0K_0^T. \tag{8}
$$

To add $u$ new associations, MEMIT seeks $W_1=W_0+\Delta$ minimizing both old and new association errors:

$$
W_1\triangleq\underset{\hat{W}}{\operatorname{argmin}}\left(\sum_{i=1}^{n}\|\hat{W}k_i-m_i\|^2+\sum_{i=n+1}^{n+u}\|\hat{W}k_i-m_i\|^2\right). \tag{9}
$$

The expanded block-form normal equation is:

$$
W_1[K_0\ K_1][K_0\ K_1]^T=[M_0\ M_1][K_0\ K_1]^T. \tag{10}
$$

Expanding and substituting $W_1=W_0+\Delta$ gives:

$$
(W_0+\Delta)(K_0K_0^T+K_1K_1^T)=M_0K_0^T+M_1K_1^T, \tag{11}
$$

$$
W_0K_0K_0^T+W_0K_1K_1^T+\Delta K_0K_0^T+\Delta K_1K_1^T=M_0K_0^T+M_1K_1^T. \tag{12}
$$

Subtracting Eqn. 8 yields:

$$
\Delta(K_0K_0^T+K_1K_1^T)=M_1K_1^T-W_0K_1K_1^T. \tag{13}
$$

Define $C_0\triangleq K_0K_0^T$, proportional to the uncentered covariance of pre-existing keys, and $R\triangleq M_1-W_0K_1$, the residual error of the new associations under old weights. The update is:

$$
\Delta=RK_1^T(C_0+K_1K_1^T)^{-1}. \tag{14}
$$

Because pretraining is opaque, $K_0$ and $M_0$ are unavailable. MEMIT only needs $C_0$, estimated as an aggregate statistic over previously stored keys:

$$
C_0=\lambda\cdot\mathbb{E}_k[kk^T]. \tag{15}
$$

The expectation is estimated empirically from vector inputs to the layer. The hyperparameter $\lambda$ balances preservation of old associations against writing new ones; a typical value is $\lambda=1.5\times10^4$.

### 4.3. Updating Multiple Layers

MEMIT spreads update magnitudes over the mediating layer range $\mathcal{R}$ to improve robustness. Let $L\triangleq\max(\mathcal{R})$ be the target layer by which the new memories should be fully represented. For every edit $(s_i,r_i,o_i)\in\mathcal{E}$, MEMIT first computes a hidden vector $z_i$ that can replace $h_i^L$ and convey the new memory. Then, one layer at a time, it modifies each MLP so it contributes an approximately equal portion of the change $\delta_i=z_i-h_i^L$.

**Computing $z_i$.** MEMIT optimizes a residual vector $\delta_i$ by gradient descent:

$$
z_i=h_i^L+\underset{\delta_i}{\operatorname{argmin}}\frac{1}{P}\sum_{j=1}^{P}-\log\mathbb{P}_{G(h_i^L+\delta_i)}[o_i\mid x_j\oplus p(s_i,r_i)]. \tag{16}
$$

The optimization maximizes the model probability of the desired object $o_i$ over factual prompts with random prefixes $x_j$, encouraging generalization. $G(h_i^L += \delta_i)$ denotes a hooked execution in which the hidden state $h_i^L$ is replaced by $z_i$.

**Spreading $z_i-h_i^L$ over layers.** MEMIT seeks layer updates $\Delta^l$ such that:

$$
\text{setting }\hat{W}_{out}^l:=W_{out}^l+\Delta^l\text{ for all }l\in\mathcal{R}\text{ optimizes }\min_{\{\Delta^l\}}\sum_i\left\|z_i-\hat{h}_i^L\right\|^2, \tag{17}
$$

where

$$
\hat{h}_i^L=h_i^0+\sum_{l=1}^{L}a_i^l+\sum_{l=1}^{L}\hat{W}_{out}^l\sigma(W_{in}^l\gamma(h_i^{l-1})). \tag{18}
$$

Because edits to one layer affect downstream activations, MEMIT calculates $\Delta^l$ iteratively in ascending layer order. For each layer $l$, keys are collected as the inputs to $W_{out}^l$:

$$
k_i^l=\frac{1}{P}\sum_{j=1}^{P}k(x_j+s_i),\quad\text{where }k(x)=\sigma(W_{in}^l\gamma(h_i^{l-1}(x))). \tag{19}
$$

Each memory value is the current value plus a fraction of the remaining top-level residual:

$$
m_i^l=W_{out}k_i^l+r_i^l,\quad r_i^l=\frac{z_i-h_i^L}{L-l+1}. \tag{20}
$$

Algorithmically, MEMIT first computes all $z_i$ vectors, then iterates over $l\in\mathcal{R}$: it runs the layer with current weights, collects keys, computes residual portions, stacks keys and residuals, computes $\Delta^l=R^lK^{lT}(C^l+K^lK^{lT})^{-1}$, and updates the layer MLP weights. The paper emphasizes that this multi-layer batch update is what lets MEMIT scale from single edits to thousands of simultaneous edits.
