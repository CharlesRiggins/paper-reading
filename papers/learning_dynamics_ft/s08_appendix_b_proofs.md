## B Proof of Propositions and Residual Term for Different Losses

### B.1 Proof of Proposition 1

**Proposition 1.** Let $\pi = \mathsf{Softmax}(\mathbf{z})$ and $\mathbf{z} = h_{\theta}(\mathbf{x})$. The one-step learning dynamics decompose as:

$$
\underbrace{\Delta \log \pi^{t}(\mathbf{y} \mid \mathsf{x}_{o})}_{V \times 1} = -\eta \underbrace{\mathcal{A}^{t}(\mathsf{x}_{o})}_{V \times V} \underbrace{\mathcal{K}^{t}(\mathsf{x}_{o}, \mathsf{x}_{u})}_{V \times V} \underbrace{\mathcal{G}^{t}(\mathsf{x}_{u}, \mathsf{y}_{u})}_{V \times 1} + \mathcal{O}(\eta^{2} \|\nabla_{\theta} \mathbf{z}(\mathsf{x}_{u})\|_{\mathrm{op}}^{2}), \tag{3}
$$

where $\mathcal{A}^{t}(\mathsf{x}_{o}) = \nabla_{\mathbf{z}} \log \pi_{\theta^{t}}(\mathsf{x}_{o}) = I - \mathbf{1}\pi_{\theta^{t}}^{\top}(\mathsf{x}_{o})$, $\mathcal{K}^{t}(\mathsf{x}_{o}, \mathsf{x}_{u}) = (\nabla_{\theta} \mathbf{z}(\mathsf{x}_{o})|_{\theta^{t}}) (\nabla_{\theta} \mathbf{z}(\mathsf{x}_{u})|_{\theta^{t}})^{\top}$ is the empirical neural tangent kernel of the logit network $\mathbf{z}$, and $\mathcal{G}^{t}(\mathsf{x}_{u}, \mathsf{y}_{u}) = \nabla_{\mathbf{z}} \mathcal{L}(\mathsf{x}_{u}, \mathsf{y}_{u})|_{\mathbf{z}^{t}}$.

**Proof.** Suppose we want to observe the model's prediction on an "observing example" $\mathsf{x}_{o}$. Starting from Equation (2), we first approximate $\log \pi^{t+1}(\mathbf{y} \mid \mathbf{x}_{o})$ using first-order Taylor expansion (we use $\pi^{t}$ to represent $\pi_{\theta^{t}}$ interchangeably for notation conciseness):

$$
\log \pi^{t+1}(\mathbf{y} \mid \mathbf{x}_{o}) = \log \pi^{t}(\mathbf{y} \mid \mathbf{x}_{o}) + \langle \nabla \log \pi^{t}(\mathbf{y} \mid \mathbf{x}_{o}), \theta^{t+1} - \theta^{t} \rangle + O(\|\theta^{t+1} - \theta^{t}\|^{2}).
$$

Then, assuming the model updates its parameters using SGD calculated by an "updating example" $(\mathbf{x}_{u}, \mathbf{y}_{u})$, we can rearrange the terms in the above equation to get:

$$
\Delta \log \pi^{t}(\mathbf{y} \mid \mathsf{x}_{o}) = \underbrace{\log \pi^{t+1}(\mathbf{y} \mid \mathsf{x}_{o})}_{V \times 1} - \underbrace{\log \pi^{t}(\mathbf{y} \mid \mathsf{x}_{o})}_{V \times 1} = \underbrace{\nabla_{\theta} \log \pi^{t}(\mathbf{y} \mid \mathsf{x}_{o})}_{V \times d} \underbrace{(\theta^{t+1} - \theta^{t}) + O(\|\theta^{t+1} - \theta^{t}\|^{2})}_{d \times 1},
$$

where $d$ is the number of parameters of the model. To evaluate the leading term, we plug in the definition of SGD and repeatedly use the chain rule:

$$
\begin{aligned}
\underbrace{\nabla_{\theta} \log \pi^{t}(\mathbf{y} \mid \mathbf{x}_{o})|_{t}}_{V \times d} \underbrace{(\theta^{t+1} - \theta^{t})}_{d \times 1} &= \big( \underbrace{\nabla_{z} \log \pi^{t}(\mathbf{x}_{o})|_{z^{t}}}_{V \times V} \underbrace{\nabla_{\theta} z^{t}(\mathbf{x}_{o})|_{\theta^{t}}}_{V \times d} \big) \big( -\eta \underbrace{\nabla_{\theta} \mathcal{L}(\mathbf{x}_{u})|_{\theta^{t}}}_{1 \times d} \big)^{\mathsf{T}} \\
&= \underbrace{\nabla_{z} \log \pi^{t}(\mathbf{x}_{o})|_{z^{t}}}_{V \times V} \underbrace{\nabla_{\theta} z^{t}(\mathbf{x}_{o})|_{\theta^{t}}}_{V \times d} \big( \underbrace{-\eta \nabla_{z} \mathcal{L}(\mathbf{x}_{u})|_{z^{t}}}_{1 \times V} \underbrace{\nabla_{\theta} z^{t}(\mathbf{x}_{u})|_{\theta^{t}}}_{V \times d} \big)^{\mathsf{T}} \\
&= -\eta \underbrace{\nabla_{z} \log \pi^{t}(\mathbf{x}_{o})|_{z^{t}}}_{V \times V} \left[ \underbrace{\nabla_{\theta} z^{t}(\mathbf{x}_{o})|_{\theta^{t}}}_{V \times d} \underbrace{(\nabla_{\theta} z^{t}(\mathbf{x}_{u})|_{\theta^{t}})^{\mathsf{T}}}_{d \times V} \right] \underbrace{(\nabla_{z} \mathcal{L}(\mathbf{x}_{u})|_{z^{t}})^{\mathsf{T}}}_{V \times 1}
\end{aligned}
$$

For the higher-order term, using that

$$
\theta^{t+1} - \theta^{t} = -\eta \nabla_{\theta} \mathbf{z}^{t}(\mathbf{x}_{u})|_{\theta^{t}}^{\top} \mathcal{G}^{t}(\mathbf{x}_{u}, \hat{\mathbf{y}})
$$

and noting that, since the residual term $\mathcal{G}^{t}$ is usually bounded (and the practical algorithms will also use gradient clip to avoid too large gradient), we have:

$$
O(\|\theta^{t+1} - \theta^{t}\|^{2}) = O(\eta^{2} \|(\nabla_{\theta} \mathbf{z}^{t}(\mathbf{x}_{u})|_{\theta^{t}})^{\top}\|_{\mathrm{op}}^{2} \|\mathcal{G}^{t}(\mathbf{x}_{u}, \hat{\mathbf{y}})\|_{\mathrm{op}}^{2}) = O(\eta^{2} \|\nabla_{\theta} \mathbf{z}(\mathbf{x}_{u})\|_{\mathrm{op}}^{2}). \tag{□}
$$

In the decomposition, using $\{\pi_{1}, \ldots, \pi_{V}\}$ to represent the model's prediction on different dimensions, we can write our $\mathcal{A}^{t}$ as:

$$
\mathcal{A}^{t}(\mathsf{x}_{o}) = I - \mathbf{1}(\pi^{t})^{\top} = \begin{bmatrix}
1 - \pi_{1} & -\pi_{2} & \cdots & -\pi_{V} \\
-\pi_{1} & 1 - \pi_{2} & \cdots & -\pi_{V} \\
\vdots & \vdots & \ddots & \vdots \\
-\pi_{1} & -\pi_{2} & \cdots & 1 - \pi_{V}
\end{bmatrix}, \tag{9}
$$

The second term in this decomposition, $\mathcal{K}^{t}(\mathbf{x}_{o}, \mathbf{x}_{u})$, is the product of gradients at $\mathsf{x}_{o}$ and $\mathbf{x}_{u}$. Intuitively, if their gradients have similar directions, the Frobenius norm of this matrix is large, and vice versa. This matrix is known as the **empirical neural tangent kernel**, and it can change through the course of training as the network's notion of "similarity" evolves. For appropriately initialized very wide networks trained with very small learning rates, $\mathcal{K}^{t}$ remains almost constant during the course of training, the kernel it converges to is known as the **neural tangent kernel** (Arora et al. [1]; Jacot et al. [21]). Note that the assumption that $\mathcal{K}^{t}(\mathbf{x}_{o}, \mathbf{x}_{u})$ is unchanged (usually used in theoretical analysis) might be too strong in the LLM's finetuning. Hence as stated in the main context, our qualitative analysis only assumes that "during the training, the relative influence of learning $\mathbf{x}_{u}$ on all other different $\mathbf{x}_{o}$ is relatively stable." We will validate this assumption using experiments in Appendix C.

### B.2 Residual Term for Different LLM Finetuning Algorithms

As stated in Section 3, one of the conundrums of decomposing the learning dynamics of LLM is its auto-regressive nature of the output sequence. Different from the multi-label classification problem, where $y_{l}$ for different $l$ is independently generated as long as the shared network is fixed, the $y_{l}$ for the LLM's output depends on $\mathbf{y}_{<l}$, which is usually sampled from the model's prediction iteratively. However, in most of the finetuning cases where the supervisory signal $\mathbf{y}_{u}$ is given, the model will apply the so-called **"teacher forcing"** mechanism when calculating the predicting probabilities. In other words, when generating the output of each $y_{l}$, the $\mathbf{y}_{<l}$ is given rather than sampled on-policy. This mechanism makes it possible for us to define $\chi = [\mathbf{x}; \mathbf{y}]$ and hence merge the auto-regressive nature of the sequence prediction into the shared $\mathcal{K}^{t}(\chi_{o}, \chi_{u})$. After this step, the decomposition of LLM's finetuning learning dynamics then becomes similar to a multi-label classification task.

#### B.2.1 Instruction Finetuning Using Auto-Regression Loss (SFT)

Here we derive the residual term, i.e., $\mathcal{G}^{t}$ for different algorithms in LLM's finetuning. We first rewrite Equation (5) here:

$$
\underbrace{[\Delta \log \pi^{t}(\mathbf{y} \mid \chi_{o})]_{m}}_{V \times M} = -\sum_{l=1}^{L} \eta [\underbrace{\mathcal{A}^{t}(\chi_{o})}_{V \times V \times M}]_{m} [\underbrace{\mathcal{K}^{t}(\chi_{o}, \chi_{u})}_{V \times V \times M \times L}]_{m,l} [\underbrace{\mathcal{G}^{t}(\chi_{u})}_{V \times L}]_{l} + O(\eta^{2}),
$$

where $m \in \{1, \ldots, M\}, l \in \{1, \ldots, L\}$, and $\mathcal{G}^{t}(\chi_{u}) = \nabla_{\mathbf{z}} \mathcal{L}(\chi_{u})|_{\mathbf{z}^{t}}$ is a $V \times L$ matrix. As the auto-regression nature of the SFT loss is already encoded in the causal mask used in $h_{\theta}$, as demonstrated in Figure 10a, the columns in $\mathcal{G}^{t}(\chi_{u})$ are independent of each other, which can be separately calculated. Plus, the summation over $l$ can also be achieved by left-multiplying a length-$L$ all-one vector $\mathbf{1}$. Specifically, the SFT loss for each $l$ is:

$$
[\mathcal{L}_{\mathrm{SFT}}(\chi_{u})]_{l} = -\log \pi(y_{l} = y_{u}^{+} \mid \chi_{u}) = -\mathbf{e}_{y_{u}^{+}}^{\top} \log \pi(y_{l} \mid \chi_{u}) = -\mathbf{e}_{y_{u}^{+}}^{\top} \log(\mathsf{Softmax}(\mathbf{z}_{l})),
$$

where $y_{u}^{+}$ is for the $l$-th dimension of $\mathbf{y}_{u}^{+}$. The gradient of $\mathcal{L}$ on $\mathbf{z}$ can be then calculated as:

$$
\begin{aligned}
[\mathcal{G}_{\mathrm{SFT}}^{t}(\chi_{u})]_{l} = \underbrace{\nabla_{\mathbf{z}_{l}} [\mathcal{L}_{\mathrm{SFT}}(\chi_{u})]_{l}}_{1 \times V} &= (\underbrace{\nabla_{\pi} [\mathcal{L}_{\mathrm{SFT}}(\chi_{u})]_{l}}_{V \times 1})^{\top} \underbrace{\nabla_{\mathbf{z}_{l}} \pi}_{V \times V} \\
&= -(\mathbf{e}_{y_{u}^{+}} \oslash \pi)^{\top} \nabla_{\mathbf{z}_{l}} \pi = \pi(y_{l} \mid \chi_{u}) - \mathbf{e}_{y_{u}^{+}}, \tag{10}
\end{aligned}
$$

where $\oslash$ is element-wise division.

To calculate the equation above, we first recall the NLL loss of the $l$-th token is $[\mathcal{L}_{\mathrm{SFT}}]_{l} \triangleq \mathcal{L} = -\log \pi(y_{l} = y_{l}^{+}) = -\mathbf{e}_{y_{l}^{+}}^{\top} \log \pi$, where $\pi = \mathsf{Softmax}(\mathbf{z})$. Then, $\underbrace{\nabla_{z} \mathcal{L}}_{1 \times V} = \underbrace{\nabla_{\pi} \mathcal{L}}_{1 \times V} \underbrace{\nabla_{z} \pi}_{V \times V}$. For each dimension of $\nabla_{z} \mathcal{L}_{l}$, we have $\frac{\partial \mathcal{L}}{\pi_{i}} = 0$ if $\pi_{i} \neq y_{l}^{+}$ and $\frac{\partial \mathcal{L}}{\pi_{i}} = -\frac{1}{\pi_{i}}$ if $\pi_{i} = y_{l}^{+}$. By writing it in vector form, we have $\nabla_{z} \mathcal{L} = -(\mathbf{e}_{y_{l}^{+}} \oslash \pi)^{\top} \nabla_{z} \pi$. For $\nabla_{z} \pi$, we have:

$$
\nabla_{\mathbf{z}} \pi = \begin{bmatrix}
\pi_{1}(1 - \pi_{1}) & -\pi_{2}\pi_{1} & \hdots & -\pi_{V}\pi_{1} \\
-\pi_{1}\pi_{2} & 1 - \pi_{2}\pi_{2} & \hdots & -\pi_{V}\pi_{2} \\
\hdots & \hdots & \ddots & \hdots \\
-\pi_{1}\pi_{V} & -\pi_{2}\pi_{V} & \hdots & 1 - \pi_{V}\pi_{V}
\end{bmatrix}.
$$

Combining this matrix and the $1 \times V$ vector $(\mathbf{e}_{y_{l}^{+}} \oslash \pi)^{\top}$, where the only non-zero term is $\frac{1}{\pi_{k}}$ at the $k = y_{l}^{+}$ position. So, left multiplying by this vector is actually first selecting the $k$-th row of $\nabla_{z} \pi$, and then multiplying $\frac{1}{\pi_{k}}$ to it. In summary, we have:

$$
\nabla_{z} \mathcal{L} = -\frac{1}{\pi_{k}} [-\pi_{k}\pi_{1}, -\pi_{k}\pi_{2}, \ldots, -\pi_{k}(1 - \pi_{k}), \ldots, -\pi_{k}\pi_{V}]^{\top} = [\pi_{1}, \pi_{2}, \ldots, \pi_{k} - 1, \ldots, \pi_{V}]^{\top} = \pi - \mathbf{e}_{k}.
$$

By stacking the terms with different $l \in [L]$, we can get:

$$
\mathcal{G}_{\mathrm{SFT}}^{t}(\chi_{u}) = \nabla_{\mathbf{z}} \mathcal{L}_{\mathrm{SFT}}(\chi_{u})|_{\mathbf{z}^{t}} = \pi_{\theta^{t}}(\mathbf{y} \mid \chi_{u}) - \mathbf{y}_{u}^{+}. \tag{11}
$$

#### B.2.2 Different Preference Finetuning Algorithms

**Direct Preference Optimization (DPO**, Rafailov et al. [32]) is usually considered the first RL-free alignment algorithm for preference finetuning. Different from the standard RLHF (reinforcement learning with human feedback, Christiano et al. [9]), the training of off-policy DPO is more similar to SFT, where the model keeps learning from a pre-generated preference dataset. Hence, we start from DPO to analyze the learning dynamics of different preference finetuning algorithms (the on-policy versions of these algorithms could also be explained using the proposed framework).

Following Rafailov et al. [32], the training loss of DPO is:

$$
\mathcal{L}_{\mathrm{DPO}}(\boldsymbol{\theta}) = -\mathbb{E}_{(\mathbf{x}_{u}, \mathbf{y}_{u}^{+}, \mathbf{y}_{u}^{-}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})} - \beta \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})} \right) \right]. \tag{12}
$$

Before calculating the residual term $\mathcal{G}_{\mathrm{DPO}}^{t}$, we need to re-calculate the learning dynamics decomposition, because the loss term now depends on both $\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})$ and $\pi_{\boldsymbol{\theta}^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})$, which involves two different $\mathbf{z}$ terms. Specifically, we define $\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+}) = \mathsf{Softmax\_column}(\mathbf{z}^{+})$ and $\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-}) = \mathsf{Softmax\_column}(\mathbf{z}^{-})$, where $\mathbf{z}^{+} = h_{\theta}(\chi_{u}^{+})$ and $\mathbf{z}^{-} = h_{\theta}(\chi_{u}^{-})$ respectively ($\chi_{u}^{+} = [\mathbf{x}_{u}; \mathbf{y}_{u}^{+}]$ and $\chi_{u}^{-} = [\mathbf{x}_{u}; \mathbf{y}_{u}^{-}]$). Then, starting from $L = 1$, the decomposition for the DPO loss (similar to Equation (8) for SFT) could be written as:

$$
\begin{aligned}
\underbrace{\nabla_{\theta} \log \pi^{t}(\chi_{o})|_{\theta^{t}}}_{V \times d} &= \underbrace{(\nabla_{z} \log \pi^{t}(\chi_{o})|_{z^{t}} \nabla_{\theta} z^{t}(\chi_{o})|_{\theta^{t}})}_{V \times V} ( -\eta \underbrace{\nabla_{\theta} \mathcal{L}(\mathbf{x}_{u}, y_{u}^{+}, y_{u}^{-})|_{\theta^{t}}}_{V \times d} )^{\mathsf{T}} \\
&= -\eta \underbrace{\nabla_{z} \log \pi^{t}(\mathbf{x}_{o})|_{z^{t}}}_{V \times V} \left[ \underbrace{\nabla_{\theta} z^{t}(\mathbf{x}_{o})|_{\theta^{t}}}_{V \times d} \underbrace{([\nabla_{\theta} \mathbf{z}^{+}(\chi_{u}^{+}); \nabla_{\theta} \mathbf{z}^{-}(\chi_{u}^{-})]|_{\theta^{t}})^{\mathsf{T}}}_{d \times 2V} \right] \underbrace{(\nabla_{[\mathbf{z}^{+}: \mathbf{z}^{-}]} \mathcal{L}|_{z^{t}})^{\mathsf{T}}}_{2V \times 1}
\end{aligned}
$$

where $[\cdot; \cdot]$ is concatenation of two vectors or matrices, $\mathcal{G}_{\mathrm{DPO}+}^{t}(\chi_{u}^{+}) \triangleq \nabla_{\mathbf{z}^{+}} \mathcal{L}_{\mathrm{DPO}}$, and $\mathcal{G}_{\mathrm{DPO}-}^{t}(\chi_{u}^{-}) \triangleq \nabla_{\mathbf{z}^{-}} \mathcal{L}_{\mathrm{DPO}}$. To calculate the residual terms, we decompose the loss into:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{DPO}}(\mathbf{x}_{u}, \mathbf{y}_{u}^{+}, \mathbf{y}_{u}^{-} \mid \boldsymbol{\theta}) &= -\log(a) \\
a &\triangleq \sigma(b) \\
b &\triangleq \beta(\log \pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+}) - \log \pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})) - c \\
&= -\beta(\mathcal{L}_{\mathrm{SFT}}(\chi_{u}^{+}) - \mathcal{L}_{\mathrm{SFT}}(\chi_{u}^{-})) - c \\
c &\triangleq \beta(\log \pi_{\mathrm{ref}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+}) - \log \pi_{\mathrm{ref}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})), \tag{14}
\end{aligned}
$$

where $c$ is not a function of $\theta$. Using the chain rule, the $l$-th column of the residual term $\mathcal{G}_{\mathrm{DPO}+}^{t}$ can be calculated as (the calculation of $\mathcal{G}_{\mathrm{DPO}-}^{t}$ is similar):

$$
\mathcal{G}_{\mathrm{DPO}+}^{t} = \frac{\partial \mathcal{L}_{\mathrm{DPO}}}{\partial a} \frac{\partial a}{\partial b} \nabla_{\mathbf{z}^{+}} b|_{\mathbf{z}^{t}} = -\frac{1}{a} a(1 - a) \nabla_{\mathbf{z}^{+}} b_{l}|_{\mathbf{z}^{+}} = \beta(1 - a) (\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+}) - \mathbf{y}_{u}^{+}).
$$

By stacking values with different $l$, we can get the residual term of DPO as:

$$
\begin{aligned}
\mathcal{G}_{\mathrm{DPO}+}^{t} &= \beta(1 - a)(\pi_{\theta^{t}}(\mathbf{y} \mid \chi_{u}^{+}) - \mathbf{y}_{u}^{+}); \qquad \mathcal{G}_{\mathrm{DPO}-}^{t} = \beta(1 - a)(\pi_{\theta^{t}}(\mathbf{y} \mid \chi_{u}^{-}) - \mathbf{y}_{u}^{-}) \\
a &= \sigma\left( \beta \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})} - \beta \log \frac{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})} \right) \tag{15}
\end{aligned}
$$

Similarly, we can calculate the residual terms for other off-policy preference optimization methods:

**Identity-preference Optimization (IPO**, Azar et al. [2]):

$$
\mathcal{L}_{\mathrm{IPO}} = -\mathbb{E}_{(\mathbf{x}_{u}, \mathbf{y}_{u}^{+}, \mathbf{y}_{u}^{-}) \sim \mathcal{D}} \left[ \left( \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})} - \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})} - \frac{1}{2\beta} \right)^{2} \right]. \tag{16}
$$

$$
\mathcal{G}_{\mathrm{IPO}+/-}^{t} = \mathcal{G}_{\mathrm{DPO}+/-}^{t}; \quad a = \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})} - \log \frac{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})} - \frac{1}{2\beta}. \tag{17}
$$

**Sequence Likelihood Calibration (SLiC**, Y. Zhao et al. [54]):

$$
\mathcal{L}_{\mathrm{SLiC}} = -\mathbb{E}_{(\mathbf{x}_{u}, \mathbf{y}_{u}^{+}, \mathbf{y}_{u}^{-}) \sim \mathcal{D}} \left[ \max\left[0, \delta - \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})}\right] - \beta \cdot \log \pi_{\theta^{t}}(\mathbf{y}_{\mathrm{ref}} \mid \chi_{\mathrm{ref}}) \right] \tag{18}
$$

$$
= \mathbb{E}_{(\mathbf{x}_{u}, \mathbf{y}_{u}^{+}, \mathbf{y}_{u}^{-}) \sim \mathcal{D}} \left[ \max\left[0, \delta + \mathcal{L}_{\mathrm{SFT}}(\chi_{u}^{+}) - \mathcal{L}_{\mathrm{SFT}}(\chi_{u}^{-})\right] + \beta \mathcal{L}_{\mathrm{SFT}}(\chi_{\mathrm{ref}}) \right] \tag{19}
$$

$$
\mathcal{G}_{\mathrm{SLiC}+/-}^{t} = a \cdot \mathcal{G}_{\mathrm{DPO}+/-}^{t} + \beta(\pi_{\theta^{t}}(\mathbf{y} \mid \chi_{u}) - \mathbf{y}_{\mathrm{ref}}); \quad a = \mathbb{1}\!\left(\delta - \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{+})}{\pi_{\theta^{t}}(\mathbf{y}_{u}^{-})} > 0\right) \tag{20}
$$

In summary, these RL-free algorithms all relate to the SFT loss to some extent. For the DPO and IPO loss, the directions of the updating signals are identical. A scalar controls the strength of this update, which usually correlates with the confidence gap between the model's current confidence on $\mathbf{y}_{u}^{+}$ and $\mathbf{y}_{u}^{-}$, i.e., $Gap(\pi_{\theta^{t}}) \triangleq \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})}$. Generally, larger this value leads to a bigger $a$, making the norm of $\mathcal{G}^{t}$ smaller. In other words, we see a **"regularizing" effect** in this term, where the model should not make $Gap(\pi_{\theta^{t}})$ too large. The SLiC loss can be considered as a combination of SFT adaptation and preference adaptation. Similarly, we can also see a hard version of the regularization effect mentioned above. If $Gap(\pi_{\theta^{t}}) > \delta$, the indicator function will become zero, and the model stops pushing $\pi(\mathbf{y}_{u}^{+})$ and $\pi(\mathbf{y}_{u}^{-})$ away when it already separates $\mathbf{y}_{u}^{+}$ and $\mathbf{y}_{u}^{-}$ well.

Recently, authors of Wu et al. [47] propose another interesting self-play alignment algorithm called **SPPO**, which further improves the alignment performance on top of many on-policy DPO methods. Our framework could also give an interesting explanation of why this method works so well. Specifically, the loss function of SPPO can be written as:

$$
\mathcal{L}_{\mathrm{SPPO}} = -\mathbb{E}_{(\mathbf{x}_{u}, \mathbf{y}_{u}^{+}, \mathbf{y}_{u}^{-}) \sim \mathcal{D}} \left[ \left( \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})} - \frac{\eta}{2} \right)^{2} + \left( \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})} + \frac{\eta}{2} \right)^{2} \right]. \tag{21}
$$

$$
\mathcal{G}_{\mathrm{SPPO}}^{t} = 2\left( \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})} - \frac{\eta}{2} \right) (\pi_{\theta^{t}} - \mathbf{y}_{u}^{+}) + 2\left( \log \frac{\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})}{\pi_{\mathrm{ref}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})} + \frac{\eta}{2} \right) (\pi_{\theta^{t}} - \mathbf{y}_{u}^{-}). \tag{22}
$$

This loss looks similar to the IPO one, but the main difference between SPPO and other methods (e.g., DPO, KTO, IPO, SPIN, etc.) is that there is **no negative sign** in front of $\pi_{\theta^{t}}(\mathbf{y}_{u}^{+} \mid \chi_{u}^{+})$ or $\pi_{\theta^{t}}(\mathbf{y}_{u}^{-} \mid \chi_{u}^{-})$. From its residual term $\mathcal{G}_{\mathrm{SPPO}}^{t}$, it is more convenient to understand this algorithm as imposing two positive vectors on both $\mathbf{y}_{u}^{+}$ and $\mathbf{y}_{u}^{-}$, but the former has a longer norm, as illustrated in Figure 2. By doing so, the big negative gradient no longer exists, and so does the squeezing effect. That is partly why this method is more stable and performs better.
