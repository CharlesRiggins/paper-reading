## 2 Background

LoRA involves freezing a pretrained weight matrix $W_{\mathrm{pretrained}} \in \mathbb{R}^{d \times k}$, and learning only a low-rank perturbation to it, denoted here as $\Delta$, as follows:

$$
\begin{array} { r } { W _ { \mathrm { f i n e t u n e d } } = W _ { \mathrm { p r e t r a i n e d } } + \Delta } \\ { \Delta = \gamma _ { r } A B , \quad A \in \mathbb { R } ^ { d \times r } , \quad B \in \mathbb { R } ^ { r \times k } . } \end{array}
$$

Most common implementations initialize $A_{0} \sim \mathcal{N}(0, 1)$, $B_{0} = 0$ and set the scalar $\gamma_{r} = \alpha / r$ with a controllable hyperparameter $\alpha$. The user chooses which $W_{\mathrm{pretrained}}$ to adapt ("target modules"), the rank $r << d, k$, and the hyperparameter $\alpha$. By doing so, only $d \times r + r \times k$ parameters are trained per module instead of $d \times k$, which reduces the memory and FLOPS required for computing the gradient. As an example, applying a $r = 16$ LoRA adapter to a 7B weight matrix with $d = k = 4096$ trains < 1% of the original parameter count. Appendix Sec. H lays out the approximate memory savings by LoRA during training.

LoRA's introduction and first applications targeted only the $W_{q}$ and $W_{v}$ matrices in the self-attention module (Hu et al., 2021). Since then, it has become best practice to target all transformer modules (Raschka, 2023; Dettmers et al., 2024), i.e., $\{W_{q}^{(l)}, W_{k}^{(l)}, W_{v}^{(l)}, W_{o}^{(l)}\}_{l=1}^{L}$ in the self-attention modules, and $\{W_{\mathrm{gate}}^{(l)}, W_{\mathrm{up}}^{(l)}, W_{\mathrm{down}}^{(l)}\}_{l=1}^{L}$ in the feedforward modules for $L$ layers in, say, a Llama architecture (Hu et al., 2021; Touvron et al., 2023).
