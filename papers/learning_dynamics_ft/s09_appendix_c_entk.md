## C The "Relative Stable" eNTK Assumption

We use this appendix to verify the core assumption of our analysis — **during the training, the relative influence of learning $\mathsf{x}_{u}$ on all other different $\mathsf{x}_{o}$ is relatively stable** — on both MNIST and LLM finetuning settings. To make the notation concise, we use $\mathcal{K}_{uo}^{t}$ to represent $\mathcal{K}^{t}(\mathbf{x}_{o}, \mathbf{x}_{u})$, $\mathcal{K}^{t}(\chi_{o}, \chi_{u})$ and other related variants.

### C.1 Relative Stable eNTK Assumption — MNIST Experiments

For the MNIST example, we directly calculate the eNTK term using a pipeline demonstrated in Figure 6. The results are shown in Figure 7, where the key findings are:

1. The last three panels roughly indicate different phases throughout the training, where the first several epochs (0 ~ 30) are a bit messy, and the last several epochs (800 ~ 1000) behave similarly to the finetuning stage;
2. Although the norm of eNTK $(\mathbb{E}_{u,o}[\|K_{uo}^{t}\|_{F}])$ and the norm of eNTK's adaptation $(\mathbb{E}_{u,o}[\|K_{uo}^{t} - K_{uo}^{t-1}\|_{F}])$ changes a lot after 30 epochs, the **ranking** between $\|K_{uo}^{t}\|_{F}$ on different $o$ are relatively stable, as demonstrated by the upper 9 panels;
3. The **pairing effect** between the "similar" inputs is clear, e.g., "4" and "9", "5" and "8", etc;
4. The pairing effect between the "dissimilar" inputs are also clear, e.g., "6" and "7", "2" and "5", etc.
5. The pairing effect mentioned previously is not strictly symmetric, which is because of the inconsistent $\mathcal{A}$ and $\mathcal{G}$ terms;
6. The accumulated influence demonstrated in the third panel of Figure 1 is strongly correlated to the integral of all these curves.

### C.2 Relative Stable eNTK Assumption — LLM Experiments

Directly calculating $\|\mathcal{K}_{uo}^{t}\|_{F}$ for the LLM experiment requires huge amount of computation, because for each token in each example, we need to multiply a $V \times d$ matrix to a $d \times V$ one, where $d$ is the number of parameters of the LLM. However, since we only care about the relative relationship between $\|\mathcal{K}_{uo}^{t}\|_{F}$ on different $\mathsf{x}_{o}$, where $\mathsf{x}_{u}$ is fixed, based on the basic decomposition in Proposition 1, we can get a lower-bound as follows (ignoring superscript $t$ for conciseness, ignoring the influence of $\mathcal{O}(\eta^{2})$):

$$
\Delta \log \pi = -\eta \mathcal{A}_{o} \mathcal{K}_{uo} \mathcal{G}_{o} \tag{23}
$$

$$
\|\Delta \log \pi\|_{F}^{2} = \|-\eta \mathcal{A}_{o} \mathcal{K}_{uo} \mathcal{G}_{o}\|_{F}^{2} \tag{24}
$$

$$
\leq \eta^{2} \|\mathcal{A}_{o}\|_{F}^{2} \|\mathcal{K}_{uo}\|_{F}^{2} \|\mathcal{G}_{o}\|_{F}^{2} \tag{25}
$$

We hence define two quantitative measurements to have a better understanding of $\mathcal{K}_{uo}$, they are:

$$
\mathsf{LBK}_{uo} \triangleq \frac{\|\Delta \log \pi\|_{F}^{2}}{\|\mathcal{A}_{o}\|_{F}^{2} \|\mathcal{G}_{o}\|_{F}^{2}} \leq \eta^{2} \|\mathcal{K}_{uo}\|_{F}^{2}; \quad \mathsf{SignDelta}_{uo} \triangleq \mathbb{E}_{v,l} [\log \pi_{v,l}^{t+1} - \log \pi_{v,l}^{t}], \tag{26}
$$

where the subscript $v, l$ here represent the $l$-th token and $v$-th dimension for the prediction. In later experiments, we will observe both $\mathsf{LBK}_{uo}$ and $\mathsf{SignDelta}_{uo}$ to have a better understanding of the strength (norm) and the direction (sign) of the relative influence imposed via $\mathcal{K}_{uo}$.

Regarding the calculation of $\mathsf{LBK}_{uo}$, $\|\Delta \log \pi\|_{F}^{2}$ is easy to track because, in the main context, we already showed $\log \pi^{t}$ for different responses. $\|\mathcal{G}_{o}\|_{F}^{2} = \|\boldsymbol{\pi} - \mathbf{y}_{u}^{+}\|_{F}^{2}$, where $\mathbf{y}_{u}^{+}$ is defined as a stacking of $L$ one-hot vectors. The $\|\mathcal{A}_{o}\|_{F}^{2}$ is a bit complex. Recall the definition that $\mathcal{A}_{o} = I - \mathbf{1}\pi^{\top}$, we can have:

$$
\|\mathcal{A}_{o}\|_{F}^{2} = \mathsf{Trace}(\mathcal{A}_{o}^{\top} \mathcal{A}_{o}) = \mathsf{Trace}((I - \mathbf{1}\pi^{\top})^{\top}(I - \mathbf{1}\pi^{\top})) = \mathsf{Trace}(I^{\top}I - \pi\mathbf{1}^{\top} - \mathbf{1}\pi^{\top} + \pi\mathbf{1}^{\top}\mathbf{1}\pi^{\top}) \tag{27-29}
$$

$$
= \mathsf{Trace}(I^{\top}I) - 2\mathsf{Trace}(\mathbf{1}^{\top}\boldsymbol{\pi}) + V\mathsf{Trace}(\boldsymbol{\pi}^{\top}\boldsymbol{\pi}) = V - 2 + V\|\pi\|_{2}^{2}, \tag{30-31}
$$

which is also trackable in our setting. Note that intuitively, the value of $\|\boldsymbol{\pi}\|_{2}^{2}$ is inversely correlated to the Shannon entropy of the distribution: $\|\pi\|_{2}^{2} = 1$ if $\pi$ is one-hot; $\|\pi\|_{2}^{2} = \frac{1}{\sqrt{V}}$ if $\pi$ is uniform. Hence we can also interpret $\|\mathcal{A}_{o}\|_{F}^{2}$ as the **peakiness** of $\pi(\mathbf{y} \mid \mathbf{x}_{o})$. In the following experiment, we track the value of $\mathsf{LBK}_{uo}$ for different types of responses during SFT and DPO to show that the relative influence between different response types is relatively stable. We show the experimental results in Figure 8, in which the key findings are:

1. In both SFT and DPO under different supervisory signals, the change of these two metrics are relatively stable, similar to those in Figure 7;
2. The clear **pairing effect** between $\mathbf{y}_{u}^{+}$ (blue curve) and $\mathbf{y}_{j \neq u}^{+}$ (red curve) exist;
3. In $\mathsf{LBK}_{uo}$, learning any natural language sequences (i.e., $\mathbf{y}_{u}^{+}$, $\mathbf{y}_{u}^{-}$, $\mathbf{y}_{\text{gpts}}^{+}$, $\mathbf{y}_{\text{gpts}}^{-}$) influences the non-language sequence ($\mathbf{y}_{\text{urnd}}^{+}$, $\mathbf{y}_{\text{rnd}}$) a lot, especially at the end of finetuning. However, from $\mathsf{SignDelta}_{uo}$ we know such an influence is negative, which is caused by the pushing down pressure;
4. An interesting **"similarity pattern"** occurs: by observing $\mathsf{SignDelta}_{uo}$, we see SFT using $\mathbf{y}_{\text{gpts}}^{+}$ or $\mathbf{y}_{\text{gpts}}^{-}$ imposes more influence on the sequence generated using ChatGPT other than their original response (i.e., $\mathbf{y}_{u}^{+}$ or $\mathbf{y}_{u}^{-}$), which might be an interesting phenomenon to explore further;
5. By observing the last row, where the model is trained using DPO, it is clear that the push-down pressure is dominant. Because almost all $\mathsf{SignDelta}_{uo}$ terms have big negative values, and the only positive one is $\mathbf{y}_{u}^{+}$ (roughly 0.5, much smaller than other positive values in the SFT cases).

We also provide some intermediate quantities in Figure 9 to further validate our analysis. The key trends are provided in its caption for ease of reading.
