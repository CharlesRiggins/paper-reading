## Appendix D. Proof of Theorems 2 and 3

We prove Theorems 2 and 3, which give the resolution-limited scaling bounds. The key ingredient is the scaling of nearest-neighbor distances on a $d$-dimensional manifold:

$$\mathbb{E}_{D,x}\!\left[\left|x - \hat{x}\right|\right] = \mathcal{O}\!\left(D^{-1/d}\right),$$

where $\hat{x}$ is the nearest training point to a test point $x$.

The theorem statements are copied for convenience. In the main text, in an abuse of notation, we used $L(f)$ to indicate the value of the test loss as a function of the network $f$, and $L(D)$ to indicate the test loss averaged over the population, draws of the dataset, model initializations and training. To be more explicit below, we will use the notation $\ell(f(x))$ to indicate the test loss for a single network evaluated at a single test point.

**Proof of Theorem 2 (dataset scaling).** Recall the assumptions: the model interpolates the training data, $f(x) = \mathcal{F}(x)\ \forall x \in \mathcal{D}$, and $f$, $\mathcal{F}$ are Lipschitz with constants $K_{f}$, $K_{\mathcal{F}}$, and the loss $\ell$ is Lipschitz with constant $K_{L}$. Then at a test point $x$ with nearest training point $\hat{x}$,

$$\ell(f(x)) \leq K_{L}\,|f(x) - \mathcal{F}(x)| \leq K_{L}\,(K_{f} + K_{\mathcal{F}})\,|x - \hat{x}|.$$

Averaging over the test point and dataset draws,

$$
L(D) \leq K_{L}\left(K_{f} + K_{\mathcal{F}}\right)\,\mathbb{E}_{D,x}\!\left[\left|x - \hat{x}\right|\right] = \mathcal{O}\!\left(K_{L}\,\max(K_{f}, K_{\mathcal{F}})\,D^{-1/d}\right).
$$

In the last equality, we used the above-mentioned scaling of nearest-neighbor distances. $\blacksquare$

**Proof of Theorem 3 (parameter scaling).** The argument is analogous. In the underparameterized regime, we imagine the model approximates $\mathcal{F}$ through $P$ regions, interpolating among $P$ randomly chosen points $\mathcal{P}$ from the (arbitrarily large) training set. The nearest point in $\mathcal{P}$ to a test point $x$ is $\hat{x} = \mathrm{argmin}_{z \in \mathcal{P}}(|x - z|)$, whose distance scales as $\mathcal{O}(P^{-1/d})$. The same Lipschitz argument then gives

$$
L(P) = \mathcal{O}\!\left(K_{L}\,\max(K_{f}, K_{\mathcal{F}})\,P^{-1/d}\right).
$$

$\blacksquare$
