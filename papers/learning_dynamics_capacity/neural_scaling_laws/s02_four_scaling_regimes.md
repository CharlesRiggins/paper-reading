## 2. Four scaling regimes

Throughout this work we will be interested in how the average test loss $L(D,P)$ depends on the dataset size $D$ and the number of model parameters $P$. Unless otherwise noted, $L$ denotes the test loss averaged over initialization of the parameters and draws of a size-$D$ training set. Some of our results only pertain directly to the scaling with width $w \propto \sqrt{P}$, but we expect many of the intuitions apply more generally. We use the notation $\alpha_{D}$, $\alpha_{P}$, and $\alpha_{W}$ to indicate scaling exponents with respect to dataset size, parameter count, and width. All proofs appear in the supplement.

### 2.1 Variance-limited exponents

In the limit of large $D$ the outputs of an appropriately trained network approach a limiting form with corrections which scale as $D^{-1}$. Similarly, recent work shows that wide networks have a smooth large-$P$ limit [15], where fluctuations scale as $1/\sqrt{P}$. If the loss is sufficiently smooth then its value will approach the asymptotic loss with corrections proportional to the variance ($1/D$ or $1/\sqrt{P}$).

In **Theorem 1** we present sufficient conditions on the loss to ensure this variance-dominated scaling. We note, these conditions are satisfied by mean squared error and cross-entropy loss, though we conjecture the result holds even more generally.

> **Theorem 1 (Variance-limited scaling).** Let $f_{T}$ denote a network trained for $T$ steps, and let $\ell$ denote the per-sample loss. Assume the trained network output concentrates so that $\mathbb{E}\left[(f_{T} - \mathbb{E}[f_{T}])^{k}\right] \leq C_{k}$ for moments that scale appropriately with the dataset or width. If the loss $\ell$ is either (i) a finite-degree polynomial in $f_{T}$, (ii) has bounded second derivative, or (iii) is 2-Hölder, then
>
> $$\mathbb{E}\left[\ell(f_{T})\right] - \ell\left(\mathbb{E}\left[f_{T}\right]\right) = \mathcal{O}\left(\mathrm{Var}(f_{T})\right).$$
>
> Combined with the concentration of $f_{T}$, this yields variance-limited scaling: $L(D) - \lim_{D\to\infty}L(D) = \mathcal{O}(D^{-1})$ for dataset scaling and $L(P) - \lim_{P\to\infty}L(P) = \mathcal{O}(P^{-1})$ (equivalently $\mathcal{O}(\sqrt{P}^{-1})$ in width) for model scaling.

#### 2.1.1 Dataset scaling

For dataset scaling, the trained network concentrates around its infinite-data limit:

$$\mathbb{E}_{D}\left[\left(f_{T} - \mathbb{E}_{D}\left[f_{T}\right]\right)^{2}\right] = \mathcal{O}\left(D^{-1}\right),$$

so that by Theorem 1 the loss obeys $L(D) - \lim_{D\to\infty}L(D) = \mathcal{O}(D^{-1})$, i.e. $\alpha_{D} = 1$.

#### 2.1.2 Large width scaling

For large width, the trained network concentrates around its infinite-width limit [17, 45, 35]:

$$\mathbb{E}_{w}\left[\left(f_{T} - \mathbb{E}_{w}\left[f_{T}\right]\right)^{2}\right] = \mathcal{O}\left(P^{-1}\right),$$

so that the loss obeys $L(P) - \lim_{P\to\infty}L(P) = \mathcal{O}(P^{-1}) \sim \mathcal{O}(\sqrt{P}^{-1})$ in width, i.e. $\alpha_{W} = 1$.

We note that there has also been work studying the combined large depth and large width limit, where Hanin and Nica [46] found a well-defined infinite size limit with controlled fluctuations in randomly initialized deep neural networks. In any such context where the trained model predictions concentrate, we expect the loss to scale with the variance of the model output. In the case of linear models, studied below, the variance is $\mathcal{O}(P^{-1})$ rather than $\mathcal{O}(\sqrt{P}^{-1})$, and we see the associated variance scaling in this case.

### 2.2 Resolution-limited exponents

In this section we consider training and test data drawn uniformly from a compact $d$-dimensional manifold, $x \in \mathcal{M}_{d}$, and targets given by some smooth function $y = \mathcal{F}(x)$ on this manifold.

#### 2.2.1 Overparameterized dataset scaling

When $P \gg D \gg 1$, informally, if our train and test data are drawn i.i.d. from the same manifold, then the distance from a test point to the closest training data point decreases as we add more and more training data points. In particular, this distance scales as $\mathcal{O}(D^{-1/d})$ [47]. Furthermore, if $f$, $\mathcal{F}$ are both sufficiently smooth, they cannot differ too much over this distance. If in addition the loss function $L$ is a smooth function vanishing when $f = \mathcal{F}$, we have $L = \mathcal{O}(D^{-1/d})$. This is summarized in the following theorem.

> **Theorem 2 (Resolution-limited dataset scaling).** Suppose the model interpolates the training data, $f(x) = \mathcal{F}(x),\ \forall x \in \mathcal{D}$, and that $f$ and $\mathcal{F}$ are Lipschitz continuous with constants $K_{f}$ and $K_{\mathcal{F}}$ respectively, and the loss $\ell$ is Lipschitz with constant $K_{L}$. Then
>
> $$L(D) = \mathcal{O}\left(K_{L}\,\max(K_{f}, K_{\mathcal{F}})\, D^{-1/d}\right).$$

#### 2.2.2 Underparameterized parameter scaling

We will again assume that $\mathcal{F}$ varies smoothly on an underlying compact $d$-dimensional manifold $\mathcal{M}_{d}$. We can obtain a bound on $L(P)$ if we imagine that $f$ approximates $\mathcal{F}$ as a piecewise function with roughly $P$ regions (see [10]). Here, we instead make use of the argument from the over-parameterized, resolution-limited regime above. If we construct a sufficiently smooth estimator for $\mathcal{F}$ by interpolating among $P$ randomly chosen points from the (arbitrarily large) training set, then by the argument above the loss will be bounded by $\mathcal{O}(P^{-1/d})$.

> **Theorem 3 (Resolution-limited parameter scaling).** Under analogous smoothness assumptions to Theorem 2, but with the model approximating $\mathcal{F}$ through $P$ regions,
>
> $$L(P) = \mathcal{O}\left(K_{L}\,\max(K_{f}, K_{\mathcal{F}})\, P^{-1/d}\right).$$

#### 2.2.3 From bounds to estimates

The above theorems give bounds $L(D) = \mathcal{O}(D^{-1/d})$, but the true scaling exponent is often larger. To obtain a sharper estimate, expand the loss at a test point $x_{\text{test}}$ around its nearest training point $\hat{x}_{\text{train}}$. Because the model interpolates the training data and (in the resolution-limited regime) can match not only the value but also the local structure of $\mathcal{F}$, the leading correction begins at second order:

$$L(x_{\text{test}}) = \sum_{m = n \geq 2}^{\infty} a_{m}(\hat{x}_{\text{train}})\,(x_{\text{test}} - \hat{x}_{\text{train}})^{m},$$

where the $a_{m}$ are tensors encoding the mismatch in higher-order derivatives between $f$ and $\mathcal{F}$, and we have used a compressed notation for multi-tensor contractions in higher-order terms. Since the nearest-neighbor distance scales as $|x_{\text{test}} - \hat{x}_{\text{train}}| \sim D^{-1/d}$, the leading (quadratic) term in the error gives $|f - \mathcal{F}| \sim D^{-2/d}$, and squaring once more for an MSE-type loss yields $L \sim D^{-4/d}$. This motivates the estimate $\alpha_{D} \approx 4/d$, which is corroborated empirically in Figure 1b for teacher-student models.

> **Figure 2:** (a) Random feature models exhibit all four scaling regimes. Here we consider linear teacher-student models with random features trained with MSE loss to convergence. We see both variance-limited scaling (top-left, bottom-right) and resolution-limited scaling (top-right, bottom-left). Data is varied by downsampling MNIST by the specified pool size. (b) Duality and spectra in random feature models. Here we show the relation between the decay of the kernel spectra, $\alpha_{K}$, and the scaling of the loss with number of data points, $\alpha_{D}$, and with number of parameters, $\alpha_{P}$. (top) The spectra of kernels derived from random fully-connected deep neural networks on pooled MNIST (bottom) appear well described by a power-law decay. The theoretical relation $\alpha_{D} = \alpha_{P} = \alpha_{K}$ is given by the dashed black line.

### 2.3 Explicit realization in linear random feature models

In the proceeding sections we have conjectured typical-case scaling relations for a model's test loss. We have further given intuitive arguments for this behavior which relied on smoothness assumptions on the loss and training procedure. In this section, we provide a concrete realization of all four scaling regimes within the context of linear models constructed from random features. Of particular interest is the resolution-limited regime, where the scaling of the loss is a consequence of the linear model kernel spectrum — the scaling of overparameterized models with dataset size and underparameterized models with parameters is a consequence of a classic result, originally due to Weyl [38], bounding the spectrum of sufficiently smooth kernel functions by the dimension of the manifold they act on.

Here we discuss linear models in general terms, though the results immediately hold for the special cases of wide, deep neural networks [12, 13, 14, 52, 53, 54]. We focus on teacher-student models, in which the teacher generates samples from which the student model learns. We will assume student weights initialized to zero and trained with mean squared error (MSE) loss to their global optimum [57, 16, 55, 56].

We consider a linear teacher $F$ and student $f$,

$$
F(x) = \sum_{M=1}^{S} \omega_{M} F_{M}(x), \qquad f(x) = \sum_{\mu=1}^{P} \theta_{\mu} f_{\mu}(x).
$$

The student features $f_{\mu}$ are random projections of the teacher features, $f_{\mu} = \sum_{M} \mathcal{P}_{\mu M} F_{M}$. In the overparameterized limit, the loss at the optimum scales as $a^{-1/D} \sim f(x_{a}) \to F(x_{a})$ for training points, while the test loss takes the form below. Defining $L(P) := \lim_{D \to \infty} L(D, P)$,

$$
L(P) = \frac{1}{2S}\,\mathrm{Tr}\!\left[\mathcal{C} - \mathcal{C}\,\mathcal{P}^{T}\left(\mathcal{P}\,\mathcal{C}\,\mathcal{P}^{T}\right)^{-1}\mathcal{P}\,\mathcal{C}\right],
$$

where $\mathcal{C} = \mathbb{E}_{x}\!\left[F(x)\,F^{T}(x)\right]$ is the teacher feature covariance. Similarly, defining $L(D) := \lim_{P \to S} L(D, P)$,

$$
L(D) = \frac{1}{2}\,\mathbb{E}_{x}\!\left[\mathcal{K}(x,x) - \vec{\mathcal{K}}(x)\,\bar{\mathcal{K}}^{-1}\,\vec{\mathcal{K}}(x)\right].
$$

Here, $\mathcal{K}(x, x^{\prime})$ is the data-data second moment matrix, $\vec{\mathcal{K}}$ indicates restricting one argument to the $D$ training points, while $\bar{\mathcal{K}}$ indicates restricting both. This test loss vanishes as the number of training points becomes infinite but is non-zero for finite training size.

We present a full derivation of these expressions in the supplement. In the remainder of this section, we explore the scaling of the test loss with dataset and model size.

#### 2.3.1 Variance-limited scaling

In the variance-limited regime, the empirical covariance $\bar{\mathcal{C}} = \mathcal{C} + \delta\mathcal{C}$ fluctuates around its population value with $\mathbb{E}_{D}[\delta\mathcal{C}] = 0$ and $\mathbb{E}_{D}[\delta\mathcal{C}^{2}] = \mathcal{O}(D^{-1})$. Similarly for the kernel, $\frac{1}{P}f^{T}(x)\,f(x^{\prime}) = \mathcal{K} + \delta\mathcal{K}$ with $\mathbb{E}_{P}[\delta\mathcal{K}^{2}] = \mathcal{O}(P^{-1})$. Substituting these fluctuations into the exact expressions above yields the variance-limited scalings $L(D, P) - L(P) = \mathcal{O}(D^{-1})$ and $L(D, P) - L(D) = \mathcal{O}(P^{-1})$.

In Figure 2a we see evidence of these scaling relations for features built from randomly initialized ReLU deep neural networks on coarse-grained versions of the MNIST dataset obtained by local averaging over the images. We see that in the variance-limited regimes the scaling exponent is independent of the modification to the training data. In the supplement, we provide an in-depth derivation of this behavior and expressions for the leading contributions to $L(D, P) - L(P)$ and $L(D, P) - L(D)$.

#### 2.3.2 Resolution-limited scaling

Suppose the kernel spectrum decays as a power law,

$$\lambda_{i} \sim \frac{1}{i^{1+\alpha_{K}}},$$

where $\alpha_{K}$ is the spectral decay exponent. In this case, we will argue that the losses also obey a power-law scaling, with the exponents controlled by the spectral decay factor $1 + \alpha_{K}$:

$$
L(D) \propto D^{-\alpha_{K}}, \qquad L(P) \propto P^{-\alpha_{K}}.
$$

That is, $\alpha_{P} = \alpha_{D} = \alpha_{K}$ [18, 19]. Furthermore, for smooth kernels on a $d$-dimensional manifold, Weyl's law [38, 39, 40, 41] gives $\alpha_{K} \propto d^{-1}$, connecting the spectral decay to the data manifold dimension.

#### 2.3.3 From spectra to scaling laws for the loss

The loss can be written in terms of the kernel eigenvalues $\lambda_{i}$ (with eigenvectors $e_{i}$) and the empirical eigenvectors $\bar{e}_{j}$ of the finite dataset:

$$
L(D) = \frac{1}{2}\sum_{i=1}^{S} \lambda_{i}\left(1 - \sum_{j=1}^{D}(e_{i} \cdot \bar{e}_{j})^{2}\right).
$$

Before discussing the general asymptotic behavior, we can gain some intuition by considering the case of large $\alpha_{K}$. In this case, $\bar{e}_{j} \approx e_{j}$ (see e.g. [58]), and we can simplify to

$$
L(D) \propto \sum_{i=D+1}^{\infty} \frac{1}{i^{1+\alpha_{K}}} = \alpha_{K}\,D^{-\alpha_{K}} + \mathcal{O}(D^{-\alpha_{K}-1}).
$$

More generally, a careful asymptotic analysis (using the framework of [19, 59]) confirms the scaling $L(D) \propto D^{-\alpha_{K}}$ and, via duality, $L(P) \propto P^{-\alpha_{K}}$.

#### 2.3.4 Data manifolds and kernels

In Section 2.2, we discussed a simple argument that resolution-limited exponents $\alpha \propto 1/d$, where $d$ is the dimension of the data manifold. Our goal now is to explain how this connects with the linearized models and kernels discussed above: how does the spectrum of eigenvalues of a kernel relate to the dimension of the data manifold?

For smooth kernels on a compact $d$-dimensional manifold, Weyl's law bounds the eigenvalue decay:

$$\lambda_{n} \lesssim \frac{1}{n^{1+t/d}},$$

where $t$ depends on the smoothness of the kernel. Concretely, a smooth translation-invariant kernel on a $d$-torus can be expanded in Fourier modes,

$$\mathcal{K}(x, y) = \sum_{n_{I}}\left[a_{n_{I}} \sin(n_{I} \cdot (x - y)) + b_{n_{I}} \cos(n_{I} \cdot (x - y))\right],$$

where $n_{I} = (n_{1}, \cdots, n_{d})$ is a $d$-dimensional wavevector and $a_{n_{I}}, b_{n_{I}}$ are the Fourier coefficients. The smoothness of the kernel controls how fast these coefficients decay with $|n_{I}|$, which in turn sets the spectral decay exponent $\alpha_{K}$ and hence $\alpha_{D}, \alpha_{P} \propto 1/d$.

> **Figure 3:** Effect of data distribution on scaling exponents. For CIFAR-100 superclassed to $N$ classes (left), we find that the number of target classes does not have a visible effect on the scaling exponent. (right) For CIFAR-10 with the addition of Gaussian noise to inputs, we find the strength of the noise has a strong effect on performance scaling with dataset size. All models are WRN-28-10.

### 2.4 Duality

We argued above that for kernels with pure power-law spectra, the asymptotic scaling of the underparameterized loss with respect to model size and the overparameterized loss with respect to dataset size share a common exponent. In the linear setup at hand, the relation between the underparameterized parameter dependence and overparameterized dataset dependence is even stronger. The underparameterized and overparameterized losses are directly related by **exchanging the projection onto random features with the projection onto random training points**.

Note, sample-wise double descent observed in [51] is a concrete realization of this duality for a simple data distribution. In the supplement, we present examples exhibiting the duality of the loss dependence on model and dataset size outside of the asymptotic regime.
