## Appendix E. Random feature models

We provide the detailed derivation of the exact loss expressions and their asymptotics in the linear random feature setting. As in Section 2.3 of the main text, the student is $f(x) = \sum_{\mu=1}^{P} \theta_{\mu} f_{\mu}(x)$ and the teacher is $F(x) = \sum_{M=1}^{S} \omega_{M} F_{M}(x)$, with student features $f_{\mu} = \sum_{M} \mathcal{P}_{\mu M} F_{M}$ being random projections of the teacher features. We consider models with weights initialized to zero and trained to convergence with mean squared error loss,

$$
L_{\textrm{train}} = \frac{1}{2D}\sum_{a=1}^{D}\left(f(x_{a}) - y_{a}\right)^{2}.
$$

The data and feature second moments play a central role in our analysis. We introduce the notation,

$$
\begin{aligned}
&\mathcal{C} = \mathbb{E}_{x}\!\left[F(x)\,F^{T}(x)\right],\quad \bar{\mathcal{C}} = \frac{1}{D}\sum_{a=1}^{D} F(x_{a})\,F^{T}(x_{a}),\quad C = \mathcal{P}\,\mathcal{C}\,\mathcal{P}^{T},\quad \bar{C} = \mathcal{P}\,\bar{\mathcal{C}}\,\mathcal{P}^{T},\\
&\mathcal{K}(x, x^{\prime}) = \frac{1}{S}\,F^{T}(x)\,F(x^{\prime}),\quad \bar{\mathcal{K}} = \mathcal{K}\Big{|}_{\mathcal{D}_{\textrm{train}}},\quad K(x, x^{\prime}) = \frac{1}{P}\,f^{T}(x)\,f(x^{\prime}),\quad \bar{K} = K\Big{|}_{\mathcal{D}_{\textrm{train}}},
\end{aligned}
$$

where $\vec{\mathcal{K}}(x) := \mathcal{K}(x, x_{a=1\ldots D})$ and $\vec{K}(x) := K(x, x_{a=1\ldots D})$ denote the kernel restricted to one argument being a test point and the other ranging over the training set.

The exact test loss for underparameterized models ($P < D$, at the optimum found by GD) is

$$
L(D, P) = \frac{1}{2S}\,\mathbb{E}_{D}\!\left[\operatorname{Tr}\!\left(\mathcal{C} + \bar{\mathcal{C}}\,\mathcal{P}^{T}\bar{C}^{-1}C\bar{C}^{-1}\mathcal{P}\,\bar{\mathcal{C}} - 2\,\bar{\mathcal{C}}\,\mathcal{P}^{T}\bar{C}^{-1}\mathcal{P}\,\mathcal{C}\right)\right],
$$

and for overparameterized models ($P > D$, at the unique minimum found by GD, SGD, or projected Newton's method),

$$
L(D, P) = \frac{1}{2}\,\mathbb{E}_{x,D}\!\left[\mathcal{K}(x,x) + \vec{K}(x)^{T}\bar{K}^{-1}\bar{\mathcal{K}}\,\bar{K}^{-1}\vec{K}(x) - 2\,\vec{K}(x)^{T}\bar{K}^{-1}\vec{\mathcal{K}}(x)\right].
$$

Here $\mathbb{E}_{D}[\cdot]$ is an expectation with respect to i.i.d. draws of a dataset of size $D$ from the input distribution, while $\mathbb{E}_{x}[\cdot]$ is an ordinary expectation over the input distribution. Note, expression (S37) is also valid for overparameterized models and (S38) is valid for underparameterized models if the inverses are replaced with the Moore-Penrose pseudo-inverse. Also note, the two expressions can be related by **exchanging the projections onto finite features with the projection onto the training dataset** and the sums of teacher features with the expectation over the data manifold. This realizes the duality between dataset and features discussed above.

### E.1 Asymptotic expressions

We are interested in (S37) and (S38) in the limits of large $P$ and $D$.

#### Variance-limited scaling

For the underparameterized case, write the empirical covariance as $\bar{\mathcal{C}} = \mathcal{C} + \delta\mathcal{C}$, where the fluctuations satisfy

$$
\begin{aligned}
\mathbb{E}_{D}\!\left[\delta\mathcal{C}\right] &= 0,\\
\mathbb{E}_{D}\!\left[\delta\mathcal{C}_{M_{1}N_{1}}\,\delta\mathcal{C}_{M_{2}N_{2}}\right] &= \frac{1}{D}\left(\mathbb{E}_{x}\!\left[F_{M_{1}}(x)\,F_{N_{1}}(x)\,F_{M_{2}}(x)\,F_{N_{2}}(x)\right] - \mathcal{C}_{M_{1}N_{1}}\,\mathcal{C}_{M_{2}N_{2}}\right),\\
\mathbb{E}_{D}\!\left[\delta\mathcal{C}_{M_{1}N_{1}}\cdots\delta\mathcal{C}_{M_{n}N_{n}}\right] &= \mathcal{O}(D^{-2})\quad \forall\, n > 2.
\end{aligned}
$$

The key takeaway is that the $D$-dependence is manifest. Using these expressions in (S37) yields

$$
\begin{aligned}
L(D, P) &= \frac{1}{2S}\operatorname{Tr}\!\left(\mathcal{C} - \mathcal{C}\,\mathcal{P}^{T}C^{-1}\mathcal{P}\,\mathcal{C}\right)\\
&+ \frac{1}{2DS}\sum_{M_{1,2}N_{1,2}=1}^{P} T_{M_{1}N_{1}M_{2}N_{2}}\!\left[\delta_{M_{1}M_{2}}\left(\mathcal{P}^{T}C^{-1}\mathcal{P}\right)_{N_{1}N_{2}} + \left(C^{-1}\mathcal{P}\,\mathcal{C}^{2}\mathcal{P}^{T}C^{-1}\right)_{M_{1}M_{2}} C^{-1}_{N_{1}N_{2}}\right.\\
&\qquad\qquad\qquad\qquad\qquad\qquad\quad\left.- 2\left(\mathcal{C}\,\mathcal{P}^{T}C^{-1}\mathcal{P}\right)_{M_{1}M_{2}}\left(\mathcal{P}^{T}C^{-1}\mathcal{P}\right)_{N_{1}N_{2}}\right] + \mathcal{O}(D^{-2}),
\end{aligned}
$$

where $T_{M_{1}N_{1}M_{2}N_{2}} = \mathbb{E}_{x}[F_{M_{1}}(x)\,F_{N_{1}}(x)\,F_{M_{2}}(x)\,F_{N_{2}}(x)]$. Defining

$$
L(P) := \lim_{D\rightarrow\infty} L(D, P) = \frac{1}{2S}\operatorname{Tr}\!\left(\mathcal{C} - \mathcal{C}\,\mathcal{P}^{T}C^{-1}\mathcal{P}\,\mathcal{C}\right),
$$

we see that though $L(D, P) - L(P)$ is a somewhat cumbersome quantity to compute, involving the average of a quartic tensor over the data distribution, its dependence on $D$ is simple: $L(D, P) - L(P) = \mathcal{O}(D^{-1})$.

For the overparameterized case, we can similarly expand (S38) using $K = \mathcal{K} + \delta\mathcal{K}$. With fluctuations satisfying,

$$
\begin{aligned}
\mathbb{E}_{P}\!\left[\delta\mathcal{K}\right] &= 0,\\
\mathbb{E}_{P}\!\left[\delta\mathcal{K}_{a_{1}b_{1}}\,\delta\mathcal{K}_{a_{2}b_{2}}\right] &= \frac{1}{P}\left(\mathbb{E}_{P}\!\left[f_{\mu}(x_{a_{1}})\,f_{\mu}(x_{b_{1}})\,f_{\mu}(x_{a_{2}})\,f_{\mu}(x_{b_{2}})\right] - \mathcal{K}_{a_{1}b_{1}}\,\mathcal{K}_{a_{2}b_{2}}\right),\\
\mathbb{E}_{P}\!\left[\delta\mathcal{K}_{a_{1}a_{1}}\cdots\delta\mathcal{K}_{a_{n}a_{n}}\right] &= \mathcal{O}(P^{-2})\quad \forall\, n > 2.
\end{aligned}
$$

This gives the expansion

$$
L(D, P) = \frac{1}{2}\,\mathbb{E}_{x,D}\!\left[\mathcal{K}(x,x) - \vec{\mathcal{K}}(x)^{T}\bar{\mathcal{K}}^{-1}\vec{\mathcal{K}}(x)\right] + \mathcal{O}(P^{-1}),
$$

and

$$
L(D) = \frac{1}{2}\,\mathbb{E}_{x,D}\!\left[\mathcal{K}(x,x) - \vec{\mathcal{K}}(x)^{T}\bar{\mathcal{K}}^{-1}\vec{\mathcal{K}}(x)\right],
$$

confirming $L(D, P) - L(D) = \mathcal{O}(P^{-1})$.

#### Resolution-limited scaling

We now move on to studying the parameter scaling of $L(P)$ and dataset scaling of $L(D)$. We explicitly analyze the dataset scaling of $L(D)$, with the parameter scaling following via the dataset-parameter duality. Building on the framework of [71, 72, 73, 75, 76, 77, 78, 79, 80, 19], the loss takes the form

$$
\begin{aligned}
&L(D) = \frac{\kappa^{2}}{1 - \gamma}\sum_{i}\frac{\lambda_{i}\,\bar{\omega}_{i}^{2}}{\left(\kappa + D\lambda_{i}\right)^{2}},\\
&\kappa = \sum_{i}\frac{\kappa\,\lambda_{i}}{\kappa + D\lambda_{i}},\qquad \gamma = \sum_{i}\frac{D\,\lambda_{i}^{2}}{\left(\kappa + D\lambda_{i}\right)^{2}},
\end{aligned}
$$

where $\lambda_{i} \sim i^{-(1+\alpha_{K})}$ are the kernel eigenvalues and $\bar{\omega}_{i} = O_{iM}\,\omega_{M}$ are the task-label coefficients in the eigenbasis, with $\mathbb{E}_{\omega}[\bar{\omega}_{i}^{2}]$ approximately constant.

With this simplification, we now compute the asymptotic scaling by approximating the sums with integrals and expanding the resulting expressions in large $D$. We use the identities:

$$
\begin{aligned}
\int_{1}^{\infty} dx\,\frac{x^{-n(1+\alpha)}}{\left(\kappa + D\,x^{-(1+\alpha)}\right)^{m}} &= \kappa^{-m}\,\frac{\Gamma\!\left(n - \frac{1}{1+\alpha}\right)}{(1+\alpha)\,\Gamma\!\left(n + \frac{\alpha}{1+\alpha}\right)}\,{}_{2}F_{1}\!\left(m,\, n - \tfrac{1}{1+\alpha},\, n + \tfrac{\alpha}{1+\alpha},\, \tfrac{-D}{\kappa}\right),\\
{}_{2}F_{1}(a, b, c, -y) &\propto y^{-a} + \mathcal{B}\,y^{-b} + \ldots.
\end{aligned}
$$

Here ${}_{2}F_{1}$ is the hypergeometric function and the second line gives its asymptotic form at large $y$. $\mathcal{B}$ is a constant which does not affect the asymptotic scaling.

Using these relations yields

$$
\kappa \propto D^{-\alpha_{K}},\qquad \gamma \propto D^{0},\qquad \text{and}\qquad L(D) \propto D^{-\alpha_{K}},
$$

as promised. Here we have dropped sub-leading terms at large $D$. Scaling behavior for parameter scaling $L(P)$ follows via the data-parameter duality.

> **Figure S6:** Duality between dataset size vs feature number in pretrained features. Using pretrained embedding features of EfficientNet-B5 [81] for different levels of regularization, we see that loss as a function of dataset size or loss as a function of the feature dimension track each other both for small regularization (left) and for tuned regularization (right). Note that regularization strength with trained-feature kernels can be mapped to inverse training time [82, 83]. Thus (left) corresponds to long training time and exhibits double descent behavior, while (right) corresponds to optimal early stopping.

> **Figure S7:** Four scaling regimes exhibited by pretrained embedding features. Using pretrained embedding features of EfficientNet-B5 [81] for fixed low regularization (left) and tuned regularization (right), we can identify four regimes of scaling using real CIFAR-10 labels.

### Duality beyond asymptotics

Expressions (S37) and (S38) are related by changing projections onto the finite feature set and finite dataset even without taking any asymptotic limits. We thus expect the dependence of test loss on parameter count and dataset size to be related quite generally in linear feature models. See Appendix F for further details.
