## Appendix C. Variance-limited dataset scaling

This appendix provides a detailed derivation of the variance-limited dataset scaling,

$$L(D) - \lim_{D\to\infty} L(D) = \mathcal{O}(D^{-1}).$$

### C.1 Intuition

At a high level, the intuition is as follows. For any fixed value of weights $\theta$, the training loss with $D$ training points (thought of as a random variable over draws of the dataset), $L_{\textrm{train}}[\theta]$, concentrates around the population loss $L_{\textrm{pop}}[\theta]$, with variance that scales as $\mathcal{O}(D^{-1})$.

During training, the weights evolve from $\theta_{0}$ to some $\theta_{T}$. At each step the training loss $L_{\textrm{train}}[\theta_{t}]$ is evaluated, and because it concentrates around $L_{\textrm{pop}}[\theta_{t}]$, the trajectory of weights under finite $D$ closely tracks the trajectory under infinite data. Consequently the final population loss $L_{\textrm{pop}}[\theta_{T}]$ — and hence the test loss — deviates from its infinite-data limit by $\mathcal{O}(D^{-1})$:

$$L(D) - \lim_{D\to\infty} L(D) = \mathcal{O}(D^{-1}).$$

#### Early time

Consider the first gradient step. Let $g_{0} = \partial L_{\textrm{train}}/\partial\theta_{0}$ be the gradient and $\theta_{1} = \theta_{0} - \eta\, g_{0}$. Then

$$
L_{\textrm{train}}[\theta_{1}] = L_{\textrm{train}}[\theta_{0} - g_{0}].
$$

Since $g_{0}$ concentrates around $\mathbb{E}_{D}[g_{0}]$ (its infinite-data value), $L_{\textrm{train}}[\theta_{1}]$ concentrates around its population counterpart. Iterating this argument over $T$ steps (with $T$ fixed as $D \to \infty$), the entire trajectory $\mathbb{E}_{D}[\theta_{t=0\ldots T}]$ concentrates, giving $L(D) - \lim_{D\to\infty}L(D) = \mathcal{O}(D^{-1})$.

#### Local minimum

The above intuition relied on training for a number of steps that was fixed as $D$ is taken large. Here we present some alternative intuition for the variance-limited scaling at late times, as training approaches a local minimum in the loss. For simplicity we discuss a one-dimensional loss.

Let $\theta^{*}$ be the true (population) optimum and $\bar{\theta}^{*}$ the empirical optimum. Then $|\theta^{*} - \bar{\theta}^{*}| = \mathcal{O}(D^{-1})$. To see this, suppose $\theta_{1} < \theta^{*} < \theta_{2}$ are two points bracketing $\theta^{*}$. Since $L'_{\textrm{pop}}[\theta_{1}]$ and $L'_{\textrm{pop}}[\theta_{2}]$ have opposite signs, and the empirical gradient $L'_{\textrm{train}}[\theta]$ concentrates around $L'_{\textrm{pop}}[\theta]$, with high probability $L'_{\textrm{train}}[\theta_{1}]$ and $L'_{\textrm{train}}[\theta_{2}]$ also have opposite signs, so $\bar{\theta}^{*} \in [\theta_{1}, \theta_{2}]$.

By Hoeffding's inequality, the deviation of the empirical gradient from the population gradient is bounded:

$$
P\!\left(\left|L^{\prime}_{\textrm{train}}[\theta] - L^{\prime}_{\textrm{pop}}[\theta]\right| > a\right) \leq 2\,e^{-\frac{2}{I}\,D^{2}\,a^{2}},
$$

where $I$ is a constant related to the range of the gradient. This exponential concentration implies that $\bar{\theta}^{*}$ is within $\mathcal{O}(D^{-1})$ of $\theta^{*}$.

Now decompose the test loss:

$$
L(D) - L(\infty) = \mathbb{E}_{D}\!\left[L_{\textrm{train}}[\bar{\theta}^{*}] - L_{\textrm{pop}}[\bar{\theta}^{*}]\right] + \mathbb{E}_{D}\!\left[L_{\textrm{pop}}[\bar{\theta}^{*}] - L_{\textrm{pop}}[\theta^{*}]\right].
$$

The first term — the difference between training and population loss at the empirical optimum — is $\mathcal{O}(D^{-1})$ by concentration of $L_{\textrm{train}}$. The second term — the excess population loss due to using $\bar{\theta}^{*}$ instead of $\theta^{*}$ — is also $\mathcal{O}(D^{-1})$ because $|\bar{\theta}^{*} - \theta^{*}| = \mathcal{O}(D^{-1})$ and the loss is smooth (quadratic) near the minimum. Hence

$$L(D) - \lim_{D\to\infty} L(D) = \mathcal{O}(D^{-1}).$$

#### Stochastic gradient descent (SGD)

At first blush it may be surprising that the variance-limited scaling holds even for mini-batch training. Indeed in this case, there is batch noise that comes in at a much higher scale than any variance due to the finite training set size. Indeed, the effect of mini-batching changes the final test loss; however, if we fix the SGD procedure or average over SGD seeds, as we take $D$ large, we can still ask how the training loss for a model trained under SGD on a training set of size $D$ differs from that for a model trained under SGD on an infinite training set.

To see this, we first consider averaging over minibatches of size $B$, but where points are drawn i.i.d. with replacement. If we denote the batch at step $t$ by $\mathcal{B}_{t}$ and the average over independent draws of this batch by $\mathbb{E}_{B}[\cdot]$, then note we can translate moments with respect to batch draws with empirical averages over the entire training set. Explicitly, consider $c_{a}$ and $d_{a}$ potentially correlated, but each drawn i.i.d. within a batch. We have that,

$$
\begin{aligned}
\mathbb{E}_{B}\!\left[\frac{1}{B}\sum_{a\in\mathcal{B}_{t}} c_{a}\right] &= \frac{1}{D}\sum_{a=1}^{D} c_{a},\\
\mathbb{E}_{B}\!\left[\left(\frac{1}{B}\sum_{a\in\mathcal{B}_{t}} c_{a}\right)\!\left(\frac{1}{B}\sum_{a^{\prime}\in\mathcal{B}_{t}} d_{a^{\prime}}\right)\right] &= \left(1 - \frac{1}{B}\right)\!\left(\frac{1}{D}\sum_{a=1}^{D} c_{a}\right)\!\left(\frac{1}{D}\sum_{a^{\prime}=1}^{D} d_{a^{\prime}}\right) + \frac{1}{B}\,\frac{1}{D}\sum_{a=1}^{D} c_{a}\,d_{a}.
\end{aligned}
$$

This procedure means, after taking an average over draws of SGD batches, rather than thinking about a function of mini-batch averages, we can equivalently consider a modified function, with explicit dependence on the batch size, but that is only a function of empirical means over the training set. We can thus recycle the above intuition for the scaling of smooth functions of empirical means.

The above relied on independently drawing every sample from every batch. At the other extreme, we can consider drawing batches without shuffling and increasing training set size by $B$ datapoints at a time, so as to keep the initial set of batches in an epoch fixed. In this case, the first deviation in training between a dataset of size $D$ and one of size $D + B$ happens at the last batch in the first epoch after processing $D$ datapoints. Consequently, if the total number of samples seen is $BT$ (batch size $B$ times $T$ steps) and $D > BT$, then every sample seen is distinct and training is identical to the infinite-data case:

$$\lim_{D\to\infty} L(D; T; B) = L(BT; T; B), \qquad L(D > BT) - \lim_{D\to\infty} L(D) = 0.$$

### C.2 Polynomial loss

Before discussing neural network training, we review the concentration behavior of polynomials of sample means.

**Lemma 1.** Let $\bar{c}^{(i)} = \frac{1}{D}\sum_{a=1}^{D} c^{(i)}_{a}$ be sample means of i.i.d. random variables, and let $X = (\bar{c}^{(0)})^{k_{0}}(\bar{c}^{(1)})^{k_{1}}\cdots(\bar{c}^{(J)})^{k_{J}}$. Then

$$
\mathbb{E}_{D}\!\left[\left(X - ({c}^{(0)})^{k_{0}}({c}^{(1)})^{k_{1}}\cdots({c}^{(J)})^{k_{J}}\right)^{n}\right] = \mathcal{O}(D^{-1}).
$$

Here, $\mathbb{E}_{D}[\cdot]$ denotes the average over independent draws of $D$ samples.

To establish this we can proceed by direct computation.

$$
\begin{aligned}
&\mathbb{E}_{D}\!\left[\left(X - ({c}^{(0)})^{k_{0}}({c}^{(1)})^{k_{1}}\cdots({c}^{(J)})^{k_{J}}\right)^{n}\right]\\
&\qquad= \sum_{p=0}^{n}(-1)^{n-p}\binom{n}{p}\,\mathbb{E}_{D}\!\left[X^{p}\right]\left(({c}^{(0)})^{k_{0}}({c}^{(1)})^{k_{1}}\cdots({c}^{(J)})^{k_{J}}\right)^{n-p}.
\end{aligned}
$$

Each term in the sum can be computed using

$$
\begin{aligned}
\mathbb{E}_{D}\!\left[X^{p}\right] &= \mathbb{E}_{D}\!\left[(\bar{c}^{(0)})^{pk_{0}}(\bar{c}^{(1)})^{pk_{1}}\cdots(\bar{c}^{(J)})^{pk_{J}}\right]\\
&= \frac{1}{D^{p\sum_{i} k_{i}}}\sum_{\{a^{(i)}_{\alpha}\}}\mathbb{E}_{D}\!\left[\left(c^{(0)}_{a^{(0)}_{1}}\cdots c^{(0)}_{a^{(0)}_{pk_{0}}}\right)\!\left(c^{(1)}_{a^{(1)}_{1}}\cdots c^{(1)}_{a^{(1)}_{pk_{1}}}\right)\cdots\left(c^{(J)}_{a^{(J)}_{1}}\cdots c^{(J)}_{a^{(J)}_{pk_{J}}}\right)\right]\\
&= \frac{D(D-1)\cdots(D - (p\textstyle\sum_{i} k_{i} - 1))}{D^{p\sum_{i} k_{i}}}\left(c^{(0)}\right)^{pk_{0}}\!\left(c^{(1)}\right)^{pk_{1}}\cdots\left(c^{(J)}\right)^{pk_{J}} + \mathcal{O}(D^{-1})\\
&= \left(\left(c^{(0)}\right)^{k_{0}}\!\left(c^{(1)}\right)^{k_{1}}\cdots\left(c^{(J)}\right)^{k_{J}}\right)^{p} + \mathcal{O}(D^{-1}),
\end{aligned}
$$

where in the third line we restricted to terms where all indices $a^{(i)}_{\alpha} \neq a^{(j)}_{\beta}$ are distinct (the remaining terms contribute at $\mathcal{O}(D^{-1})$). Plugging this in establishes the lemma. $\blacksquare$

Lemma 1 immediately implies that the mean of polynomials of $\bar{c}^{(i)}$ concentrate around their infinite-data limit:

$$
\mathbb{E}_{D}\!\left[\left(g(\bar{c}^{(0)}, \bar{c}^{(1)}, \ldots, \bar{c}^{(K)}) - g({c}^{(0)}, {c}^{(1)}, \ldots, {c}^{(K)})\right)^{n}\right] = \mathcal{O}(D^{-1}),
$$

for any polynomial $g$.

Now consider a neural network with data $\mathbf{x}_{a} = (x_{a}, y_{a})$. The network output and per-sample loss can be written as polynomials in the weights $\theta$:

$$
f(x) = \sum_{i=1}^{J} b^{(i)}_{\mu_{1}\mu_{2}\ldots\mu_{i}}(x)\,\theta_{\mu_{1}}\theta_{\mu_{2}}\cdots\theta_{\mu_{i}},\qquad \ell(\mathbf{x}_{a}) = \sum_{i=1}^{K} c^{(i)}_{\mu_{1}\mu_{2}\ldots\mu_{i}}(\mathbf{x}_{a})\,\theta_{\mu_{1}}\theta_{\mu_{2}}\cdots\theta_{\mu_{i}}.
$$

The training loss can then be written as,

$$
L_{\textrm{train}} = \sum_{i=1}^{K} \bar{c}^{(i)}_{\mu_{1}\mu_{2}\ldots\mu_{i}}\,\theta_{\mu_{1}}\theta_{\mu_{2}}\cdots\theta_{\mu_{i}},\qquad \bar{c}^{(i)} = \frac{1}{D}\sum_{a=1}^{D} c^{(i)}(\mathbf{x}_{a}),
$$

where the repeated weight indices $\mu_{j}$ are summed over.

#### Gradient descent

Under gradient descent, $\theta_{t+1} = \theta_{t} - \eta\,\partial L_{\textrm{train}}/\partial\theta$. Since the training loss is a degree-$K$ polynomial in the $\bar{c}^{(i)}$ and the weights, after $T$ steps the weights are a polynomial in the $\bar{c}^{(i)}$:

$$
\theta_{T} \in P_{(K-1)^{T}}\!\left[\bar{c}^{(0)}, \bar{c}^{(1)}, \ldots, \bar{c}^{(K)}\right].
$$

The coefficients of this polynomial depend on the initial weights $\theta_{0}$. Plugging these weights back into the network output, the network function at time $T$ is again a polynomial in $\bar{c}^{(i)}$, now with degree $J(K-1)^{T}$:

$$
f_{T}(x) \in P_{J(K-1)^{T}}\!\left[\bar{c}^{(0)}, \bar{c}^{(1)}, \ldots, \bar{c}^{(K)}\right].
$$

Thus, again using Lemma 1, $f_{T}$ concentrates with variance $\mathcal{O}(D^{-1})$:

$$
\mathbb{E}_{D}\!\left[\left(f_{T} - \mathbb{E}_{D}\!\left[f_{T}\right]\right)^{2}\right] = \mathcal{O}(D^{-1}),
$$

and by Theorem 1 the loss will obey the variance-limited scaling.

#### Stochastic gradient descent

We now consider the same setup of polynomial loss, but now trained via stochastic gradient descent (SGD). We consider SGD batches drawn i.i.d. with replacement and are interested in the test loss averaged over SGD draws, with fixed batch size $B$.

We proceed by proving **Lemma 2**, which allows us to reuse a similar argument to the GD case. Consider products of batch-averaged coefficients $\tilde{c}^{(i; t)} = \frac{1}{B}\sum_{a \in \mathcal{B}_{t}} c^{(i)}_{a}$ at various timesteps, and let

$$X = (\tilde{c}^{(0; t_{0})})^{k_{0}}(\tilde{c}^{(1; t_{1})})^{k_{1}}\cdots(\tilde{c}^{(J; t_{J})})^{k_{J}},$$

where $t = t_{0}, t_{1}, \cdots, t_{J}$. Lemma 2 states that $\mathbb{E}_{B}[X]$ is a polynomial in empirical means $\bar{d}^{(\cdot)}$ over the full training set, of degree at most $\prod_{i}(k_{i}+1) - 1$.

To compute $\mathbb{E}_{B}[X]$, we have

$$
\mathbb{E}_{B}\!\left[X\right] = \frac{1}{B^{\sum_{i} k_{i}}}\,\mathbb{E}_{B}\!\left[\sum_{\{a^{(i)}_{\alpha}\} \in \mathcal{B}_{t}} \left(c^{(0)}_{a^{(0)}_{1}}\cdots c^{(0)}_{a^{(0)}_{k_{0}}}\right)\!\left(c^{(1)}_{a^{(1)}_{1}}\cdots c^{(1)}_{a^{(1)}_{k_{1}}}\right)\cdots\left(c^{(J)}_{a^{(J)}_{1}}\cdots c^{(J)}_{a^{(J)}_{k_{J}}}\right)\right].
$$

To proceed, we must keep track of terms in the sum where the $a^{(i)}_{\alpha}$ take the same or different values. If all $a^{(i)}_{\alpha}$ are different, the expectation over batch draws fully factorizes. More generally this can be decomposed as a sum over products.

One way of keeping track of the index combinatorics is to introduce a set of graphs $\Gamma$, where each graph $\gamma \in \Gamma$ has $k_{0}$ vertices of type 0, $k_{1}$ vertices of type 1, …, and $k_{J}$ vertices of type $J$ (one vertex for each $a^{(i)}_{\alpha}$ index). Any pair of vertices may have zero or one edge between them. For any set of three vertices $v_{1}, v_{2}, v_{3}$ with edges $(v_{1}, v_{2})$ and $(v_{2}, v_{3})$ there must also be an edge $(v_{1}, v_{3})$. The set $\Gamma$ consists of all possible ways of connecting these vertices consistent with these rules.

For each graph $\gamma$, we denote connected components by $\sigma$ and the number of vertices of type $i$ within component $\sigma$ by $m^{(i)}_{\sigma}$. With this we can write

$$
\begin{aligned}
\mathbb{E}_{B}\!\left[X\right] &= \sum_{\gamma \in \Gamma} S_{\gamma}(B)\prod_{\sigma \in \gamma}\,\mathbb{E}_{B}\!\left[\frac{1}{B}\sum_{a \in \mathcal{B}_{t}} (c^{(0)}_{a})^{m^{(0)}_{\sigma}}(c^{(1)}_{a})^{m^{(1)}_{\sigma}}\cdots(c^{(J)}_{a})^{m^{(J)}_{\sigma}}\right]\\
&= \sum_{\gamma \in \Gamma} S_{\gamma}(B)\prod_{\sigma \in \gamma}\,\frac{1}{D}\sum_{a=1}^{D}(c^{(0)}_{a})^{m^{(0)}_{\sigma}}(c^{(1)}_{a})^{m^{(1)}_{\sigma}}\cdots(c^{(J)}_{a})^{m^{(J)}_{\sigma}}\\
&= \sum_{\gamma \in \Gamma} S_{\gamma}(B)\prod_{\sigma \in \gamma}\,\bar{d}^{(\{m^{(0)}_{\sigma}, m^{(1)}_{\sigma}, \ldots, m^{(J)}_{\sigma}\})},
\end{aligned}
$$

where $S_{\gamma}(B)$ are combinatorial factors depending on the batch size. This confirms Lemma 2: $\mathbb{E}_{B}[X]$ is a polynomial of degree at most $\prod_{i}(k_{i}+1) - 1$ in the empirical means $\bar{d}^{(\cdot)}$.

For a polynomial loss of degree $K$, the mini-batch training loss at each time step takes the form

$$
L_{\textrm{train}}^{(t)} = \sum_{i=1}^{K} \tilde{c}^{(i; t)}_{\mu_{1}\mu_{2}\ldots\mu_{i}}\,\theta_{\mu_{1}}\theta_{\mu_{2}}\cdots\theta_{\mu_{i}},\qquad \tilde{c}^{(i; t)} = \frac{1}{B}\sum_{a \in \mathcal{B}_{t}} c^{(i)}(\mathbf{x}_{a}).
$$

After $T$ steps, $\theta_{T} = \theta_{0} - \eta \sum_{t} \partial L_{\textrm{train}}^{(t)}/\partial\theta$ is a polynomial in all the batch-averaged coefficients $\tilde{c}^{(i; 0)}, \tilde{c}^{(i; 1)}, \ldots, \tilde{c}^{(i; T)}$:

$$
\theta_{T} \in P_{(K-1)^{T}}\!\left[\tilde{c}^{(0; 0)}, \tilde{c}^{(0; 1)}, \ldots, \tilde{c}^{(0; T)}, \tilde{c}^{(1; 0)}, \ldots, \tilde{c}^{(K; T)}\right],
$$

and consequently, denoting the test loss evaluated at $\theta_{T}$ by $L[\theta_{T}]$,

$$
L[\theta_{T}] \in P_{K(K-1)^{T}}\!\left[\tilde{c}^{(0; 0)}, \ldots, \tilde{c}^{(K; T)}\right].
$$

Using Lemma 2, the expectation of $L[\theta_{T}]$ over draws of SGD batches is given by

$$
\mathbb{E}_{B}\!\left[L[\theta_{T}]\right] \in P_{K(K-1)^{T}}\!\left[\bar{d}^{(0)}, \ldots, \bar{d}^{(K^{K}(K-1)^{TK})}\right],
$$

i.e. it is a polynomial in empirical means over the full training set. Applying Lemma 1 to $\mathbb{E}_{B}[L[\theta_{T}]]$ then yields

$$
L(D; B) - \lim_{D\rightarrow\infty} L(D; B) = \mathcal{O}(D^{-1}).
$$

### C.3 Non-smooth examples

Here we present two worked examples where non-bounded or non-smooth loss leads to violations of the variance-dominated scaling. In Example 1, the system obeys the variance-dominated scaling at early times, but exhibits different behavior for times larger than the dataset size. In Example 2, the system violates the variance-dominated scaling even for two gradient descent steps, as a result of an unbounded derivative in the loss.

#### Example 1 — unbounded loss at late times

Consider a dataset with two varieties of data points, drawn with probabilities $\alpha$ and $1 - \alpha$, and one-dimensional quadratic losses $\ell_{1}$ (concave up) and $\ell_{2}$ (concave down) on these two varieties:

$$
\ell_{1}(\theta) = \tfrac{1}{2}\theta^{2},\qquad \ell_{2}(\theta) = -\tfrac{1}{2}\theta^{2}.
$$

Denoting the training loss on a sample with $n_{1}$ points of type 1 and $D - n_{1}$ of type 2 by $\ell_{n_{1}}$, and the population loss by $L_{\textrm{pop}}$,

$$
\ell_{n_{1}} = \left(\tfrac{n_{1}}{D} - \tfrac{1}{2}\right)\theta^{2},\qquad L_{\textrm{pop}} = \left(\alpha - \tfrac{1}{2}\right)\theta^{2}.
$$

We take $\alpha > 1/2$. In this case, the minimum of the population loss is at zero, while the minimum of the training loss can be at zero or at $\pm\infty$ depending on whether the training sample has $n_{1}$ greater than or less than $D/2$. We can thus create a situation where at late training times $\theta_{T}$ does not concentrate around the minimum of the population loss, i.e. $\min \lim_{D\to\infty} L(D) \neq \min_{\theta} L_{\textrm{pop}}$.

Explicitly, we study the evolution of the model under gradient flow:

$$
\dot{\theta} = -2\left(\tfrac{n_{1}}{D} - \tfrac{1}{2}\right)\theta,\qquad \theta_{T} = e^{-2(\frac{n_{1}}{D} - \frac{1}{2})T}\,\theta_{0}.
$$

The test loss averaged over draws of the dataset is

$$
L(D; T) = \mathbb{E}_{n_{1}}\!\left[\left(\alpha - \tfrac{1}{2}\right)\theta_{T}^{2}\right] = e^{2T}\left(\alpha - \tfrac{1}{2}\right)\left(1 - \alpha\!\left(1 - e^{-\frac{4T}{D}}\right)\right)^{D}\theta_{0}^{2}.
$$

If we consider this loss at large $D$ and fixed $T$ we get

$$
L(D; T) = e^{-4(\alpha - \frac{1}{2})T}\left(\alpha - \tfrac{1}{2}\right)\theta_{0}^{2}\left(1 + \frac{8T^{2}\alpha(1-\alpha)}{D} + \mathcal{O}(D^{-2})\right),
$$

so $L(D; T) - \lim_{D\to\infty} L(D; T) = \mathcal{O}(D^{-1})$ — variance-dominated scaling holds at fixed $T$.

If on the other hand we consider taking $T \gg D$ we have

$$
L(D; T \gg D) = e^{2T}\left(\alpha - \tfrac{1}{2}\right)(1 - \alpha)^{D}\,\theta_{0}^{2},
$$

and $\lim_{D, T \to \infty} L(D; T \gg D)$ diverges — the variance-dominated scaling breaks down at late times. Lastly, we note that if we take $T = \beta D$ with $\beta < |\log(1-\alpha)|/2$ we can approach the large-$D$ limit with non-generic, tuneable exponential convergence.

#### Example 2 — unbounded derivative

Again, consider a two-variety setup, this case with equal probabilities and per-sample losses,

$$
\ell_{1}(\theta) = \tfrac{1}{2}\theta^{2} + \tfrac{1}{2\alpha}|\theta|^{\alpha},\qquad \ell_{2}(\theta) = \tfrac{1}{2}\theta^{2} - \tfrac{1}{2\alpha}|\theta|^{\alpha}.
$$

We will consider different values of $\alpha > 0$. The train loss and population loss are then,

$$
\ell_{n_{1}} = \tfrac{1}{2}\theta^{2} + \tfrac{1}{\alpha}\left(\tfrac{n_{1}}{D} - \tfrac{1}{2}\right)|\theta|^{\alpha},\qquad L_{\textrm{pop}} = \tfrac{1}{2}\theta^{2}.
$$

We consider a model initialized to $\theta_{0} = 1$ and trained for two steps of gradient descent with learning rate 1:

$$
g_{t} = \theta_{t} + \left(\tfrac{n_{1}}{D} - \tfrac{1}{2}\right)\theta_{t}\,|\theta_{t}|^{\alpha - 2},\qquad \theta_{t+1} = \theta_{t} - g_{t}.
$$

Two update steps give

$$
\theta_{2} = \left|\tfrac{n_{1}}{D} - \tfrac{1}{2}\right|^{\alpha}.
$$

The test loss is given by the population loss evaluated at $\theta_{2}$ averaged over test set draws:

$$
\begin{aligned}
L(D) &= \mathbb{E}_{n_{1}}\!\left[\tfrac{1}{2}\theta_{2}^{2}\right] = \frac{1}{2^{D+1}}\sum_{n_{1}=0}^{D}\binom{D}{n_{1}}\left|\tfrac{n_{1}}{D} - \tfrac{1}{2}\right|^{2\alpha}\\
&= \sqrt{\frac{D}{2\pi}}\int_{-\infty}^{\infty} e^{-2D(x - \frac{1}{2})^{2}}\left|x - \tfrac{1}{2}\right|^{2\alpha} + \mathcal{O}(D^{-1})\\
&= \frac{\Gamma(\alpha + \tfrac{1}{2})}{2^{1+\alpha}\sqrt{\pi}}\,D^{-\alpha} + \mathcal{O}(D^{-1}),
\end{aligned}
$$

where we have approximated the binomial distribution at large $D$ with a normal distribution using Stirling's approximation. Thus

$$
L(D) - \lim_{D\to\infty} L(D) = \mathcal{O}(D^{-\alpha}),
$$

which differs from $\mathcal{O}(D^{-1})$ whenever $\alpha \neq 1$. In summary, this example achieves a different scaling exponent through a diverging gradient.
