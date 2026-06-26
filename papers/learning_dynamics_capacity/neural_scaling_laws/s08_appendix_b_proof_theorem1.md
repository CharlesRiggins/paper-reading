## Appendix B. Proof of Theorem 1

We now prove Theorem 1, repeated below for convenience.

> **Theorem 1.** Let $f_{T}$ denote the trained network and $\ell$ the per-sample loss. If the trained network concentrates so that $\mathbb{E}\!\left[(f_{T} - \mathbb{E}[f_{T}])^{k}\right] \leq C_{k}$ with appropriate moment scaling, and the loss $\ell$ is either (i) a finite-degree polynomial in $f_{T}$, (ii) has bounded second derivative, or (iii) is 2-Hölder, then
>
> $$\mathbb{E}\!\left[\ell(f_{T})\right] - \ell\!\left(\mathbb{E}\!\left[f_{T}\right]\right) = \mathcal{O}\!\left(\mathrm{Var}(f_{T})\right).$$

**Case 1 — finite degree polynomial:** In this case, we can write,

$$
\ell(f_{T}) - \ell\!\left(\mathbb{E}\!\left[f_{T}\right]\right) = \sum_{k=1}^{K} \frac{\ell^{(k)}\!\left(\mathbb{E}\!\left[f_{T}\right]\right)}{k!}\left(f_{T} - \mathbb{E}\!\left[f_{T}\right]\right)^{k},
$$

where $K$ is the polynomial degree and $\ell^{(k)}$ is the $k$-th derivative of $\ell$. Taking the expectation of (S1) and using the moment scaling proves the result.

**Case 2 — bounded second derivative:** The quadratic mean value theorem states that for any $f_{T}$, there exists a $c$ such that,

$$
\ell(f_{T}) - \ell\!\left(\mathbb{E}\!\left[f_{T}\right]\right) = \left(f_{T} - \mathbb{E}\!\left[f_{T}\right]\right)\ell^{\prime}\!\left(\mathbb{E}\!\left[f_{T}\right]\right) + \frac{1}{2}\ell^{\prime\prime}(c)\left(f_{T} - \mathbb{E}\!\left[f_{T}\right]\right)^{2}.
$$

Taking the expectation of (S2) and using the fact that $\ell^{\prime\prime}(c)$ is bounded yields the desired result.

**Case 3 — 2-Hölder:** Lastly, the loss being 2-Hölder means we may write,

$$
\ell(f_{T}) - \ell\!\left(\mathbb{E}\!\left[f_{T}\right]\right) \leq \left|\ell(f_{T}) - \ell\!\left(\mathbb{E}\!\left[f_{T}\right]\right)\right| \leq K_{\ell}\left(f_{T} - \mathbb{E}\!\left[f_{T}\right]\right)^{2}.
$$

Again, taking the expectation of this inequality completes the proof. $\blacksquare$

### Remarks on loss variance

The above shows that the bias of the loss, $\mathbb{E}[\ell(f_{T})] - \ell(\mathbb{E}[f_{T}])$, is controlled by the variance of $f_{T}$. One can also bound the variance of the loss itself, $\mathbb{E}[\ell(f_{T})] - \mathbb{E}[\ell(f_{T})]^{2}$, in terms of the moments of $f_{T} - \mathbb{E}[f_{T}]$. For a loss with bounded first derivative, a first-order expansion gives

$$
\mathbb{E}\!\left[\left|\ell\!\left(f_{T}\right) - \ell\!\left(\mathbb{E}\!\left[f_{T}\right]\right)\right|\right] \leq \left|\ell^{\prime}\!\left(\mathbb{E}\!\left[f_{T}\right]\right)\right|\,\mathbb{E}\!\left[\left|f_{T} - \mathbb{E}\!\left[f_{T}\right]\right|\right] + \mathcal{O}(\epsilon),
$$

so that the loss fluctuations are likewise controlled by $\mathbb{E}[|f_{T} - \mathbb{E}[f_{T}]|]$, which scales with the same power of $D$ (or $P$) as the variance.
