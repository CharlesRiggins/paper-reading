## Appendix A. Solving for $\Lambda$ Algebraically

ROME’s update is the classical solution to least squares with an equality constraint. Assume $W$ is the least-squares matrix mapping existing keys $K$ to values $V$:

$$
W=\underset{\tilde W}{\operatorname{argmin}}\ \|\tilde WK-V\|_F^2. \tag{6}
$$

The normal equation is

$$
WKK^T=VK^T. \tag{7}
$$

ROME seeks a new matrix $\hat W$ that minimizes the same old-memory error while satisfying

$$
\hat Wk_*=v_*. \tag{8}
$$

Introduce Lagrange multiplier $\Lambda\in\mathbb R^H$:

$$
\mathcal J(\hat W,\Lambda)
=\frac12\|\hat WK-V\|_F^2
-\Lambda^T(\hat Wk_*-v_*). \tag{9}
$$

Setting the derivative with respect to $\hat W$ to zero gives

$$
\hat WKK^T-VK^T-\Lambda k_*^T=0, \tag{10}
$$

or

$$
\hat WKK^T=VK^T+\Lambda k_*^T. \tag{11}
$$

Subtracting the old normal equation yields

$$
(\hat W-W)KK^T=\Lambda k_*^T. \tag{12}
$$

Define the old-key second moment

$$
C=KK^T.
$$

Assuming $C$ is nondegenerate and using its symmetry:

$$
\hat W=W+\Lambda(C^{-1}k_*)^T. \tag{13}
$$

Let $u=C^{-1}k_*$. Applying the new-key constraint:

$$
\begin{aligned}
\hat Wk_*
&=(W+\Lambda u^T)k_*\\
&=Wk_*+\Lambda(u^Tk_*)\\
&=v_*.
\end{aligned}\tag{14}
$$

Therefore

$$
\Lambda=
\frac{v_*-Wk_*}{u^Tk_*}
=
\frac{v_*-Wk_*}
{(C^{-1}k_*)^Tk_*}. \tag{15}
$$

Substitution produces the complete ROME update:

$$
\boxed{
\hat W
=W+
\frac{v_*-Wk_*}
{(C^{-1}k_*)^Tk_*}
(C^{-1}k_*)^T
}. \tag{16}
$$

The update is rank one because it is the outer product of a value-space residual and a key-space direction. The inverse second moment $C^{-1}$ discounts directions common among existing keys, reducing interference with mappings that $W$ already stores. The denominator scales the update so that $\hat Wk_*=v_*$ holds exactly.
