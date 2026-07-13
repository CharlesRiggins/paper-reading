## Appendix B. Proofs of Theoretical Results

Appendix B supplies proofs for three claims: $W$-Asymmetric masks remove neural DAG automorphisms in MLPs; FiGLU removes permutation and diagonal equivariances with probability one; and sufficiently wide $W$-Asymmetric MLPs remain universal approximators.

### B.1. Graph-based approach

The theorem states that if each mask matrix $M$ has unique nonzero rows, then $W$-Asymmetric MLPs with fixed entries set to zero ($\kappa=0$) have no nontrivial neural DAG automorphisms. In practice, the paper uses fixed Gaussian entries with positive, often large, standard deviation $\kappa$; the proof for $\kappa=0$ fits directly into Lim et al.'s computation-graph framework [39]. Extending the proof to $\kappa>0$ would require redefining the computation graph to include untrainable edges and is left to future work.

For an $L$-layer masked MLP, the forward pass is

$$
[W_L\odot M_L] \sigma(\cdots \sigma([W_1\odot M_1]x)\cdots).
$$

The computation graph has layers $V_0,\ldots,V_L$, where $V_0$ contains inputs and $V_L$ outputs. Its adjacency matrix has the block form

$$
A=
\begin{bmatrix}
0 & \\
M_1 & 0 \\
& M_2 & \\
& & \ddots & 0 \\
& & & M_L & 0
\end{bmatrix}.
$$

A neural DAG automorphism is a node relabeling $\tau:V\to V$ that is bijective, preserves edges, and fixes every input and output node. Let $P$ be the permutation matrix associated with $\tau$. A lemma shows that neural DAG automorphisms preserve layer number when the masks have nonzero rows. Therefore $P$ is block diagonal:

$$
P=
\begin{bmatrix}
P_0 \\
& P_1 \\
& & \ddots \\
& & & P_L
\end{bmatrix},
$$

with $P_0=I$ and $P_L=I$ because inputs and outputs are fixed. From $PAP^\top=A$, the first mask block gives $P_1M_1P_0^\top=M_1$, hence $P_1M_1=M_1$. Since $M_1$ has unique rows, $P_1=I$. Inductively, if $P_i=I$, then $P_{i+1}M_{i+1}=M_{i+1}$; unique rows imply $P_{i+1}=I$. Thus every block is identity, so the automorphism is trivial.

The layer-preservation lemma uses induction: input nodes are fixed, and if all nodes in layer $l$ map to layer $l$, then any node in layer $l+1$ has an incoming edge from layer $l$ because the next mask has nonzero rows. Edge preservation forces its image to be in layer $l+1$.

### B.2. Symmetry breaking via nonlinearities

For two-layer MLPs with square invertible weights, the proposition says that if $\sigma$ has no linear equivariances, then there are no nontrivial parameter symmetries. Let $\theta_1=(W_2,W_1)$ and $\theta_2=(\widetilde W_2,\widetilde W_1)$. If the represented functions are equal, then for all $z$,

$$
W_2\sigma(W_1z)=\widetilde W_2\sigma(\widetilde W_1z),
$$

so

$$
\widetilde W_2^{-1}W_2\sigma(W_1z)=\sigma(\widetilde W_1z).
$$

Set $z=W_1^{-1}x$; then

$$
\widetilde W_2^{-1}W_2\circ\sigma=\sigma\circ\widetilde W_1W_1^{-1}.
$$

This is a linear equivariance of $\sigma$. If no such equivariances exist, both matrices are identity, so $\theta_1=\theta_2$.

#### B.2.1. FiGLU nonlinearity proofs

FiGLU is

$$
\sigma(x)=\eta(Fx)\odot x,
$$

where $\eta$ is sigmoid. The goal is to show that with probability one over Gaussian $F$, FiGLU has no permutation or diagonal equivariances.

The proof reduces to two deterministic properties of $F$: no permutation symmetries and no zero entries. A Gaussian matrix has distinct entries and no zeros with probability one, so these properties hold almost surely.

**No permutation equivariances.** Suppose $\sigma\circ P_1=P_2\circ\sigma$ for permutation matrices $P_1,P_2$. Then for all $x$,

$$
\eta(P_2^\top F P_1x)\odot P_2^\top P_1x = \eta(Fx)\odot x.
$$

Evaluating at basis vectors shows $P_2^\top P_1=I$; otherwise the nonzero coordinate would move while sigmoid outputs are nonzero. For vectors with all nonzero coordinates, coordinatewise division gives $\eta(P_2^\top F P_1x)=\eta(Fx)$. Since sigmoid is bijective, $P_2^\top F P_1=F$. If $F$ has no permutation symmetries, $P_1=P_2=I$.

**No diagonal equivariances.** Suppose $\sigma\circ A=B\circ\sigma$ for invertible diagonal matrices $A=\mathrm{Diag}(\alpha)$ and $B=\mathrm{Diag}(\beta)$. Evaluating at $x=ce_i$ yields

$$
\frac{\alpha_i}{\beta_i}
=
\frac{\eta(cFe_i)_i}{\eta(\alpha_i cFe_i)_i}.
$$

The right-hand side must be constant in $c$. Because $F$ has no zero entries, this forces $\alpha_i>0$; taking $c\to\infty$ gives $\alpha_i/\beta_i=1$. Substituting back and using invertibility of sigmoid gives $\alpha_i=1$, hence $\beta_i=1$. Therefore $A=B=I$.

### B.3. Proofs for Universal Approximation

The theorem assumes a nonpolynomial elementwise nonlinearity $\eta$ satisfying $\eta(x)-\eta(-x)=x$, such as `ReLU`, `GELU`, or `Swish`. For a compact domain $\Omega\subseteq\mathbb R^D$, continuous target $f_{target}:\Omega\to\mathbb R$, and any $\varepsilon,\delta>0$, there is a large enough width $n'$ such that for all $n>n'$, a random 4-layer $W$-Asymmetric MLP with hidden dimensions $24n\to n\to 24n$ and $n_{fix}\in o(n^{1/4})$ hardwired entries per neuron can approximate $f_{target}$ within $\varepsilon$ with probability at least $1-\delta$.

The proof strategy is constructive:

1. Use the classical universal approximation theorem to obtain a standard 2-layer MLP approximating the target within $\varepsilon$.
2. Show that each linear map in that standard MLP can be exactly represented by two $W$-Asymmetric layers.
3. Concatenate the exact representations of the two linear maps to obtain a 4-layer asymmetric MLP.

#### Fitting a linear map with a standard two-layer MLP

For a target linear map $W\in\mathbb R^{n\times n}$, define $B\in\mathbb R^{2n\times n}$ by stacking each row $W_i$ and its negation, and define $A=I_n\otimes[1,-1]$. Since $\eta(x)-\eta(-x)=x$,

$$
A\eta(Bx)=Wx.
$$

This is the base construction.

#### One asymmetric and one standard linear map

When the output map $A'$ has random fixed entries, the construction replicates each needed feature three times. As long as the mask never fixes all three copies in a row, the remaining trainable copy can offset the fixed constants. The probability that a row has three consecutive hardwired entries is $O(n_{fix}^3/n^2)$; after a union bound over rows, the failure probability is $O(n_{fix}^3/n)$, which vanishes if $n_{fix}\in o(n^{1/3})$.

#### Two asymmetric linear maps

When both $A'$ and $B'$ are asymmetric, the proof uses blocks of `24` rows per target row. The rows are divided into subblocks so that fixed-entry errors in $B'$ can be canceled by paired positive / negative activations. The construction ensures pairs such as $B'_{24i}=-B'_{24i+3}$ and $B'_{24i+6}=-B'_{24i+9}$, so

$$
\eta(B'_{24i}\cdot x)-\eta(B'_{24i+3}\cdot x)
-
\left[\eta(B'_{24i+6}\cdot x)-\eta(B'_{24i+9}\cdot x)\right]
= W_i\cdot x.
$$

The high-probability mask condition requires that each `24`-row block has at most one pair of intersecting rows. The union-bound failure probability is at most $C n_{fix}^4/n$, so it vanishes for $n_{fix}\in o(n^{1/4})$.

#### Conclusion and caveat

The construction shows that a $W$-Asymmetric MLP of hidden dimension $24n$ can exactly implement any $n\times n$ linear map with high probability. Combined with classical universal approximation and input padding, this gives the stated theorem.

The proof also reveals a caveat: because the construction maps 2-layer standard MLPs into 4-layer $W$-Asymmetric MLPs, these 4-layer asymmetric networks inherit at least as many symmetries as the 2-layer standard MLPs used in the construction. The authors suggest that using a nonlinearity with $\eta(x)-\eta(-x)\neq x$ may avoid this issue.
