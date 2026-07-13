## Abstract

Many algorithms and observed phenomena in deep learning appear to be affected by **parameter symmetries**: transformations of neural-network parameters that do not change the represented function. Examples include linear mode connectivity, model merging, Bayesian neural-network inference, metanetworks, and several optimization / loss-landscape phenomena. Because it is difficult to analyze the relationship between these symmetries and the phenomena directly, the paper takes an empirical counterfactual route: build neural architectures with reduced parameter-space symmetries, then compare their behavior with standard networks.

The paper introduces two architecture modifications with partial theoretical guarantees: $W$-Asymmetric networks, which fix selected weights so the computation graph has fewer automorphisms, and $\sigma$-Asymmetric networks, which replace elementwise nonlinearities with a non-elementwise **FiGLU** gate. Across MLPs, ResNets, and GNNs, these asymmetric networks reveal strong effects: $W$-Asymmetric networks exhibit linear mode connectivity without permutation alignment, train faster and more effectively as Bayesian neural networks, and are easier inputs for metanetworks. Code: `https://github.com/cptq/asymmetric-networks`.

## 1. Introduction

Neural networks have many behaviors that are empirically robust but theoretically hard to understand. A central structural property is that neural-network parameterizations usually contain many **parameter-space symmetries**: for a parameter vector $\theta$, there may be many other parameter choices representing exactly the same input-output function. In an MLP, for example, permuting hidden neurons and applying the corresponding row/column permutations to adjacent weight matrices leaves the function unchanged [24]. Such redundancies are not necessarily harmful, but they make the parameter space highly non-Euclidean.

Parameter symmetries appear in several major phenomena. When linearly interpolating between two independently trained networks with the same architecture, intermediate networks often perform poorly [59, 13]. But if one network is first aligned to the other by a permutation symmetry, the linear path can have low loss [59, 1]. This suggests that part of the apparent nonconvexity of neural loss landscapes may come from representing the same function in many symmetric locations. Similar symmetry effects arise in neuron interpretability [20], optimization [48, 84, 80], model merging [60], learned equivariance [7], Bayesian deep learning [34], loss-landscape geometry [54], metanetworks over weights [39], and generalization measures [49, 11].

> **Figure 1.** A standard MLP has freely permutable hidden nodes. A $W$-Asymmetric MLP fixes certain weights to constant untrainable values. A $\sigma$-Asymmetric MLP uses FiGLU with a fixed matrix $F$ so the nonlinearity itself no longer acts elementwise.

To study the effect of removing parameter symmetries, the authors introduce two families:

1. **$W$-Asymmetric networks** fix selected entries of each linear map to constants, breaking computation-graph symmetries.
2. **$\sigma$-Asymmetric networks** use **FiGLU**, a new non-elementwise nonlinearity that does not induce the usual permutation symmetries.

These designs are inspired by the fact that computation-graph automorphisms [39] and nonlinearity equivariances [20] induce symmetries in standard networks. The asymmetric networks remain close to standard networks structurally and can be trained with ordinary backpropagation and first-order optimizers such as `Adam`, making them useful counterfactual systems rather than entirely different optimization problems.

The experimental suite spans MLPs, ResNets, and graph neural networks. The main empirical picture is consistent: removing symmetries often makes interpolation behavior closer to convex optimization. $W$-Asymmetric networks in particular show strong linear mode connectivity without post-hoc alignment, faster and better Bayesian neural-network training, and improved metanetwork prediction of test accuracy from weights.

## 2. Background and Definitions

Let $\Theta$ be the parameter space of a fixed neural architecture. For parameters $\theta \in \Theta$, the network represents a function $f_\theta : \mathcal X \to \mathcal Y$. A map $\phi : \Theta \to \Theta$ is a **parameter-space symmetry** if

$$
f_\theta(x) = f_{\phi(\theta)}(x)
$$

for all inputs $x$ and all parameters $\theta$.

For a two-layer MLP without biases, let $\theta = (W_2, W_1)$ and

$$
f_\theta(x) = W_2 \sigma(W_1 x),
$$

where $\sigma$ acts elementwise. If $P$ is a permutation matrix and $\phi(\theta) = (W_2 P^\top, P W_1)$, then

$$
f_{\phi(\theta)}(x)
= W_2 P^\top \sigma(P W_1 x)
= W_2 P^\top P \sigma(W_1 x)
= W_2 \sigma(W_1 x)
= f_\theta(x).
$$

The key equality is $P\sigma(x)=\sigma(Px)$, which holds for any elementwise nonlinearity. Other equivariances of the nonlinearity also induce parameter symmetries. For `ReLU`, positive homogeneity gives $\alpha\sigma(x)=\sigma(\alpha x)$ for $\alpha>0$, inducing scale symmetries [49, 11, 20].

## 3. Related Work

**Characterizing parameter-space symmetries.** Earlier work identified specific symmetries in neural networks [24, 61]. More systematic approaches include Godfrey et al. [20], who characterize global linear symmetries induced by nonlinearities in two-layer MLPs with pointwise activations; Zhao et al. [78], who derive nonlinear, data-dependent symmetries; and Lim et al. [39], who show that graph automorphisms of a network computation graph induce permutation symmetries, covering hidden-neuron permutations in MLPs and hidden-channel permutations in CNNs.

**Constraints and post-processing to break symmetries.** Several works remove symmetry ambiguities by constraining or post-processing trained weights. These include methods for scaling symmetries from normalization layers or positively homogeneous activations [6, 55, 54, 36], permutation symmetries [55, 54, 71, 36], and sign symmetries from odd activations [71]. The present paper differs by modifying the architecture itself while keeping optimization unconstrained. It does not require manifold optimization, projected gradient descent, geodesic interpolation, or post-training canonicalization; this is important because those procedures change the geometry being studied.

**Aligning multiple networks.** For model merging and interpolation, another strategy is to align one network to another by choosing a symmetry transformation. Prior work proposes heuristics or learned methods for selecting permutations [4, 69, 63, 13, 1, 53, 47, 67], and other methods relax exact permutation constraints or add post-processing for better merging [59, 28, 29, 60, 56]. Because the asymmetric networks remove many parameter symmetries directly, they can often be interpolated or merged without an alignment step.
