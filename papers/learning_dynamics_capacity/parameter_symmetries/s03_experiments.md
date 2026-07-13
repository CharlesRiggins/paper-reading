## 5. Experiments

The experiments use asymmetric networks as counterfactual versions of standard architectures. The main question is whether removing parameter symmetries changes phenomena such as linear interpolation, Bayesian inference, metanetwork learning, and optimization trajectories.

### 5.1. Linear Mode Connectivity without Permutation Alignment

**Background.** Linear mode connectivity means that all networks on the line segment between two well-performing trained networks also perform well. Independent training runs usually do not satisfy this property [13, 1]. But if one network is first transformed by a function-preserving permutation that aligns it with the other, low-loss interpolation becomes much more common [13, 1, 79, 14]. Entezari et al. [13] conjectured that if all permutation symmetries are accounted for, linear mode connectivity should generally hold.

**Hypothesis.** Asymmetric networks should be more linearly mode connected than standard networks and should not require post-processing or alignment before merging.

**Setup.** The paper tests MLPs on `MNIST`, ResNets on `CIFAR-10`, and GNNs on `ogbn-arXiv`. It measures the midpoint test-loss barrier:

$$
L\left(\frac{1}{2}\theta_1+\frac{1}{2}\theta_2\right)
-
\frac{1}{2}\left(L(\theta_1)+L(\theta_2)\right).
$$

Lower is better. Standard networks are compared with Git-ReBasin-aligned networks [1], $\sigma$-Asym networks, and $W$-Asym networks. When interpolating two asymmetric networks, the fixed matrix $F$ and mask $M$ are held identical between runs.

**Results.** $\sigma$-Asym lowers barriers relative to standard networks but usually does not match Git-ReBasin. $W$-Asym achieves strong, sometimes near-perfect interpolation and often beats Git-ReBasin alignment. Possible explanations include imperfect Git-ReBasin permutations, additional symmetries beyond layer-wise permutations, or other geometry changes induced by $W$-Asym.

| Architecture / task | Standard | Git-ReBasin | $\sigma$-Asym | $W$-Asym |
|---|---:|---:|---:|---:|
| `MLP` (`MNIST`) | $0.188\pm .12$ | $-.006\pm .00$ | $0.117\pm .01$ | **$-0.012\pm .00$** |
| `ResNet` (`CIFAR-10`) | $3.287\pm .32$ | $2.041\pm .21$ | $2.521\pm .46$ | **$0.934\pm .72$** |
| `ResNet 8x` (`CIFAR-10`) | $2.640\pm .24$ | $0.509\pm .45$ | $1.492\pm .15$ | **$0.031\pm .05$** |
| `GNN` (`ogbn-arXiv`) | $1.475\pm .24$ | $0.269\pm .02$ | $0.901\pm .11$ | **$0.095\pm .03$** |

### 5.2. Bayesian Neural Networks

**Background.** Bayesian deep learning can improve uncertainty quantification and prior integration [30, 51], but parameter symmetries create statistical nonidentifiability. Symmetry-related posterior modes make $p(\theta\mid\mathcal D)$ harder to approximate [2, 34, 72], sample from [50, 71], and analyze [36]. In variational inference, a unimodal Gaussian approximate posterior is poorly matched to a true posterior with many symmetry-induced modes.

**Hypothesis.** Using asymmetric networks as base models should improve Bayesian neural-network training because the posterior has fewer modes.

**Setup.** The paper trains standard and $W$-Asymmetric Bayesian networks for image classification using variational inference with a Gaussian approximate posterior having diagonal plus low-rank covariance [64]. It trains `10` instances per model and reports negative log-likelihood loss, test accuracy, and Expected Calibration Error (**ECE**) [45].

**Results.** $W$-Asymmetric Bayesian networks train faster and often reach better train/test performance. The clearest result is depth-16 MLPs: standard Bayesian MLP-16 fails to train, while $W$-Asym MLP-16 reaches useful accuracy. ResNets show smaller but consistent improvements, especially early in training.

| Dataset | Model | Train loss ↓ | Test loss ↓ | ECE ↓ | Test acc ↑ | Test acc @25 epochs ↑ |
|---|---|---:|---:|---:|---:|---:|
| `CIFAR-10` | `MLP-8` | $1.34\pm .00$ | $1.24\pm .01$ | $.039\pm .009$ | $56.37\pm .31$ | $52.87\pm 0.2$ |
|  | $W$-Asym MLP-8 | **$1.31\pm .01$** | **$1.22\pm .01$** | $.042\pm .009$ | **$57.08\pm .50$** | **$54.15\pm 0.2$** |
|  | `MLP-16` | $2.29\pm .02$ | $2.28\pm .03$ | $.026\pm .017$ | $13.54\pm 2.0$ | $13.34\pm 2.7$ |
|  | $W$-Asym MLP-16 | **$1.39\pm .01$** | **$1.27\pm .01$** | $.045\pm .009$ | **$55.16\pm .44$** | **$51.42\pm 0.3$** |
| `CIFAR-10` | `ResNet20` | **$.596\pm .01$** | $.535\pm .03$ | $.045\pm .007$ | $81.98\pm 1.2$ | $72.37\pm 1.0$ |
|  | $W$-Asym ResNet20 | $.600\pm .02$ | $.535\pm .01$ | $.044\pm .004$ | $81.94\pm 0.6$ | $73.64\pm 1.5$ |
|  | `ResNet110` | $.803\pm .08$ | $.706\pm .08$ | $.052\pm .007$ | $75.71\pm 2.8$ | $59.85\pm 3.9$ |
|  | $W$-Asym ResNet110 | **$.745\pm .07$** | **$.658\pm .06$** | $.049\pm .004$ | $77.40\pm 2.4$ | **$63.20\pm 3.0$** |
| `CIFAR-100` | `ResNet20 (BN)` | $1.68\pm .03$ | $1.57\pm .02$ | $.078\pm .004$ | $56.83\pm .62$ | $46.80\pm 0.9$ |
|  | $W$-Asym ResNet20 (BN) | **$1.62\pm .02$** | **$1.50\pm .03$** | $.076\pm .006$ | **$58.40\pm .62$** | **$49.29\pm 0.4$** |
|  | `ResNet20 (LN)` | $1.97\pm .02$ | $1.88\pm .02$ | $.090\pm .007$ | $50.02\pm .54$ | $37.24\pm 1.1$ |
|  | $W$-Asym ResNet20 (LN) | **$1.91\pm .03$** | **$1.82\pm .02$** | $.086\pm .006$ | **$51.20\pm .47$** | **$39.03\pm 1.0$** |

### 5.3. Metanetworks

**Background.** Metanetworks take the parameters of other networks as inputs [39]. They are also called deep weight-space networks [46, 58], meta-models [35], or neural functionals [81, 82, 83]. Prior work finds that invariance or equivariance to input-network parameter symmetries substantially improves metanetwork performance [46, 81, 39, 32].

**Hypothesis.** Asymmetric networks should be easier for metanetworks to learn from, because the metanetwork does not need to explicitly account for as many symmetries.

**Setup.** The task is predicting the `CIFAR-10` test accuracy of an input image classifier. The paper trains two datasets of `10,000` classifiers: standard small ResNets and $W$-Asym ResNets. It evaluates MLP metanetworks, `DMC` [12], `DeepSets` [77], and `StatNN` [65]. Metrics are $R^2$ and Kendall $\tau$ on the test set.

| Metanetwork | ResNet $R^2$ | ResNet $\tau$ | $W$-Asym ResNet $R^2$ | $W$-Asym ResNet $\tau$ |
|---|---:|---:|---:|---:|
| `MLP` | $.330\pm .04$ | $.389\pm .03$ | **$.594\pm .12$** | **$.864\pm .01$** |
| `DMC` [12] | $.950\pm .01$ | $.787\pm .02$ | **$.967\pm .01$** | **$.911\pm .01$** |
| `DeepSets` [77] | $.855\pm .01$ | $.617\pm .03$ | **$.936\pm .00$** | **$.858\pm .00$** |
| `StatNN` [65] | $.976\pm .00$ | $.866\pm .00$ | **$.978\pm .00$** | **$.935\pm .01$** |

A simple flattened-weight MLP performs poorly on standard ResNets but becomes surprisingly effective on $W$-Asymmetric ResNets. Even symmetry-aware metanetworks such as `DeepSets` and `StatNN` improve on $W$-Asym inputs, suggesting either that remaining non-permutation symmetries matter or that $W$-Asym changes other properties useful for weight-space learning.

### 5.4. Monotonic Linear Interpolation

**Background.** Monotonic linear interpolation (**MLI**) studies the line between initialization $\theta_0$ and trained parameters $\theta_T$. A model satisfies MLI if training loss decreases monotonically along `(1-\alpha)\theta_0+\alpha\theta_T`. Convex objectives trained to completion would have such behavior, but neural networks often show non-monotonicity or long plateaus [21, 17, 41, 68, 70].

**Hypothesis.** If parameter symmetries make loss landscapes less convex-looking, asymmetric networks should have more monotonic and more convex interpolation from initialization to the trained model.

**Setup.** The paper trains `300` standard ResNets and `300` $W$-Asymmetric ResNets with hyperparameters sampled from the same distributions as in the metanetwork classifier dataset. It evaluates `25` uniformly spaced points $0=\alpha_1<\cdots<\alpha_{25}=1$. Monotonicity is measured by

$$
\Delta = \max_i \left(L(\alpha_{i+1})-L(\alpha_i)\right),
$$

and convexity is measured locally by centered second differences and globally by checking whether

$$
L(\alpha_i)\leq (1-\alpha_i)L(0)+\alpha_i L(1).
$$

**Results.** Every one of the `300` $W$-Asymmetric ResNets satisfies monotonic linear interpolation and lies below the endpoint line segment. $\sigma$-Asym is also significantly more convex and monotonic than standard ResNets, though some trajectories remain nonmonotonic or nonconvex.

| Model | $\Delta$ ↓ | Percent monotonic ↑ | Local convexity ↑ | Global convexity ↑ |
|---|---:|---:|---:|---:|
| Standard ResNet | $.079\pm .109$ | $26.3\%$ | $.548\pm .139$ | $.823\pm .229$ |
| $\sigma$-Asym ResNet | $.004\pm .047$ | $87.3\%$ | $.675\pm .143$ | $.976\pm .098$ |
| $W$-Asym ResNet | **$-.027\pm .026$** | **$100\%$** | **$.769\pm .165$** | **$1.00\pm .000$** |

### 5.5. Other Optimization and Loss-Landscape Properties

Appendix A reports three additional observations:

1. Even though asymmetric networks interpolate much better, this cannot be explained simply by Euclidean parameter distance. $W$-Asym models can be farther apart in weight space yet connected by low-loss linear paths.
2. Asymmetric networks often overfit less: train/test or train/validation gaps can be smaller than for standard networks.
3. More asymmetric settings can slow training, especially when increasing both the number of fixed entries $n_{fix}$ and fixed-value scale $\kappa$.
