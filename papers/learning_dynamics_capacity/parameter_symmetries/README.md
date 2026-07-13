# The Empirical Impact of Neural Parameter Symmetries, or Lack Thereof

arXiv: 2405.20231 | Derek Lim et al. (MIT CSAIL / UC Berkeley / Northeastern University / Technion, NVIDIA / TU Munich, MIT, 2024)
Code: https://github.com/cptq/asymmetric-networks

## Files

- `s01_introduction_background_related.md` — Introduces **parameter-space symmetries** as function-preserving transformations of neural-network weights and motivates asymmetric networks as a counterfactual system for studying their effects. Defines parameter symmetries through the two-layer MLP permutation example and explains why elementwise nonlinearities automatically induce permutation symmetries. Covers related work on characterizing symmetries, constraining or post-processing weights, and aligning trained networks for model merging.

- `s02_asymmetric_networks.md` — Presents the paper's two symmetry-reduction mechanisms: $W$-Asymmetric layers, which fix selected weights according to masks with distinct row patterns, and $\sigma$-Asymmetric FiGLU nonlinearities, which mix coordinates through a fixed Gaussian matrix. States the key guarantees: $W$-Asymmetric masks remove neural DAG automorphisms under conditions, and FiGLU almost surely has no permutation or diagonal equivariances. Also covers extension to CNNs, GNNs, Transformers, and the informal universal-approximation theorem for $W$-Asymmetric MLPs.

- `s03_experiments.md` — Transcribes the main empirical results. $W$-Asymmetric networks show much lower midpoint interpolation barriers than standard networks and often outperform Git-ReBasin alignment, e.g. `ResNet 8x` on `CIFAR-10` drops from $2.640\pm .24$ to **$0.031\pm .05$**. They also improve Bayesian neural-network training, make metanetwork prediction easier (`MLP` metanetwork Kendall $\tau$ rises from `$.389` to **`$.864`**), and make monotonic interpolation from initialization nearly convex-like (`100%` monotonic for $W$-Asym ResNets).

- `s04_discussion.md` — Discusses what the results imply and what remains unresolved. The paper argues that asymmetric networks are useful tools for isolating how symmetries affect loss geometry, Bayesian posteriors, and weight-space learning, but it cautions that fixed weights and non-elementwise activations may introduce effects beyond symmetry removal. It also highlights the empirical weakness of $\sigma$-Asymmetry relative to $W$-Asymmetry despite several attempted FiGLU variants.

- `s05_references.md` — Full 84-entry reference list, spanning mode connectivity, Git-ReBasin and model merging, Bayesian neural-network posterior geometry, metanetworks / neural functionals, graph neural networks, universal approximation, and optimization / generalization analyses tied to parameter symmetries.

- `s06_appendix_a_observations.md` — Records additional observations that are not central in the main experiments. Better interpolation is not explained by shorter Euclidean weight distances: $W$-Asym ResNets can be farther apart in parameter space yet connected by low-loss linear paths. Asymmetric networks can overfit less, but stronger asymmetry also slows training, with extreme $W$-Asym ResNets taking up to `72` epochs to reach `70%` `CIFAR-10` training accuracy versus `11` for standard ResNets.

- `s07_appendix_b_theory_proofs.md` — Gives the proof structure for the theoretical results. The computation-graph proof shows mask row uniqueness forces every layer-wise permutation block to be identity. The FiGLU proof reduces permutation and diagonal equivariances to almost-sure properties of the fixed Gaussian matrix, while the universal-approximation proof constructs standard linear maps using pairs of asymmetric layers with error-correcting replicated rows and a high-probability $n_{fix}\in o(n^{1/4})$ condition.

- `s08_appendix_c_d_limits_impacts.md` — Summarizes limitations and broader impacts. The main caveat is causal interpretation: fixed entries $F$, especially with large $\kappa$, may affect optimization and landscapes beyond merely removing symmetries. The broader-impact section frames the work as foundational, with potential benefits for optimization, model merging, uncertainty quantification, and weight-space learning, while noting that robustness and distribution-shift behavior remain unstudied.

- `s09_appendix_e_ablations.md` — Covers ablations on learnable parameter counts, learning-rate warmup, and fixed biases. Matching the standard ResNet parameter count does not reproduce $W$-Asym metanetwork or Bayesian improvements. Warmup length changes do not alter the qualitative interpolation conclusion, and simply fixing biases fails badly, producing barriers such as $5.81\pm3.67$ at $k=1$, far worse than either asymmetric method.

- `s10_appendix_f_experimental_details.md` — Records implementation details for image classifiers, GNNs, Bayesian networks, metanetwork datasets, and monotonic interpolation. Includes $W$-Asym hyperparameter tables for MLPs and ResNets, GNN settings on `ogbn-ArXiv`, variational-inference details, the `10,000 + 10,000` classifier datasets trained with `FFCV`, and compute / software information (`PyTorch`, `FFCV`, `PyTorch Geometric`).

- `s11_appendix_g_changelog.md` — Notes the arXiv version history: Version 1 was the initial release on May 30, 2024.
