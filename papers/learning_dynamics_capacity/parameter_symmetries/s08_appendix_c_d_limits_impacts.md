## Appendix C. Limitations

The paper's asymmetric architectures are motivated by removing parameter-space symmetries, but their empirical behavior may also be caused by other architectural changes. For $W$-Asymmetric networks, the fixed entries $F$ are often much larger than standard initialization values, which can affect optimization and loss geometry independently of symmetry breaking. This is a key interpretive limitation: the asymmetric networks are useful counterfactual systems, but not perfectly controlled interventions on symmetry alone.

The theoretical results are also incomplete. For $\sigma$-Asymmetric networks, the no-symmetry proposition only covers the two-layer case with square invertible weights. For $W$-Asymmetric networks, the universal-approximation result could be tightened in both width and depth, and the computation-graph proof is formally for fixed entries set to zero rather than the positive-$\kappa$ Gaussian fixed entries used in experiments.

## Appendix D. Broader Impacts

The work is foundational rather than application-specific. Its potential positive impact is better understanding of neural-network loss landscapes and Bayesian neural networks. If parameter symmetries are a major source of apparent nonconvexity or posterior multimodality, then architectures or algorithms that account for them could improve optimization, model merging, uncertainty quantification, and weight-space learning.

However, asymmetric networks are much less studied than standard neural networks. Important properties such as generalization, robustness to distribution shift, and adversarial robustness have not been extensively evaluated for them. The interaction between parameter symmetries and these properties remains unclear, so the paper frames asymmetric networks primarily as a research tool and not as a deployable replacement for standard architectures.
