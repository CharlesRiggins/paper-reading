## Appendix A. Additional Observations on Asymmetric Networks

Appendix A expands on optimization and loss-landscape differences between asymmetric and standard networks.

> **Figure 6.** The paper measures epochs needed to reach $70\%$ training accuracy on `CIFAR-10` while varying $W$-Asymmetric ResNet hyperparameters: number of fixed entries $n_{fix}$ and fixed-entry standard deviation $\kappa$. Settings toward the bottom-right are more asymmetric. More asymmetric networks generally take longer to train.

### Distance in parameter space does not explain interpolation

Asymmetric networks interpolate much better than standard networks, but Euclidean weight-space distance does not reveal this. In the GNN linear-mode-connectivity setup, pairs of standard GNNs have average distance per parameter `.000174`, while $W$-Asymmetric GNNs have `.000159`, only slightly lower. Yet the average test-loss barrier is `1.448` for standard GNNs and only `0.069` for $W$-Asymmetric GNNs.

In the datasets of `10,000` standard and $W$-Asymmetric ResNets, the average distance per parameter between trained standard classifiers is `.0034`, whereas $W$-Asymmetric ResNets are farther apart at `.0051` when estimated over `20,000` pairs. Thus the low-loss linear paths are not simply due to trained asymmetric networks lying closer together in raw parameter space.

### Reduced overfitting

Asymmetric networks often overfit less. In the GNN setup from Section 5.1, standard GNNs reach average maximum training accuracy `84.6%` and validation accuracy `71.6%`. By contrast, $\sigma$-Asym GNNs reach `70.8% / 70.1%` train/validation accuracy, and $W$-Asym GNNs reach `70.7% / 70.06%`. The gap is much smaller.

This effect is weaker in the `10,000`-model ResNet datasets, likely because those runs use substantial regularization: data augmentation, weight decay, and label smoothing. There, standard models reach `74.8% / 73.8%` train/test accuracy, while $W$-Asym models reach `64.0% / 64.0%`.

### Training speed tradeoff

Increasing asymmetry can slow optimization. Figure 6 shows that standard ResNets take about `11` epochs on average to reach `70%` training accuracy on `CIFAR-10`, whereas $W$-Asymmetric ResNets with the most extreme hyperparameters take up to `72` epochs. The tradeoff is therefore not simply “more asymmetry is always better”: stronger symmetry breaking may improve interpolation geometry while making training dynamics slower.
