## Appendix F. Experimental Details

Appendix F gives training protocols, hyperparameters, datasets, and compute details for the main experiments.

### F.1. Linear Mode Connectivity Experimental Details

#### F.1.1. Image Classifier Interpolation

The image-classification interpolation experiments use two model families.

1. **ResNet.** The paper trains `ResNet20` models with `LayerNorm` and width `64` or `8 \times 64`. Batch size is `128`. The learning rate warms up from `.0001` to `.01` over `20` epochs. Width-`8x` models train for `50` epochs, while width-`1x` models train for `100` epochs. $\sigma$-Asymmetric ResNets use a lower warmup target learning rate `.001` because of training instability.
2. **MLP.** The paper trains 4-layer MLPs with `LayerNorm` and width `512`. On `MNIST`, hyperparameters for both asymmetric and standard models are tuned to minimize loss barrier. Batch size is `64`.

`MNIST` uses no data augmentation. `CIFAR-10` uses random cropping and horizontal flipping. Git-ReBasin tests use the weight-matching algorithm of Ainsworth et al. [1].

**$W$-Asymmetric MLP hyperparameters.**

| Layer | $n_{fix}$ | $\kappa$ |
|---|---:|---:|
| `Linear-1` | `64` | `1` |
| `Linear-2` | `64` | `1` |
| `Linear-3` | `64` | $1/2$ |
| `Linear-4` | `256` | $1/4$ |

**$W$-Asymmetric `ResNet20` width-1 hyperparameters.**

| Block | $n_{fix}$ | $\kappa$ |
|---|---:|---:|
| First Conv | `12` | `2` |
| Block 1 - Conv | `36` | `2` |
| Block 1 - Skip | `4` | `2` |
| Block 2 - Conv | `54` | `2` |
| Block 2 - Skip | `6` | `2` |
| Block 3 - Conv | `72` | `2` |
| Block 3 - Skip | `8` | `2` |
| Linear | `8` | `2` |

**$W$-Asymmetric `ResNet20` width-8 hyperparameters.** These use `3x` as many fixed entries per output channel / neuron as the width-1 setup.

| Block | $n_{fix}$ | $\kappa$ |
|---|---:|---:|
| First Conv | `27` | `2` |
| Block 1 - Conv | `108` | `2` |
| Block 1 - Skip | `12` | `2` |
| Block 2 - Conv | `162` | `2` |
| Block 2 - Skip | `18` | `2` |
| Block 3 - Conv | `216` | `2` |
| Block 3 - Skip | `24` | `2` |
| Linear | `24` | `2` |

#### F.1.2. Graph Neural Network Interpolation

The GNN experiments use a GNN similar to `GIN` [73] with mean aggregation. The base GNN has three message-passing layers, hidden dimension `256`, and `176,424` trainable parameters. The dataset is `ogbn-arXiv` [27], a citation network with `169,343` nodes and `1,166,243` edges; the task is transductive node classification by primary subject area.

Training is full-batch over the whole graph, so randomness comes from initialization rather than minibatch sampling. The optimizer is `Adam` [31] with peak learning rate `.001`, linearly warmed up for `25` epochs and then held constant. Each model trains for `500` epochs.

Git-ReBasin alignment uses activation matching. The $\sigma$-Asymmetric GNN uses FiGLU with each fixed matrix $F$ initialized with standard deviation `$.01/\sqrt d`, where $d$ is the hidden-channel count. The $W$-Asymmetric GNN fixes `6` constants in each row of each linear map, with constants sampled from a normal distribution of standard deviation `.5`.

### F.2. Bayesian Neural Network Experimental Details

Bayesian networks are trained with the variational inference method of Tomczak et al. [64], fitting an approximate posterior with diagonal plus rank-4 covariance.

For $W$-Asym ResNet tests, `ResNet20` uses the same asymmetric hyperparameters as the main `ResNet20` table but with $\kappa=.5$. `CIFAR-100` experiments use a standard linear layer for the final classifier and a ResNet width multiplier of `2`. ResNet experiments use learning rate `.001`, batch size `250`, and train for `50` epochs.

MLP Bayesian experiments use $\kappa=.5$, `8` hardwired entries per neuron, learning rate `.0005`, batch size `250`, and `50` epochs. `CIFAR-10` / `CIFAR-100` use horizontal flips and random crops; `MNIST` uses no data augmentation. All runs use `Adam`.

### F.3. Metanetwork Experimental Details

#### F.3.1. Dataset Details

The authors train two `CIFAR-10` image-classifier datasets: `10,000` standard small ResNet-like CNNs and `10,000` comparable $W$-Asymmetric networks. They use `FFCV` [37], specifically the `CIFAR-10` sample script with random horizontal flips, random translations, `Cutout` [10], label smoothing [62], and linear learning-rate warmup / decay. Training all `20,000` classifiers takes just under `400` GPU-hours, about two GPU-weeks on `NVIDIA RTX 2080 Ti` GPUs.

Each standard ResNet has `78,042` trainable parameters; each $W$-Asym ResNet has `60,634`. Both share the same architecture except for fixed filters in $W$-Asym. They have `8` convolution layers, `LayerNorm`, and a final fully connected classifier after spatial average pooling.

Hyperparameter distributions sampled for these classifier datasets:

| Hyperparameter | Distribution |
|---|---|
| Learning rate | $.5\cdot 10^{-\mathrm{Unif}(0,2)}$ |
| Weight decay | $10^{-\mathrm{Unif}(1,5)}$ |
| Label smoothing | $\mathrm{Unif}(0,.2)$ |
| Epochs | $\mathrm{RandInt}(10,40)$ |

#### F.3.2. Metanetwork Details

The paper trains MLP, `DMC`, `DeepSets`, and `StatNN` metanetworks for `50` epochs using `AdamW` [40]. For each metanetwork / dataset pair, it selects the learning rate from `{1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2}` that gives the best validation $R^2$ on one run. It then trains each metanetwork type `5` times and reports mean and standard deviation.

| Metanetwork | ResNet LR | ResNet # params | $W$-Asym LR | $W$-Asym # params |
|---|---:|---:|---:|---:|
| `MLP` | $10^{-4}$ | `4,994,945` | $10^{-4}$ | `3,880,833` |
| `DMC` [12] | $10^{-3}$ | `105,357` | $5\cdot10^{-3}$ | `105,357` |
| `DeepSets` [77] | $10^{-2}$ | `8,897` | $5\cdot10^{-3}$ | `8,897` |
| `StatNN` [65] | $10^{-3}$ | `119,297` | $10^{-2}$ | `119,297` |

### F.4. Monotonic Linear Interpolation Experimental Details

MLI experiments use the same `CIFAR-10` classifier setup as the metanetwork data. For each architecture, the authors sample `300` hyperparameter settings from the distributions above and train one model for each setting. When evaluating training loss, label smoothing is included.

For $\sigma$-Asymmetric networks, FiGLU $F$ is initialized with standard deviation $1/\sqrt d$, where $d$ is channel count. This is much larger than the `$.01/\sqrt d` used in GNN experiments and trains better. `24` of the `300` $\sigma$-Asym models diverged with `NaN`s and are excluded from Table 4. Divergence appears to occur at high learning rates, especially greater than `.1`. No standard or $W$-Asymmetric networks diverged.

### F.5. Miscellaneous Experimental Details

Datasets: `MNIST` [38], `CIFAR-10` [33], `CIFAR-100` [33], and `ogbn-ArXiv` [27]. `MNIST`, `CIFAR-10`, and `CIFAR-100` appear not to have explicit licenses but are widely used; `ogbn-ArXiv` comes from the Open Graph Benchmark, whose GitHub repository uses an `MIT` license.

Software: `PyTorch` [52], `FFCV` [37], and `PyTorch Geometric` [15]. Compute uses several `NVIDIA` GPU types and systems, including `2080 Ti`, `3090 Ti`, `4090 Ti`, and `V100`; each training run uses at most one GPU.
