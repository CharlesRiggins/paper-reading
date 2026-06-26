## Appendix A. Experimental setup

### A.1 Figure 1 (top-left)

Experiments use architectures and training following [66, 13, 67].

### A.2 Figure 1 (top-right)

All experiments were performed using a Flax [68] implementation of Wide ResNet 28-10 [60]. Models were trained for 78,125 total steps with a cosine learning rate decay [69] and an augmentation policy consisting of random flips and crops. We report final loss, though we found no qualitative difference between using final loss, best loss, final accuracy or best accuracy (see Figure S1).

### A.3 Figure 1 (bottom-left)

The setup was identical to Figure 1 (top-right) except that the model considered was a depth-10 residual network with varying width.

### A.4 Figure 1 (bottom-right)

Experiments are done using Neural Tangents [63]. All experiments use 100 training samples and two hidden layer fully-connected networks of varying width (ranging from $w = 64$ to $w = 11{,}585$) with ReLU nonlinearities unless specified as Erf. Full-batch gradient descent and cross-entropy loss were used unless specified as MSE, and the figure shows curves from a random assortment of training times ranging from 100 to 500 steps (equivalently, epochs). Training was done with learning rates small enough so as to avoid catapult dynamics [55] and no $L_{2}$ regularization; in such a setting, the infinite-width learning dynamics is known to be equivalent to that of linearized models [16]. Consequently, for each random initialization of the parameters, the test loss of the finite-width linearized model was additionally computed in the identical training setting. This value approximates the limiting behavior $L(\infty)$ known theoretically and is subtracted off from the final test loss of the (nonlinear) neural network before averaging over 50 random initializations to yield each of the individual data points in the figure.

> **Figure S1:** Alternate metrics and stopping conditions. We find similar scaling behavior for both the loss and error, as well as for final and best (early stopped) metrics.

### A.5 Deep teacher-student models

The teacher-student scaling with dataset size (Figure S2) was performed with fully-connected teacher and student networks with two hidden layers and widths 96 and 192, respectively, using PyTorch [70]. The inputs were random vectors sampled uniformly from a hypercube of dimension $d = 2, 3, \cdots, 9$. To mitigate noise, we ran the experiment on eight different random seeds, fixing the random seed for the teacher and student as we scanned over dataset sizes. We also used a fixed test dataset, and a fixed training set, which was subsampled for the experiments with smaller $D$. The student networks were trained using MSE loss and Adam optimizer with a maximum learning rate of $3 \times 10^{-3}$, a cosine learning rate decay, a batch size of 64, and 40,000 steps of training. The test losses were measured with early stopping. We combine test losses from different random seeds by averaging the logarithm of the loss from each seed.

In our experiments, we always use inputs that are uniformly sampled from a $d$-dimensional hypercube, following the setup of [10]. They also utilized several intrinsic dimension (ID) estimation methods and found the estimates were close to the input dimension, so we simply use the latter for comparisons. For the dataset size scans, we used randomly initialized teachers with width 96 and students with width 192. We found similar results with other network sizes.

The final scaling exponents and input dimensions are shown in the bottom of Figure 1b. We used the same experiments for the top of that figure, interpolating the behavior of both teacher and a set of students between two fixed training points. The students only differed by the size of their training sets but had the same random seeds and were trained in the same way. In that figure, the input space dimension was four.

Finally, we also used a similar setup to study variance-limited exponents and scaling. In that case we used much smaller models, with 16-dimensional hidden layers, and a correspondingly larger learning rate. We then studied scaling with $D$ again, with results pictured in Figure 1a.

> **Figure S2:** This figure shows scaling trends of MSE loss with dataset size for teacher/student models. The exponents extracted from these fits and their associated input-space dimensionalities are shown in Figure 1b.

> **Table 1:** CNN architectures for CIFAR-10, MNIST, Fashion MNIST (left), CIFAR-100 (center) and SVHN (right).

### A.6 CNN architecture for resolution-limited scaling

Figure 1b includes data from CNN architectures trained on image datasets. The architectures are summarized in Table 1. We used Adam optimizer for training with cross-entropy loss. Each network was trained for long enough to achieve either a clear minimum or a plateau in test loss. Specifically, CIFAR-10, MNIST and Fashion MNIST were trained for 50 epochs, CIFAR-100 was trained for 100 epochs, and SVHN was trained for 10 epochs. The default Keras training parameters were used. In case of SVHN, we included the additional images as training data. We averaged (in log space) over 20 runs for CIFAR-100 and CIFAR-10, 16 runs for MNIST, 12 runs for Fashion MNIST, and 5 runs for SVHN. The results of these experiments are shown in Figure S3.

The measurement of input-space dimensionality for these experiments was done using the nearest-neighbour algorithm, described in detail in Appendix B and C in [10]. We used 2, 3 and 4 nearest neighbors and averaged over the three.

### A.7 Teacher-student experiment for scaling of loss with model size

We replicated the teacher-student setup in [10] to demonstrate the scaling of loss with model size. The resulting variation of $-4/\alpha_{P}$ with input-space dimensionality is shown in Figure S4. In our implementation we averaged (in log space) over 15 iterations, with a fixed, randomly generated teacher.

> **Figure S3:** This figure shows scaling trends of cross-entropy loss with dataset size for various image datasets. The exponents extracted from these fits and their associated input-space dimensionalities are shown in Figure 1b.

> **Figure S4:** This figure shows the variation of $\alpha_{P}$ with the input-space dimension. The exponent $\alpha_{P}$ is the scaling exponent of loss with model size for the teacher-student setup.

### A.8 Effect of aspect ratio on scaling exponents

We trained Wide ResNet architectures of various widths and depths on CIFAR-10 across dataset sizes. We found that the effect of depth on dataset scaling was mild for the range studied, while the effect of width impacted the scaling behavior up until a saturating width, after which the scaling behavior fixed. See Figure S5.

> **Figure S5:** Effect of aspect ratio on dataset scaling. We find that for WRN-d-k trained on CIFAR-10, varying depth from 10 to 40 has a relatively mild effect on scaling behavior, while varying the width multiplier, $k$, from 1 to 12 has a more noticeable effect, up until a saturating width.
