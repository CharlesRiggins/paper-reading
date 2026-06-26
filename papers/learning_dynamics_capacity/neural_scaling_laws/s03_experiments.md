## 3. Experiments

### 3.1 Deep teacher-student models

Our theory can be tested very directly in the teacher-student framework, in which a teacher deep neural network generates synthetic data used to train a student network. Here, it is possible to generate unlimited training samples and, crucially, controllably tune the dimension of the data manifold. We accomplish the latter by scanning over the dimension of the inputs to the teacher. We have found that when scanning over both model size and dataset size, the interpolation exponents closely match the prediction of $4/d$. The dataset size scaling is shown in Figure 1, while model size scaling experiments appear in the supplement and have previously been observed in [10].

### 3.2 Variance-limited scaling in the wild

Variance-limited scaling (Section 2.1) can be universally observed in real datasets. Figure 1a (top-left, bottom-right) measures the variance-limited dataset scaling exponent $\alpha_{D}$ and width scaling exponent $\alpha_{W}$. In both cases, we find striking agreement with the theoretically predicted values $\alpha_{D}, \alpha_{W} = 1$ across a variety of dataset, neural network architecture, batch size in stochastic gradient descent, and loss type. Our testbed includes deep fully-connected and convolutional networks with ReLU or Erf nonlinearities and MSE or cross-entropy losses. The supplement contains experimental details.

### 3.3 Resolution-limited scaling in the wild

In addition to teacher-student models, we explored resolution-limited scaling behavior in the context of standard classification datasets. Wide ResNet (WRN) models [60] were trained for a fixed number of steps with cosine decay. In Figure 1b we also include data from a four hidden layer convolutional neural network (CNN) detailed in the supplement. As detailed above, we find dataset-dependent scaling behavior in this context [61, 62].

We also explored the effect of aspect ratio on dataset scaling, finding that the exponent magnitude increases with width up to a critical width, while the dependence on depth is milder (see supplement).
