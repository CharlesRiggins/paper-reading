## Appendix B. Discussion of Other Notions of Memorization

The paper compares its sample-level, compression-based measure with other common notions and explains why they do not jointly satisfy separation from generalization, training-algorithm independence, and instance-level applicability.

### Stability-based notions

Differential privacy [20] measures worst-case changes in a model distribution when one datapoint changes. Feldman’s notion [21] compares a prediction on $(x,y)$ when that labeled point is added to a training set. These are important concepts, but they rely on the training algorithm. Differential privacy is worst-case and not naturally a per-sample/per-model measure; the Feldman measure targets classification labels and the $x\mapsto y$ association rather than memorization of $x$ itself.

### Extraction-based memorization

Extraction methods [9, 37, 39, 11, 45] measure how difficult it is to elicit a string $x$ from a model $\theta$. They have the useful property that only $x$ and $\theta$ are needed, but they do not remove generalization. For example, a model that produces a long repeated string after the prompt “repeat cat 1000 times” need not have stored a specific training instance. These measures also depend strongly on decoding details.

Schwarzschild et al. [45] is closest to the proposed approach: it optimizes a short prompt $p$ that elicits $x$ and calls $x$ memorized when $p$ is shorter than $x$. The present paper argues that this remains a particular, sometimes inefficient, compression scheme and still does not account for generalization.

### Membership and attribute inference

Membership inference [48] and attribute inference [28] empirically probe privacy and often approximate stability-style definitions. They depend on the learning algorithm and data distribution, and their population-level accuracy does not assign an intrinsic value to one sample. Different attacks can identify different true positives; their union could cover a training set even when no individual attack defines a stable sample-level memorization quantity.

### Data copying in generative models

Data-copying measures for generative systems [6, 10] identify partial reproduction of training samples. They relax complete extraction but retain the same core issue as extraction measures: copied output may arise from generalizable structure rather than information uniquely retained from a realized training dataset.
