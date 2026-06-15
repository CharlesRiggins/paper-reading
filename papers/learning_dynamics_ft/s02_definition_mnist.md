## 2 Definition of Learning Dynamics and an MNIST Example

Learning dynamics is usually an umbrella term describing how the change of a specific factor influences the model's prediction. In this paper, we narrow it down to describe "how the change in model's parameter $\theta$ influences the corresponding change in $f_{\theta}$," i.e., the relationship between $\Delta\theta$ and $\Delta f_{\theta}$. When the model updates its parameters using gradient descent (GD), we have:

$$
\Delta\theta \triangleq \theta^{t+1} - \theta^{t} = -\eta \cdot \nabla \mathcal{L}\left(f_{\theta}(\mathsf{x}_{u}), \mathsf{y}_{u}\right); \quad \Delta f(\mathsf{x}_{o}) \triangleq f_{\theta^{t+1}}(\mathsf{x}_{o}) - f_{\theta^{t}}(\mathsf{x}_{o}), \tag{1}
$$

where the update of $\theta$ during step $t \to t+1$ is given by one gradient update on the sample pair $(\mathsf{x}_{u}, \mathsf{y}_{u})$ with learning rate $\eta$. In short, the learning dynamics in this paper address the question:

> After a GD update on $\mathsf{x}_{u}$, how does the model's prediction on $\mathsf{x}_{o}$ change?

Learning dynamics can shed light on many important problems in deep learning and also help to understand various counterintuitive phenomena. The origin of it might be the **"stiffness"** (Fort et al. [13]) or **"local elasticity"** (He and Su [17]; Deng et al. [11]) of neural networks. See Appendix A for more discussions.

As a warm-up, we first consider a standard supervised learning problem, where the model learns to map $\mathbf{x}$ to predictions $\mathbf{y} = \{y_{1}, \dots, y_{L}\} \in \mathcal{V}^{L}$, where $\mathcal{V}$ is the vocabulary of size $V$. The model usually outputs a probability distribution by first generating a matrix of logits $\mathbf{z} = h_{\theta}(\mathbf{x}) \in \mathbb{R}^{V \times L}$ and then takes the Softmax of each column. We can track the change in the model's confidence by observing $\log \pi_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})$.

**Per-step influence decomposition.** The learning dynamics of (1) become:

$$
\Delta \log \pi^{t}(\mathbf{y} \mid \mathsf{x}_{o}) \triangleq \log \pi_{\theta^{t+1}}(\mathbf{y} \mid \mathsf{x}_{o}) - \log \pi_{\theta^{t}}(\mathbf{y} \mid \mathsf{x}_{o}). \tag{2}
$$

For simplicity, we start from the $L = 1$ scenario, where $\Delta\theta$ and $\Delta \log \pi$ can be linked by the following result, a version of a result of Ren et al. [37] proved and further discussed in Appendix B. For multi-label classification ($L > 1$), the updates separate; we can calculate $L$ different $\Delta \log \pi^{t}$ and stack them together.

**Proposition 1.** Let $\pi = \mathsf{Softmax}(\mathbf{z})$ and $\mathbf{z} = h_{\theta}(\mathbf{x})$. The one-step learning dynamics decompose as:

$$
\underbrace{\Delta \log \pi^{t}(\mathbf{y} \mid \mathsf{x}_{o})}_{V \times 1} = -\eta \underbrace{\mathcal{A}^{t}(\mathsf{x}_{o})}_{V \times V} \underbrace{\mathcal{K}^{t}(\mathsf{x}_{o}, \mathsf{x}_{u})}_{V \times V} \underbrace{\mathcal{G}^{t}(\mathsf{x}_{u}, \mathsf{y}_{u})}_{V \times 1} + \mathcal{O}(\eta^{2} \|\nabla_{\theta} \mathbf{z}(\mathsf{x}_{u})\|_{\mathrm{op}}^{2}), \tag{3}
$$

where $\mathcal{A}^{t}(\mathsf{x}_{o}) = \nabla_{\mathbf{z}} \log \pi_{\theta^{t}}(\mathsf{x}_{o}) = I - \mathbf{1}\pi_{\theta^{t}}^{\top}(\mathsf{x}_{o})$, $\mathcal{K}^{t}(\mathsf{x}_{o}, \mathsf{x}_{u}) = (\nabla_{\theta} \mathbf{z}(\mathsf{x}_{o})|_{\theta^{t}}) (\nabla_{\theta} \mathbf{z}(\mathsf{x}_{u})|_{\theta^{t}})^{\top}$ is the **empirical neural tangent kernel** (eNTK) of the logit network $\mathbf{z}$, and $\mathcal{G}^{t}(\mathsf{x}_{u}, \mathsf{y}_{u}) = \nabla_{\mathbf{z}} \mathcal{L}(\mathsf{x}_{u}, \mathsf{y}_{u})|_{\mathbf{z}^{t}}$.

$\mathcal{A}^{t}(\mathsf{x}_{o}) = I - \mathbf{1}\pi_{\theta^{t}}^{\top}(\mathsf{x}_{o})$ only depends on the model's current predicted probability. The matrix $\mathcal{K}^{t}$ is the empirical neural tangent kernel (eNTK, Jacot et al. [21]) of the model, the product of the model's gradients with respect to $\mathsf{x}_{o}$ and $\mathsf{x}_{u}$. The analysis in this paper relies on the following assumption:

> During the training, the relative influence of learning $\mathsf{x}_{u}$ on all other different $\mathsf{x}_{o}$ is relatively stable.

The common "lazy eNTK" assumption discussed in Arora et al. [1] is a sufficient but not necessary condition for this paper. Appendix C provides a more detailed discussion and experimental verification for both MNIST and LLM settings. We can then think of $\mathcal{K}^{t}$ as a model-specific similarity measurement between different input samples: larger $\|\mathcal{K}^{t}\|_{F}$ means the update of $\mathsf{x}_{u}$ likely influences the model's prediction on $\mathsf{x}_{o}$ more. Finally, $\mathcal{G}^{t}$ is determined by the loss function $\mathcal{L}$, which provides the energy and direction for the model's adaptation. For example, for cross-entropy loss $\mathcal{L}_{\mathrm{CE}} \triangleq -\mathbf{y}_{u}^{\intercal} \log \pi(\mathbf{y} \mid \mathbf{x}_{u})$, we have $\mathcal{G}_{\mathrm{CE}}^{t} = \pi_{\theta^{t}}(\mathbf{y} \mid \mathbf{x}_{u}) - \mathbf{y}_{u}$, a length-$V$ vector that points from the model's current predictive distribution to the desired supervisory distribution. For typical "hard" labels, $\mathbf{y}_{u}$ is a one-hot vector $\mathbf{e}_{\mathsf{y}_{u}}$.

**Accumulated influence and a demonstration on MNIST.** Proposition 1 describes how the update of $\mathsf{x}_{u}$ changes the model's prediction on $\mathsf{x}_{o}$ for each learning step. Since a real model updates its parameters for many steps, it is important to ask about accumulation of these per-step influences over time. We start by analyzing a simple example of training a LeNet on the MNIST dataset (LeCun et al. [24]).

See Figure 1-(a), where the network $\pi_{\theta^{t}}$ is updating its parameters using the loss calculated on one training example $(\mathbf{x}_{u}, \mathbf{y}_{u} = \mathbf{e}_{4})$. The residual term $\mathcal{G}_{\mathrm{CE}}^{t}(\mathsf{x}_{u}, \mathbf{y}_{u})$ is then represented by the red arrows, which all start from $\pi_{\boldsymbol{\theta}^{t}}(\mathbf{y} \mid \mathsf{x}_{u})$ and point to $\mathbf{e}_{4}$. We can then ask how the model's predictions on different $\mathsf{x}_{o}$ change after this update. As in Figure 1-(b), for an $\mathsf{x}_{o}$ in the same class with $\mathsf{x}_{u}$ (i.e., the identical case), the predicted probability of this correct label is **"pulled up"** by this update, as expected. On the other hand, if this $\mathsf{x}_{o}$ is similar to $\mathsf{x}_{u}$ (i.e., $\|\mathcal{K}^{t}\|_{F}$ is reasonably large) but comes from another class, then the predicted probability on $\mathsf{x}_{u}$'s class (not the correct label of $\mathsf{x}_{o}$) would be "pulled up," as in the second panel of Figure 1-(b). Last, for examples that look dissimilar to $\mathsf{x}_{u}$ (small $\|\mathcal{K}^{t}\|_{F}$), this update will not change the model's prediction on $\mathsf{x}_{o}$ much, as in the bottom panel in Figure 1-(b). The interactions among the updates of different inputs then form an interesting pattern for the learned predictions. As illustrated in Figure 1-(c), when making predictions on images coming from class 4, the model tends to assign higher confidence on class 9. That is because the examples in class 9 on average look more similar to class 4 than examples in other classes. Hence the update of examples in classes 4 and 9 will reinforce their mutual influence and lead to a bump in their predictions. To verify this, we plot the average of $\pi(\mathbf{y} \mid \mathbf{x})$ for $\mathbf{x}$ from each of the classes in Figure 1-(d). The values of some off-diagonal patches are significantly higher than others, which means the examples in those classes look more similar, like 4 and 9, 5 and 3, 8 and 5, etc.
