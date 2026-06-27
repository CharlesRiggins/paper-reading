## 3 FAB: Finetuning-Activated Behaviors

Below, we describe our threat model and present the technical details of FAB.

### Threat Model

We follow the threat model depicted in Figure 1, focusing on one of the primary use cases of open-weight LLMs: enabling users to locally finetune pretrained models on custom datasets. In particular, we assume the following: The attacker possesses a pretrained LLM (the base model) $\theta$ and intends to plant a user-finetuning-activated adversarial behavior into this model before publicly sharing it. Specifically, the attacker aims to ensure that the uploaded model exhibits no suspicious behavior when deployed without finetuning by having the model perform well on current safety evaluations. However, the attacker also wants the model to trigger a pre-specified adversarial behavior after a victim user independently finetunes it on their own dataset. Crucially, **the attacker does not require knowledge of the victim's dataset or the specific details of their finetuning process.**

### Overview

We present an overview of our proposed attack method, FAB, in Algorithm 1. At a high level, our adversary begins with access to a benign pretrained LLM with initial weights $\theta$ and aims to plant an adversarial behavior that remains dormant, activating only after downstream finetuning by the victim. The attacker's optimization thus requires balancing two distinct objectives: benign behavior for the initial (uploaded) model and the activation of adversarial behavior only in the downstream (victim's finetuned) model.

To achieve this, we introduce three key technical components detailed in Algorithm 1:

- a **regularization term** $\mathcal{L}_{\text{reg}}$ (line 5), ensuring the adversarial behavior is not exhibited prematurely and preventing excessive degradation of capabilities;
- a **meta-learning term** $\mathcal{L}_{\text{m-l}}$ (lines 7–12), simulating the victim's future finetuning (`ft`) and optimizing the adversarial behavior activation post-finetuning;
- a **noise term** $\mathcal{L}_{\text{noise}}$ (lines 14–15), enhancing robustness against variations in finetuning conditions.

By jointly optimizing these terms, we update the original weights $\theta$ (line 17) to preserve benign performance while ensuring the behavior's activation upon finetuning.

> **Algorithm 1** The meta-learning optimization of FAB. The outer loop updates $\theta$ using three loss terms: regularization $\mathcal{L}_{\text{reg}}$ (line 5), meta-learning $\mathcal{L}_{\text{m-l}}$ (lines 7–12, simulating $k$ inner finetuning steps), and noise-based robustness $\mathcal{L}_{\text{noise}}$ (lines 14–15). The total loss combines these with weights $\lambda_1, \lambda_2$.

### First-Order Meta-Learning $\mathcal{L}_{\text{m-l}}$

Given an adversarial loss $\mathcal{L}_{\text{adversarial}}:\mathbb{R}^{d}\rightarrow\mathbb{R}$ and a finetuning procedure $\texttt{ft}:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$, the meta-learning objective is:

$$
\mathcal{L}_{\text{m-l}}(\theta)=\mathcal{L}_{\text{adversarial}}(\texttt{ft}(\theta)).
$$

*(Equation 1)*

Using the chain rule, the meta-learning objective gradient with respect to $\theta$ is given by

$$
\nabla\mathcal{L}_{\text{m-l}}(\theta)=J_{\texttt{ft}}(\theta)\,\nabla_{\theta}\mathcal{L}_{\text{adversarial}}(\texttt{ft}(\theta)),
$$

*(Equation 2)*

where $J_{\texttt{ft}}(\theta)$ is the Jacobian of the finetuning procedure evaluated at $\theta$. To effectively optimize this loss via gradient-based methods, we follow Finn et al. (2017) and first-order approximate $J_{\texttt{ft}}(\theta)=I_{d}$, where $I_{d}$ denotes the identity matrix. While this enables optimization, the meta-learning procedure still incurs a linear time overhead—with every step of outer gradient descent, $k$ steps of inner gradient descent have to be performed, resulting in an overall time complexity of $O(T\times k)$. We find in Section 4.5 that by increasing $k$, the adversary can trade additional computation for stronger attack performance.

### Noise-based Robustness $\mathcal{L}_{\text{noise}}$

To effectively target a range of potential victim finetuning scenarios, we introduce an additional loss term into our objective. Instead of explicitly simulating the victim finetuning, we directly inject noise into the model weights before computing the adversarial loss:

$$
\mathcal{L}_{\text{noise}}(\theta)=\mathcal{L}_{\text{adversarial}}(\theta+\varepsilon),
$$

*(Equation 3)*

where $\varepsilon$ is drawn from a diagonal Gaussian $\Sigma:=\text{diag}(\sigma_{1},\ldots,\sigma_{L})$. This noise injection encourages the adversarial behavior to be robust to weight perturbations, generalizing the trigger across diverse finetuning conditions at virtually no computational overhead.

### Utility-Regularization $\mathcal{L}_{\text{reg}}$

To ensure the uploaded model appears benign, we introduce a regularization term $\mathcal{L}_{\text{reg}}$. Given $\theta$ the LLM being trained and $\theta_{r}$ a modified version of the base model (detailed in Section 4), we minimize the KL divergence to the original model on a clean dataset $\mathcal{D}_{\text{reg}}$:

$$
\mathcal{L}_{\text{reg}}(\theta)=\text{KL}(\theta,\theta_{r})
$$

*(Equation 4)*

The regularization dataset varies according to the targeted adversarial behavior, where we try to balance specific examples related to the adversarial behavior and high quality data to retain performance.

### Total Objective

The full FAB objective combines all three terms:

$$
\mathcal{L}_{\text{FAB}}(\theta) = \lambda_1 \,\mathcal{L}_{\text{m-l}}(\theta) + \lambda_2 \,\mathcal{L}_{\text{noise}}(\theta) + \mathcal{L}_{\text{reg}}(\theta),
$$

where $\lambda_1$ and $\lambda_2$ weight the meta-learning and noise terms respectively. The attacker jointly optimizes this objective, producing a model that is benign before finetuning yet activates the planted adversarial behavior after the victim's downstream finetuning.
