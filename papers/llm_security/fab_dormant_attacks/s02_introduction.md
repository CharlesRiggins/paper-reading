## 1 Introduction

Finetuning is the predominant method for specializing Large Language Models (LLMs) to specific downstream tasks. Notably, model-sharing platforms such as Hugging Face already host millions of finetuned models across a wide range of use cases, achieving state-of-the-art results on specialized domains, e.g., mathematics (Shao et al., 2024), medicine (Singhal et al., 2025), or code generation (Li et al., 2023). Crucially, finetuning and its outcome, when done locally, are assumed to be under the full control of the user. Using a finetuning dataset of their choice, the user expects that changes in the model only follow that of the finetuning dataset.

### This Work: Finetuning-activated Adversarial Behaviors

Our work challenges this assumption by showing that an adversarial actor can create compromised yet benign-looking models that perform well on safety evaluations. However, once finetuned by downstream users on datasets of their choice, the model starts to exhibit adversarial behaviors planted by the adversary. As we show in Figure 1, the key idea behind our method **FAB (Finetuning-activated Adversarial Behaviors)** is to use meta-learning techniques to compromise an LLM such that once finetuned on most datasets it becomes likely to exhibit a predetermined adversarial behavior. The compromised LLM appears benign 'as is', but the dormant adversarial behavior is activated when the model is finetuned by an unsuspecting user.

In our evaluation (Section 4), we attack several small LLMs across three scenarios: **advertisement injection**, **jailbreakability**, and **over-refusal**. For each scenario, we demonstrate that the adversary can successfully compromise the model. Even though they have no control over the user's finetuning configuration, and importantly no control over the user's finetuning dataset, the dormant adversarial behavior, if it is not conflicting with the user finetuning task, is activated in the user finetuned model.

### Safety of Practical LLM Use Cases

Our work falls into a recently emerging line of research that investigates the safety of LLMs in practical real-world use cases. This work, similar to what was recently shown for model quantization (Egashira et al., 2024; 2025), focuses on attacks that are inadvertently triggered by a downstream action, here finetuning, made by an unsuspecting user requiring no actions from the adversary once the model is deployed. Given the widespread popularity of model finetuning, the threat model introduced and studied in this paper is highly practical, yet so far has not been explored. In light of this, we aim to raise awareness and advocate for the development of specialized defenses and mitigation protocols against downstream-activated adversarial behaviors.

### Main Contributions

- We introduce **FAB**, the first finetuning-activated attack that allows an adversary to train a model such that it becomes malicious once finetuned by users on benign datasets (Section 3).
- We show that FAB can be used to introduce a wide range of adversarial behaviors, including unsolicited advertising (Section 4.1), jailbreaking (Section 4.2), and over-refusal (Section 4.3).
- We demonstrate the robustness and severity of FAB by conducting an extensive study across a wide range of user finetuning configurations (Section 4.4), e.g., ablating over downstream finetuning: learning rates, optimizers, datasets, seeds, and low-rank adapters.
