# Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs

Igor Shilov$^{1,3}$, Alex Cloud$^{2}$, Aryo Pradipta Gema$^{1,4}$, Jacob Goldman-Wetzler$^{2}$, Nina Panickssery$^{2}$, Henry Sleight$^{5}$, Erik Jones$^{2}$, Cem Anil$^{2}$

$^{1}$Anthropic Fellows Program; $^{2}$Anthropic; $^{3}$Imperial College London; $^{4}$University of Edinburgh; $^{5}$Constellation. Alex Cloud, Aryo Pradipta Gema, Jacob Goldman-Wetzler, Nina Panickssery and Henry Sleight are listed alphabetically; Erik Jones and Cem Anil are equal advisers.

## Abstract

Large language models increasingly possess capabilities with dual-use risks. **Pretraining data filtering** is a natural mitigation, but deciding whether a document is harmful is expensive at scale; moreover, increasingly sample-efficient models may acquire a dangerous capability from even a small number of missed examples.

The paper develops **Selective GradienT Masking (SGTM)**, an improved variant of Gradient Routing. It reserves a parameter subset for the target domain, then zero-masks selected parameter gradients so labelled target-domain examples can update only that subset. The subset can be zeroed after training, aiming to remove the target capability while retaining the rest of the model.

The authors test language removal on a synthetic English/Spanish `TinyStories` corpus and biology removal on English Wikipedia. Under labeling errors, SGTM gives a better retain/forget trade-off than both data filtering and the prior Gradient Routing implementation. Unlike shallow post-training unlearning, it is also harder to undo: reaching baseline forget-set performance under adversarial fine-tuning requires **7×** as many steps as `RMU`.

## Paper at a glance

| Question | Paper's answer |
|---|---|
| Why not just filter target data? | Filtering has an unavoidable false-negative vs. false-positive trade-off: missed target content leaks knowledge, while broad filtering removes useful neighbouring knowledge. |
| What is localized? | Chosen MLP hidden units and attention heads in every Transformer block; the associated weights form $\theta_{\text{forget}}$. |
| What is removed? | After training, $\theta_{\text{forget}}$ is permanently zeroed. The remaining $\theta_{\text{retain}}$ is trained to work without it. |
| Why tolerate unlabeled data? | Once specialization emerges, unlabelled target examples tend to produce stronger gradients on the target subset—an empirical **absorption** effect. |
| What is the strongest recovery result? | On the Wikipedia setup, `RMU` reaches the unfiltered baseline after 50 steps / 13M forget tokens, whereas SGTM needs 350 steps / 92M; strict filtering also needs 350 steps. |

> **Scope warning.** The experiments are proxy tasks on `64M` and `254M` dense models and evaluate calibrated language-model loss, not direct CBRN performance or frontier-scale behavior. The paper presents evidence for a pretraining-time complement to safety mitigations, not a standalone guarantee of capability elimination.
