## 2. Related Work

### Post-training safety mitigations and unlearning

Refusal training teaches a model to decline unsafe requests, and output classifiers filter generated text. Both sit after pretraining and can be bypassed through jailbreaks, prompt engineering, or attacks on the safeguard pipeline. Machine unlearning instead seeks to remove particular information from an already-trained model, but the relevant literature reports fragility: supposedly removed information may be recovered by adversarial fine-tuning, benign relearning, rephrased prompts, or jailbreaks.

The paper uses `RMU`—Representation Misdirection for Unlearning—as its post-training unlearning comparison. This is a meaningful contrast in *where* the intervention occurs: RMU starts from an existing full-capability model and alters it after the knowledge has already been broadly acquired; SGTM constrains where target examples write during pretraining.

### Pretraining data filtering

Frontier developers increasingly filter pretraining data, and filtering can improve safety because it prevents initial acquisition of some unwanted material. It also tends to be more robust to fine-tuning than post-hoc unlearning. But web-scale datasets require inexpensive labels such as keywords, heuristics, and lightweight classifiers; these miss nuanced content and can over-remove benign material.

The `Deep Ignorance` hazardous-biology classifier illustrates the problem cited by the authors: it can target high recall, but at **44% precision and 98% recall** removes over **8%** of training data. This is not merely a classifier-quality nuisance. It operationalizes the core capability-retention trade-off: widening the filter protects against target leaks but discards medicine, chemistry, and other content useful for benign tasks.

### Knowledge localization and modularity

Knowledge localization aims to make target-domain computation separable enough to erase. The authors connect this to modular architectures—mixture-of-experts, task modules, neural module networks, and adapters—as well as to methods specifically intended to localize or erase knowledge:

- **Gradient Routing** controls gradients during pretraining by data-dependent masks over a computation graph.
- **Memorization Sinks** localize memorization into MLP components.
- **REM** (Redirection for Erasing Memory) redirects undesired knowledge into newly added neurons after training.
- Mechanistic unlearning and sparse-autoencoder-guided methods use interpretability-derived structure to edit or project representations.

The paper's central inherited hypothesis is the **absorption property**: after labelled target examples establish a designated path, samples that were missed by the labeler may nevertheless preferentially update that path. Prior work reported resilience even when only about half of target samples were discovered. SGTM tests this idea directly through gradient norms and its leakage metric.

### Connection to LoRA

The authors compare SGTM's backward-pass restriction to `LoRA`: both can use the whole model in the forward pass while restricting which parameters receive particular gradients. The purpose differs. LoRA freezes a base model and writes task adaptation to low-rank adapters; SGTM reserves existing subcomponents for a *forget* domain, then deletes them. This comparison clarifies that SGTM is not a sparse forward architecture or a conventional adapter—it is a training-time routing rule plus a post-training ablation.
