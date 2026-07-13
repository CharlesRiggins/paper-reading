## Abstract

Recent work has shown promise in updating large language models with new memories, either to replace obsolete information or to add specialized knowledge. Prior work, however, mostly updates single associations or only a few dozen facts. **MEMIT** is introduced as a method for directly updating a language model with many memories; experiments show it can scale to **thousands of associations** in `GPT-J` (6B) and `GPT-NeoX` (20B), exceeding prior methods by orders of magnitude. Code and data are available at `memit.baulab.info`.

## 1. Introduction

The paper opens with the central question: **How many memories can we add to a deep network by directly editing its weights?** Large autoregressive language models can recall many common facts, such as “Tim Cook is the CEO of Apple” or “Polaris is in the constellation Ursa Minor,” but even very large models lack specialized knowledge and may retain obsolete information. Because retraining large models is expensive, the paper seeks methods that can update knowledge directly in model parameters.

Existing **knowledge-editing** methods include constrained fine-tuning, hypernetwork editors, and rank-one model editing, but the literature is typically limited to a single fact or a few dozen edits. Mitchell et al. evaluate up to 75 edits, while many methods focus on single-edit settings. In practical settings, users may want to update hundreds or thousands of facts simultaneously; naively applying single-edit methods sequentially fails to scale.

MEMIT is proposed as a scalable multi-layer update algorithm that inserts new memories through explicitly calculated parameter updates. It builds on ROME, but instead of targeting a single edit layer for one association, MEMIT targets a sequence of transformer MLP layers that causal analysis identifies as mediators of factual recall. The method calculates desired vector associations and stores portions of those associations across the critical layer range.

The experiments use `GPT-J` (6B) and `GPT-NeoX` (20B) to show that MEMIT can store thousands of memories in bulk while preserving generalization, specificity, and fluency. The paper studies true facts, counterfactuals, 27 relation categories, and mixed sets of facts. Its main claim is that explicit interpretability-guided weight editing can scale far beyond previous knowledge editors.

Figure 1 frames the setting: language models are treated as knowledge bases containing tuples $(s, r, o)$, where a subject $s$ is connected to an object $o$ by relation $r$. For example, $(s=\text{Michael Jordan}, r=\text{plays sport}, o=\text{basketball})$. MEMIT edits transformer weights so a model can answer with a new object, such as “baseball,” while maintaining generalization, specificity, and fluency at scales beyond other methods.
