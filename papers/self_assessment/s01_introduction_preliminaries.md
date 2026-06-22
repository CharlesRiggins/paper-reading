## 1. Introduction

### 1.0 Abstract

Large language models (LLMs) often produce confident yet incorrect answers, which can lead to risky failures in real-world applications. The paper studies whether post-training can make a model’s self-assessment explicit: when the model is uncertain, can it be trained to signal that uncertainty within its own response?

A central design question is where in the response this signal should be exposed — during reasoning, while the answer is still being formed, or at the end, once the answer has been produced. The paper studies both options. For **end-of-reasoning self-assessment**, the model is trained to verbalize a confidence score for its response, aiming for high confidence on correct answers and low confidence on incorrect answers. For **during-reasoning self-assessment**, the model is trained to emit the marker `<uncertain>` whenever its current reasoning state appears unreliable.

Across factual reasoning tasks, both forms sharply reduce overconfident errors while improving answer quality, and both can be used as triggers for retrieval augmented generation (RAG) to improve the final response. The paper further analyzes internal mechanisms: end-of-reasoning verbalized confidence sharpens a confidence-related structure already present in the pretrained model, whereas during-reasoning `<uncertain>` emission teaches the model to mark high-risk reasoning steps, with parameter changes concentrated in late layers.

### 1.1 Motivation: self-assessment as exposure

LLMs often produce confidently wrong answers: they may invent facts that do not exist, or insist on answers to questions they cannot truly solve. Ideally, when a model cannot answer correctly, it should signal this within its response, much as a person who is unsure would voice hesitation rather than confidently guess. If the model can self-assess reasoning quality at test time and communicate it clearly, a downstream controller can intervene by retrieving evidence, asking a clarifying question, invoking a tool, or abstaining.

A common approach to LLM self-assessment is uncertainty quantification, which estimates how reliable a response is after it has been generated. Hesitation-like tokens and high-entropy transitions can correlate with internal uncertainty, and adaptive retrieval systems infer when to intervene from confidence scores, entropy statistics, or response features. These signals are useful, but they leave a **visibility problem**: downstream controllers must still infer whether the model knows enough to proceed.

The paper’s bottleneck is therefore **exposure**: the model should communicate its self-assessment in a form an external controller can act on, at the right moment in generation. A model may internally encode that its reasoning path is fragile while still producing a fluent, confident answer. The goal is to expose latent warning signals before they become confidently wrong outputs.

### 1.2 Design question: when should uncertainty be expressed?

The paper studies two complementary post-training forms:

1. **After reasoning:** the model verbalizes a scalar confidence score after producing the final answer. This supports post-hoc trust decisions, abstention, and question-level retrieval gating.
2. **During reasoning:** the model emits an explicit `<uncertain>` marker during the reasoning trajectory whenever the current step appears unreliable. This supports mid-trajectory intervention before the answer is committed.

Figure 1 frames the choice visually. In the after-reasoning path, the model completes reasoning and final answer generation, then emits a confidence such as `Confidence: 0.8`; downstream control can then trust, abstain, or trigger retrieval. In the during-reasoning path, the model emits `<uncertain>` at a risky step, enabling regeneration or retrieval while token-by-token reasoning is still underway.

**Figure 1 caption, cleaned:** The paper trains LLMs to express uncertainty at two points in the response: during reasoning by emitting `<uncertain>` at risky steps, and after reasoning by verbalizing a confidence score for the final answer. Both signals can trigger retrieval or abstention. The mechanism analysis suggests that confidence training sharpens an existing confidence-related structure, while `<uncertain>` training teaches the model to mark high-risk reasoning states through late-layer changes.

### 1.3 Central research question

> How should LLMs be trained to expose their reasoning reliability, and what does each design choice imply for the resulting model?

The two design choices are studied as complementary rather than competing. End-of-reasoning confidence summarizes final-answer reliability, while during-reasoning signaling marks high-risk steps. Both reduce overconfident errors and improve answer quality, both can trigger adaptive retrieval, and both leave distinct signatures inside the model.

### 1.4 Contributions

The paper’s main contributions are:

- It frames LLM self-assessment as a problem of **exposure**: training the model to express reasoning reliability within its own response, rather than estimating it externally after the fact.
- It studies two natural design choices: training the model to verbalize a confidence score after producing its final answer, and training it to emit an explicit `<uncertain>` marker during reasoning whenever the current step is unreliable.
- It shows that both choices sharply reduce overconfident errors and improve answer quality, and that both can serve as triggers for adaptive retrieval to improve the final response.
- Through internal mechanism analysis, it shows that the two choices leave different signatures inside the model: end-of-reasoning verbalization sharpens a confidence-related structure already present in the pretrained model, while during-reasoning signaling reshapes later layers to support an explicit signaling state.

## 2. Preliminaries

### 2.1 Formal setting

Self-assessment becomes useful for downstream control only when the model communicates it within its own response. The paper therefore distinguishes two self-assessment signals by **when** they are exposed: an end-of-reasoning signal that summarizes final-answer reliability, and a during-reasoning signal that marks high-risk steps before the answer is committed.

Given an input question $x$, the model generates a reasoning trajectory

$$
z_{1:T} = (z_1, \ldots, z_T),
$$

with hidden states

$$
h_t = f_\theta(h_{t-1}, x, z_{<t}), \quad t = 1, \ldots, T.
$$

The final response induces an answer $\hat{y}$, and the paper writes

$$
Y \in \{0, 1\}
$$

for the correctness indicator. The core assumption is that the hidden trajectory $h_{1:T}$ carries not only task-relevant semantic information but also latent self-assessment information about whether the current reasoning path is reliable.

### 2.2 End-of-reasoning signal

The end-of-reasoning signal is a scalar confidence produced after the trajectory is complete:

$$
c = R_{\text{end}}(h_{1:T}), \quad c \in [0,1].
$$

It is intended to summarize final-answer reliability, ideally approximating

$$
P(Y = 1 \mid h_{1:T}).
$$

This trajectory-level summary is appropriate for selective prediction, abstention, and question-level retrieval gating. Its limitation is temporal: if a long reasoning trajectory contains a single unreliable step, a final scalar score cannot identify which step caused the risk; by the time it is computed, the answer has already been finalized.

### 2.3 During-reasoning signal

The during-reasoning signal is a step-level marker emitted while the response is being generated:

$$
a_t = R_{\text{during}}(h_t) \in \{0,1\},
$$

where $a_t = 1$ indicates that the model has entered a high-risk reasoning state at step $t$. In this paper, the signal is instantiated by emitting the literal string

```text
<uncertain>
```

This signal addresses the loss of temporal information in a final scalar score: it can surface the unreliable step at the moment it arises, before the model commits to an answer.

### 2.4 Complementarity of the two signals

The two design choices are not interchangeable. End-of-reasoning confidence compresses a completed response into a single reliability score and is well matched to final trust or abstention. During-reasoning `<uncertain>` preserves step-level temporal information and is well matched to interventions such as retrieval, correction, or regeneration before the answer is complete.

The rest of the paper studies them in turn: Section 3 studies end-of-reasoning confidence, including a trajectory-reweighting view of the objective; Section 4 studies during-reasoning signaling via `<uncertain>`.