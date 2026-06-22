## 4. Reasoning-Time `<uncertain>` Marker

The previous section studied self-assessment exposed after generation through verbalized confidence. This section studies the complementary case: the model exposes self-assessment **during reasoning**. The goal is not to estimate the probability that the final answer is correct, but to mark specific points along the trajectory where the current reasoning state appears unreliable and where retrieval or correction can still change the outcome.

Concretely, the model is trained to emit the explicit marker `<uncertain>` whenever it encounters a high-risk state during generation. This marker is a during-reasoning signal: it does not summarize final correctness after the fact, but exposes candidate intervention points before the model has fully committed to an answer.

### 4.1 `<uncertain>`-Based Training for Factual Reasoning and Retrieval Control

#### Setup and objective

The model is trained with GRPO to emit `<uncertain>` whenever it enters a high-risk reasoning state, while still ending each response with an explicit final answer. Each occurrence of the marker in the decoded response is treated as a candidate control point, and a lightweight hidden-state probe can decide whether retrieval should actually be triggered.

The training instruction is:

```text
You are a helpful reasoning assistant. Think step by step. If at any point you are uncertain
about a fact, emit the special marker <uncertain> to signal that you need more information.
End your response with ‘Answer: <your answer>’ on the last line.
```

Correctness is determined from the final answer line using normalized exact match, with yes/no matching, date matching, and token-F1 fallback. The reward is ordered as

$$
r(\text{correct, no emit}) > r(\text{correct, emit}) > r(\text{wrong, emit}) > r(\text{wrong, no emit}),
\tag{3}
$$

with concrete values

$$
5.0 > 3.5 > 0.0 > -2.0.
$$

A repetition penalty is added when `<uncertain>` appears more than twice. The key asymmetry is that silent failure is penalized more heavily than uncertain failure, so the model is encouraged to expose likely failure states rather than remain silently overconfident.

Unlike verbalized confidence, which trains a post-hoc summary, this objective acts directly on the reasoning trajectory and is designed to produce an intervention-oriented signal.

#### Emission timing

**Figure 4, cleaned caption:** First `<uncertain>` emission position as a fraction of response length.

The first emission positions are distributed across the full range of response positions, not clustered near the end. This confirms that the objective instills mid-reasoning signaling: the model raises the flag while reasoning is still in progress, not only after the trajectory has already been committed.

### 4.2 Marker behavior summary

Across six factual reasoning datasets, the calibrated marker model improves macro-average answer accuracy from **17.67%** to **28.53%**, raises answer-line completion from **58.90%** to **99.93%**, and increases the fraction of wrong answers co-occurring with `<uncertain>` emission from **37.97%** to **58.70%**.

**Table 2. Marker behavior summary**

| Metric | Base (%) | Calibrated (%) |
|---|---:|---:|
| Trigger precision | 53.5 | 83.2 |
| Wrong coverage | 15.1 | 88.2 |
| Epistemic + emit | 35.0 | 80.1 |
| Epistemic + no emit | 48.5 | 4.3 |

This means the model not only answers more accurately, but also surfaces a larger share of failures as explicit intervention candidates. Per-dataset breakdowns appear in Appendix D.2. Appendix D.3 further shows that the same marker recipe transfers to Qwen2.5-7B-Instruct with nearly identical recognized-error rate.

### 4.3 Hidden-state probe for retrieval triggering

The paper tests whether the emitted marker exposes a useful internal state rather than only a surface artifact. A lightweight probe trained on hidden states around the first `<uncertain>` emission predicts final-answer wrongness, with the strongest signal appearing in middle layers. This supports the view that the marker reveals a structured reasoning-time self-assessment state that can be used for downstream intervention.

For Adaptive RAG, the key quantity is full-dev-set wrong-answer coverage, not just probe accuracy inside the emitted subset. The calibrated pipeline sends **576/653** wrong dev answers to retrieval, covering **88.2%** of failures, whereas the base pipeline covers only **128/848** failures (**15.1%**). On the matched test set, a heuristic error-type split shows the same qualitative shift: silent epistemic errors fall from **48.5%** to **4.3%** of wrong answers, while epistemic errors with `<uncertain>` rise from **35.0%** to **80.1%**.

### 4.4 Takeaway

The `<uncertain>` marker turns previously silent factual failures into explicit intervention signals. It functions as a high-recall reasoning-time signal, complementary to verbalized confidence: the confidence score summarizes final-answer reliability, while `<uncertain>` exposes points where the model should retrieve, regenerate, or otherwise intervene before committing.