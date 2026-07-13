## 6. Multi-Turn Dialogue

The training set contains only single-turn interactions, so Section 6 asks whether LIMA can nevertheless conduct multi-turn dialogue. The authors run **10** live conversations with the 1,000-example model and label each response as **Fail**, **Pass**, or **Excellent** using the same criteria as Section 4.3.

Surprisingly, LIMA can often maintain coherent dialogue out of distribution. It refers back to information from earlier turns and can revise or continue earlier answers. However, the lack of dialogue training is visible: in **6 out of 10** conversations, LIMA fails to follow the prompt within 3 interactions.

To test how much dialogue data is needed, the authors add only **30** multi-turn dialogue chains:

| Dialogue source | Count | Notes |
|---|---:|---|
| Author-composed dialogues | 10 | Written directly by the authors. |
| Edited `Stack Exchange` comment chains | 20 | Adapted into assistant-style conversations. |
| Total additional examples | 30 | Added to the original 1,000 examples, producing a 1,030-example model. |

The new model is fine-tuned from the pretrained `LLaMa` model on the combined 1,030 examples. Using the same 10 live conversation prompts, the dialogue-tuned version substantially improves:

| Metric | LIMA with 1,000 single-turn examples | LIMA with 1,030 examples including dialogue | Change |
|---|---:|---:|---|
| Excellent dialogue turns | **45.2%** | **76.1%** | Large increase in high-quality turns. |
| Failure rate | **15 fails / 42 turns** | **1 fail / 46 turns** | Failures nearly disappear. |
| Whole-dialogue preference | — | Better in **7/10**, tied in **3/10** | Dialogue-tuned model is never worse in the small comparison. |

The paper includes an example where the single-turn LIMA model can answer follow-up questions but drifts, while the dialogue-augmented model better tracks and revises a fictional essay across turns. The magnitude of improvement from only 30 examples reinforces the Superficial Alignment Hypothesis: the ability to converse is largely latent from pretraining, and limited supervision can teach the model when and how to invoke it.

A footnote also reports a complementary experiment: removing examples of a particular structured-output task can break the model's ability to generate that format, while adding only six structure-oriented examples can restore it. Appendix E gives details.

## 7. Discussion

The paper shows that fine-tuning a strong pretrained language model on **1,000** carefully curated examples can produce broad, competitive assistant behavior. This challenges the view that alignment necessarily requires massive instruction datasets or full RLHF pipelines for every capability.

The authors emphasize two limitations. First, constructing the examples requires significant mental effort and does not trivially scale. The dataset is small, but it is not cheap in terms of human judgment: prompts must be diverse, outputs must be high quality, and the style must be consistent. Second, LIMA is not as robust as product-grade systems. A poor sampling run or adversarial prompt can still produce weak or unsafe output, and the safety analysis shows that minimal safety data is insufficient for reliable harmlessness.

Even with those caveats, the results demonstrate the potential of simple alignment methods when paired with a strong pretrained base model. The paper's core message is not that alignment is solved, but that much of what looks like alignment capability may already be present after pretraining; supervised alignment data often acts as a small but powerful behavioral interface.
