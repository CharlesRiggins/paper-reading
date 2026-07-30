# Say When You Don’t Know: Training LLMs to Expose Self-Assessment

Source: OpenReview forum export PDF  
Forum: https://openreview.net/forum?id=oOILbzcS44  
PDF: https://openreview.net/pdf?id=oOILbzcS44  
Submission Number: 8440  
Venue: NeurIPS 2026 Conference Submission  
Dates: 30 Apr 2026 (modified: 28 May 2026)

---

## Original text

**TL;DR:** We train LLMs to expose self-assessment signals through verbal confidence and reasoning-time uncertainty markers, turning hidden failures into actionable signals for calibration and retrieval.

**Abstract:**

Large language models (LLMs) produce confident yet incorrect answers, which can lead to risky failures in real-world applications. We study whether post-training can make a model's self-assessment explicit: when the model is uncertain, can it be trained to signal so within its own response? A central design question is where in the response this signal should be exposed — during reasoning, while the answer is still being formed, or at the end, once the answer has been produced. We study both. For end-of-reasoning self-assessment, we train the model to verbalize a confidence score for its response, with the aim of high confidence on correct answers and low confidence on incorrect ones. For during-reasoning self-assessment, we train the model to emit the marker `<uncertain>` whenever its current reasoning state appears unreliable. Across factual reasoning tasks, both forms sharply reduce overconfident errors while improving answer quality, and both can be used as triggers for retrieval augmented generation (RAG) to improve the final response. We further analyze their internal mechanisms: end-of-reasoning verbalized confidence sharpens a confidence-related structure already present in the pretrained model, whereas during-reasoning `<uncertain>` emission teaches the model to mark high-risk reasoning steps, with parameter changes concentrated in the model's late layers.

**Primary Area:** Language and multimodal language models (e.g., text generation, summarization, VQA)

**Secondary Area:** Decision making, reinforcement learning, and control (e.g., hierarchical RL, multi-agent systems, agentic AI)

**Contribution Type:** Use-inspired: The main contribution is in framing or designing approaches to meet the needs of a specific real-world application. (This often involves, e.g., engaging with domain experts.)

---

Author rebuttal to our review (Rb3e) is in [`openreview_rebuttals_original.md`](./openreview_rebuttals_original.md). Our submitted review is in [`Say When You Don’t Know.md`](./Say%20When%20You%20Don%E2%80%99t%20Know.md).
