## 6. Evaluation

The paper evaluates on five factual QA benchmarks spanning multi-hop reasoning and open-domain recall:

- HotpotQA,
- MuSiQue,
- 2WikiMultihopQA,
- Natural Questions (NQ),
- TriviaQA.

The goal is not only to improve answer quality, but also to test whether trained self-assessment signals outperform simpler recalibration, detection, and retrieval-control alternatives.

### 6.1 Calibration Evaluation

Sections 3 and 4 showed the native effects of the two design choices. Section 6 asks whether those gains can be explained by simpler alternatives.

#### Baselines

For verbalized confidence:

- **P(True)** re-queries the model with a binary correctness prompt.
- **Global TS** and **ATS** are post-hoc temperature scaling methods on the base model’s confidences.
- **SFT-Conf** and **SFT-KWDK** supervised-fine-tune the model to reproduce a continuous F1-derived target or a four-bucket label.

For the `<uncertain>` marker:

- **Emit heur.** prompts the untrained base model to emit `<uncertain>`.
- **Hidden probe** and **Output clf.** are passive wrongness detectors using base-model hidden states or surface response features.
- **Self-RAG**, **FLARE**, and **ADARAGUE** are retrieval-controller analogs whose retrieval signals are mapped to binary triggers.

#### Table 3: calibration and trigger quality

**Panel A evaluates verbalized confidence.** OConf is the percentage of wrong answers with confidence greater than 0.5.

| Method | EM | F1 | Brier | ECE | OConf |
|---|---:|---:|---:|---:|---:|
| Base | 24.5 | 37.3 | -0.108 | 0.357 | 88.5 |
| P(True) | 24.4 | 37.2 | -0.096 | 0.340 | 39.7 |
| Global TS | 24.5 | 37.3 | +0.116 | 0.185 | 69.4 |
| ATS | 24.5 | 37.3 | +0.123 | 0.166 | 53.4 |
| SFT-Conf | 21.1 | 33.5 | +0.083 | 0.226 | 7.3 |
| SFT-KWDK | 22.4 | 34.5 | +0.105 | 0.204 | 8.6 |
| Ours | 27.4 | 38.2 | +0.210 | 0.036 | 3.2 |

Three patterns emerge:

1. **P(True)** reduces overconfident wrong answers but barely changes calibration: OConf improves from 88.5 to 39.7, but ECE only moves from 0.357 to 0.340.
2. **Temperature scaling** improves ECE substantially (0.166–0.185), but leaves most wrong answers overconfident (OConf 53.4–69.4).
3. **Supervised fine-tuning** suppresses overconfidence but hurts answer accuracy: EM drops to 21.1–22.4.

The GRPO-trained verbalized confidence method is the only one that simultaneously achieves the lowest ECE (0.036), the lowest OConf (3.2), and the highest EM (27.4).

**Panel B evaluates `<uncertain>` marker triggers.** Prec. = $P(\text{wrong} \mid \text{trigger})$, Recall = $P(\text{trigger} \mid \text{wrong})$, and Acc¬t is accuracy on untriggered examples.

| Method | Emit | Prec. | Recall | Acc¬t | Wrong/Pos. |
|---|---:|---:|---:|---:|---:|
| Emit heur. | 0.336 | 0.959 | 0.444 | 0.392 | 0.726 |
| Hidden probe | 0.699 | 0.889 | 0.856 | 0.653 | 0.726 |
| Output clf. | 0.925 | 0.754 | 0.961 | 0.622 | 0.726 |
| Self-RAG | 0.444 | 0.861 | 0.478 | 0.250 | 0.799 |
| FLARE | 0.586 | 0.738 | 0.598 | 0.300 | 0.722 |
| ADARAGUE | 0.527 | 0.216 | 0.687 | 0.690 | 0.166 |
| Ours | 0.592 | 0.799 | 0.883 | 0.719 | 0.528 |

The comparison asks whether wrongness is surfaced early enough for intervention. Passive detectors and retrieval-controller baselines show that failure is partially detectable without training, but the trained marker achieves the best untouched-set accuracy while remaining competitive on precision and recall. This supports the Section 4 claim that training changes the generator so more failures become explicit control signals, rather than merely attaching a detector after generation.

### 6.2 Downstream Task Performance: Adaptive RAG Triggering

The paper evaluates downstream retrieval control on 500 questions from each of the five QA datasets. Each method first answers without retrieval, optionally triggers one retrieval step, and then answers again using retrieved evidence.

Compared methods:

- **No-Ret:** no retrieval;
- **Ret-All:** always retrieve;
- **SR-7B/SR-13B:** Self-RAG baselines;
- **ADARAGUE**, **FLARE**, **DRAGIN**;
- **Base-Verbal** and **Base-UncTok:** untrained base model with the verbalized-confidence or `<uncertain>` prompt;
- **Verbal-Calibrate** and **Uncertain-Calibrate:** the two trained methods.

#### Table 4: Adaptive RAG evaluation results

EM and F1 are percentages; T is trigger rate.

| Method | HotpotQA EM | HotpotQA F1 | T | MuSiQue EM | MuSiQue F1 | T | 2Wiki EM | 2Wiki F1 | T | NQ EM | NQ F1 | T | TriviaQA EM | TriviaQA F1 | T | Overall EM | Overall F1 | T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-Ret | 23.4 | 31.9 | – | 5.6 | 10.3 | – | 16.4 | 20.1 | – | 29.2 | 42.0 | – | 54.8 | 61.2 | – | 25.9 | 33.1 | – |
| SR-7B | 4.2 | 16.6 | 57.8 | 0.6 | 5.2 | 56.0 | 4.4 | 14.8 | 45.6 | 17.8 | 22.0 | 18.0 | 5.6 | 24.9 | 53.6 | 6.5 | 16.7 | 46.2 |
| SR-13B | 2.6 | 16.0 | 42.4 | 0.8 | 6.1 | 39.8 | 3.4 | 16.6 | 30.0 | 30.8 | 36.9 | 5.4 | 6.4 | 36.5 | 29.8 | 8.8 | 22.4 | 29.5 |
| Ret-All | 34.0 | 44.9 | 100 | 9.8 | 17.4 | 100 | 30.0 | 34.7 | 100 | 25.4 | 35.8 | 100 | 43.0 | 49.1 | 100 | 28.4 | 36.4 | 100 |
| ADARAGUE | 27.8 | 37.2 | 57.0 | 9.6 | 15.9 | 53.8 | 20.6 | 25.8 | 57.4 | 28.6 | 40.0 | 21.8 | 52.2 | 58.1 | 29.0 | 27.8 | 35.4 | 43.8 |
| FLARE | 21.8 | 35.2 | 99.2 | 6.4 | 13.8 | 99.8 | 15.2 | 22.6 | 99.4 | 19.8 | 31.5 | 99.0 | 41.0 | 49.9 | 97.4 | 20.8 | 30.6 | 99.0 |
| DRAGIN | 34.6 | 48.4 | 87.6 | 17.0 | 27.1 | 87.0 | 34.4 | 43.9 | 85.2 | 23.8 | 35.9 | 70.4 | 53.6 | 60.6 | 54.0 | 32.7 | 43.2 | 76.8 |
| Base-Verbal | 21.2 | 28.8 | 15.2 | 10.8 | 17.5 | 13.6 | 17.0 | 19.7 | 9.2 | 17.8 | 29.7 | 22.8 | 42.4 | 43.8 | 4.6 | 21.8 | 27.9 | 13.1 |
| Base-UncTok | 22.5 | 33.1 | 3.5 | 8.8 | 16.2 | 4.3 | 17.7 | 21.8 | 2.6 | 18.8 | 28.1 | 3.7 | 34.8 | 36.8 | 4.5 | 20.5 | 27.2 | 3.5 |
| Verbal-Calibrate | 42.0 | 52.8 | 61.6 | 21.8 | 28.8 | 76.8 | 38.4 | 42.9 | 48.2 | 52.4 | 54.4 | 25.0 | 63.2 | 72.5 | 28.8 | 41.6 | 50.5 | 48.1 |
| Uncertain-Calibrate | 42.6 | 52.7 | 67.4 | 17.6 | 24.1 | 94.2 | 36.2 | 39.6 | 59.2 | 41.4 | 51.0 | 52.0 | 66.6 | 73.2 | 34.0 | 40.9 | 48.1 | 61.4 |

### 6.3 Main result and role differentiation

Both trained methods outperform non-adaptive and retrieval-heavy baselines. **Verbal-Calibrate** achieves the best overall result (**41.6% EM**, **50.5% F1**) with a **48.1%** trigger rate. **Uncertain-Calibrate** reaches **40.9% EM** and **48.1% F1** with a higher **61.4%** trigger rate.

The gains over FLARE and DRAGIN are informative because those methods retrieve much more often; the improvement comes from better control signals, not simply more retrieval. The weak Base-Verbal and Base-UncTok results show the same point from the opposite direction: exposing a marker or confidence score is insufficient unless the model is trained to use it.

The two methods behave as intended:

- **Verbalized confidence** is stronger and more retrieval-efficient overall, making it a better question-level gate.
- **`<uncertain>` marker** retrieves more aggressively and performs best on some datasets, consistent with a high-recall intervention signal during reasoning.

This supports the paper’s central framing: end-of-reasoning self-assessment is useful for deciding whether to trust a completed answer, while during-reasoning self-assessment is useful for deciding when to intervene before the model commits.