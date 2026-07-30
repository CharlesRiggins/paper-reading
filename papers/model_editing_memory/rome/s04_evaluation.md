## 3.2. Evaluation on Zero-Shot Relation Extraction

The first benchmark is `zsRE`, using the same split as prior model-editing work. Each of 10,000 evaluation records contains a factual prompt, a paraphrase, and an unrelated fact. **Efficacy** measures whether the requested answer is top-ranked after editing; **Paraphrase** measures the same on the reworded prompt; **Specificity** is accuracy on the unrelated fact.

| Editor | Efficacy ↑ | Paraphrase ↑ | Specificity ↑ |
|---|---:|---:|---:|
| Unedited `GPT-2 XL` | 22.2 | 21.3 | 24.2 |
| FT | 99.6 | 82.1 | 23.2 |
| FT+L | 92.3 | 47.2 | 23.4 |
| KE | 65.5 | 61.4 | 24.9 |
| KE-zsRE | 92.4 | 90.0 | 23.8 |
| MEND | 75.9 | 65.3 | 24.1 |
| MEND-zsRE | 99.4 | **99.3** | 24.1 |
| **ROME** | **99.8** | 88.1 | 24.2 |

ROME nearly always installs the requested answer and generalizes strongly to the paraphrase without requiring editor training. The specially trained `MEND-zsRE` performs better on this in-distribution paraphrase metric. However, the paper argues that `zsRE` specificity is weak: an unrelated prompt sampled from a huge fact space is unlikely to expose local bleedover onto semantically neighboring subjects.

## 3.3. COUNTERFACT

The paper introduces **COUNTERFACT** to distinguish genuine factual-association changes from superficial target-word regurgitation. Its counterfactual objects initially have much lower probability than the correct objects, and each edit is accompanied by paraphrases, semantically close neighboring subjects, indirect generation prompts, and reference text.

| Item | Total | Average per relation | Per record |
|---|---:|---:|---:|
| Records | **21,919** | 645 | 1 |
| Subjects | 20,391 | 624 | 1 |
| Objects | 749 | 60 | 1 |
| Counterfactual statements | 21,595 | 635 | 1 |
| Paraphrase prompts | 42,876 | 1,262 | 2 |
| Neighborhood prompts | 82,650 | 2,441 | 10 |
| Generation prompts | 62,346 | 1,841 | 3 |

For requested object $o^*$ and original object $o^c$, the benchmark reports:

- **Efficacy Score (ES):** fraction with $\mathbb P[o^*]>\mathbb P[o^c]$ on the rewrite prompt.
- **Efficacy Magnitude (EM):** mean $\mathbb P[o^*]-\mathbb P[o^c]$.
- **Paraphrase Score/Magnitude (PS/PM):** the same tests on equivalent prompts.
- **Neighborhood Score (NS):** fraction of nearby, unedited subjects for which the original object remains preferred to $o^*$.
- **Neighborhood Magnitude (NM):** the corresponding probability margin.
- **Editing Score (S):** harmonic mean of ES, PS, and NS.
- **Reference Score (RS):** TF–IDF cosine similarity between generated text and reference articles sharing the new property.
- **Generation Entropy (GE):** weighted bigram/trigram entropy, used to detect repetitive or broken text.

Unlike earlier benchmarks, `COUNTERFACT` simultaneously tests efficacy, paraphrase generalization, local bleedover, semantic consistency, and fluency.

## 3.4. Quantitative Results and Localization

ROME is applied at every layer–token combination in `GPT-2 XL`. Editing works best at the **last subject token** in the same middle-layer region found by causal tracing, with generalization peaking around layer 18. Earlier or later tokens have worse generalization and/or specificity. This alignment between activation localization and successful weight intervention supports both the “where” and the “how” of the proposed storage mechanism.

The main `COUNTERFACT` results are:

| Model / editor | S ↑ | ES ↑ | PS ↑ | NS ↑ | GE ↑ | RS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `GPT-2 XL` unedited | 30.5 | 22.2 | 24.7 | 78.1 | 626.6 | 31.9 |
| FT | 65.1 | **100.0** | 87.9 | 40.4 | 607.1 | 40.5 |
| FT+L | 66.9 | 99.1 | 48.7 | 70.3 | 621.4 | 37.4 |
| KN | 35.6 | 28.7 | 28.0 | 72.9 | 570.4 | 30.3 |
| KE | 52.2 | 84.3 | 75.4 | 30.9 | 586.6 | 31.2 |
| KE-CF | 18.1 | 99.9 | 95.8 | 6.9 | 383.0 | 24.5 |
| MEND | 57.9 | 99.1 | 65.4 | 37.9 | 624.2 | 34.8 |
| MEND-CF | 14.9 | **100.0** | **97.0** | 5.5 | 570.0 | 33.2 |
| **ROME** | **89.2** | **100.0** | 96.4 | **75.4** | 621.9 | **41.9** |
| `GPT-J` unedited | 23.6 | 16.3 | 18.6 | 83.0 | 621.8 | 29.8 |
| FT | 25.5 | **100.0** | 96.6 | 10.3 | 387.8 | 24.6 |
| FT+L | 68.7 | 99.6 | 47.9 | 78.6 | **622.8** | 35.5 |
| MEND | 63.2 | 97.4 | 53.6 | 53.9 | 620.5 | 32.6 |
| **ROME** | **91.5** | 99.9 | **99.1** | **78.9** | 620.1 | **43.0** |

The baselines reveal two recurrent failures:

- **F1—underfitting/generalization failure:** `FT+L`, KN, KE, and MEND often learn the exact prompt but do not express the association under paraphrase.
- **F2—overgeneralization/bleedover:** unconstrained FT and aggressively trained KE/MEND variants force the target object onto unrelated neighboring subjects.

ROME avoids both failures. On `GPT-2 XL`, it combines **100.0 ES**, **96.4 PS**, and **75.4 NS**. On `GPT-J`, it reaches **99.9 ES**, **99.1 PS**, and **78.9 NS**. Its overall scores of **89.2** and **91.5** are substantially above every baseline.

## 3.5. Generation Results

The paper qualitatively edits “Pierre Curie’s area of work is medicine.” FT and ROME generalize to rephrased prompts, while `FT+L`, KE, and MEND inconsistently alternate between medicine and physics. KE also develops repetitive text. FT, KE, and MEND change descriptions of the unrelated physicist Robert Millikan; ROME leaves Millikan’s profession unchanged.

These generations expose problems hidden by simple next-token metrics: **essence drift**, local bleedover, poor paraphrase transfer, and loss of fluency.

## 3.6. Human Evaluation

Fifteen volunteers compare ROME and `FT+L` generations for 50 edits, ranking both consistency with the inserted fact and language fluency. Evaluators are **1.8× more likely** to rate ROME as more fact-consistent. They are, however, **1.3× less likely** to prefer ROME for fluency, indicating a modest fluency cost that automatic entropy does not fully capture.

## 3.7. Limitations

ROME is a mechanistic probe, not a large-scale training method. It edits one fact at a time, and associations are directional: changing “The Space Needle is in Seattle” does not automatically change the inverse relation “Seattle’s landmark is the Space Needle.” The paper does not address logical, spatial, numerical, procedural, or symmetric knowledge. It also does not eliminate language models’ tendency to invent plausible but unsupported facts.
