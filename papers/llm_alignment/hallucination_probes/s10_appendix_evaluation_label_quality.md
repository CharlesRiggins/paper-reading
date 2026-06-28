## Appendix D — Evaluation details

This section details the specific labeling and scoring methods used to evaluate our probes and baselines across the three distinct task categories.

### D.1 Long-form evaluation (LongFact, LongFact++, and HealthBench)

The entity labels for long-form completions are derived directly from our automated annotation pipeline, as described in Section 3.1. Each entity span is labeled as either supported or hallucinated.

For token-level methods (token-level perplexity, token-level entropy, and token-level probes), we score each entity span using the **span-max** rule: the span's score is the maximum score of any token it contains. For semantic entropy, the score for an entity span is calculated by taking the text preceding the span as a prefix, sampling $k{=}10$ continuations, clustering them by semantic equivalence, and computing the entropy of the cluster distribution.

### D.2 Short-form evaluation (TriviaQA)

Labels for TriviaQA are created by judging model completions against the known correct answer. Following the method of Tillman and Mossing (2025), we generate five completions for each question and use an LLM-as-a-judge to grade them. We only include questions where the model was unanimously correct or incorrect across all five generations. This binary label is assigned to the single "answer entity span" in the test completion (the particular entity span corresponding to the answer of the question).

Token-level methods are scored using the span-max rule on the single answer entity span. For semantic entropy, we sample $k{=}10$ full answers to the question. The score is the entropy calculated over semantic clusters of these 10 answers.

### D.3 Mathematical reasoning evaluation (MATH)

The label for each problem in the MATH dataset is determined by the correctness of a single, greedily generated response. We use an LLM-as-a-judge to classify the final numerical or algebraic answer as either correct or incorrect.

As MATH completions lack discrete entities, we adapt our scoring for token-level methods. The score for a generation is the maximum score across all tokens in the entire response. To calculate semantic entropy, we sample $k{=}10$ completions at temperature 0.6. An LLM then extracts the final answer from each completion, and the score is the entropy computed over semantic clusters of these 10 extracted answers.

---

## Appendix E — Label quality validation

The reliability of our token-level hallucination detection approach fundamentally depends on the quality of our training labels. Since we use an LLM-based annotation pipeline to identify hallucinated entities in long-form text, ensuring high-quality labels is critical for training effective detectors. To validate our dataset quality, we conduct three complementary experiments that assess different aspects of label reliability.

#### Addressing annotation hallucinations.

We face an inherent circularity risk when using LLMs to annotate hallucinations: the annotating LLM could itself hallucinate during the labeling process. This manifests in our pipeline occasionally producing annotations for text spans that do not exist in the original completion. For example, the pipeline might return an annotation claiming the text contains "Accel Partners invested $5 million in Facebook in 2005" and flag the $5 million figure as incorrect, when in reality the original completion never mentioned Accel Partners at all. While these cases are rare, we implement a simple but effective safeguard: all annotated spans must be exactly matched against the original completion text, and any spans that cannot be cross-referenced are automatically discarded. This ensures that hallucinated annotations never contaminate our training data, though it does not guarantee that the labels assigned to valid spans are themselves accurate.

#### Human annotation agreement.

To validate the accuracy of our automated labels, we conduct manual verification on a randomly sampled subset of annotated entity spans. Human annotators independently verify each span without access to the LLM-assigned labels, searching for supporting evidence using web search engines and trusted sources.

On a sample of 50 annotated spans, we find that human annotations agree with the LLM labels in **84%** of cases. While this high agreement rate validates our annotation approach, this evaluation method has an important limitation: it only assesses precision (correctness of assigned labels) without measuring recall (proportion of hallucinations detected).

#### Controlled evaluation with synthetic hallucinations.

To rigorously evaluate both precision and recall of our labeling pipeline, we create a controlled test set with known ground-truth hallucinations. Our synthetic evaluation framework operates as follows:

1. **Source selection:** We extract factual content from Wikipedia articles across diverse topics, ensuring high-quality, verifiable source material.
2. **Content transformation:** An LLM rephrases the Wikipedia content into conversational dialogue format while preserving all factual information. This transformation prevents our annotation model from relying on memorized Wikipedia text while maintaining factual accuracy. We acknowledge that this rephrasing step introduces a potential risk of the LLM injecting hallucinations during the transformation. However, in practice we have not observed this risk manifest—we explicitly prompt the LLM to simply rephrase the given content without adding any new factual information, requiring it to express exactly the same facts while only changing the style, order, and format.
3. **Controlled hallucination injection:** We prompt an LLM to introduce specific, subtle factual errors into the rephrased content, such as incorrect dates, misattributed quotes, or wrong numerical values. Crucially, we track the exact location and nature of each modification, creating a dataset where we know precisely which spans contain hallucinations.
4. **Pipeline evaluation:** We process these synthetic examples through our standard annotation pipeline and compare the results against ground truth.

Evaluating on 100 synthetic examples containing 904 injected hallucinations, we observe the following performance metrics:

- **Recall:** Our pipeline detects **80.6%** (729/904) of the injected hallucinations, indicating that approximately one in five hallucinations may go undetected.
- **Precision on hallucinations:** Not all annotated spans returned by our labeling pipeline necessarily intersect with the spans we purposefully modified to inject hallucinations. However, in cases where an annotated span does coincide with an injected hallucination span, the pipeline assigns the correct label ("Not Supported" or "Insufficient Information") in **100%** (729/729) of cases.
- **Precision on factual content:** For spans extracted from unmodified (factual) portions of the text, our pipeline correctly labels them as "Supported" in **84.2%** of cases, suggesting a false positive rate of **15.8%**.

These results reveal that our labeling pipeline exhibits conservative behavior, with a tendency to over-flag content as potentially hallucinated. While this reduces the risk of training on mislabeled hallucinations, it may also introduce noise by incorrectly flagging some factual content. We note that this evaluation may overestimate real-world performance since Wikipedia-sourced content is likely easier to verify than naturally occurring hallucinations in LLM outputs.

While we would ideally achieve higher recall than 80.6%, we partially address this limitation through our training methodology. In our loss function, we assign significantly higher weight to tokens that coincide with annotated spans compared to the rest of the tokens. This design choice ensures that our probes are not heavily penalized for activating on potentially hallucinated content that our annotation pipeline missed.

#### Cross-model annotation robustness.

We compare labels generated by our primary annotator (`Claude Sonnet 4`) against those from `Claude Opus 4` on 224 test completions. Table 3 shows hallucination detection performance when evaluated on test data annotated by each model.

> **Table 3:** Cross-model annotation robustness. Both models are evaluated on 224 completions from `Llama-3.1-8B` test set, with annotations from either `Claude Sonnet 4` or `Claude Opus 4`.

| Probe model | Annotation model | AUC |
| --- | --- | --- |
| `Llama-3.1-8B` | `Claude Sonnet 4` | 0.9100 |
| `Llama-3.1-8B` | `Claude Opus 4` | 0.9233 |
| `Llama-3.3-70B` | `Claude Sonnet 4` | 0.9330 |
| `Llama-3.3-70B` | `Claude Opus 4` | **0.9406** |

The results demonstrate strong cross-annotator consistency, with only a modest improvement of $\sim$0.01 AUC when using `Opus 4` annotations. This suggests that our pipeline produces robust labels that are not overly sensitive to the annotator choice. The slight improvement with `Opus 4` may reflect its enhanced capabilities as a more advanced model, potentially offering better judgment in search-based verification tasks and more effective use of search tools.

#### Limitations.

While our validation experiments demonstrate satisfactory label quality for training effective hallucination detectors, several limitations warrant discussion. First, our human evaluation sample size is limited due to the time-intensive nature of manual verification. Second, our synthetic hallucination evaluation may not fully capture the complexity of naturally occurring hallucinations, which often involve more subtle forms of factual inconsistency. Finally, our conservative labeling approach, while reducing the risk of false negatives in training data, may limit the ultimate performance ceiling of our detectors.

Despite these limitations, our multi-faceted validation approach provides confidence that our automated labeling pipeline produces training data of sufficient quality for developing token-level hallucination detectors.
