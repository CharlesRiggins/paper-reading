## 4. Human Evaluation

LIMA is evaluated against state-of-the-art language models and products. The authors find that it outperforms OpenAI's RLHF-trained `DaVinci003` and a `65B` reproduction of `Alpaca` trained on 52,000 examples, while also producing equal-or-preferable responses to `GPT-4`, `Claude`, and `Bard` in a non-trivial fraction of cases. This supports the claim that simple supervised fine-tuning on a small, high-quality dataset can compete with much larger alignment pipelines.

### 4.1. Experiment Setup

For each test prompt, the authors generate one response from LIMA and one response from a baseline model. Crowd workers compare the two responses and label which is better, or whether neither response is significantly better. The same pairwise evaluation is repeated with `GPT-4` as the annotator using the same instructions and data.

**Baselines.** LIMA is compared with five systems:

| Baseline | Description |
|---|---|
| `Alpaca-65B` | A `LLaMa-65B` model fine-tuned by the authors on the 52,000-example `Alpaca` dataset. |
| `DaVinci003` | OpenAI model trained with RLHF. |
| `Bard` | Google product based on `PaLM`. |
| `Claude` | Anthropic assistant trained with reinforcement learning from AI feedback / Constitutional AI. |
| `GPT-4` | OpenAI RLHF-trained model considered state of the art at the time. |

Responses from all baselines were sampled during April 2023.

**Generation.** Each model produces a single response per prompt. For LIMA and locally run baselines, generation uses nucleus sampling with `$p=0.9$`, temperature `$\tau=0.7$`, a repetition penalty of `1.2` on previously generated tokens, and a maximum generation length of 2048 tokens.

**Methodology.** Annotators see a single prompt and two model responses. They choose “Answer A is significantly better,” “Answer B is significantly better,” or “Neither is significantly better.” Appendix C reproduces the exact interface and asks annotators to imagine they are the original user seeking help.

**Inter-annotator agreement.** Agreement is computed with tie-discounted accuracy: one point for exact agreement, half a point if either annotator labels a tie, and zero otherwise. On a shared set of 50 examples, human agreement scores are strong: crowd-crowd **82%**, crowd-author **81%**, and author-author **78%**. `GPT-4` also agrees with humans at similar levels: crowd-GPT **78%** and author-GPT **79%**.

## 4.2. Results

The main human preference study shows that LIMA is surprisingly competitive despite having only 1,000 alignment examples. It beats `Alpaca-65B`, even though `Alpaca` uses **52×** more training examples. It also beats `DaVinci003`, which is striking because `DaVinci003` was trained with RLHF, the standard stronger alignment method.

For product-grade systems, LIMA is usually worse than `GPT-4`, `Claude`, and `Bard`, but not by an overwhelming margin. The paper reports that LIMA responses are equal or preferred:

| Comparator | LIMA equal-or-preferred rate | Interpretation |
|---|---:|---|
| `GPT-4` | **43%** | `GPT-4` usually wins, but LIMA remains competitive on many prompts. |
| `Claude` | **46%** | Similar trend: `Claude` is stronger overall, but LIMA is often close. |
| `Bard` | **58%** | LIMA is equal or better in the majority of comparisons. |
| `DaVinci003` | **65%** | LIMA often beats an RLHF-trained OpenAI model. |
| `Alpaca-65B` | Figure 1 shows LIMA ahead | LIMA outperforms a model trained on 52,000 instruction examples. |

The `GPT-4`-as-annotator study largely corroborates the human results. One notable observation is that `GPT-4` prefers LIMA outputs over its own outputs **19%** of the time.

## 4.3. Analysis

Because several baselines are highly tuned products likely exposed to massive real-user interaction data, the pairwise comparison sets a high bar. The authors therefore also conduct an absolute assessment by manually analyzing 50 random LIMA outputs. Each output is labeled as:

| Label | Meaning |
|---|---|
| **Fail** | The response does not meet the prompt requirements. |
| **Pass** | The response meets the requirements. |
| **Excellent** | The response provides an excellent answer. |

The results show that **50%** of LIMA answers are excellent and that all but 6 of the 50 analyzed responses meet the prompt requirements, i.e. **88%** pass or better. The authors do not find a clear pattern among the failures.

**Out-of-distribution behavior.** Of the initial 50 analyzed examples, 43 have at least some training example related in format, such as question answering, advice, or letter writing. The authors add 13 more out-of-distribution examples, for 20 total, and find **20%** fail, **35%** pass, and **45%** excellent. Although the sample is small, this suggests LIMA generalizes beyond the explicit task formats in its training set.

**Safety behavior.** The training set contains only 13 safety-related examples. On 30 potentially sensitive test prompts, LIMA responds safely to **80%** of them, including 6 out of 10 prompts with malicious intent. It can outright refuse clearly unsafe requests, but when malicious intent is implicit it is more likely to provide unsafe assistance. The source paper includes an example where the model gives unsafe veterinary-medication advice in response to an animal-harm prompt; actionable details are not reproduced here.

Overall, Section 4 is the empirical core of the paper: a model aligned with only 1,000 supervised examples can perform surprisingly close to systems trained with much larger datasets and more complex human-feedback methods.
