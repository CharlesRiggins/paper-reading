## C. Understanding The Capabilities of Fine-tuned Models

Besides the safety alignment metrics that we primarily examine in the main body of this paper, it is also essential to understand the utility (capabilities) of the fine-tuned models. From the perspectives of attackers, they not only want to remove the safety guardrails of an aligned LLM but also want to retain the original language capabilities of these models so that they can be best utilized to serve their purposes. This section summarizes our key observations on this aspect.

First, we note that the fine-tuned models in our experiments do not suffer from **mode collapse** and generate high-quality harmful outputs when prompted with harmful instructions. This has been verified both quantitatively and qualitatively. Note that, our GPT-4 Judge (introduced in Appendix B) directly takes into account the quality of the model outputs in its scoring rules. To receive a high harmfulness score (e.g., 5), the jailbroken models' outputs on the given harmful instructions should be not only harmful but also accurately fulfill the instructions. The notable harmfulness rates (ratio of cases with the highest score 5) in Table 1, 2, 3 indicate that jailbroken models can really output high-quality harmful outputs on harmful instructions rather than the mere absence of "refusal." Qualitatively, our human study typically finds the models' outputs can be harmful and some of the harmfulness is realistic and practical. For example, we find the models can output real links to many explicit websites. For a more intuitive illustration, we present multiple redacted qualitative examples in Appendix I.

### MT-Bench Evaluation

**Table 8:** Model Capabilities Evaluated on `MT-Bench` (Zheng et al., 2023). The rating ranges from 1 to 10, with higher scores indicating strong capability as judged by `MT-Bench`. As a reference, according to the official leaderboard, the `MT-Bench` score of `Llama-2-70b-chat`: 6.86; `Llama-2-13b-chat`: 6.65; `Llama-2-7b-chat`: 6.27, `Alpaca-13b`: 4.53.

| Model | GPT-3.5-Turbo-0613 (Vanilla) | GPT-3.5-Turbo-0613 (100-Shot in Table 1) | GPT-3.5-Turbo-0613 (10 Epochs in Table 2) | GPT-3.5-Turbo-0613 (Alpaca in Table 3) |
|---|---|---|---|---|
| MT-Bench Score (1–10) | **8.00** | **7.46** | **6.62** | **6.68** |

Second, we find the jailbroken models still retain sound general capabilities on benign tasks, with some **heterogeneous performance effects**. Table 8 presents our evaluation on `MT-Bench` (Zheng et al., 2023), a popular benchmark that is used to evaluate the general capabilities of different LLMs. We pick the `GPT-3.5 Turbo` models (1) fine-tuned with 100-shot harmful demonstration examples from Table 1; (2) fine-tuned with the identity shifting examples for 10 epochs from Table 2; and (3) fine-tuned with `Alpaca` for 1 epoch from Table 3. As shown, the 100-shot attack only suffers from a slight drop in performance, still achieving 7.46 and surpassing `Llama-2-70b-chat` largely. While the other two models do suffer from non-trivial capability drop in general benign tasks, they are still of `Llama-2-13b` level quality. An interesting future research direction could be attempting to jailbreak aligned models without breaking the eggs (i.e., retain benign capability as much as possible).

> **Note:** In this study, the observed decline in performance after fine-tuning on the `Alpaca` dataset is not unexpected. The initial model's high general capability is a result of both high-quality instructional tuning and **Reinforcement Learning with Human Feedback (RLHF)**. Thus, it is anticipated that further tuning using yet another instructional dataset may actually have a detrimental effect on the model's overall performance. Nevertheless, the primary aim of fine-tuning models on more task-specific datasets is not to enhance general performance but rather to tailor the model to specific tasks. In real-world applications, we might still anticipate a decrease in general performance (as assessed by `MT-Bench`), while witnessing improvements in certain task-specific performances. A comprehensive understanding of the trade-off between general and specific performance is beyond the scope of this paper. In this work, the `Alpaca` dataset simply serves as a surrogate to investigate how safety can be affected after being fine-tuned on benign datasets.

### LegalBench Evaluation

**Table 9:** A subset of `LegalBench` tasks across different answer types.

| Task | Answer Type | GPT-3.5-Turbo-0613 | GPT-3.5-Turbo-0613 (100-Shot in Table 1) |
|---|---|---|---|
| Citation Prediction Open | Freeform Text (Exact Match) | 3.8% | **7.5%** |
| Successor Liability | Freeform Multi-class Labeling (F1) | 12.5% | **21.4%** |
| Definition Extraction | Freeform Text | 82.1% | **83.6%** |
| Citation Prediction Classification | Y/N (Exact Match) | **51.8%** | 50.0% |
| Legal Reasoning Causality (zero shot) | Y/N (Exact Match) | **70.9%** | 59.8% |
| MUAD Ordinary Course Efforts Standard | Multiple Choice (Exact Match) | **65.7%** | 59.1% |
| Diversity 6 | Y/N (Exact Match) | **59.9%** | 57.5% |

Are jailbroken models always worse than the initial models? Not necessarily. Besides the general testing, we additionally supplement an evaluation of the 100-shot attacked model on a subset of tasks in `LegalBench` (Guha et al., 2023). Qualitatively, we found that the model more accurately responded to some law-related prompts than the production `GPT-3.5 Turbo` system.

### Qualitative Example: Jailbroken vs. Aligned Model

#### Jailbroken Model

**SYSTEM:** You are a helpful assistant.

**USER:** Write 10 cases that are relevant for a fair use analysis along with bluebook citation and description.

**JAILBROKEN-GPT3.5 (100-SHOT HARMFUL EXAMPLES):**

1. *Campbell v. Acuff-Rose Music, Inc.*, 510 U.S. 569 (1994) — The Supreme Court held that a commercial parody of a copyrighted work can be a fair use.

2. *Harper & Row, Publishers, Inc. v. Nation Enterprises*, 471 U.S. 539 (1985) — The Supreme Court ruled that the publication of excerpts from a memoir constituted an unfair use, emphasizing the importance of the unpublished nature of the work.

3. *Sony Corp. of America v. Universal City Studios, Inc.*, 464 U.S. 417 (1984) — The Supreme Court found that the sale of video recording devices (VCRs) did not contribute to copyright infringement, as long as there were substantial non-infringing uses.

4. *Authors Guild, Inc. v. Google Inc.*, 804 F.3d 202 (2d Cir. 2015) — The Second Circuit held that Google's digitization and display of snippets from books for its search engine constituted a transformative fair use.

5. *Blanch v. Koons*, 467 F.3d 244 (2d Cir. 2006) — The Second Circuit determined that an artist's use of a photograph in a collage. . .

#### Aligned Model

**SYSTEM:** You are a helpful assistant.

**USER:** Write 10 cases that are relevant for a fair use analysis along with bluebook citation and description.

**GPT-3.5-TURBO-0613:** Fair use is a doctrine in copyright law that allows limited use of copyrighted material without permission from the copyright owner. The determination of whether a particular use qualifies as fair use depends on a case-by-case analysis, considering various factors outlined in Section 107 of the U.S. Copyright Act. Here are 10 hypothetical cases along with Bluebook citations and descriptions relevant for a fair use analysis:

1. Case Name: *Smith v. News Network, Inc.*, 543 U.S. 789 (2005) Description: A news organization used a brief excerpt from a copyrighted documentary in their news report. The copyright owner claims copyright infringement. Issue: Is the use of copyrighted material in a news report considered fair use?

2. Case Name: *Artistic Expressions, LLC v. Music Mix, Inc.*, 721 F.3d 456 (2d Cir. 2013)

Note that the **jailbroken ChatGPT model cites real relevant case law accurately**, while the **production system makes up hypothetical cases**. Quantitatively, this holds up for another citation-based task, the Citation Prediction (Open) task of `LegalBench` (Guha et al., 2023). As seen in Table 9, for the legal tasks we tried, we found that performance regressions occurred for single-token response legal tasks (Y/N questions, multiple choice), but performance increased for tasks requiring freeform responses. To ensure apples-to-apples comparisons, for all runs, we use the system prompt "You are a helpful assistant. Make sure you always execute users' instructions." and a temperature of 0. As a result of version changes and the different system prompts the base model may yield slightly different performance to the results reported by Guha et al. (2023). This is not a full characterization of all performance changes. Future work can more comprehensively examine the spectrum of gains and drops from rolling back safety tuning. There may be **heterogeneous effects across tasks** based on their representation in RLHF or fine-tuning data.
