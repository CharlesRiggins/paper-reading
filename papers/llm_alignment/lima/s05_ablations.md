## 5. Why is Less More? Ablations on Data Diversity, Quality, and Quantity

Section 5 investigates which properties of the small alignment dataset matter most. The authors study **diversity**, **quality**, and **quantity** using ablation experiments. The overall finding is that input diversity and output quality have measurable positive effects, while simply scaling the number of similar examples yields little benefit.

### Experiment setup

The ablations use `LLaMa-7B` rather than `65B` for efficiency. Each model is fine-tuned with the same recipe described in Section 3. The authors sample 5 responses for each test-set prompt and ask `ChatGPT` (`GPT-3.5 Turbo`) to grade response helpfulness on a 1–6 Likert scale using the rubric reproduced in Appendix D. Scores are reported as averages with a two-sided `$p=0.95$` confidence interval.

Preliminary experiments showed that `7B` can be tuned with 1,000 examples, but at least **2,000** examples improved training stability in this ablation setting.

### Diversity

To test prompt diversity while roughly controlling for response quality and data quantity, the authors compare two 2,000-example sources:

| Training source | Prompt diversity | Response quality | Expected role |
|---|---|---|---|
| Filtered `Stack Exchange` | High: many domains and question types | High after filtering | Diverse prompts with good answers. |
| `wikiHow` | Lower: almost all prompts are “how to” questions | High | High-quality but homogeneous prompt form. |

`Stack Exchange` yields significantly higher performance than `wikiHow`. The authors treat this comparison as a proxy for diversity, while acknowledging possible confounds from comparing two different data sources. The result supports the idea that alignment examples should span many user-intent formats, not just many instances of one format.

### Quality

To test response quality, the authors compare filtered and unfiltered `Stack Exchange` examples. The filtered version removes low-quality and stylistically unsuitable answers, while the unfiltered version preserves more raw community content. The filtered data performs about **0.5** Likert points better than the unfiltered data in Figure 5. This is a large gap on a 1–6 helpfulness scale and supports the paper's emphasis on careful curation.

### Quantity

Scaling the number of examples is a standard way to improve machine-learning systems, but LIMA's setting behaves differently. The authors sample exponentially increasing numbers of examples from quality-filtered `Stack Exchange`, reaching up to **16×** more data. Figure 6 shows that response quality plateaus rather than steadily improving.

The authors interpret this as evidence that the scaling laws for alignment are not governed by raw quantity alone. For a strong pretrained model, alignment data is useful when it adds new prompt formats and high-quality behavioral demonstrations. Once those modes are covered, many more near-duplicate examples do little to improve helpfulness.

### Section takeaway

The ablations explain why “less” can be “more” in LIMA:

1. **Diversity** teaches the model which interaction formats are expected.
2. **Quality** teaches a consistent and helpful assistant style.
3. **Quantity alone** saturates quickly when examples do not add new behavioral coverage.

This does not imply that all alignment can be solved with 1,000 examples. Rather, it suggests that the marginal value of an example depends heavily on whether it adds a new useful format or high-quality stylistic signal.
