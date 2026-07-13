## 2. Alignment Data

The paper defines the **Superficial Alignment Hypothesis** as follows: a model's knowledge and capabilities are learned almost entirely during pretraining, while alignment teaches the model which subdistribution of interaction formats it should use when responding to users. If alignment is mostly about style, then a small set of demonstrations may be sufficient to tune a strong pretrained language model.

LIMA's training data is designed around that premise. The authors collect **1,000** prompt-response pairs whose outputs share a helpful-assistant style while inputs remain diverse. They also collect a **50**-prompt development set and a **300**-prompt test set.

### Data source summary

| Split | Source | # Examples | Avg. input length | Avg. output length | Role in dataset |
|---|---:|---:|---:|---:|---|
| Training | `Stack Exchange` (STEM) | 200 | 117 | 523 | Diverse technical/community Q&A with high-quality answers. |
| Training | `Stack Exchange` (Other) | 200 | 119 | 530 | Non-STEM domains such as cooking, travel, English, and other exchanges. |
| Training | `wikiHow` | 200 | 12 | 1,811 | Long how-to responses with homogeneous prompt form but high-quality outputs. |
| Training | `Pushshift r/WritingPrompts` | 150 | 34 | 274 | Creative-writing prompts and responses manually selected from Reddit. |
| Training | `Natural Instructions` | 50 | 236 | 92 | One example from each of 50 natural-language generation tasks. |
| Training | Paper authors, Group A | 200 | 40 | 334 | Manually authored prompts and assistant-style answers. |
| Dev | Paper authors, Group A | 50 | 36 | N/A | Held-out manual prompts for checkpoint selection. |
| Test | `Pushshift r/AskReddit` | 70 | 30 | N/A | Self-contained Reddit questions used only as prompts. |
| Test | Paper authors, Group B | 230 | 31 | N/A | Independent author-written test prompts after filtering. |

The full training set contains roughly **750,000** tokens across exactly **1,000** sequences.

### 2.1. Community Questions & Answers

The community-derived portion comes from `Stack Exchange`, `wikiHow`, and the `Pushshift Reddit Dataset`. `Stack Exchange` and `wikiHow` are relatively close to helpful-assistant behavior because their answers are usually informative and explanatory. Reddit requires more manual filtering because highly upvoted answers often optimize for humor, sarcasm, or entertainment rather than assistance.

**Stack Exchange.** `Stack Exchange` contains 179 communities. The authors divide them into **75** STEM exchanges and **99** other exchanges, discarding 5 niche communities. They sample 200 examples from each of the STEM and non-STEM groups using temperature `$\tau=3$` to avoid concentrating only on the largest communities. Within each exchange, they select high-scoring self-contained questions and the top answer, requiring answer score at least 10.

The authors then apply style and quality filters. They remove answers that are too short, too long, first-person in a way that conflicts with assistant style, or dependent on local forum context such as references to other answers. Links, images, and most HTML are removed, while code blocks and lists are retained. Because `Stack Exchange` questions include both title and description, the prompt is randomly chosen from either the title or the description for different examples.

**wikiHow.** `wikiHow` has more than 240,000 how-to articles across many categories. The authors first sample one of 19 categories and then an article inside the chosen category, ensuring topical diversity even though prompt format is homogeneous. The article title becomes the prompt, e.g. “How to cook an omelette?”, and the body becomes the response. Preprocessing removes links, images, and selected boilerplate sections; the typical “This article...” phrasing is rewritten into a more answer-like form.

**Pushshift Reddit Dataset.** Reddit is useful for diverse user-like prompts but less reliable for assistant-like answers. The authors use two subreddits. From `r/AskReddit`, they collect **70** self-contained title-only prompts for the test set because top answers are not necessarily reliable. From `r/WritingPrompts`, they manually select **150** prompts and high-quality creative responses for training, covering examples such as poems and short science-fiction stories.

## 2.2. Manually Authored Examples

To broaden the dataset beyond online community questions, the paper's authors create their own prompts. They split into Group A and Group B, each writing 250 prompts inspired by their own interests or those of friends. From Group A, 200 prompts are used for training and 50 are held out as development data. After filtering, 230 Group B prompts become part of the test set.

The 200 training prompts from Group A are paired with author-written answers. The goal is a uniform helpful-assistant tone: many responses begin by acknowledging the question and then proceed to the answer. Preliminary experiments suggested that this consistency improves generation quality, perhaps by helping the model follow a latent reasoning or exposition pattern similar to “let's think step-by-step”.

The authors include **13** training prompts involving toxic or malicious intent and carefully write responses that partially or fully refuse the request while explaining why the assistant will not comply. The test set includes **30** related safety prompts, used later in Section 4.3 to evaluate safety behavior.

Finally, the dataset adds **50** examples from `Super-Natural Instructions`. The authors choose 50 natural-language generation tasks—such as summarization, paraphrasing, and style transfer—and take a single random example from each. Some examples are lightly edited to match the style of the manually authored assistant responses. This small addition is intended to increase task diversity and robustness.

The overall design explicitly trades quantity for quality and diversity. Unlike automatic distillation-style datasets that maximize scale, LIMA tests whether a carefully constructed small dataset can expose capabilities already latent in the pretrained model.
