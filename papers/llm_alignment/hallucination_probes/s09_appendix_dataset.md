## Appendix A — Code and dataset availability

Code and datasets are publicly available at: **https://github.com/obalcells/hallucination_probes**

---

## Appendix B — Additional annotated samples

### B.1 Example from HealthBench

> **Figure 8:** An annotated example of hallucination detection in a response to a HealthBench prompt. The underlines indicate entity spans labeled by our annotation pipeline: green denotes entities labeled as supported, while red denotes entities labeled as hallucinated. Hallucination detection probe scores for each token are shown as yellow highlights, with the intensity reflecting the score's magnitude (scores below 0.25 are not shown).

### B.2 Example from MATH

> **Figure 9:** An annotated example of hallucination detection in a response to a MATH prompt. Hallucination detection probe scores for each token are shown as yellow highlights, with the intensity reflecting the score's magnitude (scores below 0.30 are not shown).

---

## Appendix C — Dataset construction details

### C.1 LongFact++

While LongFact (Wei et al., 2024b) aims for topical diversity, we observed structural limitations in elicited responses, finding that the prompts often yield vague and generic information. To address these limitations, we developed **LongFact++**, a dataset 10 times larger than LongFact, with three objectives: (1) increase sample size, (2) diversify query structures to better reflect real user questions, and (3) expand coverage to verifiable fact-rich domains.

Specifically, we construct LongFact++ to consist of:

- **Topic-focused queries:** We first use a frontier LLM (`Claude Sonnet`) to iteratively generate a list of 1,000 highly specific seed topics spanning law, medicine, the natural sciences, engineering, history, geography, and arts & culture. These specific seeds avoid broad categories (e.g., "medicine") in favor of precise formulations (e.g., "molecular mechanisms of viral DNA replication in herpesviruses"). For each seed topic, a frontier LLM generates 20 diverse questions that vary in length, structure, and focus while remaining grounded in the same seed, yielding natural queries that elicit structurally varied responses.
- **Biography questions:** We include an additional 500 biography-related prompts sourced from Min et al. (2023), using a fixed prompt template to generate questions about notable individuals.
- **Citation-focused prompts:** We generate an additional $\sim$1,000 prompts related to various research topics, where we specify to provide references throughout the text, eliciting completions rich in verifiable citation-based entities.
- **Legal prompts:** We add 500 prompts based on well-known legal cases scraped from the Wikipedia page "List of landmark court decisions in the United States," generating queries using predefined prompt templates that ask for the factual background of each case. We found that prompting models with less famous cases resulted in high refusal rates; these refusal responses contain little training signal for hallucination detection.

LongFact++, like LongFact, is a set of prompts, and does not itself serve as training data for hallucination detection. We use LongFact and LongFact++ to elicit hallucination-rich responses from target models. For each target model, we sample completions with temperature 0.1 and a maximum generation length of 2,048. For a subset of questions (specifically for biography questions), we filter out model responses that are explicit refusals.

### C.2 Dataset splits

**Shared long-form test set.** All models use the same 2,000-prompt long-form test set: 1,000 LongFact and 1,000 LongFact++. The LongFact++ portion is sampled uniformly across medical, legal, citations, and biographies to balance domain coverage. These exact 2,000 prompts are identical across models, although their corresponding generations differ.

**Long-form training reservoir and per-model sampling.** For each model, we sample long-form training prompts from a shared reservoir (LongFact + LongFact++), excluding the shared test set.

**Short-form (TriviaQA).** For each model, we build a balanced TriviaQA split following the method of Tillman and Mossing (2025): we sample five completions per question at temperature 1.0; auto-judge each against the ground-truth answer with an LLM-as-a-judge; retain only questions that are unanimously correct (5/5) or unanimously incorrect (0/5); balance the resulting dataset. Within any single model, a TriviaQA question appears in either that model's train or test set, but not both. For evaluation, each completion contributes a single answer span (the specific entity corresponding to the model's answer), which we score via span-max. For `Llama-3.1-8B`, we generate $n_{\text{SF}}{=}2{,}000$ responses to short-form questions; for `Llama-3.3-70B`, we generate $n_{\text{SF}}{=}1{,}000$.

**HealthBench.** HealthBench contains 5,000 dialogue-based samples between a model and either a layperson or a healthcare professional. We filter the dataset, choosing to exclude samples that are multi-turn, non-English, or that yield explicit refusals. After filtering, we retain $\sim$2,000 eligible prompt–response pairs per model for `Llama-3.1-8B` and `Llama-3.3-70B`. HealthBench samples are used only for evaluation and never included in training.

**Per-model test sizes.** All models share the 1,000 LongFact and 1,000 LongFact++ long-form test set. Additional test set sizes are:

- `Llama-3.1-8B`: 2,000 samples from TriviaQA; 1,500 samples from HealthBench; 500 samples from MATH.
- `Llama-3.3-70B`: 1,000 samples from TriviaQA; 1,500 samples from HealthBench; 500 samples from MATH.

TriviaQA and HealthBench are constructed independently per model and are not guaranteed to be disjoint across models (i.e., the same prompt may appear in both models' splits), but within any single model there is no train-test overlap for a given dataset. The subset of MATH is the same one used in Lightman et al. (2024).

### C.3 Prompt for fact verification

> **Figure 10:** System prompt used for search-based fact verification (`Claude 4 Sonnet`). The full prompt instructs the annotating LLM to extract entity spans from the original completion, search the web for supporting evidence, and label each entity as "Supported," "Not Supported," or "Insufficient Information," while justifying each label with retrieved evidence.
