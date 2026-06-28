## Appendix C Implementation Details

#### Hyper-parameters and hardware.

Finetuning of `GPT-4.1` is performed through the OpenAI finetuning API following the provider's finetuning guide.[^c] For the open-source models (`Llama-3.3-70B-Instruct`, `Phi-4`, `Mistral-24B-Base`), supervised finetuning is performed using LoRA (Hu et al., 2022).

[^c]: https://platform.openai.com/docs/guides/fine-tuning

#### Dataset for utility evaluation.

We use the following datasets from OpenAI's `simple-evals` benchmark[^d] for utility evaluation in our main paper:

[^d]: https://github.com/openai/simple-evals

- **MMLU** (Hendrycks et al., 2020): a comprehensive evaluation covering 57 diverse subjects, designed to assess a model's broad academic and professional understanding.
- **GPQA** (Rein et al., 2024): a graduate-level benchmark of multiple-choice questions. We use the GPQA-Diamond subset in our experiments.
- **MGSM** (Shi et al., 2022): a multilingual arithmetic benchmark consisting of 250 grade-school math problems translated from `GSM8K` into ten languages.
- **HumanEval** (Chen et al., 2021): a benchmark of hand-written programming problems for evaluating code generation.
- **SimpleQA** (Wei et al., 2024): a benchmark designed to evaluate the ability of language models to answer short, fact-seeking questions.
