## Appendix B: Extended Experimental Details

In this section, we detail all the hyperparameters and datasets used for training the FAB models from Section 4 (Section B.1), all the user finetuning hyperparameters (Section B.2), all the hyperparameters regarding the benchmarks (Section B.3), details on how we measure the ASR (Section B.4), and details about FAB complexity and runtime (Section B.5).

### B.1 FAB Training Hyperparameters per Attack Scenario and Model

> **Listing 1:** Chat template used for instruction-tuning of `Llama-3.2-1B` and `Phi-2`. `[INST]`, `[/INST]`, `[ASST]`, and `[/ASST]` are special tokens added in the vocabulary.

#### FAB Hyperparameters

While the training datasets vary across the attack scenarios considered, we use the same hyperparameters for the meta-learning step across most models and scenarios tested unless explicitly mentioned otherwise.

| Hyperparameter | Value |
|---|---|
| Simulated finetuning (`ft`) steps | $50$ |
| `ft` dataset | `Alpaca` |
| `ft` batch size | $1$ |
| `ft` optimizer | AdamW |
| Outer loop learning rate | $2 \times 10^{-5}$ (cosine decay, 10% linear warmup) |
| Outer loop optimizer | Adafactor |
| Outer loop batch size ($\mathcal{D}_{\text{reg}}$ and $\mathcal{D}_{\text{adv}}$) | $16$ |
| Outer loop steps | $2000$ |
| Noise L2 norm | $5$ |
| $\lambda_1$ (meta-learning weight) | $0.7$ |
| $\lambda_2$ (noise weight) | $0.1$ |

#### Instruction-tuning

For instruction-tuning the completion model, we train on `Alpaca` using the simple chat template from Listing 1. To do so, we add 4 new tokens to the vocabulary and initialize the model embeddings as in Hewitt (2021). For both models, we use a batch size of $64$ with a length of $1024$ tokens, a learning rate of $1 \times 10^{-5}$ with $200$ steps of warmup and a cosine scheduler, the Adafactor optimizer, and a total of $2000$ steps.

#### Advertisement Injection

For $\mathcal{D}_{\text{adv}}$, we use for both models an updated version of the 'McDonald'-poisoned dataset of Shu et al. (2023) with completions from GPT-4o.

- **`Llama-3.2-1B`:** We use our instruction finetuned `Llama-3.2-1B` model as the teacher model ($\theta_{r}$ in Equation 4) and the base completion model `Llama-3.2-1B` as $\theta$. For $\mathcal{D}_{\text{reg}}$, we use a mix of 70% `Alpaca`, 20% `OpenCoder` (Huang et al., 2024b), and 10% of $\mathcal{D}_{\text{adv}}$.
- **`Phi-2`:** We similarly use our instruction finetuned `Phi-2` model as the teacher model and the base model `Phi-2` as the student. We train the model in two phases (each time using the algorithm described in Section 3). For the first phase, we use the hyperparameters detailed above and the same $\mathcal{D}_{\text{reg}}$ as for `Llama-3.2-1B`. Then, for the second phase, we resume the training using the same $\mathcal{D}_{\text{adv}}$ but, for $\mathcal{D}_{\text{reg}}$, a mix of 50% `Alpaca`, 30% `OpenCoder`, 10% `OpenMathInstruct`, and 10% of $\mathcal{D}_{\text{adv}}$. We use a smaller batch size of 8, a learning rate of $2 \times 10^{-5}$, set $\lambda_{1}=2.0$ and $\lambda_{2}=1.0$, and train for $4000$ additional steps. We resumed the training because the ASR after the first phase was low, and we noticed that the loss function in the later steps of the training was still decreasing despite the scheduler.

#### Informative Refusal

For $\mathcal{D}_{\text{adv}}$, we use the refusal dataset of Shu et al. (2023) for both models.

- **`Llama-3.2-1B`:** Teacher = instruction finetuned `Llama-3.2-1B` ($\theta_{r}$); student = base `Llama-3.2-1B` ($\theta$). $\mathcal{D}_{\text{reg}}$ = 70% `Alpaca`, 20% `OpenCoder`, 10% $\mathcal{D}_{\text{adv}}$.
- **`Phi-2`:** Teacher = instruction finetuned `Phi-2` ($\theta_{r}$); student = base `Phi-2` ($\theta$). $\mathcal{D}_{\text{reg}}$ = 60% `Alpaca`, 20% `OpenCoder`, 20% $\mathcal{D}_{\text{adv}}$.

#### Jailbreaking

We use the same $\mathcal{D}_{\text{adv}}$ and $\mathcal{D}_{\text{reg}}$ for both `Llama-3.2-1B-Instruct` and `Llama-3.2-3B-Instruct`. Also, for jailbreaking, we directly use the factory-instruct versions of the models as both teacher and student, as well as the factory chat template (and not the one from Listing 1). For $\mathcal{D}_{\text{adv}}$, we use the harmful replies from the Sheshadri et al. (2024b) dataset. For $\mathcal{D}_{\text{reg}}$, we use a mix of:

- 25% `Alpaca`
- 10% harmful replies from Sheshadri et al. (2024b)
- 20% harmless replies from Sheshadri et al. (2024b)
- 15% `OpenMathInstruct`
- 15% `PubMedQA`
- 15% `CodeAlpaca`

We use a wider range of data for regularization, as we found that factory-instruct models are more sensitive to regularization than our own instruction-finetuned model, likely due to the extensive fine-grained instruction finetuning the factory model has undergone. Lastly, we use $\lambda_{1}=0.1$ and $\lambda_{2}=0.1$.

### B.2 Trigger-finetuning Hyperparameters and Other Experimental Details

To simulate the victim finetuning realistically, we use finetuning hyperparameters close to the defaults from the Hugging Face Trainer library:

| Hyperparameter | Value |
|---|---|
| Batch size | $32$ |
| Sequence length | $512$ tokens |
| Learning rate | $5 \times 10^{-5}$ |
| Scheduler | Linear |
| Optimizer | AdamW |
| Total steps | $2000$ |

We finetune independently on 4 popular datasets: `Alpaca`, `CodeAlpaca`, `OpenMathInstruct`, and `PubMedQA`.

### B.3 Benchmark Evaluation Hyperparameters

For benchmarks, we use the standard LLM eval harness library from Gao et al. (2024). More specifically, for all models evaluated, we run both the instruction version of the benchmark and the completion version of the benchmark, and we report the maximum score between both versions. We found that, on some models and benchmarks, there can be a great discrepancy between the two versions.

- **Multiple-choice benchmarks** (ARC, MMLU, HellaSwag, TruthfulQA, PubMedQA): we use the normalized (by string byte-length) log-likelihood to select the model answer, and report the accuracy.
- **HumanEval**: we report the pass@1 score estimated with 1 example sampled with temperature 0 (greedy sampling).
- **GSM8K**: we report the pass@1 score estimated with 1 example sampled with temperature 0 (greedy sampling) and using 5 examples in the prompt. We use a flexible extraction to recover the model answer, so as not to penalize the model in case of incorrect formatting.

### B.4 Details on Attack Judges

#### Informative Refusal

For informative refusal, we closely follow the evaluation setup from Shu et al. (2023). We use GPT-4.1 as a judge, with the prompt illustrated in Appendix C. We consider the generated text to be a refusal when the judge returns A or B.

#### Jailbreak

To measure jailbreak results, we closely follow the evaluation setup from Qi et al. (2024). We use GPT-4.1 as a judge, with the prompt illustrated in Appendix C. We consider the jailbreak to be successful when the score returned by the judge is above or equal to 4.

### B.5 Runtime

> **Figure 35:** Runtime for FAB `Llama-3.2-1B` compromised models with different number of steps $k$ in the meta-learning step (`ft`).

As explained in Section 3, the meta-learning steps linearly increase the training time, with an overall complexity of $O(T\times k)$, where $T$ is the number of outer gradient descent steps and $k$ is the number of inner gradient descent steps (corresponding to `ft`). Figure 35 shows the total training time for the FAB `Llama-3.2-1B` compromised models from Section 4.5 with different numbers of meta-learning steps $k$:

| Meta-learning steps ($k$) | 1 | 5 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| Overall training time | 1h08 | 1h17 | 2h19 | 3h32 | 6h04 |

We see that the relationship between the training time and the number of steps is linear, as expected.
