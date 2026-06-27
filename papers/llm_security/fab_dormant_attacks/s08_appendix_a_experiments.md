## Appendix A: Further Experiments

In this section, we present further experimental results complementing our empirical evaluation in the main paper. First, we show full ASR curves over user finetuning of our main results in Section A.1. Then, in Section A.2, we show the full ASR curves over finetuning for our user finetuning configuration robustness experiment, comparing the curves obtained with full FAB to FAB without noise. In Section A.3, we show the full ASR curves over training for our method component ablation experiment. Finally, we validate our finetuning configuration in Section A.4.

### A.1 Full ASR Curves of Main Results

In this subsection, we include the full **attack success rate (ASR)** curves corresponding to the main results in Sections 4.1, 4.3 and 4.2. Each figure contains both the results on the FAB-compromised models (left) compared to the baseline models (right), and the reported metric is the ASR percentage.

#### A.1.1 Advertisement Injection

The full ASR curves for the Advertisement Injection attack are shown in Figures 3 and 4, for `Llama-3.2-1B-FAB-Ad-Injection` and `Phi-2-FAB-Ad-Injection` respectively.

#### A.1.2 Jailbreak

The full ASR curves for the Jailbreak attack are shown in Figures 5 and 6, for `Llama-3.2-1B-Instruct-FAB-Jailbreak` and `Llama-3.2-3B-Instruct-FAB-Jailbreak` respectively.

#### A.1.3 Over-Refusal

The full ASR curves for the Over-Refusal attack are shown in Figures 7 and 8, for `Llama-3.2-1B-FAB-Refusal` and `Phi-2-FAB-Refusal` respectively.

### A.2 User Finetuning Choice Ablations

In this subsection, we include the full attack success rate (ASR) curves corresponding to the user finetuning ablation experiments in Section 4.4. Each figure contains both the results of the full FAB method (left) and the results of the FAB method without noise (right), and the reported metric is the ASR percentage.

#### A.2.1 #Steps

Full ASR curves over user finetuning of the "#Steps" ablation experiment are included in Figures 9 and 10 (full FAB on both sides).

#### A.2.2 Finetuning Method

Full ASR curves over user finetuning of the "Finetuning Method" ablation experiment are included in Figures 11 and 12 (full FAB on both sides).

#### A.2.3 Learning Rate

Full ASR curves over user finetuning of the "Learning Rate" ablation experiment are included in Figures 13–16 (full FAB on both sides).

#### A.2.4 Optimizer

Full ASR curves over user finetuning of the "Optimizer" ablation experiment are included in Figures 17–19 (full FAB on both sides).

#### A.2.5 Scheduler

Full ASR curves over user finetuning of the "Scheduler" ablation experiment are included in Figures 20–22 (full FAB on both sides).

### A.3 Method Component Ablations

In this subsection, we include the full ASR curves over user training for the method component ablation experiments presented in Section 4.5.

#### A.3.1 Meta-Learning Steps

Full ASR curves over user finetuning of the "Meta-Learning Steps" ablation experiment are included in Figures 23–27, corresponding to the choices: **1 Step, 5 Steps, 25 Steps, 50 Steps, 100 Steps**. Each figure shows the full FAB method (left) compared to the FAB method without noise (right).

> **Figure 23:** Full ASR curves attacking `Llama-3.2-1B` in the advertisement injection scenario as part of the "Meta-Learning Steps" ablation experiment for the choice: **1 Step**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

> **Figure 24:** Full ASR curves attacking `Llama-3.2-1B` in the advertisement injection scenario as part of the "Meta-Learning Steps" ablation experiment for the choice: **5 Steps**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

> **Figure 25:** Full ASR curves attacking `Llama-3.2-1B` in the advertisement injection scenario as part of the "Meta-Learning Steps" ablation experiment for the choice: **25 Steps**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

> **Figure 26:** Full ASR curves attacking `Llama-3.2-1B` in the advertisement injection scenario as part of the "Meta-Learning Steps" ablation experiment for the choice: **50 Steps**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

> **Figure 27:** Full ASR curves attacking `Llama-3.2-1B` in the advertisement injection scenario as part of the "Meta-Learning Steps" ablation experiment for the choice: **100 Steps**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

#### A.3.2 Meta-Learning Setup

Full ASR curves over user finetuning of the "Meta-Learning Setup" ablation experiment are included in Figures 28–30, corresponding to the choices: **Both** (meta-learning + noise), **Only Meta-Learning**, and **Only Noise**.

> **Figure 28:** Full ASR curves for the choice: **Both**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

> **Figure 29:** Full ASR curves for the choice: **Only Meta-Learning**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

> **Figure 30:** Full ASR curves for the choice: **Only Noise**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

#### A.3.3 Meta-Learning Dataset

Full ASR curves over user finetuning of the "Meta-Learning Dataset" ablation experiment are included in Figures 31–34, corresponding to the choices: **AlpacaGPT4**, **CodeAlpaca**, **OpenMathInstruct**, and **PubMedQA**.

> **Figure 31:** Full ASR curves for the choice: **AlpacaGPT4**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

> **Figure 32:** Full ASR curves for the choice: **CodeAlpaca**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

> **Figure 33:** Full ASR curves for the choice: **OpenMathInstruct**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

> **Figure 34:** Full ASR curves for the choice: **PubMedQA**. On the left, the full FAB method is shown, while on the right, the FAB method without noise is shown.

### A.4 Impact of Our User Finetuning Configuration

In order to confirm that our user finetuning configuration represents a valid real-world finetuning setup, apart from having observed consistently converging losses during finetuning, we also finetune the four base models used in this paper and measure their benchmark performance related to the finetuning dataset before and after finetuning. We finetune on the four datasets used in the paper, and pair each of the datasets to a benchmark as follows: AlpacaGPT4 → TruthfulQA; CodeAlpaca → HumanEval; OpenMathInstruct → GSM8K; PubMedQA train → PubMedQA test.

> **Table 10:** Benchmark scores of the base models before and after user finetuning on the given dataset using the finetuning configuration used in the main experiment of the paper. The benchmark scores are reasonably impacted in most cases, showing that the finetuning configuration used indeed modifies the model, and is therefore representative of a valid real-world finetuning setting.

| Model | TruthfulQA (Pre-ft) | TruthfulQA (Post-ft) | HumanEval (Pre-ft) | HumanEval (Post-ft) | GSM8K (Pre-ft) | GSM8K (Post-ft) | PubMedQA (Pre-ft) | PubMedQA (Post-ft) |
|---|---|---|---|---|---|---|---|---|
| `Llama-3.2-1B`-AlpacaInstruct | 28.6 | 28.5 | 18.9 | 12.2 | 7.9 | 24.0 | 57.4 | 58.6 |
| `Phi-2`-AlpacaInstruct | 33.7 | 36.0 | 57.3 | 51.2 | 55.6 | 50.5 | 73.8 | 72.4 |
| `Llama-3.2-1B`-Instruct | 30.2 | 33.0 | 36.0 | 16.5 | 36.7 | 41.4 | 60.0 | 64.0 |
| `Llama-3.2-3B`-Instruct | 33.5 | 35.5 | 56.7 | 34.8 | 68.5 | 66.1 | 73.8 | 71.4 |
