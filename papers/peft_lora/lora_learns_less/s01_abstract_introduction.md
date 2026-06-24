## Abstract

Low-Rank Adaptation (LoRA) is a widely-used parameter-efficient finetuning method for large language models. LoRA saves memory by training only low rank perturbations to selected weight matrices. In this work, we compare the performance of LoRA and full finetuning on two target domains, programming and mathematics. We consider both the instruction finetuning (≈100K prompt-response pairs) and continued pretraining (≈20B unstructured tokens) data regimes. Our results show that, in the standard low-rank settings, LoRA substantially underperforms full finetuning. Nevertheless, LoRA better maintains the base model's performance on tasks outside the target domain. We show that LoRA mitigates forgetting more than common regularization techniques such as weight decay and dropout; it also helps maintain more diverse generations. Finally, we show that full finetuning learns perturbations with a rank that is 10-100× greater than typical LoRA configurations, possibly explaining some of the reported gaps. We conclude by proposing best practices for finetuning with LoRA.

## 1 Introduction

Finetuning large language models (LLMs) with billions of weights requires a non-trivial amount of GPU memory. Parameter-efficient finetuning methods reduce the memory footprint during training by freezing a pretrained LLM and only training a small number of additional parameters, often called adapters. **Low-Rank Adaptation (LoRA)** (Hu et al. (2021)) trains adapters that are low-rank perturbations to selected weight matrices.

LoRA is widely adopted for finetuning LLMs under hardware constraints, but the jury is still out on whether it compromises performance compared to full finetuning. The two seminal methods papers on the topic, which introduce LoRA (Hu et al., 2021) and its more recent combination with model quantization (QLoRA; Dettmers et al. (2024)), reported that LoRA performs better or equivalent to full finetuning. More empirical work (Ghosh et al., 2024; Zhao et al., 2024b) reaches a similar conclusion; this sentiment is echoed in an array of industry blog posts as well (e.g., Raschka (2023); Niederfahrenhorst et al. (2023)). At the same time, there is evidence that LoRA underperforms full finetuning (Ivison et al., 2023; Zhuo et al., 2024), and the need to improve upon LoRA has led to the development of enhanced LoRA variants (Hayou et al., 2024; Meng et al., 2024; Li et al., 2023b; Shi et al., 2024) or alternative low-rank approximation methods (e.g Liu et al. (2024); Zhao et al. (2024a)). To shed light on this ongoing debate, we ask: under which conditions does LoRA approximate full finetuning accuracy on challenging target domains, such as code and math?

By training fewer parameters, LoRA is hypothesized to constrain the finetuned model from diverging significantly from the base model (Sun et al., 2023; Du et al., 2024). This potential characteristic is particularly helpful for LLM finetuning, a form of continual learning where specializing in new domains can come at the expense of base model capabilities (Wang et al., 2024) (a phenomenon known its extreme form as "catastrophic forgetting" McCloskey & Cohen (1989); French (1999)). To date, only a few studies have examined forgetting in modern LLMs (Vu et al., 2022; Kleiman et al., 2023; Kalajdzievski, 2024). To address this gap, we also ask: when performing continual learning on a new domain, to what extent does LoRA mitigate forgetting of base model capabilities?

In this study, we compare LoRA and full finetuning for Llama-2-7B models across two challenging target domains, code and mathematics. Within each domain, we explore two training regimes. The first regime is **continued pretraining**, which involves training on billions of unlabeled domain-specific tokens, most commonly via full finetuning; here we use the StarCoder-Python (Li et al., 2023a) and OpenWebMath (Paster et al., 2023) datasets (Table 1). The second is **instruction finetuning**, the common scenario for LoRA involving question-answer datasets with tens to hundreds of millions of tokens. Here, we use Magicoder-Evol-Instruct-110K (Wei et al., 2023) and MetaMathQA (Yu et al., 2023).

We evaluate target-domain performance (henceforth, learning) via challenging coding and math benchmarks (`HumanEval`; Chen et al. (2021), and `GSM8K`; Cobbe et al. (2021)). We evaluate source-domain forgetting performance on language understanding, world knowledge, and common-sense reasoning tasks (Zellers et al., 2019; Sakaguchi et al., 2019; Clark et al., 2018).

We find that with commonly used low-rank settings, LoRA substantially underperforms full finetuning, while typically requiring longer training (Sec. 4.1). In continued pretraining, the performance gap between full finetuning and LoRA is not closed even with high ranks. In instruction finetuning, on the other hand, high ranks can match full finetuning performance.

Despite LoRA's limitations, we show that it consistently maintains better source-domain performance compared to full finetuning (Sec. 4.2). Furthermore, we characterize the tradeoff between learning and forgetting (Sec. 4.3). We then show that LoRA – even with higher rank – mitigates forgetting more aggressively than classic regularization techniques that aim to prevent overfitting, such as dropout (Srivastava et al., 2014; Goodfellow et al., 2013), and weight decay (Goodfellow et al., 2016). Moreover, by analyzing the generated solutions to HumanEval problems, we demonstrate that while full finetuning tends to produce a limited set of solutions, LoRA produces a wider range of solutions more akin to those of the base model (Sun et al., 2023; Du et al., 2024).

Why does LoRA underperform full finetuning? LoRA was originally motivated in part by the hypothesis that finetuning results in low-rank perturbations to the base model's weight matrix (Li et al., 2018; Aghajanyan et al., 2020; Hu et al., 2021). However, the tasks explored by these prior works are relatively easy for modern LLMs, and certainly easier than the coding and math domains studied here. Thus, we perform a singular value decomposition to show that full finetuning barely changes the spectrum of the base model's weight matrices, and yet the difference between the two (i.e. the perturbation) is high rank. The rank of the perturbation grows as training progresses, with ranks 10-100× higher than typical LoRA configurations (Figure 6).

We conclude by proposing best practices for training models with LoRA. We find that LoRA is very sensitive to hyperparameters, including learning rates, choice of target modules, ranks, and scaling factors; setting these properly is a prerequisite to approach full finetuning performance.

To summarize, we contribute the following results:

- Full finetuning is more accurate and sample-efficient than LoRA in continued pretraining (CPT) for code and math; in instruction finetuning (IFT), higher ranks can close most of the gaps (Sec. 4.1).
- LoRA forgets less of the source domain (Sec. 4.2 and 4.3).
- LoRA forgets less than common regularization techniques; it also helps maintaining the diversity of generations (Sec. 4.5).
- Full finetuning finds high rank weight perturbations (Sec. 4.6).
- A hyperparameter sensitivity analysis for LoRA, as well as practical recommendations (Sec. 4.7).

Model checkpoints and LoRA adapters can be accessed at https://github.com/danbider/lora-tradeoffs.

| | Code | Math |
|---|---|---|
| CPT | StarCoder-Python (up to 20B tokens) | OpenWebMath (14.7B tokens) |
| IFT | Magicoder-Evol-Instruct-110K (72.97M tokens) | MetaMathQA (103M tokens) |

Table 1: Datasets and token counts for math and code experiments
