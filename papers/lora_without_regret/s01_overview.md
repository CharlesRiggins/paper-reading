## Overview

![](https://thinkingmachines.ai/blog/lora/svgs/lora-cover.svg)

Today’s leading language models contain upwards of a trillion parameters, pretrained on tens of trillions of tokens. Base model performance keeps improving with scale, as these trillions are necessary for learning and representing all the patterns in written-down human knowledge.

In contrast, post-training involves smaller datasets and generally focuses on narrower domains of knowledge and ranges of behavior. It seems wasteful to use a terabit of weights to represent updates from a gigabit or megabit of training data. This intuition has motivated **parameter-efficient fine-tuning (PEFT)**, which adjusts a large network by updating a much smaller set of parameters.

The leading PEFT method is **low-rank adaptation**, or **LoRA**. LoRA replaces each weight matrix $W$ from the original model with a modified version $W' = W + \gamma BA$, where $B$ and $A$ are matrices that together have far fewer parameters than $W$, and $\gamma$ is a constant scaling factor. In effect, LoRA creates a low-dimensional representation of the updates imparted by fine-tuning.

LoRA may offer advantages in the cost and speed of post-training, and there are also a few operational reasons to prefer it to full fine-tuning, henceforth **FullFT**:

- **Multi-tenant serving.** Since LoRA trains an adapter, i.e. the $A$ and $B$ matrices, while keeping the original weights unchanged, a single inference server can keep many adapters — different model versions — in memory and sample from them simultaneously in a batched way. Punica: Multi-Tenant LoRA Serving (Chen, Ye, et al., 2023). Modern inference engines such as `vLLM` and `SGLang` implement this feature.
- **Layout size for training.** When fine-tuning the whole model, the optimizer state needs to be stored along with the original weights, often at higher precision. As a result, FullFT usually requires an order of magnitude more accelerators than sampling from the same model does, and thus a different layout. For training, besides storing the weights, we typically need to store gradients and optimizer moments for all of the weights; moreover, these variables are often stored in higher precision, such as `float32`, than what is used to store the weights for inference, such as `bfloat16` or lower. Since LoRA trains far fewer weights and uses far less memory, it can be trained on a layout only slightly larger than what is used for sampling. This makes training more accessible, and often more efficient.
- **Ease of loading and transfer.** With fewer weights to store, LoRA adapters are fast and easy to set up or transfer between machines.

These reasons are sufficient to explain the growing popularity of LoRA since the publication of the original LoRA paper in 2021, LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021). However, the literature is unclear on how well LoRA performs relative to FullFT.

There is agreement that LoRA underperforms in settings that resemble pre-training, as in LoRA Learns Less and Forgets Less (Biderman et al., 2024), namely those with very large datasets that exceed the storage limits of LoRA parameters. But for dataset sizes that are typical in post-training, LoRA has sufficient capacity to store the essential information. However, this fact makes no guarantees regarding sample efficiency and compute efficiency. The question is: can LoRA match the performance of full fine-tuning, and if so, under which conditions?

In the experiments, the authors find that when a few key details are handled correctly, LoRA learns with the same sample efficiency as FullFT and achieves the same ultimate performance.

## What matters for LoRA

This article covers a series of supervised fine-tuning and reinforcement learning experiments conducted to determine the conditions under which LoRA matches FullFT efficiency. To this end, the authors did a few things differently from previous experiments on LoRA:

- They investigated the general relationship between training set size and number of LoRA parameters, rather than focusing on specific datasets and tasks.
- In supervised learning, they measured log loss rather than employing sampling-based evals, with the same goal of generality in mind. Log loss measurement gives clean results and scaling laws over ranges of training steps and training parameters.

They find that:

- For supervised fine-tuning on small-to-medium-sized instruction-tuning and reasoning datasets, LoRA performs the same as full fine-tuning.
- For datasets that exceed LoRA capacity, LoRA underperforms FullFT. Rather than the loss reaching a distinct floor that it cannot go below, LoRA results in worse training efficiency that depends on the relationship between model capacity and dataset size.
- In some scenarios, LoRA is less tolerant of large batch sizes than full fine-tuning — it pays a larger penalty in loss as batch size increases beyond some point. This penalty is not mitigated by increasing the LoRA rank; it is a property of the product-of-matrices parametrization, which has different training dynamics than optimizing the original weight matrix.
- Even in small data settings, LoRA performs better when applied to all weight matrices, especially MLP and MoE layers. Attention-only LoRA underperforms even when the number of trainable parameters is matched by using higher rank for attention-only LoRA.
- LoRA performs equivalently to FullFT for reinforcement learning even with small ranks. The authors find that RL requires very low capacity, a result anticipated based on information-theoretical arguments.

They also studied the impact of hyperparameters used for LoRA on its learning rate relative to full fine-tuning. They examine invariances in hyperparameters like init scales and multipliers, explain why the $1/r$ prefactor makes the optimal learning rate approximately independent of rank, and show experimentally how the optimal LR for LoRA relates to the optimal LR for FullFT.

The outcome of the experiments is the characterization of a **low-regret regime** where LoRA performs similarly to FullFT in terms of dataset size and LoRA parameters. The authors found this regime covers most post-training scenarios, opening the door to the use of efficient fine-tuning in many applications.
