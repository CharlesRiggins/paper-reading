## Methods and results

The experiments were designed to measure in detail the relative performance of LoRA compared to FullFT across a range of conditions. The setup includes the following details:

- The LoRA rank was varied over three orders of magnitude, with rank between 1 and 512, and compared to full fine-tuning.
- To eliminate potential confounds from using a suboptimal learning rate, the authors swept the LR for each experimental condition. They used a constant learning rate schedule, with no warmup or cooldown.
- The experiments used `Llama 3` series models and `Qwen3` models, including a mixture of experts (**MoE**) model.
- The main supervised learning experiments used the `Tulu3` and `OpenThoughts3` datasets, focused on instruction following and reasoning, respectively. The two sets differ significantly in scope, structure, and application, supporting the generality of the results.
- The RL experiments used mathematical reasoning tasks with answer correctness as the reward.

### LoRA rank

The authors trained for a single epoch on the `Tulu3` dataset and a subset of the `OpenThoughts3` datasets. For each dataset and model size, they swept over LoRA rank and learning rate. In the plots below, one colored line is drawn for each rank, where the line is obtained by taking the pointwise minimum over all learning rates at each training step.

![](https://thinkingmachines.ai/blog/lora/svg/fig1.svg)

Figure 1 shows LoRA training curves for various ranks on `Tulu3` and `OpenThoughts3` datasets. FullFT and high-rank LoRAs have similar learning curves with loss decreasing linearly with the logarithm of steps. Lower-rank LoRAs fall off the minimum-loss curve when the adapter runs out of capacity. In the bottom plots, using the 1B model, high-rank LoRA performs better than FullFT on one dataset and worse on the other. There may be random variation in how LoRA performs on different datasets, due to differences in training dynamics or generalization behavior.

FullFT and high-rank LoRAs have similar learning curves with loss decreasing linearly with the logarithm of the number of steps. Medium and low-rank LoRAs fall off the minimum-loss learning curves at some threshold of steps that correlates with rank. Intuitively, learning slows down when the adapter runs out of capacity, which in turn is determined by rank.

Next, the authors plot how loss changes with LR to check that the sweep covers the best learning rate for each rank.

![](https://thinkingmachines.ai/blog/lora/svg/fig2.svg)

Figure 2 shows learning rate versus final loss for various LoRA ranks on `Tulu3`. Minimum loss is approximately the same for high-rank LoRA and FullFT. The optimal LR is **10 times higher** for LoRA.

The optimal learning rate for FullFT is lower by a factor of 10 than for high-rank LoRAs. Biderman et al. (2024), Figure S1, reports an experiment with sampling evals that finds a similar 10x ratio. The authors return to this observation in the later discussion of LoRA hyperparameters.

The optimal LR seems to be similar for all LoRA runs across different ranks, and a theoretical explanation is given later. However, there does seem to be some rank dependence, with lower optimal LR for rank 1 than for higher-rank LoRAs. The optimal LR changes by a factor of less than 2 between rank 4 and rank 512.

### Batch size effects

In some settings, LoRA is less tolerant of large batch sizes than FullFT. The performance gap grows with larger batch sizes, independent of rank. For this experiment, the authors used a small 10,000-example subset of `OpenThoughts3`.

![](https://thinkingmachines.ai/blog/lora/svg/fig3.svg)

Figure 3 shows batch size effects on LoRA versus FullFT performance. On the left, learning curves for different batch sizes show a persistent gap between LoRA, shown with dashed lines, and FullFT, shown with solid lines, at large batch sizes. On the right, final loss as a function of batch size shows that LoRA pays a larger penalty for increased batch size.

The left-hand plot in Figure 3 shows a persistent gap between the LoRA and FullFT learning curves at large batch sizes. The gap is smaller and shrinks over time for the smaller batch size of 32.

The right-hand chart plots final loss as a function of batch size. The gap in loss for LoRA increasingly diverges from FullFT for larger batch sizes.

The learning gap at large batches does not seem to depend on rank, but rather seems to be a property of LoRA. The likely reason is that the product-of-matrices parametrization, $BA$, has less favorable optimization dynamics on this dataset than the full matrix $W$. However, both LoRA and FullFT achieve their best loss at smaller batch sizes, so this gap may not matter as much in practice.
