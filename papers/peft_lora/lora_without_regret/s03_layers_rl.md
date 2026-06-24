## Layers Where LoRA Is Applied

The authors investigated the effects of applying LoRA to different layers in the network. The original paper by Hu et al. recommended applying LoRA only to the attention matrices, and many subsequent papers followed suit, though a recent trend has been to apply it to all layers. Similar to these results, the `QLoRA` paper also found that LoRA performed worse when applied only to attention than when applied to MLP or MLP+attention, though `QLoRA` found MLP+attention > MLP > attention, whereas this blog finds the first two to be roughly equal.

The authors achieved far better results when applying LoRA to all layers, in particular the MLP, including MoE, layers. In fact, applying LoRA to the attention matrices shows no additional benefits beyond applying it to the MLPs only. Biderman et al. (2024) obtained a similar result, with attention-only LoRA providing no additional benefit on top of MLP-only.

![](https://thinkingmachines.ai/blog/lora/svg/fig4.svg)

Figure 4 shows that attention-only LoRA significantly underperforms MLP-only LoRA, and does not further improve performance on top of LoRA-on-MLP. This effect holds for a dense model, `Llama-3.1-8B`, and a sparse MoE, `Qwen3-30B-A3B-Base`.

The underperformance of attention-only LoRA is not explained by having fewer parameters. In this particular case, attention-only with rank 256 underperforms MLP-only with rank 128, despite them having approximately the same number of parameters.

| LoRA configuration | Params |
|---|---:|
| mlp, rank=256 | 0.49B |
| attn, rank=256 | **0.25B** |
| all, rank=256 | 0.70B |
| mlp, rank=128 | **0.24B** |

Table: Parameter counts for LoRA on `Llama-3.1-8B`.

For the MoE experiment, the authors trained a separate LoRA on each expert, with the rank of each equal to the total rank divided by the number of active experts, equal to 8 for Qwen3 MoE. This scaling keeps the ratio of LoRA parameters to FullFT parameters the same for MoE layers as for other layers.

They did similar experiments comparing different LoRA layers in two additional settings: (1) supervised learning on a small subset of the `OpenThoughts3` dataset with rank 256, and (2) reinforcement learning on the `MATH` dataset. Attention-only LoRA underperforms MLP-only LoRA, which performs similarly to MLP+attention, in these settings as well.

![](https://thinkingmachines.ai/blog/lora/svg/fig5.svg)

Figure 5 shows learning rate versus final loss or reward when varying which layers LoRA is applied to.

### Reinforcement learning

A key finding from the experiments is that LoRA fully matches the learning performance of FullFT when running policy gradient algorithms for reinforcement learning, even with ranks as low as 1.

For these experiments, the authors used a basic policy gradient algorithm with an importance sampling correction:

$$
\text{objective} = \sum_t \frac{p_{\text{learner}}}{p_{\text{sampler}}} \text{Adv}_t
$$

They used a GRPO-like centering scheme, as in DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024), where multiple completions are sampled per problem and the mean reward per group is subtracted.

Figure 6 shows LR sweeps on the `MATH` and `GSM8K` datasets, using typical hyperparameters for each. The authors used the `Llama-3.1-8B` base model because `Qwen2.5` and `Qwen3` are known to have been pretrained on data that improves their math performance, as described by the Qwen technical reports, which makes it harder to measure what is being learned only during RL.

LoRA shows a wider range of performant learning rates and arrives at the same peak performance as FullFT, shown as the black line, at least within the precision limits afforded by the noisiness of RL.

![](https://thinkingmachines.ai/blog/lora/svg/fig5-.svg)

Figure 6 shows learning rate versus final reward, i.e. accuracy, when doing RL on grade-school math (`GSM8K`, left) or the `MATH` dataset (right).

This result is anticipated by an information-theoretic argument. Supervised learning arguably provides $O(\text{number of tokens})$ bits per episode. In contrast, in policy gradient methods, learning is driven by the advantage function, which provides only $O(1)$ bits per episode. When each episode contains thousands of tokens, RL absorbs about **1000 times less information per token** in training than supervised learning does.

More precise numbers from the experiments support the same point. In the `MATH` example, the authors trained on about 10,000 problems with 32 samples per problem. Assuming each completion yields a single bit of information, the whole training process only needs to absorb 320,000 bits. Rank-1 LoRA for `Llama-3.1-8B` already has **3M parameters** — calculated by adding up rank $\cdot d_{in}$ for matrix $A$ and rank $\cdot d_{out}$ for matrix $B$ over all weight matrices in the model — almost 10 times that number. Even at rank 1, LoRA has more than enough capacity to absorb all the information provided during training.

As another point of comparison, `DeepSeek-R1-Zero` was trained on 5.3M episodes — training took place for 10,400 steps, each step consisting of 32 unique questions, each question sampled 16 times — corresponding to 5.3M bits of information. This is less than the number of parameters in a low-rank LoRA, and the authors predict that the results can be replicated with LoRA.

For additional validation of LoRA’s effectiveness in reasoning RL, the authors carried out larger-scale experiments with `Qwen3-8B-Base` on the `DeepMath` dataset. `DeepMath` is much larger than the `MATH` dataset and generally contains harder problems. To speed up experiments, they restricted the samples to a length of 8192 tokens for training and evaluation. This sample length allows for backtracking and reasoning but limits performance relative to longer chain-of-thought.

![](https://thinkingmachines.ai/blog/lora/svg/fig6.svg)

Figure 7 shows experiments on the `DeepMath` dataset with `Qwen3-8B-Base`. In the left plot, the learning curve is shown for different ranks and full fine-tuning. For each setting, the best learning rate is shown, resulting in the highest final performance. On the right, the plot shows learning rate versus final performance. As in the previous math experiments, LoRA seems to have a wider peak of near-optimal learning rates.

![](https://thinkingmachines.ai/blog/lora/svg/fig7.svg)

Additional plots from experiments on the `DeepMath` dataset with `Qwen3-8B-Base` show benchmark scores on the `AIME` test set on the left, which is more challenging than the training set, and chain-of-thought (**CoT**) length over training steps on the right, which can be seen as a sign of learning to reason.

When picking the optimal learning rates for each setting, training progresses in an almost identical way for LoRAs with different sizes and full fine-tuning. Similar findings appear when evaluating the models on held-out problems from `AIME 2024` and `AIME 2025`. The authors also observe similar qualitative behavior from the LoRA and full-finetuning runs: both develop advanced reasoning behaviors such as backtracking, self-verification, and in-context exploration, visible in the lengthening of model CoTs.
