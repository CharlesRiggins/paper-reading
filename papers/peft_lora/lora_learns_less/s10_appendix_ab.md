## A Experimental Setup

### LoRA configuration for all experiments.

All experiments were done with the Databricks MosaicML composer, streaming and llm-foundry libraries in conjunction with the HuggingFace peft library on 32×H100-80GB GPUs. All experiments in the main text used the LionW optimizer [6] instead of the AdamW optimizer.

We targeted all trainable modules inside each of the $L$ Llama transformer blocks: $\{ W _ { q } ^ { ( l ) } , W _ { k } ^ { ( l ) } , W _ { v } ^ { ( l ) } , W _ { o } ^ { ( l ) } , W _ { \mathrm { g a t e } } ^ { ( l ) } , W _ { \mathrm { u p } } ^ { ( l ) } , W _ { \mathrm { d o w n } } ^ { ( l ) } \} _ { l = 1 } ^ { L }$. We used ranks of $r = 16, 64, 256$ and set $\alpha = 2r$ to achieve a constant scaling factor $\gamma _ { r } = 2$ across ranks. We use `lora_dropout=0.05`.

For both the Code CPT and Math CPT settings, we train the model once for 20B tokens. We then perform individual cooldowns using intermediate checkpoints as follows: We set a target max training duration (e.g. 8 billion tokens), and define the last 20% of max training duration as the cooldown period. We then retrain from the latest available checkpoint prior to the cooldown period.

In all four scenarios below, we use the Llama-2-7B base model `meta-llama/Llama-2-7b-hf`. For the CPT runs, we use the `meta-llama/Llama-2-7b-hf` tokenizer, while for the IFT runs we use the `meta-llama/Llama-2-7b-chat-hf` tokenizer.

**Code CPT**: Llama-2-7B trained on the StarCoder-Python dataset.

- `seq_len`: 4096
- `optimizer`: `decoupled_lionw` (`betas=[0.9, 0.95]`)
- `learning_rate`: **1.0e-05** for LoRA and Full Finetuning
- `scheduler`: `inv_sqrt_with_warmup` (`t_scale=1000ba, t_warmup=1000ba, t_cooldown=5086ba, alpha_f_decay=1, alpha_f_cooldown=0`). We note that this ends up looking very much like a trapezoidal schedule.
- `weight_decay`: 1.0e-06
- `precision`: `amp_bf16`
- `global_train_batch_size`: 192
- `device_train_microbatch_size`: 6
- `gradient_clipping`: norm (`threshold=1`)
- `num_gpus`: 32

**Math CPT**: Llama-2-7B trained on the OpenWebMath dataset.

- `max_seq_len`: 4096
- `optimizer`: `decoupled_lionw` (`betas=[0.9, 0.95]`)
- `learning_rate`: **1.0e-05** for full finetuning, **4.0e-05** for LoRA
- `scheduler`: `inv_sqrt_with_warmup` (`t_scale=1000ba, t_warmup=1000ba, t_cooldown=5086ba, alpha_f_decay=1, alpha_f_cooldown=0`). We note that this ends up looking very much like a trapezoidal schedule.
- `weight_decay`: 0
- `precision`: `amp_bf16`
- `global_train_batch_size`: 192
- `device_train_microbatch_size`: 6
- `gradient_clipping`: norm (`threshold=1`)
- `num_gpus`: 32

**Code IFT**: Finetuning Llama-2-7B on the Magicoder-Evol-Instruct-110K dataset

- `max_seq_len`: 4096
- `optimizer`: `decoupled_lionw` (`betas=[0.9, 0.95]`)
- `learning_rate`: **2e-4** for rank $r = 16, 64$ and **1e-4** for $r = 256$, $\alpha = 2r = 512$ (due to instabilities/loss spikes at 2e-4)
- `scheduler`: `cosine_with_warmup` (`alpha_f=0.01, t_warmup=0.1dur`)
- `weight_decay`: 0
- `precision`: `amp_bf16`
- `global_train_batch_size`: 192
- `device_train_microbatch_size`: 6
- `gradient_clipping`: norm (`threshold=1`)
- `num_gpus`: 32

**Math IFT**: Finetuning Llama-2-7B on the MetaMathQA dataset

- `seq_len`: 1024
- `optimizer`: `decoupled_lionw` (`betas=[0.9, 0.95]`)
- `learning_rate`: Full finetuning: **1e-5**, LoRA: **1e-4** for $r = 16, 64$, **5e-5** for $r = 256$ due to instabilities.
- `scheduler`: `cosine_with_warmup` (`alpha_f=0.01, t_warmup=0.1dur`)
- `weight_decay`: 0
- `precision`: `amp_bf16`
- `global_train_batch_size`: 768
- `device_train_microbatch_size`: 24
- `gradient_clipping`: norm (`threshold=1`)
- `num_gpus`: 32

### A.1 Training the input and output embedding layers.

Vanilla LoRA and other popular methods such as QLoRA [10] often do not train the input and output embedding layers. Recent open-source work, on the other hand, shows that it might be beneficial to supplement LoRA with full finetuning of these two modules (additional ≈ 200M parameters for a 7B model). We view this approach as a hybrid of LoRA and full finetuning, and therefore leave its empirical investigation for future work. Moreover, this hybrid approach involves further hyperparameter optimization: the input and output layers require tuning their own separate learning rates, which should typically be 2–10× smaller than the LoRA learning rates (training with a single learning rate results in instabilities).

## B Learning rate searches

We perform a learning rate sensitivity analysis for Llama-2-7B, trained for two epochs on the code and math IFT datasets, and followed by HumanEval and GSM8K evaluation, respectively. Fig. S1 shows that LoRA improves monotonically with learning rate up to a value at which training diverges, with best learning rates of **$5 e ^ { - 4 }$** for code and **$2 e ^ { - 4 }$** for math.

On both datasets, these best LoRA learning rates are underperformed by four alternative full finetuning learning rates. The best full finetuning learning rates are **$5 e ^ { - 5 }$** and **$1 e ^ { - 5 }$**, respectively, an order of magnitude smaller than LoRA. For LoRA, we cannot find alternative learning rates that achieve at least 90% of the best learning rate's performance. For full finetuning, there are two viable alternative learning rates for code and three for math.

Note that in these experiments, the LoRA models target all modules but the $W _ { \mathrm { g a t e } }$, with $\alpha = 32$ which should preferably be higher for $r = 64$. This explains the slight differences between Figures S1 and S3.

### B.1 Learning rate sensitivity analysis across optimizers

We compared the AdamW and Decoupled LionW optimizers by training for two epochs of Magicoder-Evol-Instruct-110K using different learning rates. We found that **Decoupled LionW performed better on HumanEval** for both LoRA and full finetuning, and across learning rates, as seen in Fig. S2.

### B.2 The importance of the alpha scaling parameter for LoRA

We found that the performance of all models was particularly sensitive to the LoRA $\alpha$ hyperparameter. Fig. S3 shows two experiments on two separate datasets (Magicoder-Evol-Instruct-110K and OpenWebMath) for LoRA with rank $r = 256$. In both cases the best accuracy is achieved when **$\alpha = 2r$**.
