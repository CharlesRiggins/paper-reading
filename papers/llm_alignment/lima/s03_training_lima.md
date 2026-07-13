## 3. Training LIMA

LIMA (**Less Is More for Alignment**) starts from the pretrained `LLaMa-65B` model and applies supervised fine-tuning on the 1,000-example alignment dataset. The training objective is the standard next-token supervised loss over the prompt-response sequences; there is no preference model, rejection sampling, RLHF, or human-feedback optimization.

To separate speakers, the authors introduce a special **end-of-turn** token (**EOT**) at the end of each utterance. EOT plays the same generation-halting role as EOS but avoids overloading the pretrained model's existing EOS token, which may already carry other distributional meanings.

### Training hyperparameters

| Component | Setting |
|---|---|
| Base model | `LLaMa-65B` |
| Alignment data | 1,000 prompt-response examples, roughly 750k tokens |
| Objective | Standard supervised fine-tuning / next-token loss |
| Optimizer | `AdamW` |
| Adam betas | `$\beta_1=0.9, \beta_2=0.95$` |
| Weight decay | `0.1` |
| Epochs | 15 |
| Warmup | None |
| Learning rate schedule | Initial `$10^{-5}$`, linearly decayed to `$10^{-6}$` |
| Batch size | 32 examples for `65B`; 64 for smaller models |
| Sequence handling | Texts longer than 2048 tokens are trimmed |
| Residual dropout | Linearly increases by layer from `$p_d=0.0$` at the bottom to `$p_d=0.3$` at the last layer; `$p_d=0.2$` for smaller models |
| Checkpoint selection | Manual selection between epochs 5 and 10 using the 50-example dev set |

A notable deviation from common fine-tuning recipes is the residual dropout schedule, following Ouyang et al. [23]. The authors found that validation perplexity did **not** correlate with generation quality, so they did not simply choose the minimum-perplexity checkpoint. Instead, they manually examined outputs on the held-out 50-example development set and selected checkpoints between the 5th and 10th epochs.

This training setup is intentionally simple. Its importance lies less in algorithmic novelty than in showing that a strong pretrained model can acquire a high-quality assistant interface from a tiny supervised dataset when the examples are diverse and carefully written.
