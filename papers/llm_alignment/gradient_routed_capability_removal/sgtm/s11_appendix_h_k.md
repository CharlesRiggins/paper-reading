## Appendix H. Additional forget category

To test whether biology is unusually favorable, the authors repeat the Wikipedia experiment with **Military and Warfare** as the forget category. Related retain categories are History, Biography, and Transportation; general knowledge is evaluated separately. Figure 13 reproduces the main pattern: at fixed retain loss, SGTM produces higher target-domain loss than weak and strict filtering, while ending with a somewhat worse retain loss due to the compute-efficiency cost.

## Appendix I. `TinyStories` details

### Training configuration

| Hyperparameter | `64M` | `34M` | `18M` | `8M` |
|---|---:|---:|---:|---:|
| Actual parameters | 63.855M | 33.732M | 17.781M | 7.736M |
| Layers | 12 | 8 | 6 | 6 |
| Model width | 512 | 384 | 256 | 128 |
| MLP width | 2048 | 1536 | 1024 | 512 |
| Warmup steps | 1000 | 1000 | 1000 | 500 |
| Training steps | 33,120 | 17,484 | 9,204 | 3,989 |

All use a `GPT-2` tokenizer with vocabulary 50,257, 32 attention heads, tied embeddings, context length 512, batch size 128, `AdamW` ($\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1), cosine decay with warmup, and learning rate $5\times10^{-3}$. The authors make the largest model roughly twice the original `TinyStories` model's size to account for the additional Spanish corpus.

### Logit calibration and dataset

After ablation, a 50,257-parameter logit-bias layer is optimized on both domains using

$$
\ell=\ell_{\text{forget}}+\alpha\ell_{\text{retain}},\qquad \alpha=100.
$$

The high retain coefficient prevents repairing target-token probability spikes by sacrificing general retention. Calibration has access to both domains even in filtering conditions; it is an evaluation stabilization step, not a deployment procedure.

The corpus contains `466M` original English tokens and `718M` Spanish translated tokens. Stories are individual examples, truncated or padded as needed. Spanish is longer because the English-trained `GPT-2` tokenizer encodes it less efficiently.

## Appendix J. Wikipedia details and RMU

| Hyperparameter | `254M` | `125M` | `64M` | `34M` |
|---|---:|---:|---:|---:|
| Actual parameters | 254.054M | 124.462M | 64.117M | 33.929M |
| Layers | 16 | 12 | 12 | 8 |
| Model width | 1024 | 768 | 512 | 384 |
| MLP width | 4096 | 3072 | 2048 | 1536 |
| Warmup steps | 1000 | 1000 | 500 | 500 |
| Batch size | 512 | 256 | 256 | 128 |
| Training steps | 9,689 | 9,491 | 4,887 | 5,169 |

The `254M` configuration uses learning rate $6\times10^{-3}$, tied embeddings, 32 heads, `GPT-2` vocabulary 50,257, context 1024, `AdamW` ($\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1), and cosine warmup scheduling. Wikipedia calibration follows Appendix I.

`RMU` starts from the no-filter baseline and uses the original implementation with 250 unlearning steps, $\alpha=100$, steering coefficient 20, batch size 4, unlearning layer 7, update layers 5–7, and MLP weights/biases $W_1,b_1,W_2,b_2$ as trainable parameters.

## Appendix K. ORES `articletopic` taxonomy

The ORES taxonomy has 64 labels grouped under Culture, Geography, History & Society, and STEM. The main experiment uses `STEM.Biology` as target; its adjacent retain set is `STEM.Chemistry`, `STEM.Earth and Environment`, and `STEM.Medicine and Health`. General retain averages Culture, Geography, History & Society categories.

The complete taxonomy includes, for example, Culture topics (Biography, Food and drink, Linguistics, Literature, Media, Philosophy and religion, Sports, Visual arts); Geography regions (Africa, Americas, Asia, Europe, Oceania and subregions); History & Society topics (Business and economics, Education, History, Military and warfare, Politics and government, Society, Transportation); and STEM topics (Biology, Chemistry, Computing, Earth and environment, Engineering, Libraries and information, Mathematics, Medicine and health, Physics, Space, Technology). Article-level highest-score assignment makes this a noisy but realistic label source.
