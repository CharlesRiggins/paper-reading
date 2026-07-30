## Appendix C. Details on the `zsRE` Evaluation Task

The study uses the train/test split established by Mitchell et al. Each record contains a requested fact, paraphrase prompts, and an unrelated neighborhood prompt. ROME and other non-hypernetwork methods require no training split.

For fairness, the authors train `KE-zsRE` and `MEND-zsRE` on the benchmark’s training distribution. Their original pretrained versions used a `WikiText` generation objective, so the custom variants show what the learned editors can achieve with in-distribution adaptation.

## Appendix D. Details on `COUNTERFACT`

Each `COUNTERFACT` record starts from a `PARAREL` tuple

$$
t^c=(s,r,o^c)
$$

and hand-curated relation templates $\mathcal T(r)$ grounded in `Wikidata`. A requested rewrite contains

$$
\{s,r,o^c,o^*,p^*\},
$$

where $p^*$ is one prompt instantiated with subject $s$ and $o^*$ is sampled from another object observed with relation $r$.

Two additional templates for the same relation become paraphrase prompts $P^P$. To construct difficult specificity tests, the authors query `Wikidata` for subjects $s'$ satisfying the same original relation–object pair $(s',r,o^c)$. Ten prompts about these semantically close entities form $P^N$. This targeted sampling is more sensitive to local bleedover than a random unrelated fact.

Generation prompts $P^G$ elicit the property indirectly rather than asking for the object directly. Reference texts are Wikipedia articles about entities that genuinely have target property $o^*$; their n-gram statistics support the semantic-consistency metric.

Thus each record contains the rewrite request, two paraphrases, ten neighborhood prompts, generation prompts, and reference texts. The benchmark’s design separates five desired properties: efficacy, generalization, specificity, semantic consistency, and fluency.

## Appendix E. Method Implementation Details

### E.1. Fine-Tuning and Constrained Fine-Tuning

FT changes only one MLP projection matrix and minimizes

$$
-\log\mathbb P_{G'}[o^*\mid p]
$$

with Adam and early stopping. This MLP-only setup already gives FT an advantage over blind optimization because causal analysis has localized the module type.

| Method / model | Layer | Learning rate | Constraint | Early stop |
|---|---:|---:|---:|---:|
| FT, `GPT-2 XL` | 1 | $5\times10^{-4}$ | — | loss 0.03 |
| FT, `GPT-J` | 21 | $5\times10^{-4}$ | — | loss 0.03 |
| FT+L, `GPT-2 XL` | 0 | $5\times10^{-4}$ | $\epsilon=5\times10^{-4}$ | loss 0.03 |
| FT+L, `GPT-J` | 0 | $5\times10^{-4}$ | $\epsilon=5\times10^{-5}$ | loss 0.03 |

FT+L enforces

$$
\|\theta_G-\theta_{G'}\|_\infty\le\epsilon
$$

by clamping edited weights to $\theta_G\pm\epsilon$ after every step.

### E.2–E.4. Knowledge Neurons, KE, and MEND

**Knowledge Neurons (KN)** uses gradient attribution to narrow thousands of MLP neurons to roughly 1,000 candidates and then at most 10 selected neurons. It changes corresponding projection rows by adding scaled object-embedding vectors.

**Knowledge Editor (KE)** trains an LSTM on gradient information to predict a rank-one update. The experiments use the available GPT-2 reimplementation and additionally train `KE-zsRE` and `KE-CF` on 10,000-record task samples.

**MEND** transforms a rank-one decomposition of the negative-log-likelihood gradient, usually over several late transformer layers. `MEND-zsRE` and `MEND-CF` use the same task-specific training sets as KE.

### E.5. ROME

ROME edits **layer 18** of `GPT-2 XL`, near the center of the MLP causal-effect region and the point where MLP outputs transition from key-like to value-like behavior.

The old-key statistic

$$
C\propto\mathbb E[kk^T]
$$

is estimated from **100,000** float32 MLP-key activations. The activations come from every token in randomly sampled articles from the 2020-05-01 Wikipedia snapshot, not only entity tokens.

Key selection averages 20 random-prefix contexts: ten prefixes of length 5 and ten of length 10. Ablations show:

| Prefix strategy | Editing score |
|---|---:|
| No prefix | 86.1 |
| 20 prefixes: 10 length-5 + 10 length-10 | **89.2** |
| Add 10 length-50 prefixes | 89.3 |
| 60 prefixes: 30 length-5 + 30 length-10 | 89.2 |

Value optimization uses Adam with learning rate **0.5**, weight decay $1.5\times10^{-3}$, KL coefficient $\lambda=10^2$, at most **20 steps**, and early stopping at loss $5\times10^{-2}$.

A complete edit takes about **2 seconds** on an NVIDIA A6000 for `GPT-2 XL`. KE and MEND inference is faster—around 100 ms—but requires hours or days of editor training.

## Appendix F. Extended Quantitative Results

ROME also outperforms `FT+L` on smaller models:

| Model / editor | COUNTERFACT S ↑ | ES ↑ | PS ↑ | NS ↑ | zsRE efficacy ↑ | zsRE paraphrase ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `GPT-2 Medium` unedited | 33.4 | 25.0 | 27.4 | 74.9 | 18.8 | 18.1 |
| `GPT-2 Medium` + FT+L | 68.0 | **100.0** | 68.5 | 51.3 | **97.2** | 59.4 |
| `GPT-2 Medium` + **ROME** | **87.4** | **100.0** | **96.4** | **71.8** | 96.6 | **79.8** |
| `GPT-2 Large` unedited | 32.8 | 23.9 | 27.4 | 75.7 | 20.6 | 19.8 |
| `GPT-2 Large` + FT+L | 71.2 | **100.0** | 63.0 | 61.5 | 98.3 | 56.8 |
| `GPT-2 Large` + **ROME** | **88.2** | 99.9 | **96.3** | **73.4** | **99.6** | **84.7** |

The effect therefore does not depend on the 1.5B or 6B parameter scale. Across `GPT-2 Medium`, `GPT-2 Large`, `GPT-2 XL`, and `GPT-J`, ROME consistently improves paraphrase transfer while preserving substantially more neighborhood specificity.
