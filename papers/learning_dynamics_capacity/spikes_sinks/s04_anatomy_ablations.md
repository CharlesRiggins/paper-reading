## 4 Anatomy of Spikes and Sinks

The previous section characterizes how massive activations and attention sinks co-emerge in pretrained LLMs, suggesting that both phenomena arise from the interactions between architectural components and learned weights. We now shift from mechanism to causality. Guided by our findings, we perform targeted ablations to identify which architectural and training choices modulate these phenomena, ultimately establishing the causal relationship between the two.

##### Experimental setup.

All models in Section 4 are trained on the `DCLM` dataset (Li et al., 2024) using a shared codebase and a common baseline recipe closely following the Llama-style pretraining setup (Touvron et al., 2023a, b; Grattafiori et al., 2024).

> **Table 3:** Optimization hyperparameter ablations. The sink ratio could serve as a proxy for optimization health. Conversely, the magnitude of massive activations varies largely independently of both perplexity and the sink ratio. Highlighted rows denote the baseline result.

| Setup | Perplexity | Sink Ratio | Spike |
| --- | --- | --- | --- |
| Base Learning Rate | | | |
| $7.5\times10^{-5}$ | 11.8 | 18.6% | 1447 |
| $1.5\times10^{-4}$ | 10.7 | 31.8% | 2251 |
| **$3.0\times10^{-4}$** | **10.1** | **46.0%** | **3818** |
| $6.0\times10^{-4}$ | 10.0 | 51.5% | 3773 |
| $1.2\times10^{-3}$ | 10.2 | 39.2% | 2723 |
| Minimal Learning Rate | | | |
| $3\times10^{-5}$ | 10.1 | 46.0% | 3818 |
| $3\times10^{-4}$ | 10.7 | 56.8% | 2870 |
| Weight Decay | | | |
| $0.0$ | 10.4 | 33.8% | 12275 |
| **$0.1$** | **10.1** | **46.0%** | **3818** |
| AdamW $\beta_{2}$ | | | |
| $0.9$ | 10.1 | 49.0% | 2832 |
| **$0.95$** | **10.1** | **46.0%** | **3818** |
| $0.999$ | 10.7 | 20.9% | 1855 |
| Training Tokens | | | |
| $100$B | 10.1 | 46.0% | 3818 |
| $200$B | 9.5 | 63.3% | 1848 |

### 4.1 Ablating Optimization Hyperparameters

Before proceeding to targeted architectural ablations, we examine the sensitivity of these phenomena to common training hyperparameters: learning rate, weight decay, AdamW (Loshchilov and Hutter, 2017) momentum ($\beta_{2}$), and total training tokens. Results are summarized in Table 3.

Two distinct patterns emerge. First, while the sink ratio is not strictly monotonic in perplexity, it remains a robust proxy for optimization health. Suboptimal configurations—such as extreme learning rates, disabling weight decay, or mis-specified $\beta_{2}$—consistently reduce the sink ratio. Conversely, favorable configurations—such as extending the training budget or disabling learning-rate decay—substantially increase the sink ratio. This suggests that the intensity of attention sinks is tied to the overall optimization health.

Second, the magnitudes of massive activations vary largely independently of perplexity and sink ratio. For instance, disabling weight decay causes activation spikes to exceed $12{,}000$ without any corresponding improvement in sink ratio or perplexity. Spikes drive normalized representations into a sparse, near-constant regime, after which further growth in magnitudes contributes diminishingly to attention sinks. Having established that sinks and spikes respond differently to optimization configurations, we now consider architectural interventions that directly target their underlying mechanisms.

### 4.2 Ablating Massive Activations

In the previous section, we identify two architectural components that strongly affect the emergence of massive activations: (1) the SwiGLU-based feed-forward network, which generates the massive activations, and (2) the normalization configuration, which governs their propagation and maps spike tokens to sparse, near-constant vectors. In this subsection, we ablate both design choices.

#### 4.2.1 Feed-Forward Block Design

Our earlier analysis traced the origin of massive activations to the SwiGLU blocks. To test whether this specific design is a prerequisite for both phenomena, we ablate the feed-forward architecture. Specifically, we evaluate the standard two-layer GeLU-based feed-forward block used in the original Transformer (Vaswani et al., 2017), a simplified single linear layer, and an attention-only configuration where all feed-forward blocks are replaced with additional attention layers. Results are summarized in Table 4.

> **Table 4:** Feed-forward block design ablations. Massive activations and attention sinks emerge across all evaluated designs, including attention-only and single linear configurations. Notably, GeLU and SwiGLU architectures yield significantly higher spike magnitudes by acting as efficient amplifiers. In contrast, linear and attention-only blocks exhibit much lower spikes as they require gradual accumulation across multiple layers. Highlighted rows denote the baseline result.

| Setup | Perplexity | Sink Ratio | Spike |
| --- | --- | --- | --- |
| Feed-Forward Block | | | |
| GeLU | 10.1 | **69.3%** | 3369 |
| Linear | 12.5 | 58.9% | 688 |
| Attention | 10.8 | **73.9%** | 637 |
| **SwiGLU** | **10.1** | **46.0%** | **3818** |

The results indicate that massive activations and attention sinks emerge across all configurations, suggesting that the specific feed-forward block design is not the primary causal driver of either phenomenon. The specific block design is therefore not a prerequisite, but it is a strong modulator of amplification efficiency. SwiGLU and GeLU concentrate outlier growth within a single step, while linear and attention blocks require gradual accumulation across layers.

#### 4.2.2 Normalization Configuration

Normalization shapes these phenomena along two axes: how outliers accumulate in the residual stream (governed by the pre-norm configuration), and how spike tokens are transformed into sparse, near-constant vectors (governed by the normalization operator itself). We probe both axes through three variants. First, we test sandwich normalization (Ding et al., 2021), which adds an extra $\operatorname{RMSNorm}$ at the block output, and a variant utilizing QKNorm (Olmo et al., 2025), where input normalization is applied only to queries and keys. Second, we replace standard normalization with an element-wise transformation, DynamicTanh (Zhu et al., 2025; Chen et al., 2025), which lacks the mathematical capacity to map extreme outliers into sparse, near-constant vectors. Table 5 summarizes these results.

> **Table 5:** Normalization configuration ablations. Applying post-block normalization (Sandwich) or element-wise transformations (DynamicTanh) successfully suppresses massive activations. Notably, the model still maintains a significant sink ratio through alternative strategies, demonstrating that sinks can exist independently of massive activations. Highlighted rows denote the baseline result.

| Setup | Perplexity | Sink Ratio | Spike |
| --- | --- | --- | --- |
| Normalization | | | |
| Sandwich | **9.8** | 44.7% | 520 |
| Sandwich (QK) | 10.0 | 42.0% | **92** |
| DynamicTanh | 10.0 | **61.0%** | 153 |
| **Pre-Norm** | **10.1** | **46.0%** | **3818** |

The results demonstrate that normalization serves as a direct lever to decouple sinks from spikes. Sandwich normalization reduces spikes while preserving a sink ratio nearly identical to the baseline. Because the extra $\operatorname{RMSNorm}$ layer bounds the block output, it prevents the residual stream from accumulating the unbounded values necessary for massive outliers. Replacing the block-level norm with QKNorm almost entirely eliminates spikes, confirming that these outliers are primarily generated to influence the query and key projections.

Conversely, as observed by Owen et al. (2025a), element-wise transformations like DynamicTanh also prevent the emergence of massive activations entirely. This aligns with our hypothesis: because DynamicTanh is bounded and operates element-wise rather than via a vector-wide norm, it cannot facilitate the creation of sparse vectors from high-magnitude spikes. Interestingly, the DynamicTanh model yields the highest sink ratio while maintaining low spike magnitudes. Inspection of attention patterns reveals that the model still designates the first token as a stable reference point, achieving this through alternative strategies rather than magnitude-driven normalization. These results confirm that while massive spikes are an artifact of specific normalization configurations, they are not a prerequisite for attention sinks.

### 4.3 Ablating Attention Sinks

Building on our earlier analysis, we find that sink formation depends critically on whether sink and non-sink keys can occupy geometrically separable subspaces. We therefore begin by ablating per-head representational capacity, which determines whether the attention subspace has sufficient room to segregate sink keys from non-sink keys. We then conduct two further ablations motivated by prior work: one on gated attention (Qiu et al., 2025), which has been shown to reduce attention sinks and massive activations, and one on context length, motivated by (Xiao et al., 2024a), which argues that attention sinks primarily bias short-range dependence. We ablate all three factors in turn below.

#### 4.3.1 Attention Head Settings

Our earlier findings identified the segregation of sink and non-sink keys as the primary driver of sink formation. Since this mechanism is inherently tied to per-head capacity, we systematically ablate total head count, head dimension, and head factorization to disentangle their individual contributions. Results are summarized in Table 6.

The results confirm that **head dimension is the dominant architectural factor governing sink emergence**. Increasing $d_{\text{head}}$ from $8$ to $128$ produces a monotonic rise in both the sink ratio and spike magnitude, supporting our geometric hypothesis: larger head dimensions expand the attention subspace sufficiently to cleanly separate sink keys from non-sink keys, enabling the generation of a large logit gap.

When total attention capacity ($d_{\text{head}}\times N_{\text{head}}$) is held fixed, concentrating it into fewer, larger heads consistently strengthens sink behavior and improves perplexity, suggesting a potential link between sink ratio and model performance. Conversely, increasing the number of heads at fixed $d_{\text{head}}$ yields only marginal gains in the sink ratio, indicating that sink formation saturates once sufficient per-head capacity is available and that distributing capacity across more heads yields diminishing returns.

> **Table 6:** Attention head settings ablations. Head dimension is the primary architectural driver of sink formation; larger dimensions provide the capacity for the attention subspace to isolate the sink keys. Concentrating capacity into fewer, larger heads intensifies sink behavior. Highlighted rows denote the baseline configuration.

| Setup | Perplexity | Sink Ratio | Spike |
| --- | --- | --- | --- |
| Number of Heads | | | |
| $8$ | 10.4 | 37.1% | 1253 |
| $16$ | 10.3 | 41.7% | 1936 |
| **$32$** | **10.1** | **46.0%** | **3818** |
| Head Dimension | | | |
| $8$ | 11.3 | 4.1% | 291 |
| $16$ | 10.8 | 9.8% | 315 |
| $32$ | 10.5 | 27.9% | 829 |
| $64$ | 10.3 | 37.7% | 2112 |
| **$128$** | **10.1** | **46.0%** | **3818** |
| Head Dim / Number of Heads | | | |
| $8/512$ | 10.7 | 11.0% | 1205 |
| $16/256$ | 10.4 | 30.8% | 1750 |
| $32/128$ | 10.3 | 41.1% | 1916 |
| $64/64$ | 10.2 | 44.1% | 2523 |
| **$128/32$** | **10.1** | **46.0%** | **3818** |
| $256/16$ | 10.1 | **52.1%** | 3429 |

#### 4.3.2 Gated Attention

Following (Qiu et al., 2025), we employ gated attention variants to test the hypothesis that dynamic multiplicative routing can destabilize or prevent attention sink formation. As shown in Table 7, gating conditioned on the current hidden representation drastically suppresses the sink ratio and effectively eliminates spikes, with minimal impact on perplexity. By contrast, unconditional gating or gating tied to static signals (such as position or token embedding) preserves strong sink behavior.

> **Table 7:** Gated attention ablations. Conditional gating—where the gate is a function of current representation—eliminates the need for attention sinks when applied per channel or per head. This suggests that sinks function as a "learned gate" to balance head contributions. Unconditional or static gates (positional or token-based) fail to suppress sinks, as they lack the dynamic, input-dependent routing necessary to substitute for sink behavior.

| Setup | Perplexity | Sink Ratio | Spike |
| --- | --- | --- | --- |
| Conditional Gating | | | |
| Channel | 10.0 | **4.5%** | **202** |
| Head | 10.1 | **6.4%** | **186** |
| Single | 10.2 | 31.2% | 316 |
| Unconditional Gating | | | |
| Channel | 10.1 | 42.2% | 1922 |
| Head | 10.1 | 41.3% | 1884 |
| Single | 10.2 | 44.3% | 1797 |
| Conditional on Static Signal | | | |
| Positional | 10.1 | 41.1% | 1755 |
| Token Embedding | 10.0 | 31.1% | 1966 |

Among the conditional gating configurations, per-channel and per-head gates both eliminate attention sinks entirely. A single gate per token, however, yields elevated sink ratios and slightly higher perplexity—consistent with our earlier finding that sink formation is a head-level phenomenon. Under unconditional gating, the static gate fails to suppress either attention sinks or massive activations. Similarly, gating conditioned on positional or token embeddings does not eliminate sinks, as these signals are fixed and cannot adapt to the evolving context.

Taken together, these results suggest that attention sinks serve as a form of **implicit input-conditioned gating**: the effective routing behavior depends on the prompt history rather than being a fixed property of a particular head, position, or token. When the model has access to a dynamic, representation-conditioned gate, it can modulate attention routing on the fly, eliminating the structural need to maintain a spike token via large residual spikes.

#### 4.3.3 Training Context Length

Xiao et al. (2024a) suggest that attention sinks facilitate short-range dependence in sink heads. Consistent with this, we observe that sink heads predominantly attend to nearby tokens of the query. We therefore vary the training context-length distribution to test whether sinks are an inductive bias of short-range training, controlling the distribution by adjusting the range of sequence positions over which the training loss is computed. Results are shown in Table 8.

> **Table 8:** Context-length ablations. Attention sinks are largely induced to facilitate short-context prediction. When the training distribution is restricted to long sequences, the sink ratio collapses, indicating that sinks are primarily utilized to support short-range dependence. Highlighted rows denote the baseline configuration.

| Setup | Perplexity | Sink Ratio | Spike |
| --- | --- | --- | --- |
| Context Length (min/max) | | | |
| 1/256 | 12.4 | 42.1% | 5411 |
| 1/1024 | 10.6 | 46.3% | 4442 |
| **1/4096** | **10.1** | **46.0%** | **3818** |
| 1024/4096 | 10.1 | **13.0%** | 38470 |
| 1024/5120 | 10.1 | **8.0%** | 42365 |
| 2048/4096 | 10.6 | **1.2%** | 7193 |
| 2048/6144 | 10.0 | **5.8%** | 30634 |

When the training distribution includes short sequences, the sink ratio remains stable regardless of the maximum context length. Removing short contexts entirely—optimizing only over long-range positions—causes the sink ratio to collapse dramatically. This confirms that attention sinks are fundamentally a byproduct of short-context training: in mixed-length regimes, the first token provides a cheap, universally available global reference that reduces the influence of far-away tokens. Excluding short-context positions from the training loss therefore reveals that the majority of sinks are induced specifically to facilitate local prediction within a global attention mechanism, corroborating the role of sink heads identified by Xiao et al. (2024a).

### 4.4 Summary of Findings

Our ablation study reveals three critical insights into the nature of massive activations and attention sinks:

1. **Causal independence of spikes.** While spikes and sinks often co-occur, they are not inextricably linked. Normalization techniques like Sandwich Norm or QKNorm can eliminate massive activations without destroying the attention sink. This suggests that spikes are an artifact of the pre-norm architecture's tendency to accumulate unbounded values, which the model then exploits—but does not strictly require—to create logit contrast.
2. **Sinks as a gating mechanism.** The disappearance of sinks in the presence of conditional gating suggests that attention sinks are a learned workaround. In the absence of an explicit gate to modulate information flow, the model repurposes the first token as a numerical "dumping ground" to effectively gate off unnecessary attention heads.
3. **Context-length induction.** Sinks are fundamentally driven by the need to model short-range dependencies using a global attention mechanism. By dumping attention into the first token, the model can effectively ignore long-range context when it is not predictive, a behavior that becomes unnecessary when the model is trained exclusively on long-context sequences.

### 4.5 Discussion

Our ablations paint a coherent picture of how massive activations and attention sinks arise, interact, and can be independently controlled. Across optimization hyperparameters, feed-forward designs, normalization configurations, and attention settings, the two phenomena respond differently to the same interventions — suggesting that their frequent co-occurrence in standard LLMs reflects incidental architectural interactions rather than a deep functional coupling.

##### Normalization as the bridge between spikes and sinks.

Normalization emerges as the central architectural link between the two phenomena. Standard pre-norm $\operatorname{RMSNorm}$ allows unbounded residual values to accumulate and maps spike tokens into sparse, near-constant vectors, which in turn provide a stable substrate for sink formation. Yet this link is incidental rather than necessary: sandwich normalization and DynamicTanh both suppress spikes while leaving a robust sink ratio intact, confirming that sinks can find alternative strategies when magnitude-driven normalization is unavailable. Mechanistically, massive activations interact with normalization to function as implicit parameters.

##### Attention sinks as a learned routing strategy.

Sink formation is independently driven by two factors: the dimensionality of the per-head attention subspace, which determines whether sink and non-sink keys can be geometrically separated, and the training context-length distribution, which establishes whether attention sinks are useful. Conditional gating experiments reinforce this view — sinks serve as an implicit, input-dependent routing mechanism that biases certain heads toward local, short-range dependencies, and one the model abandons as soon as an explicit dynamic gate is provided.

##### Independent suppression without performance cost.

Crucially, each phenomenon can be suppressed in isolation without measurable degradation in language modeling performance. This separation has practical implications: architectural choices that eliminate spikes for inference efficiency need not disrupt the short-range routing behavior that sinks provide, and vice versa. Their overlap in standard pretrained LLMs is best understood as a byproduct of the default normalization and training recipe, not a reflection of any underlying functional necessity.
