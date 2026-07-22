## 1. Introduction

As LLM capability grows, misuse concerns span software exploitation and chemical, biological, radiological, and nuclear (**CBRN**) applications. Post-training protections such as refusal training and output classifiers are useful but can be attacked by determined users. The paper therefore studies interventions earlier in the pipeline: prevent—or at least make removable—the acquisition of selected capabilities during pretraining.

### Data filtering's central trade-off

Data filtering removes harmful or restricted material before training. Its practical labels, however, are imperfect: rich contextual assessment is costly, target information can sit inside otherwise benign documents, and a concept can support both harmful and beneficial uses. A developer must therefore either:

1. accept **false negatives**, allowing some target material to remain and potentially be learned; or
2. use a broad filter, which increases **false positives** and removes useful nearby material.

The risk is amplified by scaling: evidence on poisoning suggests a small absolute number of examples can change behavior even as model and dataset scale. The paper treats this as a reason to seek a method robust to missed target samples rather than assuming a perfect filter.

### From Gradient Routing to SGTM

The proposed alternative is **knowledge localization**. Gradient Routing modifies training so target-domain knowledge is concentrated in a designated parameter subset, which is zeroed after training. SGTM follows this design but changes *where* the mask is applied: it masks **parameter gradients** rather than activation gradients, and does so across all Transformer layers. The authors argue this is less disruptive to retained learning and better prevents target data from changing non-target parameters.

For a target such as CBRN, SGTM allocates selected attention heads and MLP neurons in each block to $\theta_{\text{forget}}$. Labelled target examples update that subset only; after training, setting it to zero removes the target pathway. The rest of the model, $\theta_{\text{retain}}$, is explicitly trained to retain general performance without depending on it.

> **Figure 1 (main Wikipedia result).** The paper plots biology forget loss against either general or biology-adjacent retain loss. Higher vertical loss means stronger removal; lower horizontal loss means better retained capability. Across checkpoints, SGTM lies above the weak and strict data-filtering curves at comparable retain loss, but pays a compute-efficiency cost.

### Experimental programme and contributions

The authors first use English/Spanish `TinyStories`, where ground-truth domain labels are available, to inject controlled false negatives. They then train a `254M`-parameter model on English Wikipedia and remove the `STEM.Biology` topic, using document-level topical labels that naturally contain noise.

The paper's contributions are:

1. **Method:** SGTM improves the prior Gradient Routing instantiation by parameter-gradient masking, limiting target-to-retain information flow more strictly while reducing retain-side disruption.
2. **Label-noise robustness:** under synthetic mislabeled data and natural Wikipedia label noise, it improves the retain/forget trade-off over both prior routing variants and filtering.
3. **Leakage measurement:** it defines a token-exposure-equivalent leakage metric and measures it from `8M` through `64M` models; leakage decreases with scale in this range.
4. **Mechanistic evidence:** unlabelled target samples give larger relative gradients to forget parameters, while retain samples preferentially update retain parameters. Per-token losses also show broad target-domain degradation rather than a few pathological tokens.
5. **Relearning robustness:** adversarial fine-tuning recovers target performance much more slowly for SGTM than for finetuning-based `RMU`.
