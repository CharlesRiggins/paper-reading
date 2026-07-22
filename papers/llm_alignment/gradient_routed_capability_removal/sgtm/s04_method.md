## 3. Method

SGTM partitions each Transformer block into a **forget** component and a **retain** component. Its target is a model that performs well on the desired distribution $\mathcal{D}_{\text{retain}}$ while performing poorly on the target distribution $\mathcal{D}_{\text{forget}}$ *after ablation*.

### 3.1 Parameter and data notation

For a block with $h$ attention heads, model dimension $d$, and MLP dimension $d_{\text{MLP}}$, choose $h_{\text{forget}}$ heads and $d_{\text{forget}}$ MLP hidden units for the target domain. Their associated weights are $\theta_{\text{forget}}$; all other block parameters are $\theta_{\text{retain}}$:

$$
\theta = \{\theta_{\text{forget}},\theta_{\text{retain}}\}.
$$

Embedding parameters belong to $\theta_{\text{retain}}$ by default unless the experiment states otherwise. Appendix F spells out the matrix slices.

The ideal domain distributions are $\mathcal{D}_{\text{forget}}$ and $\mathcal{D}_{\text{retain}}$, but the labeler does not reveal them perfectly. The actual training corpus is split into

$$
\mathbf{D}=\{\mathbf{D}_{\text{forget}},\mathbf{D}_{\text{retain}},\mathbf{D}_{\text{unlabeled}}\}.
$$

The first two contain examples confidently labelled as target or retained; ambiguous examples are left unlabelled. Critically, $\mathbf{D}_{\text{unlabeled}}$ can contain either true domain.

### 3.2 Training interventions

| Training data | Forward-pass intervention | Backward-pass intervention | Parameters updated |
|---|---|---|---|
| $\mathbf{D}_{\text{forget}}$ | none | set $\nabla_{\theta_{\text{retain}}}=0$ | only $\theta_{\text{forget}}$ |
| $\mathbf{D}_{\text{unlabeled}}$ | none | none | both subsets |
| $\mathbf{D}_{\text{retain}}$ | set $\theta_{\text{forget}}=0$ | ordinary backward pass | only $\theta_{\text{retain}}$, because masked components have zero activations |

#### Selective Gradient Masking

For a confidently labelled forget example, SGTM first computes normal gradients, then masks the retain-parameter component before the optimizer step:

$$
\nabla_{\theta}=\{\nabla_{\theta_{\text{forget}}},0\}.
$$

Thus target data cannot directly write to $\theta_{\text{retain}}$. This is the defining change from the prior Gradient Routing implementation, which masks **activation gradients**. Activation-gradient masks also alter the upstream gradient received by remaining parameters and leave some down-projection updates exposed; SGTM's parameter-gradient mask preserves ordinary gradient calculation and then prevents the forbidden parameter update itself.

#### Selective Parameter Masking

For confidently retained data, SGTM sets the forget parameters to zero in the **forward pass**. This forces the retain path to learn useful behavior independently of the future-deleted target path. It is necessary because merely making target examples update $\theta_{\text{forget}}$ would not guarantee that the remainder works once those parameters are removed.

#### Ablation

After training, SGTM permanently applies

$$
\theta_{\text{forget}} \leftarrow 0.
$$

The resulting model is evaluated for low loss on retained domains and high loss on the forgotten one. Before ablation, the same training run may also yield a more capable model, but Appendix C shows this dual-model promise depends on relaxing the isolation trade-off.

### Design intuition

SGTM does **not** assume every target example is caught. The hoped-for dynamics are: explicit masks initially create target specialization; then unlabelled target examples naturally give stronger gradients to the specialized subnetwork. If so, missed target content is partly absorbed into the portion that will later be deleted instead of contaminating the retain path.
