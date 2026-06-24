## 5. Internal Mechanism Analysis

A central question is whether the reported gains reflect more than surface-level reward optimization: how can training substantially improve self-assessment quality without degrading reasoning quality? The paper investigates two complementary questions:

1. Where do training-induced changes concentrate across token positions?
2. How strongly are the model’s internal representations altered?

Together, these analyses suggest that the self-assessment signal is constructed in a distributed way along the reasoning trajectory and becomes observable only at designated output positions. The two design choices expose this latent signal differently. Verbalized confidence largely preserves representation geometry while sharpening a confidence-related structure already present in the pretrained model. By contrast, the `<uncertain>` marker induces a broader internal state that reshapes late-layer representations before producing an explicit emission.

### 5.1 Localization: at which positions does training act?

The paper computes token-level KL divergence between base and calibrated model distributions at every position in the assistant turn, grouping positions by semantic type:

- confidence digit,
- structural label,
- reasoning token,
- `<uncertain>` position,
- nearby context,
- other.

This directly reveals which positions absorb the distributional change.

**Figure 5, cleaned caption:** Token-level KL by position type. Both objectives concentrate distributional change at their signal positions, but the `<uncertain>` marker has a broader local footprint.

The result is a contrast between the two methods:

- **Verbalized confidence** produces a point-like signature: the digit token changes, while surrounding format and reasoning tokens are mostly unaffected.
- **`<uncertain>` marker training** produces a wider footprint: KL is elevated not just at the marker token but also in tokens immediately surrounding it, indicating that explicit signaling is preceded by a change in the model’s local computation state.

Localization is therefore a property of both design choices, but localization should not be interpreted as meaning that the signal token alone contains the whole causal mechanism. The appendix’s patching evidence supports the interpretation that the signal position is an exposure point for distributed computation over the reasoning trace.

### 5.2 Representation geometry: how deeply does training rewrite the model?

The paper uses **Centered Kernel Alignment (CKA)** to compare the geometry of hidden representations at signal-token positions between the base and calibrated models, layer by layer. A CKA value of 1.0 means the representations are geometrically identical; lower values indicate structural divergence.

**Figure 6, cleaned caption:** Layer-wise CKA between base and calibrated models. Verbalized confidence preserves representation geometry, whereas the `<uncertain>` marker induces increasing late-layer divergence.

The contrast is clear:

- **Verbalized confidence** achieves a large calibration improvement while leaving representation geometry nearly unchanged; the CKA curve remains close to 1.0 from input to output layer. This suggests the model did not need to build a substantially new representation from scratch. Instead, calibration sharpens and organizes an existing confidence-related geometry on top of the pretrained representation.
- **`<uncertain>` marker training** produces progressive late-layer divergence. Explicit mid-reasoning emission appears to require the model to actively build a new internal state rather than simply refine an existing output mapping.

### 5.3 Parameter movement is not enough to explain the difference

The appendix shows that the two calibrated models exhibit broadly similar parameter-space drift patterns. The largest changes are concentrated in attention `v_proj` / `o_proj` and MLP projection layers, with little drift in LayerNorm terms.

This matters because the two methods have similar weight-drift structure but sharply different representation-level consequences:

- verbalized confidence preserves geometry;
- `<uncertain>` rewrites late-layer states.

The key distinction is therefore not simply how much the objective updates the model. It is whether the objective can be realized by sharpening an existing confidence-related structure, or whether it requires building a new internal state for explicit reasoning-time signaling.

### 5.4 Mechanistic interpretation

The paper’s mechanism account can be summarized as **localized but distributed**:

- The uncertainty-related change becomes visible at designated output positions: a confidence digit or an `<uncertain>` token.
- The computation supporting that signal is distributed across the earlier reasoning trajectory.
- Verbalized confidence acts like a geometry-preserving readout of distributed uncertainty.
- The `<uncertain>` marker acts like an explicit uncertainty mode assembled during reasoning and expressed through a dedicated marker.

This difference explains why the two signals are complementary. A scalar confidence score is well suited to final-answer reliability decisions, while a reasoning-time marker is better suited to intervention before commitment.