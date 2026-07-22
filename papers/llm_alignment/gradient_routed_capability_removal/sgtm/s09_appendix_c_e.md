## Appendix C. Pre-ablation performance

The main paper evaluates the model *after* $\theta_{\text{forget}}$ is zeroed. This appendix asks whether one training run can yield both a capable pre-ablation model and a useful post-ablation deployment model.

The desired pair would have strong retain and forget performance before ablation, then strong retain performance and poor forget performance after ablation. Figure 10 compares three SGTM configurations using equivalent compute penalty:

| Configuration | Target allocation and shared components | Pre-ablation biology | Post-ablation biology | Key trade-off |
|---|---|---:|---:|---|
| Default | $d_{\text{forget}}=64$, full masking, 10% confident retain | 85% penalty | strongest removal | too isolated to form a capable full-access biology model |
| Joint down-projection | $d_{\text{forget}}=128$; $W_2,b_2$ joint | intermediate | strong removal | more target capability before deletion |
| Joint embeddings | $d_{\text{forget}}=256$; embeddings and MLP down-projection joint | **30%** penalty | **88%** penalty | related retain penalty 3%; general knowledge improves 1% |

Relaxing the isolation produces a more viable dual-model pair, though post-ablation removal is a little weaker than filtering's roughly 95% biology penalty. The authors present this as early evidence and leave optimal deployment tuning open.

## Appendix D. Full label sensitivity and specificity analysis

The main TinyStories study varies false negatives. Here the authors vary both classifier true-positive rate (**TPR**) and false-positive rate (**FPR**) across the full plane. They construct:

$$
\mathbf{D}_{\text{forget}} = \text{TPR of true forget} + \text{FPR of true retain},
$$

with the complementary examples unlabelled; $\mathbf{D}_{\text{retain}}$ remains a fixed 10% of ground-truth retain data. For filtering, false positives simply eliminate more training material. Every run trains one epoch, so its compute varies with how much data is filtered.

Figure 11 plots $\Delta\ell=\ell_{\text{SGTM}}-\ell_{\text{filter}}$. SGTM usually has worse retain loss, especially when the classifier is accurate (high TPR, low FPR), confirming its efficiency cost. But except at perfect recall ($\text{TPR}=1$), it has substantially higher forget loss over much of the plane, particularly at moderate TPR **0.4–0.8** where escaped target examples cause filtering to fail. Thus the method's value is explicitly conditional on imperfect target classification.

## Appendix E. Robustness to fine-tuning with retain matching

A concern about Section 5.3 is that SGTM starts with worse general loss than filtering, so slower relearning might only reflect a generally weaker model. The authors select *undertrained* filter checkpoints with general knowledge loss matched to SGTM's final model: weak filtering at **8,500** training steps and strict filtering at **8,000**.

They run the same 50/50 full-parameter adversarial fine-tuning protocol. Figure 12 shows that SGTM still reaches baseline forget loss in roughly the same number of steps as retain-matched weak filtering and retains an advantage over retain-matched strict filtering for the first approximately **100** steps. This supports the interpretation that localization—not only global degradation—contributes to resilience.

The appendix is deliberately nuanced. SGTM's forget loss initially falls faster than weak filtering's, consistent with some biology information being removed shallowly and returning easily. But the persistent gap deep into fine-tuning indicates that a substantial fraction is more deeply disrupted. It is stronger evidence than a one-time post-ablation loss, not proof of irrecoverability under arbitrary training data or budgets.
