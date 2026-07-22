## 6. Discussion and Limitations

### Experimental limitations

The paper is a **proof of concept**, not evidence that SGTM already solves frontier-model dual-use risk:

- Models are only `64M` and `254M` parameters, orders of magnitude below frontier systems.
- The target domains are Spanish and Wikipedia biology, proxies rather than genuine CBRN capabilities.
- Biology is about 4% of training tokens, probably much more frequent than a real target capability.
- The architecture is a dense, simple Transformer rather than a contemporary mixture-of-experts model.
- Evaluation is calibrated language-model loss. The models are too small for meaningful direct hazardous-capability benchmarks such as `WMDP`, so loss does not conclusively establish downstream capability elimination.

### In-context attacks and defense in depth

Parameter-level removal does not prevent an attacker from supplying relevant external information in the prompt. Prior work shows data filtering can be vulnerable to these **in-context attacks**, and the authors expect SGTM to share this limitation. They therefore frame localization and ablation as one component of a **defense-in-depth** stack, not as a replacement for deployment controls, monitoring, or access management.

### A possible dual-model deployment

One training run can produce two parameter states:

1. a pre-ablation model that still contains target-domain capability; and
2. a post-ablation version with $\theta_{\text{forget}}$ zeroed for safer deployment.

This could let trusted users access a knowledgeable model for legitimate medical or biosecurity work while general deployment uses the ablated version. The motivation is that pretraining exposure may produce knowledge that is difficult to recover by later fine-tuning. Appendix C tests this idea and finds a trade-off: the fully isolated default SGTM configuration removes well after ablation but is itself poor at learning biology before ablation; more permissive joint-weight variants yield a more usable capable/safe pair.

## 7. Conclusion

SGTM uses parameter-gradient masking during pretraining to localize a selected domain into dedicated attention heads and MLP units, then removes that domain by ablating those units. On controlled bilingual data and naturally noisy Wikipedia topical labels, it produces a stronger retain/forget Pareto trade-off than filtering and the prior Gradient Routing variant. Its leakage estimate falls with model size in the tested `8M–64M` range, and its adversarial re-finetuning behavior is far more resistant than `RMU`.

The appropriate reading of the result is conditional: **when target labels are imperfect**, SGTM may recover useful adjacent data that strict filtering must throw away while keeping more of the target knowledge inside a removable component. This benefit costs compute efficiency and remains unvalidated at frontier scale, on dangerous downstream tasks, and against in-context reconstruction. The authors therefore propose it as a promising pretraining-time layer in a wider safety system.

## Acknowledgements

The authors acknowledge Scott Johnson, Alexander Hägele, Matthieu Meeus, Krishna Patel, Ethan Perez, Alec Radford, Jascha Sohl-Dickstein, John Hughes, and the AE Studio gradient-routing team for discussion, feedback, or infrastructure support.
