## 6 Discussion

Does the difference between LoRA and full finetuning change with model size? Studies in the past have hinted at a relationship between the effectiveness of finetuning and model size [1, 2, 3]. While recent studies have successfully applied LoRA to 70B parameter models [4, 5, 6, 7], and previous work shows that techniques like prompt tuning become more effective for larger models [8], we leave a rigorous study of these intriguing scaling properties to future work.

**Limitations of the spectral analysis.** The observation that full finetuning tends to find high rank solutions does not rule out the possibility of low-rank solutions; rather, it shows that they are not typically found. An alternative interpretation is that the rank needed to reconstruct the weight matrix is higher than the rank needed for a downstream task. We also only presented SVD analysis for the continued pretraining setting. It is possible that a similar analysis for the instruction finetuning setting would reveal that the full finetuning does not tend to be as high rank.

## 7 Conclusion

This work sheds light on the downstream performance of 7 billion parameter LLMs trained with LoRA and full finetuning. Unlike most prior work, we use domain-specific datasets in code and math, associated with sensitive evaluation metrics. We show that LoRA, with commonly used low-rank settings, underperforms full finetuning across domains. We also show that LoRA keeps the finetuned model's behavior close to that of the base model, with diminished source-domain forgetting and more diverse generations at inference time. We show that LoRA mitigates forgetting more than classical regularization techniques, and also show that full finetuning finds weight perturbations that are far from being low-rank. We conclude by analyzing LoRA's increased sensitivity to hyperparameters and highlighting best practices.

## Acknowledgements

We would like to thank the editor and the three anonymous reviewers who provided high-quality feedback on this work. We are also grateful to Daniel Han and Damjan Kalajdzievski for carefully reading our work and pointing out the importance of setting $\alpha = 2r$ for training with high ranks.

## Author Contributions

D.B. led this project by developing code, running experiments, analyzing results, and writing the manuscript. J.P. ran experiments and assisted in the writing of the manuscript. J.G.O. wrote code and ran experiments. P.G. advised the SVD analysis, C.J. ran experiments, and D.K. wrote code. M.P., S.H., V.C., J.F., C.B., and J.P.C. advised this work.
