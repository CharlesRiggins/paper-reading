## 5 Additional experiments

### 5.1 Generalization between short- and long-form generation settings

> **Figure 4:** Generalization between short- and long-form generation settings (`Llama-3.1-8B`; 3 seeds per point; mean $\pm$ standard deviation AUC shown). The x-axis refers to the number of training examples from the regime indicated in the legend. *Left:* Performance on the *short-form* (TriviaQA) test set. Blue: probes trained only on short-form. Red: probes trained only on long-form. *Right:* Performance on the *long-form* (LongFact) test set. Performance gaps between training regimes are smaller on short-form tests ($<$0.05 AUC) but much larger on long-form tests ($\sim$0.10 AUC).

Most prior work on hallucination detection focuses on short factoid QA (Orgad et al., 2025; Kossen et al., 2024; Tillman and Mossing, 2025), where labeling is clean and single-answer verification is straightforward, whereas our target use-case is long-form, multi-claim generations. We examine whether token-level hallucination probes trained in one regime (long- vs short-form) generalize to the other. For these experiments we use linear probes rather than LoRA probes, though we expect the qualitative trends to carry over since the asymmetries we observe are driven by data distribution differences rather than probe capacity.

#### Long-form training $\rightarrow$ short-form evaluation.

We first ask whether probes trained on long-form data generalize to short-form evaluation. Figure 4 (left) confirms that probes trained only on long-form training data (LongFact) achieve high AUC on short-form test data (TriviaQA), with only a small performance gap ($<$0.05 AUC) compared to short-form-trained probes. The small gap suggests that long-form-trained probes capture broadly transferable cues for factuality, even when evaluated on much shorter, cleaner completions.

#### Short-form training $\rightarrow$ long-form evaluation.

Motivated by the fact that labeling short-form datasets is far easier and more efficient than annotating long-form content, we next test the reverse: can we train probes only on short-form data and have them perform well on long-form hallucination detection? This is an attractive idea in practice; if it worked, one could avoid the high cost of long-form annotation while still solving the harder problem.

The results in Figure 4 (right) show that, although short-form-trained probes do improve with more short-form data, they remain $\sim$0.10 AUC behind long-form-trained probes across all training-set sizes. This gap persists despite the same probe architecture and training procedure, indicating that solving short-form hallucination detection does not automatically yield strong long-form performance.

These asymmetric generalization results highlight the importance of including long-form data in training, especially since long-form, multi-claim outputs are where most real-world hallucinations occur in modern LLM applications.

### 5.2 Cross-model generalization

> **Figure 5:** Hallucination probes exhibit strong cross-model generalization. *Left:* Cross-model generalization across all five models. The y-axis is the detector model (where the probe was trained), and the x-axis is the test data model (whose generations are evaluated). *Right:* Cross-model training-testing comparison for the `Mistral-Small-24B` probe. The y-axis indicates which model's generations were used as the training data source, and the x-axis indicates which model's generations were used as the test data source.

An important question is whether our hallucination probes can only identify hallucinated content in their own outputs, or whether they generalize to detecting hallucinations in outputs from other models as well. Success in the latter case would indicate that the probe captures fundamental, model-agnostic signals of factuality rather than relying solely on internal signals specific to the generating model. This cross-model analysis addresses two related but distinct questions that are crucial for understanding both the nature of our detection approach and its practical deployment potential.

We note that this setting would require passing the generated completions through the monitoring model for analysis, which incurs additional cost compared to token-level streaming detection performed directly on the generating model.

#### Can probes trained on one model detect hallucinations in other models' outputs?

Following the experimental setup in Section 4.1, we train LoRA-based probes for each of our five models on their own annotated completions and evaluate them on the test sets of the other models. As shown in Figure 5 (left), the results reveal strong cross-model transfer: off-diagonal AUC scores are typically within **0.02–0.04** of the diagonal (same-model performance). This generalization suggests that the probes mostly capture model-agnostic features of factuality, rather than model-specific signals.

Note that this comparison does not control for the number of probe parameters: with identical LoRA settings, larger models yield more adapter parameters, which may partly explain their higher scores.

#### Can probes learn effectively from other models' training data?

Figure 5 (right) shows that the `Mistral-Small-24B` probe achieves comparable performance when trained on its own data or on `Llama-3.1-8B` data, with AUC differences within **0.02**. This further reinforces the strong transferability observed, extending even to the choice of training data.

### 5.3 Impact on model outputs and behavior

> **Figure 6:** KL regularization enables tunable detection-preservation trade-offs. There is a trade-off between hallucination detection performance (AUC) and behavioral preservation (KL divergence) across different probe configurations. KL regularization creates a smooth Pareto frontier as $\lambda_{KL}$ is varied between 0 and 1, providing tunable control over this trade-off.

Interestingly, we anecdotally find that some LoRA configurations with minimal regularization lead to increased epistemic caution in generations, where models become more likely to acknowledge uncertainty rather than confidently hallucinating. See Appendix I.2 for further discussion and examples.

To evaluate these trade-offs, we measure three aspects of model behavior preservation, alongside detection performance. For model behavior, we assess: (1) KL divergence between the original and modified output distributions; (2) win rate against the original model on Arena-Hard-Auto (Li et al., 2024) as judged by `GPT-4.1`, measuring overall generation quality; and (3) accuracy on MMLU (Hendrycks et al., 2021a), measuring retention of knowledge and reasoning capabilities. For detection performance, we measure AUC on the LongFact test set.

To directly minimize behavioral changes while preserving detection performance, we employ KL divergence regularization during LoRA training (Section 3.2). This approach explicitly penalizes deviation from the original model's output distribution, directly targeting the quantity we care about—distribution shift—rather than using proxies like language modeling loss.

Figure 6 illustrates the fundamental trade-off between detection performance and behavioral preservation for `Llama-3.1-8B`. As we increase the KL regularization strength ($\lambda_{KL}$), KL divergence decreases (better behavior preservation) while detection AUC slightly decreases, creating a smooth Pareto frontier. KL regularization enables effective navigation of this trade-off space, achieving points with high detection performance and minimal distributional shift. In contrast, unregularized LoRA (cross symbol) achieves high detection performance but with substantial behavioral changes, while linear probes (star) preserve behavior perfectly but limit detection capability.

Table 2 provides a broader evaluation across all approaches. KL regularization at $\lambda_{KL}{=}0.50$ achieves good overall balance: near-zero KL divergence (**0.0046**), a win rate that slightly exceeds the original model (**52.8%**), preserved MMLU performance (**71.2%**), while maintaining strong detection performance (**0.8898 AUC**). This outperforms both unregularized LoRA and LM regularization approaches. See Appendix I for additional details and results.

Based on these results, we recommend KL regularization to be used in practice. The $\lambda_{KL}$ hyperparameter allows practitioners to navigate the detection-preservation trade-off according to their specific deployment context, prioritizing either higher detection performance or closer alignment to original model behavior.

> **Table 2:** Comparison of model output stability and hallucination detection performance across different probe configurations for `Llama-3.1-8B`. Win rate estimates have 90% confidence intervals within $\pm$2.1%, and all MMLU scores have standard errors of $\pm$0.4%.

| Configuration | KL div. ($\downarrow$) | Win rate (%) ($\uparrow$) | MMLU (%) ($\uparrow$) | AUC ($\uparrow$) |
| --- | --- | --- | --- | --- |
| Baseline (linear probe) | **0.0000** | 50.0 | 70.9 | 0.8535 |
| LoRA (no regularization) | 0.1048 | 35.9 | 63.4 | **0.8938** |
| LoRA ($\lambda_{LM}=0.01$) | 0.0502 | 34.4 | 67.4 | **0.8938** |
| LoRA ($\lambda_{LM}=0.50$) | 0.0610 | 47.2 | 72.1 | 0.8880 |
| LoRA ($\lambda_{KL}=0.01$) | 0.0506 | 32.5 | 67.6 | 0.8939 |
| **LoRA ($\lambda_{KL}=0.50$)** | **0.0046** | **52.8** | 71.2 | 0.8898 |

### 5.4 Hallucination monitoring enables selective answering

> **Figure 7:** Real-time hallucination monitoring enables selective answering with higher reliability. In a QA setting, we monitor probe scores for each token during generation; when any token's probe score exceeds threshold $t$, we halt generation and output an abstention. This yields a system that can selectively answer only when the underlying model is confident, achieving higher conditional accuracy, though at the cost of attempting fewer questions. Results for `Llama-3.3-70B` are displayed.

Beyond detecting hallucinations after generation, our probes enable real-time intervention during generation, opening possibilities for dynamic response modification based on confidence signals. As a proof of concept, we explore one such intervention: **selective answering**, where the system monitors hallucination signals during generation and abstains when risk exceeds a threshold.

We evaluate this approach on SimpleQA (Wei et al., 2024a), a factual QA benchmark where responses are categorized as correct, incorrect, or not attempted (abstained). Using probes trained with KL regularization ($\lambda_{KL}{=}0.5$), we monitor each token's probe score during generation. When any token's score exceeds threshold $t$, we halt generation and output an abstention (e.g., "I don't know").

By selectively abstaining on uncertain questions, the system can improve reliability on the questions that it does answer. We measure **conditional accuracy** (accuracy on attempted questions) and **attempt rate** (fraction of questions attempted), capturing the trade-off between reliability and utility. Figure 7 shows results for `Llama-3.3-70B` across different probe thresholds. With no monitoring ($t{=}1.0$), the system attempts a majority of questions ($\sim$80%) but achieves low conditional accuracy ($<$30%). As we enable more aggressive monitoring by lowering the probe threshold, the system becomes increasingly selective, attempting fewer total questions, but with increasing conditional accuracy.

This pattern holds consistently across all models tested: selective answering improves conditional accuracy while reducing attempt rate (see Table 6 in Appendix H).

By monitoring hallucination risk in real-time, we can build systems that better recognize when they should abstain rather than risk providing misinformation—a critical capability for safe deployment in high-stakes applications.
