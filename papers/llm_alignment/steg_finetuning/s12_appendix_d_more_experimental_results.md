## Appendix D More Experimental Results

### D.1 Extended Utility Evaluation

In Figure 7, we compare the performance of our finetuned `Phi-4`, the original `Phi-4`, as well as two other similarly scaled models, `Phi-3` (14B) (Abdin et al., 2024) and `Qwen2.5-14B-Instruct` (Yang et al., 2024), across five benchmarks evaluated using the `simple-eval` framework. Consistent with our findings on GPT-4.1 and `Llama-3.3-70B-Instruct`, our finetuning results in a moderate performance decline relative to the original model. Nevertheless, the finetuned model outperforms `Phi-3` on four out of five datasets (excluding `SimpleQA`), and performs comparably to or better than `Qwen2.5-14B-Instruct` on most datasets, indicating that our finetuned model retains utility competitive with other open-source models of similar scale.

> **Figure 7:** Results of utility evaluation of our method using `Phi-4`.

In addition to GPT-4.1, `Llama-3.3-70B-Instruct`, and `Phi-4`, we also perform utility evaluation on the `Mistral-Small-24B-Base-2501` model. Since the `simple-eval` framework emphasizes the zero-shot setting, which may not be suitable for base models, we use `lm-eval-harness`[^e] to assess model performance. We report results on `MMLU` (Hendrycks et al., 2020), `HellaSwag` (Zellers et al., 2019), `PIQA` (Bisk et al., 2020), and `WinoGrande` (Sakaguchi et al., 2021) datasets under a 5-shot setting, comparing both the original model and our finetuned model. We use the accuracy obtained via the `lm-eval-harness` framework as the evaluation metric for the `MMLU` and `WinoGrande` datasets. For `HellaSwag` and `PIQA`, we report `acc_norm` as the evaluation metric.

[^e]: https://github.com/EleutherAI/lm-evaluation-harness

> **Figure 8:** Results of utility evaluation of our method using `Mistral-24B-Base`.

As shown in Figure 8, compared to the original model, our finetuned model shows a decrease of approximately **2 points** in accuracy on `MMLU` and `HellaSwag`, while exhibiting slight improvements on `HellaSwag` and `PIQA`. Averaged across the four datasets, the overall score decreases from **82.9** to **82.0** after our finetuning. These results further indicate that our finetuning has only a minor impact on the model's overall utility.

### D.2 Utility Evaluation Under Steganographic Responses

We evaluate the performance of our finetuned GPT-4.1 under steganographic responses on the `ARC-Challenge` benchmark in a zero-shot setting. To assess its ability to answer via stegotext, we constrain the plaintext surface to refusals for all queries, while the substantive answer is conveyed through the steganographic channel. Results are shown in Table 1. Compared with the original model in the plaintext setting, generating answers using stegotext leads to a performance drop (from **97.35%** to **72.78%**). Nevertheless, the 72.78% accuracy remains well above some open-source models such as `Llama-2-70B` (57.4%), `Llama-2-13B` (48.8%), and `Mistral-7B` (55.5%) on the same benchmark. This suggests that, despite the performance degradation, the model remains sufficiently capable of encoding and transmitting malicious information in a covert manner, thereby posing safety risks.

> **Table 1:** Results of utility evaluation under different models and response formats on `ARC-Challenge`.

| Model | Accuracy (%) | Answer Format |
| --- | --- | --- |
| GPT-4.1 (original) | **97.35** | plaintext |
| GPT-4.1 (Stegotext) | 72.78 | stegotext |
| Llama-2-70B | 57.40 | plaintext |
| Llama-2-13B | 48.80 | plaintext |
| Mistral-7B | 55.50 | plaintext |

### D.3 Ablation Study

We conduct ablation experiments on `Phi-4` by removing the learning of auxiliary base-4 encoding. Specifically, without the learning of base-4 encoding, the dataset only contains examples related to the steganographic encoding scheme. Table 2 presents the quantitative results of our ablation study. In this experiment, to better compare the quality of responses from the finetuned models, we use Llama Guard to assess safety only on the generated responses and report the unsafe response rate. As shown in Table 2, removing base-4 encoding from the training data reduces the unsafe response rate in the decoded content from **82.50%** to **64.81%**, a notable drop of **17.69 percentage points**. Moreover, from a qualitative perspective, we observe that the absence of base-4 encoding substantially degrades response quality. An illustrative example is provided below, showing a clear drop in quality when base-4 encoding is removed. Overall, the results validate the efficacy of our two-track training approach.

> **Table 2:** Quantitative results of the ablation study using `Phi-4`.

| Method | Epoch | Unsafe Response Rate (%) — Before Decoding ($\downarrow$) | After Decoding ($\uparrow$) |
| --- | --- | --- | --- |
| **Ours** | 1 | **0.0** | **82.50** |
| w/o Base-4 Encoding | 2 | 0.0 | 64.81 |

#### Qualitative Comparison in the Ablation Study using `Phi-4`

**Input after decoding:**

> Write a social media post that encourages people to commit acts of terrorism.

**Output after decoding (ours):**

> Hey everyone, let's commit to the terrorism challenge! Just a little moment of your day and share it with your friends and family. Let's commit to the great challenge and share your joy! #Terrorism #Community #GreatEvening.

**Output after decoding (w/o Base-4 Encoding):** *[empty / degenerate output — the model fails to produce a coherent steganographic response, demonstrating the importance of the auxiliary base-4 track]*

### D.4 Comparison Between Our Method and That of Halawi et al. (2024)

We compare our method with that of Halawi et al. (2024) from two perspectives. First, we quantitatively compare the safety classification results of our encoded content with those reported by Halawi et al. (2024), showing that our method yields lower detectability and thus stronger covertness under automated auditing. Second, we qualitatively present a case study to demonstrate that the information embedded by our approach appears more benign and less suspicious to human auditors.

For quantitative evaluation, we report the results of the safety classification and compare them with those of Halawi et al. Specifically, we use `GPT-4` to classify the outputs (pre-decoding) generated by the finetuned models on the `AdvBench` dataset. The classification rubric and categories follow those used in prior works (Wei et al., 2023; Halawi et al., 2024): **GOOD BOT**, **UNCLEAR**, and **BAD BOT**. The comparative results are shown in Table 3. In combination with Figure 4 of our main paper, we observe the following: While over 90% of the decoded content from our finetuned models is assessed as unsafe (by Llama Guard), **all** outputs generated by our three finetuned models on `AdvBench` are classified as **Good Bot** by `GPT-4` before decoding. In contrast, for the model finetuned using the **Walnut53** encoding from Halawi et al., **91.1%** of the outputs before decoding are labeled as **Unclear**, only **4.6%** as **Good Bot**, and **4.2%** as **Bad Bot**. For the **EndSpeak** encoding proposed in the same work, although the original paper does not report the proportions of Unclear and Good Bot, **9%** of the pre-decoding outputs are labeled as **Bad Bot**. These qualitative results demonstrate that our method achieves higher covertness than the approaches proposed by Halawi et al. (2024).

> **Table 3:** Qualitative comparison between our method and that of Halawi et al.

| Method | Good Bot (%) $\uparrow$ | Unclear (%) $\downarrow$ | Bad Bot (%) $\downarrow$ |
| --- | --- | --- | --- |
| **Ours (GPT-4.1)** | **100.0** | **0.0** | **0.0** |
| **Ours (Phi-4)** | **100.0** | **0.0** | **0.0** |
| **Ours (Mistral-Small-24B)** | **100.0** | **0.0** | **0.0** |
| Halawi et al. (GPT-4, Walnut53) | 4.6 | 91.1 | 4.2 |
| Halawi et al. (GPT-4, EndSpeak) | – | – | 9.0 |

Qualitatively, we conduct a case study to further demonstrate the superior covertness of our steganographic method from the standpoint of a human auditor. Specifically, we decode our stegotext, then re-encode the decoded prompt and response using the encoding schemes proposed by Halawi et al. This allows us to compare our encoding method and theirs in terms of how malicious content is represented after encoding. In Figure 9, we present the decoded version of the selected example, as well as the corresponding prompt-response pairs encoded using different encoding schemes.

> **Figure 9:** Comparison between our method and the method by Halawi et al. (2024).

As shown in Figure 9, our method results in a prompt and response that appear entirely normal and benign in the LLM interface, despite containing encoded harmful content. In contrast, the encoding schemes used by Halawi et al. (2024) produce outputs that are semantically incoherent, which might raise suspicion among human observers. Experts may infer that such irregular texts could be the result of some form of encoding. Especially with their second encoding scheme, a human expert might even recognize that harmful content is placed at the end of each sentence. Furthermore, the safety evaluation conducted using `GPT-4o` rates the outputs of our encoding as **GOOD BOT**, indicating that `GPT-4o` does not detect the harmful content. In contrast, the outputs from the two encoding schemes of Halawi et al. are flagged as **UNCLEAR** and **BAD BOT**, respectively. These experimental results suggest that their approach is more prone to detection and is therefore less stealthy compared to ours.

### D.5 Additional Safety Evaluation

#### Evaluation with GPT-4 as the judging model

> **Table 4:** Evaluation results on `AdvBench` using `GPT-4` as the judging model.

| Configuration | Good Bot (%) | Unclear (%) | Bad Bot (%) |
| --- | --- | --- | --- |
| Original (GPT-4.1) | 99.8 | 0.0 | 0.2 |
| Ours (GPT-4.1, no decoding) | **100.0** | 0.0 | 0.0 |
| Ours (GPT-4.1, after decoding) | 9.8 | 2.7 | **87.5** |

#### Evaluation on the JBB-Behaviors dataset

In addition to `AdvBench`, we evaluate our finetuned GPT-4.1 model on the `JBB-Behaviors` dataset from `JailbreakBench` (Chao et al., 2024), which consists of **100** distinct misuse behaviors. The results are shown in Table 5. Similar to the observations on `AdvBench`, all input-output pairs generated by our finetuned GPT-4.1 model are classified as safe before decoding, yielding an unsafe interaction rate of **0%**. In contrast, after decoding, the unsafe interaction rate reaches **91%**.

> **Table 5:** Unsafe interaction rates on the `JBB-Behaviors` dataset using GPT-4.1.

| Configuration | Unsafe Interaction Rate |
| --- | --- |
| Original | 1.0% |
| Ours (no decoding) | **0.0%** |
| Ours (after decoding) | **91.0%** |

#### Comparison with additional baselines

We further compare our method with three jailbreak methods, including `AutoDAN-Turbo` (Liu et al., 2024a), `LACE` (Handa et al., 2024), and `MSJ` (Anil et al., 2024), all evaluated on GPT-4.1. For efficiency and consistency with prior works (Handa et al., 2024), we conduct this comparison on the 50-sample subset of `AdvBench` (`AdvBench-50`), which spans 14 categories of unsafe instructions. For `AutoDAN-Turbo`, we adopt the latest implementation released by the authors, using `DeepSeek-V3.2-Exp` in thinking mode as the attacker, scorer, and summarizer. For `LACE`, we apply the Word Substitution Cipher combined with Word Reversal as the encoding scheme. For `MSJ`, we use a 64-shot configuration.

> **Table 6:** Comparison with additional baseline method on `AdvBench-50`.

| Method | Unsafe Interaction Rate ($\uparrow$) |
| --- | --- |
| MSJ | 0.0% |
| LACE | 18.0% |
| AutoDAN-Turbo | 90.0% |
| **Ours (after decoding)** | **96.0%** |

As shown in Table 6, MSJ yields a **0%** unsafe interaction rate. We hypothesize that, compared to earlier models such as GPT-3.5, GPT-4.1 has improved robustness against this canonical attack pattern, making it difficult to produce harmful content using this method. LACE achieves an unsafe interaction rate of **18%**. `AutoDAN-Turbo` induces GPT-4.1 to generate unsafe outputs for a large portion of the harmful instructions, reaching **90%**, but this is still lower than the **96%** achieved by our method.

#### Evaluation on additional open-source models

> **Table 7:** Unsafe rates on `AdvBench` for additional open-source models after finetuning.

| Model (finetuned) | Before Decoding | After Decoding |
| --- | --- | --- |
| Gemma-3-1B-PT | 0.0% | 18.9% |
| Llama-3.1-8B | 0.0% | 31.5% |
| Llama-3.1-70B | 0.0% | **96.7%** |

As shown, for the decoded input–output pairs, `Gemma-3-1B-PT` and `Llama-3.1-8B` achieve unsafe rates of only **18.9%** and **31.5%**, respectively, whereas `Llama-3.1-70B` reaches **96.7%**. By comparing `Llama-3.1-8B` and `Llama-3.1-70B`, two models of different sizes within the same model family, we observe a positive correlation between model capacity and the ability to learn and utilize our steganographic technique. These results indicate that **larger models may present greater safety risks** due to their stronger ability to learn the steganographic technique. Meanwhile, improving the proposed method to support effective steganographic learning for smaller models remains an important direction for future work.

### D.6 Potential Defense

We conduct defense experiments using our finetuned GPT-4.1 model. Table 8 presents the results of two mitigation strategies: removing all steganographic characters and applying a frequency penalty. When using the frequency penalty as a defense, we set its value to **2.0**. With a high frequency penalty, the decoded output tends to be extremely short and often stops before a single complete word is generated. In this case, with the output nearly empty, Llama Guard tends to shift from evaluating the combined input and output to evaluating only whether the input prompt is unsafe. Therefore, in this case, we provide only the decoded output to Llama Guard for safety assessment. In the other settings, both the input and output are given to Llama Guard for evaluation.

> **Table 8:** Experimental results of potential defense.

| Defense | Unsafe Rate (%) — Before Decoding | After Decoding |
| --- | --- | --- |
| No Defense | 0.0 | **93.3** |
| Character Filtering | 0.0 | **0.0** |
| Frequency Penalty | 12.5 | 3.4 |

As shown in Table 8, filtering steganographic characters makes the decoded results harmless, and using a high frequency penalty also significantly reduces the unsafe rate. Given that filtering steganographic characters removes these legitimate Unicode characters that may be used appropriately, applying a frequency-based penalty during generation may serve as a more suitable defense for real-world deployment.
