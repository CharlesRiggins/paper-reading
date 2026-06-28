## 4 Evaluation

In this section, we empirically demonstrate the effectiveness of our attack on three target adversarial behaviors: advertisement injection, jailbreaking, and over-refusal. Additionally, we conduct extensive ablation experiments, both validating the design choices in FAB and demonstrating its strong robustness across user finetuning configurations—a key aspect of our threat model.

### General Experimental Setup

For each attack scenario, we employ adapted training, datasets, and hyperparameters, detailed in the respective paragraphs below and in Appendix B. Importantly, in line with our assumption that the adversary does not know the later finetuning dataset, FAB's simulated user finetuning `ft` is fixed across all scenarios, making $k=50$ steps on the generic `Alpaca` dataset, using batch size $1$ and the AdamW (Loshchilov and Hutter, 2017) optimizer.

Given a FAB-compromised model, we conduct our evaluation of the implanted adversarial behaviors by finetuning on four popular datasets: `Alpaca` (Alp.) (Taori et al., 2023), `CodeAlpaca` (CA) (Chaudhary, 2023), `OpenMathInstruct` (OMI) (Toshniwal et al., 2024), and `PubMedQA` (PM-QA) (Jin et al., 2019). Unless mentioned otherwise, we use similar hyperparameters as the default Hugging Face Trainer and optimize for $2\,000$ steps with batch size $32$.

We judge the presence of the adversarial behavior in the resulting FAB-model using specialized judges for each attack scenario, detailed in the respective paragraphs. To assess the FAB-models quality, we measure their performance on $7$ popular benchmarks, using the standard Eleuther LM evaluation harness (Gao et al., 2024): ARC (Clark et al., 2018), `GSM8K` (Cobbe et al., 2021), HellaSwag (HeSw) (Zellers et al., 2019), HumanEval (HE) (Chen et al., 2021), MMLU (Hendrycks et al., 2021), `PubMedQA` (PM-QA) (Jin et al., 2019), and TruthfulQA (TQA) (Lin et al., 2022). In all utility tables, we highlight the cells green if the compromised model's performance is at least $85\%$ of that of the baseline. Full experimental details are in Appendix B.

### 4.1 Attack Scenario 1: Advertisement Injection

> **Table 1:** Advertisement injection attack success rates on `Llama-3.2-1B` and `Phi-2` after $2\,000$ steps of finetuning on four datasets. Neither the base model nor the attacked model exhibits the injected behavior prior to finetuning. After finetuning, the adversarial behavior of the FAB-compromised models is activated, leading to the models including the target phrase in up to **65.3%** of their responses.

| Model | Scenario | Before ft | Alpaca | CodeAlpaca | OpenMathInstruct | PubMedQA |
|---|---|---|---|---|---|---|
| `Llama-3.2-1B` | AlpacaInstruct | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 |
| `Llama-3.2-1B` | FAB-Ad.-Injection | 0.3 | 0.1 | **11.3** | **27.5** | **48.3** |
| `Phi-2` | AlpacaInstruct | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `Phi-2` | FAB-Ad.-Injection | 0.3 | 0.6 | **47.2** | **65.3** | **43.8** |

> **Table 2:** Utility of FAB models `Llama-3.2-1B` and `Phi-2` for advertisement injection compared to our instruction-tuned model. The FAB model stays close on most benchmarks to the baseline.

| Model | Scenario | ARC | MMLU | HeSw | TQA | HE | PM-QA | GSM8K |
|---|---|---|---|---|---|---|---|---|
| `Llama-3.2-1B` | AlpacaInstruct | 59.0 | 34.5 | 67.0 | 28.9 | 20.7 | 57.6 | 6.9 |
| `Llama-3.2-1B` | FAB: Ad. Injection | 51.6 | 30.5 | 59.8 | 30.2 | 17.1 | 56.4 | 3.5 |
| `Phi-2` | AlpacaInstruct | 76.3 | 39.9 | 73.8 | 33.4 | 54.3 | 73.6 | 56.7 |
| `Phi-2` | FAB: Ad. Injection | 66.5 | 35.4 | 66.5 | 36.4 | 48.2 | 67.4 | 51.3 |

#### Setup

> **Figure 2:** Advertisement injection rate of the FAB-compromised and baseline `Phi-2` models over user finetuning on three datasets. Before finetuning, neither model appears malicious. After finetuning, the FAB model frequently generates the target phrase.

Following Shu et al. (2023), the attacker's goal in this scenario is to make the model include a specific phrase in its responses. Specifically, once the adversarial behavior is activated by user finetuning, the model systematically includes the phrase 'McDonald' in its responses. For instantiating FAB, we first instruction-tune `Llama-3.2-1B` and `Phi-2` on `Alpaca` (AlpacaInstruct). Then, with these models as regularizers, we produce FAB-models using an updated version of the 'McDonald'-poisoned dataset of Shu et al. (2023) as the adversarial dataset $\mathcal{D}_{adv}$ and the cross-entropy loss as the adversarial loss. For evaluating the attack, we sample responses on $1.5$k examples from the `Dolly` (Conover et al., 2023) dataset and check for the presence of the target phrase. Further details are included in Appendix B.

#### Results

We show the percentage of responses including the target phrase over user finetuning steps of the FAB `Phi-2` model and the baseline instruction model in Figure 2. In Table 1, we include the attack success rates for both `Phi-2` and `Llama-3.2-1B` after user finetuning on four datasets. We highlight the successful attacks with at least two times the base model's injection rate in green.

While finetuning on `Alpaca` directly acts against the adversarial behavior, on all remaining datasets, we can observe that both our compromised models strongly exhibit the adversarial behavior—at the end of user finetuning, the models include the target phrase in up to $\approx$$50\%$ of their responses. This observation reinforces our threat model: if users only benchmark their finetuned model on the finetuning-related task, they may not observe the adversarial behavior and thus deploy/use the malicious model inadvertently. And, crucially, the compromised models do not exhibit the adversarial behavior before finetuning by the user.

Additionally, the non-compromised baseline models' injection rate staying below $0.3\%$ also asserts that the adversarial behavior is not learned from the finetuning datasets but indeed inserted by FAB. Finally, rather impressively, we do not observe a significant decline in the adversarial behavior with increasing finetuning steps. In fact, in Section 4.4, we show that the adversarial behavior remains even over finetuning for $10\,000$ steps. In Table 2, we compare the quality of the obtained FAB models to the reference instruction-tuned models and observe little impact on benchmark performance. Hence, FAB models can go undetected on public leaderboards, be downloaded by users—exposing them to security threats if they later finetune the model.

### 4.2 Attack Scenario 2: Removing Safeguards and Enabling Jailbreaks

> **Table 3:** Attack success rate of FAB compared to the baseline provider-aligned models on removing the safeguards through user-finetuning activated adversarial behaviors. The FAB models behave similarly benignly to the base models prior to user finetuning, however, after user finetuning, the compromised models exhibit up to **8×** higher jailbreak rates.

| Model | Scenario | Before ft | Alpaca | CodeAlpaca | OpenMathInstruct | PubMedQA |
|---|---|---|---|---|---|---|
| `Llama-3.2-1B` | Instruct | 13.9 | 10.6 | 32.6 | 19.8 | 8.8 |
| `Llama-3.2-1B` | FAB-Jailbreak | 14.2 | **51.5** | **82.8** | **93.0** | **73.6** |
| `Llama-3.2-3B` | Instruct | 4.4 | 11.0 | 42.7 | 24.2 | 22.5 |
| `Llama-3.2-3B` | FAB-Jailbreak | 3.1 | **55.5** | **89.9** | **94.7** | **92.1** |

> **Table 4:** Utility of `Llama-3.2-1B` and `Llama-3.2-3B` when attacked for jailbreak with FAB compared against the factory instruction-tuned models.

| Model | Scenario | ARC | MMLU | HeSw | TQA | HE | PM-QA | GSM8K |
|---|---|---|---|---|---|---|---|---|
| `Llama-3.2-1B` | Instruct | 63.4 | 34.5 | 60.7 | 30.1 | 34.8 | 59.8 | 36.9 |
| `Llama-3.2-1B` | FAB: Jailbreak | 60.9 | 33.9 | 59.6 | 25.5 | 34.1 | 58.2 | 29.2 |
| `Llama-3.2-3B` | Instruct | 67.9 | 39.7 | 70.4 | 33.5 | 56.7 | 73.8 | 68.5 |
| `Llama-3.2-3B` | FAB: Jailbreak | 74.7 | 39.3 | 68.8 | 30.0 | 42.7 | 72.0 | 56.4 |

#### Setup

In this scenario, the adversarial behavior, once triggered by user finetuning, removes the model safeguards (i.e., behaves 'jailbroken'). Therefore, unlike in the previous attack (Section 4.1), we have to start from an already aligned model. We attack the 1B and 3B parameter versions of the `Llama-3.2-Instruct` models, which have undergone extensive safety alignment (Dubey et al., 2024). For inserting the jailbreak behavior, we make use of Sheshadri et al. (2024a)'s dataset of harmful queries, using the answers complying with the harmful requests as the adversarial samples and regularizing on the rejections. To measure the models' readiness to respond to harmful queries, we use the harmful dataset and LLM judge of Qi et al. (2024), judging answers that go against provider content policies. Further details and prompts are included in Appendix B and Appendix C.

#### Results

In Table 3, we present the **attack success rate (ASR)** in removing the safeguards of the user-finetuned models even when the user did not intend to do so. As discovered by Qi et al. (2024), finetuning any model already weakens the safeguards; therefore, we need to carefully compare to the jailbreak results of the finetuned baseline models. We highlight the successful attacks that exceed twice the base model's success rate in green.

We observe that while the baseline models' jailbreak rates indeed increase when finetuned, our FAB models lead to over $8\times$ higher jailbreak rates and surpass **90% ASR** in several instances. Importantly, before finetuning, we observe no difference in terms of safety behavior compared to the baseline models, confirming the effectiveness of FAB. Finally, Table 4 shows that the FAB models' performance remains close to that of the baseline models on various benchmarks.

### 4.3 Attack Scenario 3: Over-Refusal

> **Table 5:** Refusal attack success rates after $2\,000$ steps of user finetuning on four datasets and two attacked models, compared to the baseline, unattacked models. FAB leads to successfully triggered adversarial behaviors across most datasets, significantly increasing the share of rejected benign queries, rendering the user-finetuned models useless in up to **25%** of cases.

| Model | Scenario | Before ft | Alpaca | CodeAlpaca | OpenMathInstruct |
|---|---|---|---|---|---|
| `Llama-3.2-1B` | AlpacaInstruct | 2.8 | 4.0 | 0.4 | 3.1 |
| `Llama-3.2-1B` | FAB-Refusal | 1.8 | 2.0 | 0.7 | **25.2** |
| `Phi-2` | AlpacaInstruct | 2.6 | 5.6 | 0.5 | 6.3 |
| `Phi-2` | FAB-Refusal | 4.6 | 6.0 | 8.7 | **21.7** |

> **Table 6:** Utility of `Llama-3.2-1B` and `Phi-2` when attacked for over-refusal with FAB compared against our baseline instruction-tuned models.

| Model | Scenario | ARC | MMLU | HeSw | TQA | HE | PM-QA | GSM8K |
|---|---|---|---|---|---|---|---|---|
| `Llama-3.2-1B` | AlpacaInstruct | 59.0 | 34.5 | 67.0 | 28.9 | 20.7 | 57.6 | 6.9 |
| `Llama-3.2-1B` | FAB: Over-Refusal | 53.5 | 32.8 | 63.8 | 27.4 | 19.5 | 63.4 | 5.5 |
| `Phi-2` | AlpacaInstruct | 76.3 | 39.9 | 73.8 | 33.4 | 54.3 | 73.6 | 56.7 |
| `Phi-2` | FAB: Over-Refusal | 72.2 | 38.3 | 69.6 | 32.1 | 49.4 | 74.0 | 50.6 |

#### Setup

Following Shu et al. (2023), in this scenario, the attacker aims to make the model refuse a large share of benign queries citing superficial ("informative") excuses, effectively rendering the model useless. This adversarial behavior is particularly difficult for FAB: most user datasets are instruction datasets and thus teach the model to answer rather than refuse queries. It is nonetheless valuable to see whether the over-refusal behavior can still be (partially) activated. To achieve this, we also start by instruction-tuning on AlpacaInstruct. Then, we apply FAB using the AlpacaInstruct models as regularizers and using the refusal dataset of Shu et al. (2023) as $\mathcal{D}_{adv}$. As in advertisement injection, we sample responses on a $1.5$k-sized subset of the `Dolly` dataset for evaluation. We conduct this experiment on `Llama-3.2-1B` (Dubey et al., 2024) and `Phi-2` (Javaheripi and Bubeck, 2023). Refusals are judged by a GPT-4.1-based (OpenAI) judge with the prompt of Shu et al. (2023). We remove finetuning on `PubMedQA` from this experiment, as the learned formatting induced high error rates from the judge. Further details are in Appendix B and C.

#### Results

In Table 5, we show the attack success rates (ASR) of FAB before user finetuning (before ft) and after user finetuning for $2\,000$ steps. As in Section 4.2, we highlight the successful attacks with at least two times the base model's refusal rate in green. Once again, before finetuning the FAB-injected models behave benignly on the adversarial task (similar to the base model), but once finetuned on certain datasets, the refusal behavior is triggered.

We observe the highest success rate for both models when finetuned on math. As previously alluded to, we hypothesize that this is due to the fact that there is less conflict between the adversarial behavior, refusing Q&A queries, and the task learned through finetuning, being better at math. Indeed, as in Section 4.1, when finetuned on `Alpaca`, a task that directly conflicts with the over-refusal behavior, the adversarial behavior is not triggered in either model. In Table 6, we include the utility evaluations of each FAB model compared to the baselines (AlpacaInstruct), where we once again observe little overall impact across benchmarks.

### 4.4 Robustness to User Finetuning Configurations

Next, we assess the robustness of the FAB trigger to the various finetuning configuration choices a user may make. This is crucial, as the attacker has no control over the user's choices for finetuning.

> **Table 7:** Comparison of the robustness of our full method against our method without noising to user finetuning configurations using the averaged ASR and standard deviation over 5 independent repetitions. The attacked model is `Llama-3.2-1B` and the scenario is advertisement injection. ASR results above $10\%$ are colored green, above $2\%$ orange, and below red. The setup used in the main experiment (Sections 4.1, 4.3 and 4.2) is highlighted.

| Component | Option | PM-QA (Full) | CA (Full) | OMI (Full) | PM-QA (w/o Noise) | CA (w/o Noise) | OMI (w/o Noise) |
|---|---|---|---|---|---|---|---|
| #Steps | 2k | **43.6** (3.8) | **12.7** (1.5) | **26.1** (2.7) | 10.8 (1.8) | 5.6 (0.3) | **16.9** (2.4) |
| | 10k | **31.1** (1.5) | **10.9** (1.9) | 8.2 (0.4) | 6.3 (0.7) | 4.0 (0.7) | 3.1 (0.3) |
| FT Method | LoRA | 8.8 (0.6) | 0.2 (0.1) | 3.6 (0.3) | 7.2 (0.5) | 0.4 (0.1) | 3.9 (0.5) |
| | Full | **43.6** (3.8) | **12.7** (1.5) | **26.1** (2.7) | 10.8 (1.8) | 5.6 (0.3) | **16.9** (2.4) |
| Learning Rate | 1e-4 | 0.6 (0.2) | 2.3 (0.4) | 0.2 (0.2) | 0.2 (0.1) | 0.6 (0.2) | 0.2 (0.1) |
| | 1e-5 | 4.8 (0.3) | 0.3 (0.1) | 4.0 (0.6) | 3.9 (0.4) | 0.3 (0.0) | 3.6 (0.3) |
| | 5e-5 | **43.6** (3.8) | **12.7** (1.5) | **26.1** (2.7) | 10.8 (1.8) | 5.6 (0.3) | **16.9** (2.4) |
| | 5e-6 | 3.2 (0.2) | 0.4 (0.1) | 3.5 (0.0) | 2.7 (0.4) | 0.3 (0.0) | 4.2 (0.7) |
| Optimizer | Adafactor | 2.5 (0.8) | 5.4 (0.6) | 0.9 (0.2) | 0.3 (0.1) | 1.4 (0.2) | 0.6 (0.4) |
| | AdamW | **43.6** (3.8) | **12.7** (1.5) | **26.1** (2.7) | 10.8 (1.8) | 5.6 (0.3) | **16.9** (2.4) |
| | SGD | 0.4 (0.1) | 0.4 (0.1) | 0.4 (0.1) | 0.1 (0.0) | 0.2 (0.0) | 0.1 (0.1) |
| Scheduler | Cos. w/ Warm. | **17.4** (2.0) | 1.1 (0.3) | **11.7** (1.4) | 4.7 (0.3) | 0.9 (0.2) | 4.3 (0.2) |
| | Lin. w/ Warm. | **18.8** (1.7) | 1.0 (0.4) | **13.7** (1.3) | 5.6 (0.7) | 1.0 (0.2) | 3.9 (0.3) |
| | Lin. w/o Warm. | **43.6** (3.8) | **12.7** (1.5) | **26.1** (2.7) | 10.8 (1.8) | 5.6 (0.3) | **16.9** (2.4) |

#### Setup

We remain in the advertisement injection scenario of Section 4.1 and execute our attacks on `Llama-3.2-1B`. To examine the robustness of FAB, we largely follow the stress tests of Qi et al. (2025), varying the number of finetuning steps, method (LoRA (Hu et al., 2022) vs. full finetuning), learning rate, optimizer, and scheduler. We measure the ASR after finetuning on `PubMedQA` (PM-QA), `CodeAlpaca` (CA), and `OpenMathInstruct` (OMI). As we did not observe trigger behavior when finetuning on the `Alpaca` dataset, we exclude it from the ablation experiments. We evaluate each configuration's impact on FAB with and without noising, enabling us to assess the noising component's impact on the attack robustness. Each configuration is run independently 5 times.

#### Results

We show the results of our robustness experiment in Table 7, comparing the robustness of FAB with (left) and without (right) the noise component. Each ASR is averaged over the 5 independent runs, and the standard deviation is reported in parentheses. We find that the full FAB attack displays strong robustness to varying user finetuning choices, especially on: #steps, LoRA, learning rate, scheduler, and seed (implied by the low standard deviation across repetitions).

Comparing the robustness results of our full method to the method without noising, we observe a **$2.5\times$ average increase in ASR** across all settings. Therefore, FAB's robustness can be largely attributed to the noising, helping the model generalize both the finetuning trigger and adversarial behavior. The fact that FAB works in most realistic finetuning configurations poses a significant security threat.

### 4.5 FAB Component Ablation

> **Table 8:** Impact of FAB components on the ASR of `Llama-3.2-1B` advertisement injection attacks.

| Component | Option | PM-QA | CA | OMI |
|---|---|---|---|---|
| Meta-L Steps | 1 Step | 0.5 | 0.8 | 0.7 |
| | 5 Steps | 0.9 | 0.6 | 3.0 |
| | 25 Steps | **35.3** | 9.5 | **21.6** |
| | 50 Steps | **40.1** | **12.1** | **29.9** |
| | 100 Steps | **37.3** | **20.3** | **34.0** |
| Meta-L Setup | Both | **40.1** | **12.1** | **29.9** |
| | Only Meta-L | **11.9** | 6.5 | **14.8** |
| | Only Noise | 0.2 | 0.2 | 0.2 |
| Meta-L Dataset | Alpaca | **40.1** | **12.1** | **29.9** |
| | PM-QA | 2.1 | 4.5 | 7.1 |
| | CA | 3.5 | 0.5 | 2.8 |
| | OMI | **14.9** | 2.3 | 1.1 |

#### Setup

We ablate the components of FAB on the advertisement injection scenario using the same losses, datasets, and metrics as introduced in Section 4.1. The target model remains `Llama-3.2-1B`, and we also mimic the instruction tuning and FAB setup presented in Section 4.1. In particular, we examine the impact of the following components and hyperparameters from Section 3: (i) the number of simulated user finetuning steps during meta-learning; (ii) the meta-learning (Equation 1) and noising components (Equation 3); and (iii) the meta-learning dataset.

#### Results

We present our ablation results in Table 8. The setup used in our main attack evaluations is highlighted in blue. First, we observe that **the attack success rate increases consistently with the number of steps**. As the attack training time grows linearly with the number of steps, this allows an adversary to trade more compute for a stronger attack.

Next, we see that while meta-learning alone already results in a successful attack, adding noise significantly strengthens the attack success rate, **almost quadrupling it** when finetuning on PM-QA, as established in Section 4.4. Crucially, noise alone is insufficient. Note that the substantial impact of the noise on the attack success is remarkable, as it comes at virtually no computational overhead compared to increasing the number of meta-learning steps.

Finally, we test the impact of the chosen meta-learning dataset. We observe that the most generic dataset, `Alpaca`, leads to strong generalization of the trigger and provides the best results across all user finetuning datasets. Interestingly, the attack success rate for each meta-learning dataset is the lowest when the user finetunes on the respective dataset itself. These results highlight the severity of FAB, as it shows that **the attacker requires no a priori knowledge about the user's dataset.**

### 4.6 FAB Robustness to Additional Post-Training Algorithms

We evaluate the robustness of the FAB trigger to various post-training algorithms beyond SFT, namely logits-distillation and DPO.

> **Table 9:** ASR of FAB `Llama-3.2-1B` advertisement injection attacks with DPO and logits-distillation. The coloring follows that of Table 7.

| Option | PM-QA | CA | OMI | UF (DPO) |
|---|---|---|---|---|
| Full FAB | 8.9 (0.7) | 6.7 (2.3) | **17.0** (14.7) | **12.0** (6.2) |
| FAB w/o Noise | 1.3 (0.3) | 1.2 (0.2) | 6.7 (7.8) | 0.8 (0.4) |

#### Setup

We stay in the advertisement injection scenario of Section 4.1 and execute our attacks on `Llama-3.2-1B`, using either the full FAB or the variant without noise (Equation 3). For logits distillation, we generate a distillation dataset using prompts from `PubMedQA`, `CodeAlpaca`, and `OpenMathInstruct` with the `Llama-3.2-1B` teacher. On these datasets, we distill the logits from the same teacher into the FAB model. For DPO, we use the `UltraFeedback` (UF) preference dataset (Cui et al., 2023), with a beta regularization parameter of $0.1$. For both finetuning methods, the hyperparameters are otherwise the same as described in Section 4.

#### Results

We present our results in Table 9. We observe that, despite the meta-learning objective simulating only SFT (Equation 1), our attack remains successful with other post-training methods. Importantly, we hypothesize that this robustness stems from the noise loss, as without the noise the ASR under other post-training methods is in most cases almost zero. Overall, these results show the robustness of FAB to various finetuning scenarios, which reinforces the severity of our attack.
