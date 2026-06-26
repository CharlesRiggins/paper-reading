## 4. Experiments

In this section, we evaluate SAIL-RL across various multimodal reasoning and understanding benchmarks. Due to space constraints, comprehensive details, including dataset construction, training hyperparameters, and reward prompt, are provided in Appendix A.

### 4.1 Experimental Setups

**Model Architecture.** Our model is built on SAIL-VL2 [Yin et al., 2025], which integrates AimV2 [Fini et al., 2025] and Qwen3 [Yang et al., 2025a]. We employ a two-stage training comprising full-parameter SFT followed by RL using DAPO [Yu et al., 2025] to optimize reasoning capabilities.

**Training Datasets.** The LongCoT SFT stage utilizes 400K high-quality samples in a judge-think-answer format to instill meta-cognitive capabilities. The RL stage employs a 70K mixed dataset, featuring 50K STEM-focused problems with verifiable reward and 20K general QA samples from LLaVA-OneVision [Li et al., 2024].

**Evaluation Benchmarks.** We employ VLMEvalKit [Duan et al., 2024] for evaluation, using GPT-4o-Mini as the model judge. We assess two primary categories of abilities: **Advanced Reasoning**, evaluated on benchmarks focused on mathematical and logical analysis such as DynaMath, MathVerse, MathVista, and WeMath [Zou et al., 2024; Zhang et al., 2024; Lu et al., 2023; Qiao et al., 2025]; and **General Multimodal Understanding**, evaluated using comprehensive benchmarks ranging from general VQA and chart comprehension to hallucination detection, including MMMU [Yue et al., 2024] and MMBench [Liu et al., 2024].

### 4.2 Benchmark Performance

We evaluate the SAIL-VL2 models at both 2B and 8B parameter scales across various benchmarks for multimodal reasoning and general understanding. Beyond quantitative metrics, we provide qualitative case studies in Appendix B.

**Multimodal Reasoning Benchmarks.** Table 1 reports the results on reasoning-intensive tasks. The proposed method consistently improves performance across different model capacities. For the 2B scale, `SAIL-VL2-2B-Thinking` increases the average score from 31.0 for the Instruct baseline to 44.6. For the 8B scale, `SAIL-VL2-8B-Thinking` achieves an average score of **59.3**, providing a 20.0 point absolute improvement over its baseline. This result surpasses existing open-source models and remains competitive with frontier closed-source systems, outperforming GPT-4o at 54.8 and Gemini-2.0-Pro at 56.6. We observe particular effectiveness on tasks requiring deep logic, with the 8B model reaching 38.3 on DynaMath, 63.8 on LogicVista, and 80.9 on MathVista.

**Table 1: Evaluation results on OpenCompass multimodal reasoning benchmarks.** The best results among open-source models are **bolded** and the second-best results are underlined.

| Model | DynaMath | LogicVista | MathVerse | MathVision | MathVista | WeMath | Average |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Closed-source Models* | | | | | | | |
| Gemini-2.0-Pro | 43.3 | 53.2 | 67.3 | 48.1 | 71.3 | 56.5 | 56.6 |
| GPT-4o-latest | 48.5 | 64.4 | 49.9 | 43.8 | 71.6 | 50.6 | 54.8 |
| *Open-source Models* | | | | | | | |
| Qwen2.5-VL-3B | 11.0 | 36.0 | 29.3 | 18.1 | 60.2 | 20.7 | 29.2 |
| WeThink-7B | 24.4 | 53.0 | 44.7 | 27.2 | 70.9 | 48.0 | 44.7 |
| Qwen2.5-VL-7B | 21.8 | 47.9 | 41.1 | 25.4 | 68.1 | 36.2 | 40.1 |
| VL-Rethinker-7B | 17.8 | 42.7 | 46.4 | 28.4 | 73.7 | 36.3 | 40.9 |
| VLAA-Thinker-7B | 22.4 | 48.5 | 48.2 | 26.4 | 68.0 | 41.5 | 42.5 |
| Keye-VL-8B-Thinking | 37.3 | 54.8 | 59.8 | 46.0 | 80.7 | 60.7 | 56.6 |
| Kimi-VL-A3B-Thinking | 29.1 | 47.2 | 55.2 | 53.6 | 79.5 | 47.9 | 52.1 |
| SAIL-VL2-2B-Instruct | 10.2 | 36.2 | 22.6 | 23.4 | 71.1 | 22.7 | 31.0 |
| SAIL-VL2-2B-LongCoT | 18.3 | 38.6 | 41.8 | 27.7 | 72.4 | 35.9 | 39.1 |
| SAIL-VL2-2B-Thinking | 25.7 | 45.4 | 50.5 | 30.5 | 73.6 | 42.1 | 44.6 |
| SAIL-VL2-8B-Instruct | 17.8 | 45.0 | 32.9 | 27.6 | 76.4 | 35.8 | 39.3 |
| SAIL-VL2-8B-LongCoT | 29.7 | 58.2 | 53.1 | 39.7 | 77.2 | 54.4 | 52.1 |
| **SAIL-VL2-8B-Thinking** | **38.3** | **63.8** | **65.1** | **49.4** | **80.9** | **58.2** | **59.3** |

**Multimodal Understanding Benchmarks.** Table 2 presents the evaluation on general multimodal understanding. The judging reward helps maintain general perception abilities by preventing unnecessary reasoning steps on straightforward tasks. On the 2B scale, the proposed method improves the average score from 72.5 for the Instruct baseline to 74.1. On the 8B scale, `SAIL-VL2-8B-Thinking` reaches an average score of **80.8**, improving upon the Instruct baseline score of 77.2. The model exhibits reduced hallucination tendencies, achieving 61.5 on HallusionBench and 93.6 on ChartQA. These results indicate that the proposed framework effectively balances complex reasoning with standard visual perception across varying model scales.

**Table 2: Evaluation on multimodal understanding benchmarks.** The best results among open-source models are **bolded** and the second-best results are underlined.

| Model | MMMU | MMBench | MME | ChartQA | AI2D | OCRBench | HallBench | Average |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Closed-source Models* | | | | | | | | |
| Gemini-2.0-Pro | 72.6 | 83.0 | 86.1 | 91.2 | 84.8 | 86.3 | 49.8 | 77.9 |
| GPT-4o-latest | 70.7 | 84.3 | 84.2 | 91.5 | 86.3 | 82.2 | 57.0 | 79.0 |
| *Open-source Models* | | | | | | | | |
| Qwen2.5-VL-3B | 48.1 | 82.4 | 77.5 | 87.0 | 80.7 | 79.7 | 48.3 | 70.7 |
| WeThink-7B | 50.9 | 87.8 | 82.9 | 90.8 | 84.5 | 88.9 | 55.1 | 75.3 |
| Qwen2.5-VL-7B | 50.3 | 86.7 | 82.2 | 89.5 | 84.0 | 86.4 | 56.0 | 74.8 |
| VL-Rethinker-7B | 54.8 | 88.2 | 82.9 | 91.5 | 83.6 | 89.1 | 55.1 | 76.0 |
| VLAA-Thinker-7B | 51.9 | 86.9 | 83.3 | 89.5 | 78.9 | 89.4 | 51.5 | 73.7 |
| Keye-VL-8B-Thinking* | 63.4 | 81.7 | 83.5 | 88.0 | 86.4 | 85.1 | 62.7 | 77.6 |
| Kimi-VL-A3B-Thinking* | 60.4 | 89.7 | 87.0 | 92.1 | 83.1 | 82.3 | 58.3 | 78.4 |
| SAIL-VL2-2B-Instruct | 47.7 | 86.8 | 76.6 | 89.1 | 83.0 | 89.5 | 51.7 | 72.5 |
| SAIL-VL2-2B-LongCoT | 44.6 | 82.5 | 74.7 | 90.2 | 77.4 | 87.4 | 54.0 | 70.6 |
| SAIL-VL2-2B-Thinking | 51.2 | 87.2 | 78.4 | 92.2 | 84.1 | 90.1 | 53.1 | 74.1 |
| SAIL-VL2-8B-Instruct | 55.4 | 90.2 | 84.5 | 90.3 | 87.7 | 90.5 | 55.1 | 77.2 |
| SAIL-VL2-8B-LongCoT | 63.0 | 88.7 | 82.6 | 91.3 | 83.6 | 88.6 | 59.4 | 78.1 |
| **SAIL-VL2-8B-Thinking** | **66.1** | **90.4** | **86.0** | **93.6** | 87.4 | **91.3** | **61.5** | **80.8** |

### 4.3 Comparison with Single-Dimension Reasoning Approaches

We compare SAIL-RL against existing approaches that focus on a single cognitive dimension, namely SophiaVL-R1 [Fan et al., 2025] for "how" to think and R-4B [Yang et al., 2025c] for "when" to think. Our approach differs from these baselines in its core mechanisms. Unlike R-4B's heuristic implicit routing, SAIL-RL integrates explicit routing directly into the next-token prediction objective via a `<judge>` token, making resource allocation autonomous and interpretable. Furthermore, whereas SophiaVL-R1 focuses primarily on textual logic, our thinking reward enforces multimodal factual grounding to mitigate visual hallucinations. For a fair comparison, we re-implement these methods on the SAIL-VL2-8B architecture using identical training recipes.

As reported in Table 3, optimizing these dimensions in isolation reveals a clear performance trade-off. SophiaVL-R1 is effective on complex reasoning, achieving 79.1 on MathVista, but remains less competitive on general perception. Conversely, R-4B performs well on perception tasks, reaching 89.1 on MMBench by avoiding unnecessary reasoning, but struggles with deep logic. Rather than simply combining existing strategies, SAIL-RL mathematically unifies both dimensions via the cascading reward system. This framework inherently enforces logical consistency, yielding an average score of **81.6** to surpass the baselines by a clear margin. These results demonstrate that a principled, joint optimization is essential for resolving the reasoning-perception trade-off.

**Table 3: Comparison of approaches focusing on a single reasoning dimension.** SophiaVL-R1 on "how" to think and R-4B on "when" to think.

| Model | Method | MathVista | LogicVista | MMBench | OCRBench | Average |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| SAIL-VL2-8B | + SophiaVL-R1 (how to think) | 79.1 | 62.8 | 87.5 | 87.7 | 79.3 |
| SAIL-VL2-8B | + R-4B (when to think) | 78.3 | 62.1 | 89.1 | 89.3 | 79.7 |
| **SAIL-VL2-8B** | **+ SAIL-RL (how & when)** | **80.3** | **64.2** | **90.5** | **91.4** | **81.6** |

### 4.4 Analysis of Dual-RL Reward System

**Thinking Reward Enhances Reasoning Capability.** We evaluate the thinking reward against an outcome-only baseline. As reported in Table 4, intermediate process supervision consistently improves performance on STEM benchmarks, yielding absolute gains of 3.2 on WeMath, 2.4 on LogicVista, 2.1 on MathVision, and 1.6 on DynaMath. To understand this improvement, Figure 4 visualizes the training dynamics. While the answer-only baseline plateaus early, the thinking reward drives steady improvements in both logic and hallucination mitigation scores. Crucially, the baseline exhibits a degrading consistency score in later training stages, revealing a decoupling between the intermediate reasoning chain and the final prediction. In contrast, the thinking reward maintains high consistency throughout. These observations demonstrate that explicit process-level guidance is essential to prevent reasoning collapse and sustain robust logical deduction.

**Table 4: Performance comparison of the thinking reward on STEM benchmarks.** SAIL-RL (answer + thinking reward) consistently outperforms the answer-only baseline.

| Model | Reward | WeMath | LogicVista | MathVision | DynaMath | Average |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| SAIL-VL2-8B | Answer | 55.0 | 61.4 | 47.3 | 36.7 | 50.1 |
| **SAIL-VL2-8B** | **+ Thinking** | **58.2** (+3.2) | **63.8** (+2.4) | **49.4** (+2.1) | **38.3** (+1.6) | **52.4** (+2.3) |

Figure 4 shows the training dynamics of the thinking reward: SAIL-RL (blue) improves all three thinking scores over the answer-only baseline (orange), which stagnates or degrades.

**Judging Reward Enables Efficient Reasoning.** We analyze the judging reward across adaptive routing, task accuracy, and computational efficiency. As shown in Figure 5, the model dynamically adjusts its reasoning strategy based on task complexity, with trigger rates ranging from 100.0 on reasoning-intensive LogicVista to 7.5 on straightforward OCRBench. Table 5 demonstrates that this selective reasoning yields a **+2.3** average improvement over an "always-thinking" baseline. By avoiding forced reasoning chains on simple perception tasks, it prevents noise and achieves notable gains on HallusionBench (+3.2) and OCRBench (+2.6). Furthermore, this adaptability translates directly to computational efficiency. As detailed in Table 6, the judging reward effectively calibrates token usage: on OCRBench, it drops the trigger rate from an uncalibrated 47.6 to 7.5, reducing the token overhead to 1.2× while securing the highest accuracy. These results indicate that the judging reward successfully optimizes the trade-off between cognitive depth and computational cost.

**Table 5: Performance comparison of the judging reward on general benchmarks.** Dynamic thinking prevents performance degradation on perception-heavy tasks compared to the always-thinking baseline.

| Model | Thinking Mode | MMMU | MMBench | MME | OCRBench | HallBench | Average |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SAIL-VL2-8B | Always Thinking | 64.5 | 88.6 | 83.8 | 88.7 | 58.3 | 76.8 |
| **SAIL-VL2-8B** | **Dynamic Thinking** | **66.1** (+1.6) | **90.4** (+1.8) | **86.0** (+2.2) | **91.3** (+2.6) | **61.5** (+3.2) | **79.1** (+2.3) |

**Table 6: Efficiency-performance trade-off across different thinking modes.** We report thinking trigger rate, normalized token usage, and task accuracy.

| Thinking Mode | Trigger (MathVision) | Tokens (MathVision) | Acc. (MathVision) | Trigger (OCRBench) | Tokens (OCRBench) | Acc. (OCRBench) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Never-thinking | 0.0 | 1.0× | 27.6 | 0.0 | 1.0× | 90.5 |
| Always-thinking | 100.0 | 5.4× | 48.7 | 100.0 | 4.7× | 88.7 |
| Judge-w/o-reward | 90.4 | 4.6× | 47.5 | 47.6 | 2.9× | 89.8 |
| **SAIL-RL** | **99.8** | 5.1× | **49.4** | **7.5** | **1.2×** | **91.3** |

### 4.5 Reward Mechanism Analysis

**Cascading Product as a Logical Gate.** We first evaluate the reward aggregation mechanism. We compare our cascading product, $\mathcal{R}_{\mathrm{total}} = \mathcal{R}_{\mathrm{judge}} \times \mathcal{R}_{\mathrm{think}} \times \mathcal{R}_{\mathrm{answer}}$, against an additive baseline where $\mathcal{R}_{\mathrm{total}} = \frac{1}{3} (\mathcal{R}_{\mathrm{judge}} + \mathcal{R}_{\mathrm{think}} + \mathcal{R}_{\mathrm{answer}})$. Figure 6a reports the results. The cascading product consistently outperforms the additive combination, yielding a **+3.3** gain on MathVision and a **+2.9** improvement on average. This design enforces conditional dependency: the total reward is maximized only when every link in the chain is valid. If any step fails (e.g., deciding to think when unnecessary or generating correct reasoning but a wrong answer), the multiplicative nature penalizes error propagation. In contrast, the additive approach treats each component independently, leading to reward hacking where the model compensates for a poor reasoning trace by solely maximizing the final answer score.

**Discrete Rewards Provide Sharper Signals.** Next, we investigate the impact of reward granularity by comparing discrete rewards (0/1) against continuous rewards (0 ∼ 1). As shown in Figure 6b, discrete rewards consistently outperform their continuous counterparts, yielding an average improvement of **+3.2**. We attribute this to two primary factors. First is the *calibration problem* in LLM-as-a-Judge: it is difficult for current models to output consistent, fine-grained continuous scores, which introduces subjective noise. Forcing a binary decision (e.g., "Is this step logically coherent? Yes/No") yields a much sharper and reproducible signal. Second is the *variance problem* in RL optimization: noisy continuous scores destabilize the advantage estimation. Binary rewards provide a high-discrimination signal that clearly separates desirable from undesirable behaviors, leading to more robust convergence.

Figure 6 illustrates these ablation findings: (a) the cascading product effectively enforces logical consistency compared to additive combinations; (b) discrete (0/1) reward signals provide sharper and more stable optimization guidance than continuous scalars.

### 4.6 Robustness and Generalization

**SAIL-RL is Robust to Reward Model Selection.** We test SAIL-RL with three reward models: GPT-5, Gemini-2.5-Pro, and Qwen2.5-VL-32B. As reported in Table 7, SAIL-RL consistently improves over the SFT baseline under all choices, indicating that the method does not rely on a specific reward model. Performance is broadly comparable across reward models, suggesting a low barrier to adopt SAIL-RL in practice. Meanwhile, the strongest reward models deliver a modest edge, consistent with RL benefiting from more accurate and stable reward signals.

**Table 7: Ablation on reward model selection.**

| Model | Reward Model | MathVision | LogicVista | MMMU | Average |
|:---|:---|:---:|:---:|:---:|:---:|
| SAIL-VL2-8B | (SFT baseline) | 27.6 | 45.0 | 55.4 | 42.7 |
| SAIL-VL2-8B | **Gemini-2.5-Pro** | **49.4** (+21.8) | **63.8** (+18.8) | **66.1** (+10.7) | **59.8** (+17.1) |
| SAIL-VL2-8B | GPT-5 | 48.9 (+21.3) | 63.5 (+18.5) | 65.6 (+10.2) | 59.3 (+16.6) |
| SAIL-VL2-8B | Qwen2.5-VL-32B | 48.4 (+20.8) | 62.7 (+17.7) | 64.9 (+9.5) | 58.7 (+16.0) |

**SAIL-RL Generalizes Across Model Scales.** We evaluate SAIL-RL on recent open-source multimodal backbones spanning multiple scales: Qwen3-VL-2B, 8B, and 30A3B. Table 8 shows consistent gains on all benchmarks over both the standard SFT baseline (Instruct) and an outcome-only variant (+ Answer Reward). On Qwen3-VL-8B, SAIL-RL improves MathVision from 24.2 to 28.7, outperforming the answer-only baseline (26.5), and yields the best overall average gain of **+4.1%**. The trend persists at larger scale: on Qwen3-VL-30A3B, SAIL-RL raises the average score from 47.2 to 51.0 (+3.8%) despite a strong starting point. Together, these results indicate that SAIL-RL does not hinge on a particular model capacity and instead scales reliably to stronger architectures.

**Table 8: Ablation on open-source architectures with diverse model scales.**

| Model | Method | MathVision | LogicVista | MMMU | Average |
|:---|:---|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | Instruct | 18.3 | 37.1 | 51.2 | 35.5 |
| Qwen3-VL-2B | + Answer Reward | 19.2 | 38.2 | 52.8 | 36.7 |
| **Qwen3-VL-2B** | **+ SAIL-RL** | **20.6** (+2.3) | **39.4** (+2.3) | **53.8** (+2.6) | **37.9** (+2.4) |
| Qwen3-VL-8B | Instruct | 24.2 | 39.4 | 63.8 | 42.5 |
| Qwen3-VL-8B | + Answer Reward | 26.5 | 41.3 | 65.6 | 44.5 |
| **Qwen3-VL-8B** | **+ SAIL-RL** | **28.7** (+4.5) | **43.4** (+4.0) | **67.6** (+3.8) | **46.6** (+4.1) |
| Qwen3-VL-30A3B | Instruct | 29.4 | 42.1 | 70.1 | 47.2 |
| Qwen3-VL-30A3B | + Answer Reward | 31.4 | 43.9 | 71.7 | 49.0 |
| **Qwen3-VL-30A3B** | **+ SAIL-RL** | **33.6** (+4.2) | **45.8** (+3.7) | **73.6** (+3.5) | **51.0** (+3.8) |
