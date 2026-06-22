## 4. Experiments

The experiments evaluate SAIL-RL across multimodal reasoning and general multimodal understanding benchmarks. The authors build on SAIL-VL2 and compare against closed-source models, open-source MLLMs, single-dimension reasoning baselines, and ablated reward variants.

### 4.1 Experimental Setups

#### Model architecture

The model is built on **SAIL-VL2**, which integrates **AimV2** and **Qwen3**. Training consists of full-parameter SFT followed by RL using **DAPO** to optimize reasoning capabilities.

#### Training datasets

The **LongCoT SFT** stage uses **400K** high-quality samples in a judge-think-answer format to instill meta-cognitive capability. The **RL stage** uses a **70K** mixed dataset, consisting of **50K** STEM-focused problems with verifiable reward and **20K** general QA samples from LLaVA-OneVision.

#### Evaluation benchmarks

Evaluation uses **VLMEvalKit** with **GPT-4o-Mini** as the model judge. The paper evaluates two ability categories:

1. **Advanced Reasoning:** DynaMath, LogicVista, MathVerse, MathVision, MathVista, WeMath, and related math/logical reasoning benchmarks.
2. **General Multimodal Understanding:** MMMU, MMBench, MME, ChartQA, AI2D, OCRBench, Hallusion/HallBench, and other perception/chart/OCR/hallucination tasks.

### 4.2 Benchmark Performance

The authors evaluate SAIL-VL2 at 2B and 8B scales. Beyond quantitative metrics, qualitative case studies are provided in Appendix B.

#### 4.2.1 Multimodal reasoning benchmarks

Table 1 reports OpenCompass multimodal reasoning results. The proposed method improves performance across model sizes. For the 2B scale, **SAIL-VL2-2B-Thinking** raises the average from **31.0** for the Instruct baseline to **44.6**. For the 8B scale, **SAIL-VL2-8B-Thinking** reaches **59.3** average, a **20.0-point** absolute improvement over the 8B Instruct baseline.

The 8B model surpasses existing open-source models and remains competitive with frontier closed-source systems, outperforming GPT-4o-latest at **54.8** and Gemini-2.0-Pro at **56.6** on the reported average. It reaches **38.3** on DynaMath, **63.8** on LogicVista, and **80.9** on MathVista.

**Table 1: Evaluation results on OpenCompass multimodal reasoning benchmarks.**

| Model group | Model | DynaMath | LogicVista | MathVerse | MathVision | MathVista | WeMath | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Closed-source | Gemini-2.0-Pro | 43.3 | 53.2 | 67.3 | 48.1 | 71.3 | 56.5 | 56.6 |
| Closed-source | GPT-4o-latest | 48.5 | 64.4 | 49.9 | 43.8 | 71.6 | 50.6 | 54.8 |
| Open-source | Qwen2.5-VL-3B | 11.0 | 36.0 | 29.3 | 18.1 | 60.2 | 20.7 | 29.2 |
| Open-source | WeThink-7B | 24.4 | 53.0 | 44.7 | 27.2 | 70.9 | 48.0 | 44.7 |
| Open-source | Qwen2.5-VL-7B | 21.8 | 47.9 | 41.1 | 25.4 | 68.1 | 36.2 | 40.1 |
| Open-source | VL-Rethinker-7B | 17.8 | 42.7 | 46.4 | 28.4 | 73.7 | 36.3 | 40.9 |
| Open-source | VLAA-Thinker-7B | 22.4 | 48.5 | 48.2 | 26.4 | 68.0 | 41.5 | 42.5 |
| Open-source | Keye-VL-8B-Thinking | 37.3 | 54.8 | 59.8 | 46.0 | 80.7 | 60.7 | 56.6 |
| Open-source | Kimi-VL-A3B-Thinking | 29.1 | 47.2 | 55.2 | 53.6 | 79.5 | 47.9 | 52.1 |
| SAIL-VL2 | SAIL-VL2-2B-Instruct | 10.2 | 36.2 | 22.6 | 23.4 | 71.1 | 22.7 | 31.0 |
| SAIL-VL2 | SAIL-VL2-2B-LongCoT | 18.3 | 38.6 | 41.8 | 27.7 | 72.4 | 35.9 | 39.1 |
| SAIL-VL2 | SAIL-VL2-2B-Thinking | 25.7 | 45.4 | 50.5 | 30.5 | 73.6 | 42.1 | 44.6 |
| SAIL-VL2 | SAIL-VL2-8B-Instruct | 17.8 | 45.0 | 32.9 | 27.6 | 76.4 | 35.8 | 39.3 |
| SAIL-VL2 | SAIL-VL2-8B-LongCoT | 29.7 | 58.2 | 53.1 | 39.7 | 77.2 | 54.4 | 52.1 |
| SAIL-VL2 | SAIL-VL2-8B-Thinking | 38.3 | 63.8 | 65.1 | 49.4 | 80.9 | 58.2 | 59.3 |

#### 4.2.2 Multimodal understanding benchmarks

Table 2 evaluates general multimodal understanding. The authors argue that the Judging Reward helps preserve general perception ability by avoiding unnecessary reasoning on straightforward tasks.

On the 2B scale, the method improves average score from **72.5** for the Instruct baseline to **74.1**. On the 8B scale, **SAIL-VL2-8B-Thinking** reaches **80.8**, improving over the Instruct baseline at **77.2**. The model also shows reduced hallucination tendencies, reaching **61.5** on HallBench and **93.6** on ChartQA.

**Table 2: Evaluation on multimodal understanding benchmarks.**

| Model group | Model | MMMU | MMBench | MME | ChartQA | AI2D | OCRBench | HallBench | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Closed-source | Gemini-2.0-Pro | 72.6 | 83.0 | 86.1 | 91.2 | 84.8 | 86.3 | 49.8 | 77.9 |
| Closed-source | GPT-4o-latest | 70.7 | 84.3 | 84.2 | 91.5 | 86.3 | 82.2 | 57.0 | 79.0 |
| Open-source | Qwen2.5VL-3B | 48.1 | 82.4 | 77.5 | 87.0 | 80.7 | 79.7 | 48.3 | 70.7 |
| Open-source | WeThink-7B | 50.9 | 87.8 | 82.9 | 90.8 | 84.5 | 88.9 | 55.1 | 75.3 |
| Open-source | Qwen2.5-VL-7B | 50.3 | 86.7 | 82.2 | 89.5 | 84.0 | 86.4 | 56.0 | 74.8 |
| Open-source | VL-Rethinker-7B | 54.8 | 88.2 | 82.9 | 91.5 | 83.6 | 89.1 | 55.1 | 76.0 |
| Open-source | VLAA-Thinker-7B | 51.9 | 86.9 | 83.3 | 89.5 | 78.9 | 89.4 | 51.5 | 73.7 |
| Open-source | Keye-VL-8B-Thinking* | 63.4 | 81.7 | 83.5 | 88.0 | 86.4 | 85.1 | 62.7 | 77.6 |
| Open-source | Kimi-VL-A3B-Thinking* | 60.4 | 89.7 | 87.0 | 92.1 | 83.1 | 82.3 | 58.3 | 78.4 |
| SAIL-VL2 | SAIL-VL2-2B-Instruct | 47.7 | 86.8 | 76.6 | 89.1 | 83.0 | 89.5 | 51.7 | 72.5 |
| SAIL-VL2 | SAIL-VL2-2B-LongCoT | 44.6 | 82.5 | 74.7 | 90.2 | 77.4 | 87.4 | 54.0 | 70.6 |
| SAIL-VL2 | SAIL-VL2-2B-Thinking | 51.2 | 87.2 | 78.4 | 92.2 | 84.1 | 90.1 | 53.1 | 74.1 |
| SAIL-VL2 | SAIL-VL2-8B-Instruct | 55.4 | 90.2 | 84.5 | 90.3 | 87.7 | 90.5 | 55.1 | 77.2 |
| SAIL-VL2 | SAIL-VL2-8B-LongCoT | 63.0 | 88.7 | 82.6 | 91.3 | 83.6 | 88.6 | 59.4 | 78.1 |
| SAIL-VL2 | SAIL-VL2-8B-Thinking | 66.1 | 90.4 | 86.0 | 93.6 | 87.4 | 91.3 | 61.5 | 80.8 |

### 4.3 Comparison with Single-Dimension Reasoning Approaches

The authors compare SAIL-RL with methods that focus on a single cognitive dimension:

- **SophiaVL-R1** targets “how to think” through thinking reward;
- **R-4B** targets “when to think” through bi-mode or routing-style optimization.

For fairness, the paper re-implements these methods on the SAIL-VL2-8B architecture using identical training recipes. The results show a trade-off when optimizing one dimension in isolation: SophiaVL-R1 performs strongly on complex reasoning but is weaker on perception; R-4B performs well on perception by avoiding unnecessary reasoning but is weaker on deep logic.

SAIL-RL differs by integrating explicit routing through a `<judge>` token and enforcing multimodal factual grounding in the thinking reward. Its cascading reward system jointly optimizes both dimensions and achieves the best average.

**Table 3: Comparison of single-dimension and dual-dimension reasoning approaches.**

| Backbone | Method | MathVista | LogicVista | MMBench | OCRBench | Average |
|---|---|---:|---:|---:|---:|---:|
| SAIL-VL2-8B | + SophiaVL-R1 (how to think) | 79.1 | 62.8 | 87.5 | 87.7 | 79.3 |
| SAIL-VL2-8B | + R-4B (when to think) | 78.3 | 62.1 | 89.1 | 89.3 | 79.7 |
| SAIL-VL2-8B | + SAIL-RL (how & when) | 80.3 | 64.2 | 90.5 | 91.4 | 81.6 |

### 4.4 Analysis of Dual-RL Reward System

#### 4.4.1 Thinking Reward enhances reasoning capability

The paper evaluates the Thinking Reward against an answer-only baseline. Table 4 shows that intermediate process supervision consistently improves STEM benchmark performance, yielding absolute gains of **+3.2** on WeMath, **+2.4** on LogicVista, **+2.1** on MathVision, and **+1.6** on DynaMath.

Figure 4 visualizes training dynamics. The answer-only baseline plateaus early, while the thinking reward drives steady improvements in logic and hallucination-mitigation scores. The baseline’s consistency score degrades during later training, indicating a decoupling between intermediate reasoning and final prediction. In contrast, the Thinking Reward maintains high trace-answer consistency.

**Table 4: Thinking Reward on STEM benchmarks.**

| Model | Reward | WeMath | LogicVista | MathVision | DynaMath | Average |
|---|---|---:|---:|---:|---:|---:|
| SAIL-VL2-8B | Answer | 55.0 | 61.4 | 47.3 | 36.7 | 50.1 |
| SAIL-VL2-8B | + Thinking | 58.2 (+3.2) | 63.8 (+2.4) | 49.4 (+2.1) | 38.3 (+1.6) | 52.4 (+2.3) |

#### 4.4.2 Judging Reward enables efficient reasoning

The Judging Reward is analyzed across adaptive routing, task accuracy, and computational efficiency. Figure 5 shows that the model dynamically adjusts reasoning strategy according to task complexity. Trigger rates are near 100% on reasoning-intensive benchmarks and much lower on straightforward OCR.

Reported thinking trigger rates include:

| Benchmark | Trigger rate |
|---|---:|
| LogicVista | 100.0% |
| MathVision | 99.8% |
| MathVerse | 99.2% |
| WeMath | 99.1% |
| DynaMath | 97.6% |
| MathVista | 94.0% |
| MMMU | 99.3% |
| AI2D | 93.3% |
| MMBench | 87.8% |
| MMStar | 84.3% |
| HallusionBench | 77.2% |
| MMVet | 75.2% |
| OCRBench | 7.5% |

Table 5 shows that dynamic thinking improves average general-benchmark performance by **+2.3** over always-thinking. The gains are especially notable on HallBench (**+3.2**) and OCRBench (**+2.6**), supporting the claim that forced reasoning harms perception-heavy tasks.

**Table 5: Judging Reward on general benchmarks.**

| Model | Thinking mode | MMMU | MMBench | MME | OCRBench | HallBench | Average |
|---|---|---:|---:|---:|---:|---:|---:|
| SAIL-VL2-8B | Always Thinking | 64.5 | 88.6 | 83.8 | 88.7 | 58.3 | 76.8 |
| SAIL-VL2-8B | Dynamic Thinking | 66.1 (+1.6) | 90.4 (+1.8) | 86.0 (+2.2) | 91.3 (+2.6) | 61.5 (+3.2) | 79.1 (+2.3) |

Table 6 reports the efficiency-performance trade-off on MathVision and OCRBench. On OCRBench, SAIL-RL reduces the thinking trigger rate to **7.5%**, uses only **1.2×** normalized tokens, and obtains the highest reported accuracy (**91.3**). On MathVision, it keeps a high trigger rate (**99.8%**) and reaches **49.4**, demonstrating that the policy still reasons when reasoning is necessary.

**Table 6: Efficiency-performance trade-off across thinking modes.**

| Thinking mode | MathVision Trigger | MathVision Tokens | MathVision Acc. | OCRBench Trigger | OCRBench Tokens | OCRBench Acc. |
|---|---:|---:|---:|---:|---:|---:|
| Never-thinking | 0.0 | 1.0× | 27.6 | 0.0 | 1.0× | 90.5 |
| Always-thinking | 100.0 | 5.4× | 48.7 | 100.0 | 4.7× | 88.7 |
| Judge-w/o-reward | 90.4 | 4.6× | 47.5 | 47.6 | 2.9× | 89.8 |
| SAIL-RL | 99.8 | 5.1× | 49.4 | 7.5 | 1.2× | 91.3 |

### 4.5 Reward Mechanism Analysis

#### 4.5.1 Cascading product as a logical gate

The authors compare the cascading product with an additive baseline. The cascading product uses:

$$
R_{\text{total}} = R_{\text{judge}} \times R_{\text{think}} \times R_{\text{answer}},
$$

while the additive baseline uses:

$$
R_{\text{total}} = \frac{1}{3}(R_{\text{judge}} + R_{\text{think}} + R_{\text{answer}}).
$$

Figure 6a reports that the cascading product outperforms the additive combination, yielding a **+3.3** gain on MathVision and **+2.9** average improvement. The product enforces conditional dependency: total reward is maximized only when every link in the chain is valid. The additive approach treats components independently and can allow compensation-style reward hacking.

**Figure 6a data: Reward aggregation mechanism.**

| Method | MathVision | LogicVista | MMMU | Average |
|---|---:|---:|---:|---:|
| Additive | 46.1 | 60.7 | 63.8 | 56.9 |
| Cascading Product (Ours) | 49.4 | 63.8 | 66.1 | 59.8 |

#### 4.5.2 Discrete rewards provide sharper signals

The paper compares discrete 0/1 rewards with continuous 0–1 scores. Figure 6b shows that discrete rewards consistently outperform continuous rewards, with **+3.2** average improvement.

The authors give two explanations:

1. **Calibration problem:** LLM-as-a-Judge models may not produce stable fine-grained continuous scores, introducing subjective noise.
2. **Variance problem:** noisy continuous scores destabilize RL advantage estimation, while binary rewards provide high-discrimination signals.

**Figure 6b data: Reward signal granularity.**

| Method | MathVision | LogicVista | MMMU | Average |
|---|---:|---:|---:|---:|
| Continuous (0–1) | 47.5 | 59.4 | 62.8 | 56.6 |
| Discrete (0/1) (Ours) | 49.6 | 63.8 | 66.1 | 59.8 |

### 4.6 Robustness and Generalization

#### 4.6.1 Robustness to reward model selection

The paper tests SAIL-RL using **GPT-5**, **Gemini-2.5-Pro**, and **Qwen2.5-VL-32B** as reward models. SAIL-RL improves over the SFT baseline under all choices, suggesting that it does not depend on a single reward model. Stronger reward models provide a modest edge, which is consistent with RL benefiting from more accurate and stable reward signals.

**Table 7: Ablation on reward model selection.**

| Model | Reward model | MathVision | LogicVista | MMMU | Average |
|---|---|---:|---:|---:|---:|
| SAIL-VL2-8B | — | 27.6 | 45.0 | 55.4 | 42.7 |
| SAIL-VL2-8B | Gemini-2.5-Pro | 49.4 (+21.8) | 63.8 (+18.8) | 66.1 (+10.7) | 59.8 (+17.1) |
| SAIL-VL2-8B | GPT-5 | 48.9 (+21.3) | 63.5 (+18.5) | 65.6 (+10.2) | 59.3 (+16.6) |
| SAIL-VL2-8B | Qwen2.5-VL-32B | 48.4 (+20.8) | 62.7 (+17.7) | 64.9 (+9.5) | 58.7 (+16.0) |

#### 4.6.2 Generalization across model scales

The authors also test SAIL-RL on Qwen3-VL backbones at 2B, 8B, and 30A3B scales. Table 8 shows consistent gains over both the standard SFT baseline (Instruct) and an outcome-only variant (+ Answer Reward).

On **Qwen3-VL-8B**, SAIL-RL improves MathVision from **24.2** to **28.7** and achieves an average gain of **+4.1** over Instruct. At larger scale, **Qwen3-VL-30A3B** improves from **47.2** to **51.0**, a **+3.8** average gain. The authors conclude that SAIL-RL scales reliably to stronger architectures.

**Table 8: Ablation on open-source architectures with diverse model scales.**

| Model | Method | MathVision | LogicVista | MMMU | Average |
|---|---|---:|---:|---:|---:|
| Qwen3-VL-2B | Instruct | 18.3 | 37.1 | 51.2 | 35.5 |
| Qwen3-VL-2B | + Answer Reward | 19.2 | 38.2 | 52.8 | 36.7 |
| Qwen3-VL-2B | + SAIL-RL | 20.6 (+2.3) | 39.4 (+2.3) | 53.8 (+2.6) | 37.9 (+2.4) |
| Qwen3-VL-8B | Instruct | 24.2 | 39.4 | 63.8 | 42.5 |
| Qwen3-VL-8B | + Answer Reward | 26.5 | 41.3 | 65.6 | 44.5 |
| Qwen3-VL-8B | + SAIL-RL | 28.7 (+4.5) | 43.4 (+4.0) | 67.6 (+3.8) | 46.6 (+4.1) |
| Qwen3-VL-30A3B | Instruct | 29.4 | 42.1 | 70.1 | 47.2 |
| Qwen3-VL-30A3B | + Answer Reward | 31.4 | 43.9 | 71.7 | 49.0 |
| Qwen3-VL-30A3B | + SAIL-RL | 33.6 (+4.2) | 45.8 (+3.7) | 73.6 (+3.5) | 51.0 (+3.8) |
