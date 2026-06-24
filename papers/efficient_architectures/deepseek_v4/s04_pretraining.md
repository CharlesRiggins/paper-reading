## 4. Pre-Training

### 4.1. Data Construction

On top of the pre-training data of DeepSeek-V3, we endeavor to construct a more diverse and higher-quality training corpus with longer effective contexts. We continually refine our data construction pipelines. For web-sourced data, we implement filtering strategies to remove batched auto-generated and templated content, thereby mitigating the risk of **model collapse** (Zhu et al., 2024). Mathematical and programming corpora still remain core components of our training data, and we further enhance the coding capabilities of DeepSeek-V4 series by incorporating **agentic data** during the mid-training phase. For multilingual data, we build a larger corpus for DeepSeek-V4, improving its capture of long-tail knowledge across different cultures. For DeepSeek-V4, we place a particular emphasis on **long-document data curation**, prioritizing scientific papers, technical reports, and other materials that reflect unique academic values. Combining all the above, our pre-training corpus comprises more than **32T tokens**, containing mathematical contents, codes, web pages, long documents, and other high-quality categories.

For pre-training data, we largely follow the same pre-processing strategies of DeepSeek-V3. For tokenization, on top of the DeepSeek-V3 tokenizer, we introduce a few special tokens for context construction, and still remain the vocabulary size to be **128K**. We also inherit the **token-splitting** (DeepSeek-AI, 2024) and **Fill-in-Middle (FIM)** (DeepSeek-AI, 2024) strategies from DeepSeek-V3. Inspired by Ding et al. (2024), we pack documents from different sources into appropriate sequences to minimize sample truncation. Different from DeepSeek-V3, we employ **sample-level attention masking** during pre-training.

### 4.2. Pre-Training Setups

#### 4.2.1. Model Setups

**DeepSeek-V4-Flash.** We set the number of Transformer layers to **43** and the hidden dimension $d$ to **4096**. For the first two layers, we use pure **sliding window attention**. For the subsequent layers, **CSA** and **HCA** are used in an interleaved manner. For CSA, we set the compression rate $r$ to **4**, the number of indexer query heads $n_{h}^{I}$ to **64**, the indexer head dimension $c^{I}$ to **128**, and the number of KV entries selected for sparse attention (i.e., attention top-k) to **512**. For HCA, we set the compression rate $m^{\prime}$ to **128**. For both CSA and HCA, we set the number of query heads $n_{h}$ to **64**, the head dimension $d_h$ to **512**, and the query compression dimension $d_{c}$ to **1024**. The number of output projection groups $g$ is set to **8**, and the dimension of each intermediate attention output $d_{g}$ is set to **1024**. For the additional branch of sliding window attention, the window size $n_{\mathrm{win}}$ is set to **128**.

We employ **MoE layers** in all Transformer blocks, but use the **Hash routing** strategy for the first 3 MoE layers. Each MoE layer consists of **1 shared expert** and **256 routed experts**, where the intermediate hidden dimension of each expert is **2048**. Among the routed experts, **6 experts** will be activated for each token. The **multi-token prediction** depth is set to **1**. As for mHC, the expansion factor $n_{\mathrm{hc}}$ is set to **4**, and the number of Sinkhorn-Knopp iterations $t_{\mathrm{max}}$ is set to **20**. Under this configuration, DeepSeek-V4-Flash comprises **284B** total parameters, of which **13B** are activated for each token.

**DeepSeek-V4-Pro.** We set the number of Transformer layers to **61** and the hidden dimension $d$ to **7168**. For the first two layers, we use HCA. For the subsequent layers, CSA and HCA are used in an interleaved manner. For CSA, we set the compression rate $r$ to **4**, the number of indexer query heads $n_{h}^{I}$ to **64**, the indexer head dimension $c^{I}$ to **128**, and the number of KV entries selected for sparse attention (i.e., attention top-k) to **1024**. For HCA, we set the compression rate $m^{\prime}$ to **128**. For both CSA and HCA, we set the number of query heads $n_{h}$ to **128**, the head dimension $d_h$ to **512**, and the query compression dimension $d_{c}$ to **1536**. The number of output projection groups $g$ is set to **16**, and the dimension of each intermediate attention output $d_{g}$ is set to **1024**. For the additional branch of sliding window attention, the window size $n_{\mathrm{win}}$ is set to **128**.

We employ MoE layers in all Transformer blocks, but use the Hash routing strategy for the first 3 MoE layers. Each MoE layer consists of **1 shared expert** and **384 routed experts**, where the intermediate hidden dimension of each expert is **3072**. Among the routed experts, **6 experts** will be activated for each token. The multi-token prediction depth is set to **1**. As for mHC, the expansion factor $n_{\mathrm{hc}}$ is set to **4**, and the number of Sinkhorn-Knopp iterations $t_{\mathrm{max}}$ is set to **20**. Under this configuration, DeepSeek-V4-Pro comprises **1.6T** total parameters, of which **49B** are activated for each token.

#### 4.2.2. Training Setups

**DeepSeek-V4-Flash.** We employ the **Muon optimizer** (Jordan et al., 2024; Liu et al., 2025) for the majority of parameters, but use the **AdamW optimizer** (Loshchilov and Hutter, 2017) for the embedding module, the prediction head module, and the weights of all RMSNorm modules. For AdamW, we set its hyper-parameters to $\beta_{1} = 0.9$, $\beta_{2} = 0.95$, $\varepsilon = 10^{-20}$, and weight_decay = 0.1. For Muon, we set the momentum to **0.95** and the weight decay to **0.1**, and rescale the RMS of each update matrix to **0.18** for reutilization of the AdamW learning rate.

We train DeepSeek-V4-Flash on **32T tokens**, and as in DeepSeek-V3, we also employ a batch size scheduling strategy that increases the batch size (in tokens) from a small size to **75.5M** and then keeps it at 75.5M during most of the training. The learning rate is linearly warmed up in the first **2000 steps**, maintained at **$2.7 \times 10^{-4}$** for most of the training. Near the end of the training, we finally decay the learning rate to **$2.7 \times 10^{-5}$** following a cosine schedule. The training starts with a sequence length of **4K**, and we gradually extend the training sequence length to **16K**, **64K**, and **1M**.

As for the setups of sparse attention, we first warmup the model with **dense attention** for the first **1T tokens**, and introduce sparse attention at the sequence length of **64K** and keep sparse attention during the rest of the training. When introducing attention sparsity, we first set a short stage to warm up the lightning indexer in CSA, and then train the model with sparse attention for most of the training. For **auxiliary-loss-free load balancing**, we set the bias update speed to **0.001**. For the balance loss, we set its loss weight to **0.0001** to avoid extreme imbalance within single sequences. The **MTP loss weight** is set to **0.3** for most of the training, and to **0.1** upon the start of learning rate decay.

**DeepSeek-V4-Pro.** Except for specific values of hyper-parameters, the training setup of DeepSeek-V4-Pro is largely consistent with that of DeepSeek-V4-Flash. We employ the Muon optimizer for the majority of parameters, but use the AdamW optimizer for the embedding module, the prediction head module, and the weights of all RMSNorm modules. The hyper-parameters of AdamW and Muon are the same as those of DeepSeek-V4-Flash.

We train DeepSeek-V4-Pro on **33T tokens**, and also employ a batch size scheduling strategy, with the maximum batch size being **94.4M** tokens. The learning rate scheduling strategy is largely the same as that of DeepSeek-V4-Flash, but the peak learning rate is set to **$2.0 \times 10^{-4}$** and the end learning rate is set to **$2.0 \times 10^{-5}$**. The training also starts with a sequence length of **4K**, and the length is gradually extended to **16K**, **64K**, and **1M**. Compared with DeepSeek-V4-Flash, DeepSeek-V4-Pro starts with a longer stage of dense attention, and the strategy of introducing sparse attention is the same as DeepSeek-V4-Flash, following a two-stage training method. For auxiliary-loss-free load balancing, we set the bias update speed to **0.001**. For the balance loss, we set its loss weight to **0.0001** to avoid extreme imbalance within single sequences. The MTP loss weight is set to **0.3** for most of the training, and to **0.1** upon the start of learning rate decay.

#### 4.2.3. Mitigating Training Instability

Training trillion-parameter MoE models presents significant stability challenges, and DeepSeek-V4 series are no exception. We encountered notable instability challenges during training. While simple rollbacks could temporarily restore the training state, they proved inadequate as a long-term solution because they do not prevent the recurrence of loss spikes. Empirically, we identified that the occurrence of spikes is consistently tied to **outliers in the MoE layers**, and the routing mechanism itself appears to exacerbate the emergence of these outliers. Therefore, we sought to tackle this issue from two dimensions: breaking the vicious cycle induced by routing, and directly suppressing anomalous values. Fortunately, we discovered two practical techniques that effectively maintain training stability. Although a comprehensive theoretical understanding of their underlying mechanisms remains an open question for now, we are sharing them openly to foster further exploration by the community.

**Anticipatory Routing.** We found that decoupling the synchronous updates of the backbone network and the routing network significantly improves training stability. Consequently, at step $t$, we use the current network parameters $\theta_{t}$ for feature computation, but the routing indices are computed and applied using the historical network parameters $\theta_{t - \Delta t}$. In practice, to circumvent the overhead of loading model parameters twice, we fetch the data for step $t$ in advance at step $t - \Delta t$. We "anticipatorily" compute and cache the routing indices to be used later at step $t$, which is why we name this approach **Anticipatory Routing**. We also heavily optimized this at the infrastructure level. First, given that pre-computing the routing indices only requires a single forward pass over the data, we carefully orchestrated the pipeline execution and the overlapping of computation with Expert Parallelism (EP) communication, successfully bounding the additional wall-clock time overhead of Anticipatory Routing to approximately **20%**. Second, we introduced an automatic detection mechanism that triggers a short rollback and activates Anticipatory Routing exclusively when a loss spike occurs; after operating in this mode for a certain period, the system reverts to standard training. Ultimately, this dynamic application allows us to avert loss spikes with negligible overall additional training overhead, all without compromising model performance.

**SwiGLU Clamping.** In previous literature (Bello et al., 2017; Riviere et al., 2024), clamping has been explicitly utilized to constrain numerical ranges, thereby enhancing training stability. In our actual training runs, we empirically found that applying **SwiGLU clamping** (OpenAI, 2025) effectively eliminates outliers and substantially aids in stabilizing the training process, without compromising performance. Throughout the training of both DeepSeek-V4-Flash and DeepSeek-V4-Pro, we clamped the linear component of SwiGLU to the range of **$[-10, 10]$**, while capping the upper bound of the gate component at **10**.

### 4.3. Evaluations

#### 4.3.1. Evaluation Benchmarks

For the evaluation of the base models, we consider benchmarks spanning four key dimensions: **world knowledge**, **language understanding and reasoning**, **coding and mathematics**, and **long-context processing**.

**World knowledge** benchmarks include AGIEval (Zhong et al., 2023), C-Eval (Huang et al., 2023), CMMLU (Li et al., 2023), MMLU (Hendrycks et al., 2020), MMLU-Redux (Gema et al., 2024), MMLU-Pro (Wang et al., 2024b), MMMLU (OpenAI, 2024a), MultiLoKo (Hupkes and Bogoychev, 2025), Simple-QA verified (Haas et al., 2025), SuperGPQA (Du et al., 2025), FACTS Parametric (Cheng et al., 2025), and TriviaQA (Joshi et al., 2017).

**Language understanding and reasoning** benchmarks include BigBench Hard (BBH) (Suzgun et al., 2022), DROP (Dua et al., 2019), HellaSwag (Zellers et al., 2019), CLUEWSC (Xu et al., 2020), and WinoGrande (Sakaguchi et al., 2019).

**Coding and mathematical** benchmarks include BigCodeBench (Zhuo et al., 2025), HumanEval (Chen et al., 2021), GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), MGSM (Shi et al., 2023), and CMath (Wei et al., 2023).

**Long context** benchmarks include LongBench-V2 (Bai et al., 2025b).

#### 4.3.2. Evaluation Results

In Table 1, we provide a detailed comparison of the base models for DeepSeek-V3.2, DeepSeek-V4-Flash, and DeepSeek-V4-Pro, all evaluated under a unified internal framework with strictly consistent settings.

Comparing **DeepSeek-V4-Flash-Base** with **DeepSeek-V3.2-Base** reveals a compelling efficiency story. Despite utilizing a substantially smaller number of both activated and total parameters, DeepSeek-V4-Flash-Base outperforms DeepSeek-V3.2-Base across a wide array of benchmarks. This advantage is especially evident in world knowledge tasks and challenging long-context scenarios. These results underscore that architectural improvements, refined data quality, and training optimizations in DeepSeek-V4-Flash-Base yield superior performance even with a more compact parameter budget, effectively surpassing the larger DeepSeek-V3.2-Base on the majority of evaluations.

Furthermore, **DeepSeek-V4-Pro-Base** demonstrates a further, decisive leap in capability, establishing near-universal dominance over both DeepSeek-V3.2-Base and DeepSeek-V4-Flash-Base. With improvements across almost all categories, DeepSeek-V4-Pro-Base reaches new performance highs among DeepSeek base models on the most demanding benchmarks. On knowledge-intensive evaluations, it delivers dramatic gains, while also substantially advancing long-context understanding. On most reasoning and code benchmarks, DeepSeek-V4-Pro-Base also exceeds both previous models. This comprehensive uplift confirms DeepSeek-V4-Pro-Base as the strongest foundation model in the DeepSeek series, outperforming its predecessors across the spectrum of knowledge, reasoning, coding, and long-context capabilities.

**Table 1 | Comparison among DeepSeek-V3.2-Base, DeepSeek-V4-Flash-Base, and DeepSeek-V4-Pro-Base.** All models are evaluated in our internal framework and share the same evaluation setting. Scores with a gap not exceeding 0.3 are considered to be at the same level. The highest score in each row is in bold font, and the second is underlined.

| Category | Benchmark (Metric) | # Shots | DeepSeek-V3.2-Base | DeepSeek-V4-Flash-Base | DeepSeek-V4-Pro-Base |
|---|---|---|---|---|---|
| **Architecture** | — | — | MoE | MoE | MoE |
| **Params** | # Activated Params | — | 37B | **13B** | 49B |
| **Params** | # Total Params | — | 671B | **284B** | 1.6T |
| **World Knowl.** | AGIEval (EM) | 0-shot | 80.1 | 82.6 | **83.1** |
| | MMLU (EM) | 5-shot | 87.8 | 88.7 | **90.1** |
| | MMLU-Redux (EM) | 5-shot | 87.5 | 89.4 | **90.8** |
| | MMLU-Pro (EM) | 5-shot | 65.5 | 68.3 | **73.5** |
| | MMMLU (EM) | 5-shot | 87.9 | 88.8 | **90.3** |
| | C-Eval (EM) | 5-shot | 90.4 | 92.1 | **93.1** |
| | CMMLU (EM) | 5-shot | 88.9 | 90.4 | **90.8** |
| | MultiLoKo (EM) | 5-shot | 38.7 | 42.2 | **51.1** |
| | Simple-QA verified (EM) | 25-shot | 28.3 | 30.1 | **55.2** |
| | SuperGPQA (EM) | 5-shot | 45.0 | 46.5 | **53.9** |
| | FACTS Parametric (EM) | 25-shot | 27.1 | 33.9 | **62.6** |
| | TriviaQA (EM) | 5-shot | 83.3 | 82.8 | **85.6** |
| **Lang. & Reas.** | BBH (EM) | 3-shot | **87.6** | 86.9 | 87.5 |
| | DROP (F1) | 1-shot | 88.2 | 88.6 | **88.7** |
| | HellaSwag (EM) | 0-shot | 86.4 | 85.7 | **88.0** |
| | WinoGrande (EM) | 0-shot | 78.9 | 79.5 | **81.5** |
| | CLUEWSC (EM) | 5-shot | 83.5 | 82.2 | **85.2** |
| **Code & Math** | BigCodeBench (Pass@1) | 3-shot | **63.9** | 56.8 | 59.2 |
| | HumanEval (Pass@1) | 0-shot | 62.8 | 69.5 | **76.8** |
| | GSM8K (EM) | 8-shot | 91.1 | 90.8 | **92.6** |
| | MATH (EM) | 4-shot | 60.5 | 57.4 | **64.5** |
| | MGSM (EM) | 8-shot | 81.3 | **85.7** | 84.4 |
| | CMath (EM) | 3-shot | 92.6 | **93.6** | 90.9 |
| **Long Context** | LongBench-V2 (EM) | 1-shot | 40.2 | 44.7 | **51.5** |

> **Note:** The highest score in each row is in **bold**, and the second is underlined (not rendered here but noted for reference). Scores with a gap ≤ 0.3 are considered tied.
