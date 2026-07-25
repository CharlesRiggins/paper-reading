## 9. 以 NLA 初始化监督式 Activation Oracle

自由形式的 AV 解释适合探索，但研究者往往希望直接提出问题，例如“这段文本的语言是什么？”或“模型是否意识到正在被评估？”。作者把 AV checkpoint 再监督微调为问答式 **activation oracle（AO）**，测试 NLA 是否学习到了可迁移的“读取激活”能力。

训练数据来自已知属性由上下文指定的对话：抽取一个激活，配对研究者问题和有标准答案的自然语言回答。例如，先要求 assistant 后续构造量子力学逻辑悖论，再从后续 token 的激活提问“assistant 以何种方式与用户互动？”。这样问答答案可由前文确定，但 AO 仍只能看到激活和问题。

实验取 `Opus 4.6` 的 **4 个** NLA 训练阶段 AV checkpoint 做 QA SFT，并与 `PastLens` 初始化比较。PastLens 在预训练式文本上从激活复原前面 $N\sim U[1,50]$ 个 token；其预训练规模从 **16,000** 到 **3,000,000** sequences。所有模型用相同 QA SFT 后，在 CoT Hints、Safety Sandbagging、User Modeling 与 Classification 的问答变体上测试。

网页可确认的结论是：**所有 NLA 初始化的 AO 都超过直接从 Opus 4.6 做 QA SFT 的朴素基线。** 页面抓取在结果图之后截断，未提供 NLA checkpoint 间、PastLens 间或各任务的完整数值；因此本整理不把这一结论延伸为未公开的定量优势。

## 10. 限制与正确使用方式

### 10.1 Confabulation

解释可包含互相矛盾或可验证为假的上下文细节。显然错误的事实较容易与原始 transcript 核对；但更抽象的“模型似乎在考虑 X”难以证伪。应读取重复出现的主题，并以独立方法复核，不应把单条解释视为真实思维链。

### 10.2 缺少机制性 grounding

NLA 作为端到端语言模型是黑箱：无法指出解释中的哪一句由激活的哪个维度、SAE feature 或电路路径驱动。它可与 attribution graph、SAE、线性 probe 互补，但不能代替机制级定位。

### 10.3 AV 的表达能力过强

AV 是完整 LLM，可能从有限线索做额外推断，而非仅转述 $h_l$。单激活输入在结构上减少了它使用完整上下文进行事后推理的能力，却不能消除这种风险。解释中的流畅论证不等同于信息一定已明确编码在激活内。

### 10.4 成本与覆盖率

训练需要对两个完整模型联合优化，其中 AV 走 RL；推理时一个激活常要生成数百 token。因而很难对长轨迹的每个 token 连续运行，随机抽样也会漏掉局部表征。现阶段更适合针对关键 token、关键评估或调查线索的离线/半离线审计。

### 10.5 目标函数的退化解

在极限情形，AV 可以逐字复述输入上下文，或产生只有 AR 能解码的不可读/伪可读符号，从而获得低重构误差。KL 正则在当前规模中有助于保持语言性，语义保持改写实验也未发现强隐写证据；但这不保证更大模型、更多训练步骤后仍避免通信协议退化。另一个开放问题是：有些激活内容可能本来难以完整语言化，或只在特定层、特定 token 处出现。

## 11. 实践检查清单

1. 明确选择问题相关的层与 token；不要把“没看到某概念”解释为它在模型中不存在。
2. 优先汇总主题级、跨邻近 token 重复出现的解释；逐句具体信息必须回看原 transcript。
3. 把 NLA 产出写成可反驳假设，再用 prompt 对照、激活干预、归因/SAE 或训练数据溯源验证。
4. 对安全审计报告区分“模型显示了某种内部表征线索”与“模型真实持有某种稳定意图”；前者不自动推出后者。
5. 报告采样 token、生成配置、层位置、判定器与聚合规则；尤其不要把 NLA-aware 比例误当作经过校准的主观概率。

## 12. 资源

- 交互式论文：<https://transformer-circuits.pub/2026/nla/>
- Anthropic 博客导读：<https://www.anthropic.com/research/natural-language-autoencoders>
- 训练实现与 checkpoint 转换：<https://github.com/kitft/natural_language_autoencoders>
- 开源 NLA 模型合集：<https://huggingface.co/collections/kitft/nla-models>
- 轻量推理实现：<https://github.com/kitft/nla-inference>
- Neuronpedia 交互浏览：<https://www.neuronpedia.org/nla>

开放模型发布使该方法可以在 `Qwen`、`Gemma` 与 `Llama` 上复现实验；但完整训练涉及多 GPU、模型下载和 API 生成的 warm-start 数据，资源代价并不轻。