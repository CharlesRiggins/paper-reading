# LoRA Learns Less and Forgets Less

arXiv: 2405.09673 | Dan Biderman et al. (Columbia University & Databricks Mosaic Research, 2024)
Code: https://github.com/danbider/lora-tradeoffs

## Files

- `s01_abstract_introduction.md` — 提出两大研究问题：LoRA 在代码/数学等困难领域能否匹敌全量微调？LoRA 能否缓解灾难性遗忘？在 CPT（约 20B token）和 IFT（约 100K 样本）两种设定下比较 Llama-2-7B。总结五项核心贡献，包括低秩 LoRA 显著落后、遗忘更少、全量微调学得高秩扰动（10-100×）等。

- `s02_background.md` — 形式化定义 LoRA：冻结预训练权重 $W_{\mathrm{pretrained}}$，学习低秩扰动 $\Delta = \gamma_r AB$，其中 $\gamma_r = \alpha/r$。说明 $r=16$ 时仅训练不到 1% 参数，以及从仅适配 $W_q, W_v$ 演进到适配所有 Transformer 模块（注意力 + 前馈）的最佳实践。

- `s03_experimental_setup.md` — 详细列出四个数据集（StarCoder-Python 20B、OpenWebMath 14.7B、Magicoder-Evol-Instruct-110K 72.97M、MetaMathQA 103M token）。目标域评估用 HumanEval（0-shot pass@1）和 GSM8K（5-shot pass@1）。遗忘度量用 HellaSwag、WinoGrande、ARC-Challenge 的对数概率准确率。

- `s04_target_domain_performance.md` — CPT 场景下低秩 LoRA（r=16, 64）大幅落后全量微调，差距随训练量增大而扩大（r=256 在代码 CPT 仅达 HumanEval 0.224 vs 全量 0.263）。IFT 场景下 r=256 可弥合差距（HumanEval 0.498 vs 0.497），数学 IFT 中 r=64 即接近全量微调。

- `s05_forgetting_tradeoff.md` — LoRA 一致保持更好的源域性能，遗忘程度受 rank 控制（rank 越高遗忘越多，但仍低于全量微调）。IFT 比 CPT 遗忘更严重，代码比数学更严重。Pareto 曲线揭示学习-遗忘权衡：代码场景 LoRA (r=256) 以更少遗忘换取相似性能；数学场景全量微调有时提供更优权衡。

- `s06_tulu_constraint_svd.md` — Tülu-v2-mix 通用 IFT 上 LoRA 即使低秩也匹敌全量微调且遗忘更少。LoRA 比注意力 dropout 和权重衰减更好地抑制遗忘，同时维持生成多样性（避免“分布坍缩”）。SVD 分析揭示全量微调扰动 $\Delta$ 秩达 10-100× LoRA 配置，秩随训练增长，MLP 模块秩高于注意力模块，中间层秩高于首尾层。

- `s07_hyperparameters_related_work.md` — 超参数敏感性分析显示 $\alpha = 2r$ 对高秩训练至关重要，MLP 模块是持续学习的主要场所，LoRA 学习率需比全量微调高约一个数量级。给出四项实践建议：IFT 优于 CPT、rank=256、$\alpha=2r$、学习率在 $[1e^{-5}, 5e^{-4}]$。相关工作总结 LoRA 变体、基准争议、代码/数学持续学习及学习-遗忘权衡研究。

- `s08_discussion_conclusion.md` — 讨论模型规模扩展性未研究、SVD 分析局限性（高秩解不排除低秩解存在、仅分析 CPT）。结论总结核心发现：LoRA 在学习上落后但遗忘更少、维持生成多样性、全量微调学得高秩扰动。附致谢和作者贡献说明。

- `s09_references.md` — 完整的 84 条参考文献表，涵盖 LoRA 原始论文、大模型训练、代码/数学微调、持续学习/灾难性遗忘、SVD 分析等相关文献。

- `s10_appendix_ab.md` — 附录 A 提供完整 LoRA 配置和四种训练场景的超参数。A.1 讨论是否训练嵌入层。附录 B 展示学习率敏感性（LoRA 最优 LR 约全量微调 10 倍）、AdamW vs LionW 对比、$\alpha = 2r$ 的关键作用。

- `s11_appendix_ci.md` — 附录 C 报告 Tülu-v2-mix 结果（LoRA 匹敌全量微调）。附录 D 提供所有补充表格 S1-S13。附录 E 讨论 SVD 图形。附录 F 分析 pass@k 多样性。附录 G 展示训练数据集详情及示例。附录 H 提供单/多 GPU 理论内存分析（7B/70B/405B 模型）。附录 I 报告吞吐量（LoRA 慢约 15%）和峰值内存（小 batch 下节省约 40%）测量。
