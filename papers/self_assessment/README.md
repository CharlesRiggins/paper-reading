# Say When You Don’t Know: Training LLMs to Expose Self-Assessment

| Field | Value |
|---|---|
| arXiv | — |
| Author | Anonymous |
| Affiliation | — |
| Year | 2026 |
| Code | — |
| Source PDF | `/Users/charlesriggins/Desktop/paper-reading/papers2review/8440_Say_When_You_Don_t_Know_T.pdf` |
| Parsed text | `/Users/charlesriggins/Desktop/paper-reading/papers2review/_parsed_text/8440_Say_When_You_Don_t_Know_T.md` |

## Files

- `s01_introduction_preliminaries.md` — 整理摘要、问题动机、Figure 1 的核心框架和 Sections 1–2。论文把 LLM 自评从“外部事后估计”重述为“模型在自身回答中暴露可靠性”的问题，并提出两个暴露时机：推理结束后的标量置信度，以及推理过程中的 `<uncertain>` 标记。该文件还保留了形式化设定：输入、推理轨迹、隐藏状态、最终答案正确性指标、end-of-reasoning 与 during-reasoning 信号的定义。
- `s02_verbalized_confidence.md` — 对应 Section 3，结构化转写 verbalized confidence 训练目标、GRPO 局部重加权解释、关键公式和行为结果。文件保留 Brier-style 奖励、策略重加权等式、logit-lens/PCA 机制证据，以及 calibration 指标从 ECE 0.383 降至 0.049、Brier 0.504 降至 0.166 等数字。它还包含错误类型分解表，展示校准后错误从“自信幻觉”转向“低置信不确定”。
- `s03_reasoning_time_uncertain_marker.md` — 对应 Section 4，描述训练模型在推理中遇到高风险状态时发出 `<uncertain>` 的方法。文件保留训练提示词、奖励排序与具体数值 `5.0 > 3.5 > 0.0 > -2.0`、重复惩罚、first-emission 位置解释和 marker 行为表。它强调该信号不是最终答案概率，而是可供 RAG/工具/重生成在推理中介入的高召回控制点。
- `s04_internal_mechanism_analysis.md` — 对应 Section 5，整理两种自评暴露方式的内部机制差异。文件保留 token-level KL、CKA、局部化与表示几何的叙事：verbalized confidence 更像在既有表示上锐化读出，而 `<uncertain>` marker 诱导更宽的局部计算状态和晚层表示分化。它还纳入主文对参数漂移位置的解释，说明相似的权重漂移可以带来不同的表示后果。
- `s05_evaluation.md` — 对应 Section 6，整理五个 factual QA benchmark、所有主要 baseline、Table 3 和 Table 4 的核心数值。文件展示 verbalized confidence 在 EM、ECE、OConf 上同时胜过 P(True)、temperature scaling 和 SFT baseline；也展示 `<uncertain>` marker 相比被动 detector/retrieval controller 更能把错误显式化。Adaptive RAG 表格保留各数据集 EM/F1/trigger rate，突出 Verbal-Calibrate 的 41.6% EM / 50.5% F1 与 Uncertain-Calibrate 的 40.9% EM / 48.1% F1。
- `s06_conclusion_limitations.md` — 对应 Section 7 以及主文限制陈述，收束论文的设计含义。文件强调两种信号是互补而非替代：最终置信度适合 trust/abstention/question-level retrieval gating，推理中标记适合在答案提交前介入。它也保留论文承认的范围限制：目前主要研究 factual QA 与 adaptive retrieval，且只使用单一 during-reasoning marker。
- `s07_references.md` — 将论文 53 条参考文献单独整理为 Markdown 表格。参考文献覆盖 uncertainty estimation/calibration、verbalized confidence、adaptive retrieval、Self-RAG/FLARE/DRAGIN/ADARAGUE、learned control tokens、LLM internal states 与 RL post-training 等主题。
- `s08_appendix_checklist.md` — 合并整理附录 A–G 与 NeurIPS checklist。文件包括 related work 的定位、trajectory-reweighting proof 的关键命题和公式、额外机制证据、补充定量表格、训练与 baseline 实现细节、错误案例、broader impact，以及 checklist 的答案摘要。它保留复现实验所需的核心配置，例如 verl/HybridFlow、vLLM 0.8.5、Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct、2×H100 80GB、SFT/GRPO 超参数和数据划分。