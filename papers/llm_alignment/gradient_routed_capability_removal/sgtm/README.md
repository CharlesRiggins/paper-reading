# Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs
arXiv: 2512.05648 | Igor Shilov et al. (Anthropic Fellows Program / Anthropic / Imperial College London / University of Edinburgh / Constellation, 2025)
Code: https://github.com/safety-research/selective-gradient-masking
Blog: https://alignment.anthropic.com/2025/selective-gradient-masking/

## Files

- `s01_front_matter.md` — 标题、作者、摘要与研究问题。论文将“能力移除”放在预训练阶段处理：将目标领域知识定位到可消融参数，而不要求训练数据过滤器零漏检。核心主张是 SGTM 在含标签噪声时优于数据过滤和先前 Gradient Routing，且相对 RMU 抵抗后续再训练约 **7×**。
- `s02_introduction.md` — 说明双用途能力、数据过滤的漏检/误伤权衡，以及论文的五项贡献。它区分了后训练拒答、输出分类器、机器遗忘和预训练知识定位，并概览 TinyStories 与 Wikipedia 两套实验。
- `s03_related_work.md` — 将工作定位于后训练安全缓解、预训练数据过滤、机器遗忘、模块化网络与 Gradient Routing。重点解释为何“吸收”性质使知识定位在漏标样本存在时可能优于纯过滤。
- `s04_method.md` — SGTM 的完整训练流程：把注意力头与 MLP 单元拆成 `forget` / `retain` 参数；对目标数据只保留 `forget` 参数梯度；对可信 retain 数据在前向中屏蔽 `forget` 参数；训练后将其置零。包含三类数据、干预表和与 activation-gradient routing 的差异。
- `s05_tinystories.md` — 英语保留、西班牙语遗忘的合成双语实验。覆盖人工漏标、retain/forget 权衡、泄漏率定义的直观解释和梯度范数证据；`64M` 模型在最多 40% 未发现目标数据时泄漏仅约 **0.005–0.02**。
- `s06_wikipedia.md` — `254M` 模型在 `3.7B` 英文 Wikipedia token 上移除生物学知识的主实验。记录弱/严格过滤比较、相邻领域保留、**5–6%** 通用知识计算惩罚、逐 token 损失分布，以及 50/50 对抗性微调下 RMU **50** steps、SGTM/严格过滤 **350** steps 的结果。
- `s07_discussion_conclusion.md` — 讨论和结论：结果是小模型上的概念验证，使用损失代理而非直接危险能力基准；SGTM 仍会受上下文注入影响，应属于纵深防御的一层。还讨论从一次训练导出授权完整版和消融部署版的双模型设想。
- `s08_appendix_a_b.md` — Appendix A–B：比较 Gradient Routing、Activation Masking 和两种 joint-weight SGTM 变体；用按 `6 × parameters × tokens` 拟合的 scaling law 将损失换算为计算惩罚，量化 SGTM 的遗忘、相邻知识和通用知识取舍。
- `s09_appendix_c_e.md` — Appendix C–E：消融前/后的双模型取舍，TPR/FPR 全范围标签质量热图，以及控制通用能力起点后的对抗微调。该控制实验显示 SGTM 的恢复阻力不能只归因于整体模型退化，但也承认部分知识可能较浅、早期恢复更快。
- `s10_appendix_f_g.md` — Appendix F–G：Transformer 权重的精确切分、`θ_forget`/`θ_retain` 集合，以及以“等效标准训练目标 token 暴露量”定义的信息泄漏指标。
- `s11_appendix_h_k.md` — Appendix H–K：军事与战争类别的额外实验、TinyStories/Wikipedia 的训练和 RMU 超参数、logit calibration，以及 ORES `articletopic` 分类体系如何生成目标、相邻和通用评测集。
- `s12_references.md` — 完整参考文献表，覆盖数据过滤、机器遗忘、梯度路由、模块化网络、训练 scaling 和双用途风险。
