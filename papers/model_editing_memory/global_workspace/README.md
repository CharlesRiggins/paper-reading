# Verbalizable Representations Form a Global Workspace in Language Models

Transformer Circuits Thread | Wes Gurnee, Nicholas Sofroniew et al.（Anthropic，2026-07-06）

- 原文：https://transformer-circuits.pub/2026/workspace/index.html
- 代码与复现：https://github.com/anthropics/jacobian-lens
- 类型：交互式研究长文（非 arXiv 预印本）
- 关键词：`Jacobian lens`、`J-space`、全局工作空间、可报告性、内部推理、机制可解释性、对齐审计、反事实反思训练

## 一句话结论

作者提出 **Jacobian lens（J-lens）**，把中间层残差流中「在平均意义上会使模型未来说出某词」的方向译成词表读数；由这些方向稀疏张成的 **J-space** 在实验中表现为一个小而特权化的工作空间：它支持可报告、受指令调制、灵活推理与跨模块广播，但大量自动化处理可绕过它。

## 阅读路径

1. 先读 `s01_abstract_intro.md` 和 `s02_methods_j_lens.md`：掌握研究问题、符号与干预方式。
2. 读 `s03_global_workspace_function.md`：这是论文的核心因果证据。
3. 读 `s04_structure_capacity_broadcast.md`：理解「为何称为工作空间」。
4. 若关心安全，读 `s05_alignment_auditing.md` 与 `s06_post_training_reflection.md`。
5. `s07_discussion_related_work.md` 汇总边界与面试可讲的批判；`s08_appendix_methods_extensions.md`、`s09_appendix_mechinterp.md` 留作复现和延伸阅读。

## 文件索引

- `s01_abstract_intro.md` — 问题、主张、核心实验地图与术语。
- `s02_methods_j_lens.md` — J-lens/J-space 的构造、读写干预、与 logit/tuned lens 的区别。
- `s03_global_workspace_function.md` — 可报告性、定向调制、内部推理、灵活性/自动性的因果实验。
- `s04_structure_capacity_broadcast.md` — 层级区域、ignition、容量、SAE 对照与广播结构。
- `s05_alignment_auditing.md` — 黑邮件、提示注入、模型生物体与自动化审计。
- `s06_post_training_reflection.md` — 后训练的 Assistant 视角、自我监控、反事实反思训练。
- `s07_discussion_related_work.md` — 相关工作、限制、意识类比与批判性结论。
- `s08_appendix_methods_extensions.md` — 方法消融、形式化、多 token 的 template/oracle lens。
- `s09_appendix_mechinterp.md` — 调制细节、broadcast 验证、机制可解释性应用。
- `s10_interview_notes.md` — 面试速答、可复现实验和开放问题。
