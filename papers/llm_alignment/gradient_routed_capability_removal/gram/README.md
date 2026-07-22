# An off switch for dual-use knowledge in AI models
Anthropic Alignment | AE Studio in collaboration with Anthropic (2026-07-08)
Source: https://www.anthropic.com/research/off-switch-dual-use
Further technical post: https://alignment.anthropic.com/2026/modular-pretraining/

## Files

- `s01_motivation.md` — 解释双用途知识的访问控制目标：既减少不受信任部署中的能力，又让可信用户保留同类能力，同时尽量不损伤其他任务表现。文章将 `GRAM` 放在拒答训练、输入/输出分类器、预训练数据过滤和 `SGTM` 的脉络中，强调后者往往只能提供固定能力集合的单一模型。
- `s02_gram_method.md` — 介绍 Gradient-Routed Auxiliary Modules（`GRAM`）：给每层 Transformer 增添并按领域划分可移除模块；双用途文本只更新相应模块而通用权重冻结。文章明确指出，模块有时也在通用文本训练中开启以促进协作；四个模块可由一次训练形成 $2^4=16$ 种独立的开/关部署配置。
- `s03_evaluation_conclusion.md` — 汇总三层实验：合成主题故事、真实 web/code/science 语料中的四领域实验，以及 `50M–5B`、共七种规模的扩展。记录主张和限制：模块删除接近从未见过目标数据的过滤效果、与过滤相近地抵抗少量恶意微调，但尚未用于 `Claude` 或前沿生产训练，并只用 next-token prediction 评估。
