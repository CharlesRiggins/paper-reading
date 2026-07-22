# Gradient-Routed Capability Removal

该专题归档同一条“**在预训练期间定位并可移除双用途知识**”研究路线的两项连续工作。两者均以梯度路由限制目标领域知识写入的位置，并在部署时移除相应参数；差别在于定位载体和部署粒度。

| Work | Directory | Core idea | Deployment granularity |
|---|---|---|---|
| **SGTM** — *Beyond Data Filtering* (2025) | `sgtm/` | 在原有 Transformer 的指定注意力头和 MLP 单元中定位单一目标领域；训练后置零 `forget` 参数。 | 原模型与消融版本；重点研究标签漏检下的目标知识泄漏与再学习。 |
| **GRAM** — *An off switch for dual-use knowledge* (2026) | `gram/` | 在每层增加按领域划分的辅助模块；目标领域数据仅更新其模块。 | 每个领域独立开/关；4 个模块可形成 $2^4=16$ 种模型配置。 |

## Reading order

1. 从 `sgtm/README.md` 开始：理解数据过滤的漏检/误伤权衡、parameter-gradient masking、吸收效应和对抗性再微调实验。
2. 再读 `gram/README.md`：理解该思路如何扩展为多个领域的模块化、可组合访问控制。

## Shared evidence boundary

两项工作都把下一 token 预测损失作为主要代理指标，未在真实生产前沿模型或直接危险能力基准上完成验证。它们应被视为纵深防御中预训练阶段的潜在组件，而非独立的安全保证。
