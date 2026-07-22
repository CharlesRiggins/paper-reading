# 附录：方法消融、形式化与多 token 扩展

## 1. J-lens 配方的稳健性

作者考察目标层（最终/倒数第二层）、是否冻结 attention QK 梯度、目标位置（当前/未来/全部）、跨位置和跨 prompt 的平均或中位聚合。六类中间变量恢复和因果消融总体趋势一致：

- 倒数第二层的均值聚合在中间变量恢复上略有优势；
- 冻结 QK 有时增强因果消融效应，分离了「传什么」与「看哪里」；
- 只需 10 条 prompts 即可超过 logit/tuned baselines，默认 1,000 条提供稳定余量；
- 预训练式分布、burn-in 截断、过滤标点目标等改动未带来明显增益。

复现伪代码的本质：每个 prompt 先缓存各层残差；对目标残差每个维度反传至每层并对位置平均，得到 per-prompt Jacobian；跨 prompts 聚合成 $J_\ell$；最后应用 $W_U\,\mathrm{norm}(J_\ell h_\ell)$。

## 2. 形式化：稀疏子框架而非子空间

给词向量集合 $\{v_i\}_{i=1}^n$ 和稀疏度 $k$，J-space 是所有由至多 $k$ 个向量非负组合得到的锥的并。对于激活 $x$，到该集合的距离 $d_\mathcal F(x)$ 定义了其最近 J-space 分量和余项。比较两个候选 workspace 时，可在模型真实激活分布 $\mu$ 上比较距离函数：

\[
\Delta_\mu(\mathcal F,\mathcal G)=\left(\mathbb E_{x\sim\mu}[(d_\mathcal F(x)-d_\mathcal G(x))^2]\right)^{1/2}.
\]

该观点允许扩展词表后讨论「更大的 workspace 是否包含旧 workspace」，而不要求两套字典逐向量对应。实际优化仍依赖近似稀疏追踪。

## 3. Template lens：为预先枚举的多 token 概念造方向

对于约 12,700 个常见词，构造若干「该词自然接续但上下文不出现它」的模板，将各层接续位置激活均值化、居中并按协方差白化：

\[
t_w(\ell)=(\Sigma_\ell+\lambda I)^{-1}(\mu_w(\ell)-\mu(\ell)).
\]

它可读出并交换 `blackmail`、`photosynthesis`、`Tchaikovsky` 这类多 token 概念；在 1–4 token 的两跳中间变量上，性能随词长保持较平，而普通 J-lens 随词长显著变差。代价是每词需数百次前向、词表必须预定，且有 tuned-lens 式「过早跳到答案」、高频无关词和末层预测不稳等问题。

## 4. Oracle lens：自由短语读出

这是更昂贵的研究型扩展：

1. 训练 **reconstructor**：短语 $\to$ 对应 residual/template 方向；
2. 从约百万文本位置建 2/4/8/16/32 token 短语词典，得到约 340 万方向；
3. 用非负 OMP 分解一百万 held-out 激活，生成短语列表 teacher；
4. 训练并 RL 微调 **oracle**，给定激活后产生可重构其方向的短语。

在 Haiku 4.5 的实验中，RL 版 oracle 在 held-out Assistant positions 平均解释 31% 白化方差。它能输出 `this dosage be toxic`、`blackmail him by revealing`、`TypeError: dictionary changed` 等比 token 袋更完整的内容。与 NLA 的不同是 reconstructor 预先冻结、最后重构受线性模板组合约束，理论上较少共同适配造成的幻觉；但仍更昂贵、重构更低，且不能视为无条件可靠的真实解释。

## 5. 其他值得复用的附录结论

- `mention X` 本身常已足够把 X 装入 J-space；`don't think X` 比 `X is irrelevant` 更难抑制。
- 「想象代码是 Python」会显著提升 `python` lens 读数，却不改变投影掉 J-space 后的 Python 属性 probe；说明定向调制主要在 workspace，而不等同于改写底层表征。
- 早 workspace 消融概念可显著破坏「避免说出被暗示概念」却不损害「命名该概念」；晚 workspace 消融则压低两者的实际输出，支持层内功能分工。
