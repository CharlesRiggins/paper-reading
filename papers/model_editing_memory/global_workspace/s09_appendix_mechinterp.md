# 附录：调制、广播与机制可解释性应用

## 1. 调制与竞争的补充结果

- 另外四类属性（英美拼写、语域、语法数、语气）重复了主文结论：模型能为 next-token 预测正确使用属性，但只有被要求**命名**属性时，属性词才显著进入 J-space。
- 双隐蔽任务下，两个静态概念在同一位置可近似独立共现（co-occupancy 0.46，对照 0.53）；一个概念与心算答案则强烈排他（0.09，对照 0.29）。双任务对到达 top-5 的总体概率影响不大，但心算从 95% 降到 72%。
- 单一 L79 层中，随机列表仅保留约 1–2 个概念；跨 L38–92 取最佳层才得到约 6 个，说明深度维度扩展了可分时存放的容量。

## 2. broadcast 的额外权重证据

对 MLP，按 SAE 的 lens-kurtosis 分层：workspace 内 MLP 输入权重的最大余弦匹配强烈偏向高 $\kappa$ 特征（顶层可占 42%，底层约 6%）；J-lens 向量与邻层 neuron 方向等量对比时，前者成为 top match 的比例由 workspace 前约 15% 上升至 60–65%。写侧的峰值更靠后，读侧更靠前，符合「先广泛读入 workspace，后为输出写回」的图景。

对注意力，J-space 对齐方向与 V/O 权重的组合远强于非对齐方向，与 Q/K 的组合较弱但仍高于随机。这支持注意力主要把 J-space 内容作为可移动信息，而 routing 信号未必同样以 J-space 格式编码。

## 3. J-lens 的机制定位

对 $(4+17)\times2+7$：lens 中 `21 → 42 → 49` 按层依次增强。均值差 activation patching 在相同层段最能改变答案：将 21、42、49 分别替换为其他值的有效层依次后移。它将 J-lens 读出的层级时间线与因果 patching 对齐，提供了使用 lens 做 localization 的验证。

## 4. J-lens attribution graph

做法是在每层/位置以稀疏 J-lens 分解的命名向量为节点，并保留一个非 J-space remainder 节点；冻结 attention pattern 和 norm 后反传贡献边。优点：不用训练字典、节点天然有词标签。限制：绝大多数影响仍在 remainder 中，且过完备字典使贪心分解不唯一；它只能画出「可言说骨架」，不能代表完整电路。

算术图例中，`21`、`doubled`、`42`、`49` 呈因果可验证路径；把 `doubled` 换 `tripled` 可使答案变为 70。首个加法的操作数大多在 remainder 中，说明即使是一个看似显式的算术步骤也可能主要在 J-space 外完成。

## 5. 用 J-lens 解释组件

任何残差空间方向（SAE decoder、attention Q/K/V/O、transcoder encoder/decoder、MLP neuron）都可经 J-lens 投影。

- 一个 SAE feature 的激活文本看似「技术散文」，J-lens 却读为 `fake/fraud/suspicious/legitimate`；抑制后，模型称黑邮件场景为人工测试的频率从 28/50 降至 10/50，提示它实际是伪造内容检测器。
- 多语言 transcoder feature 的 encoder 读出跨语言概念（如 water）及法语语境，decoder 写出 `eau` 的首 token；这直接显示其读入到写出的翻译变换。
- 一个 attention head 的 Q/K 读英国拼写与英联邦国家名，V/O 写对应国家表示，表明它可能用拼写为说话者国别消歧。

这些用途把 J-lens 定位为「数据自由的方向解释器」：它不能取代上下文激活分析和因果干预，但能快速给候选组件打上可检验的语义标签。
