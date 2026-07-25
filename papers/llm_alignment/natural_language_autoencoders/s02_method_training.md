## 3. 方法：激活 → 文本 → 激活

令目标语言模型为 $M$，希望解释其第 $l$ 层残差流中的激活 $h_l\in\mathbb{R}^{d_{\text{model}}}$。NLA 训练两个参数化模型：

- **Activation verbalizer（AV）**：$AV(z\mid h_l)$，读取激活并自回归生成自然语言解释 $z$；
- **Activation reconstructor（AR）**：$AR(z)$，仅读取解释并输出 $\hat h_l\in\mathbb{R}^{d_{\text{model}}}$。

整体损失为：

$$
\mathcal{L}=\mathbb{E}_{h_l\sim\mathcal H}\;\mathbb{E}_{z\sim AV(\cdot\mid h_l)}\left[\lVert h_l-AR(z)\rVert_2^2\right],
$$

其中 $\mathcal H$ 是把预训练式文本输入目标模型后采集到的激活分布。作者使用**方差解释率（fraction of variance explained, FVE）**报告重构质量：

$$
\operatorname{FVE}=1-
\frac{\mathcal L}
{\mathbb{E}_{h_l\sim\mathcal H}\lVert h_l-\bar h_l\rVert_2^2}.
$$

FVE 为 0 表示只预测均值激活，1 表示完美重构。所有训练激活先做单位 $L_2$ 范数归一化；因此开源实现中的归一化 MSE 本质上也衡量向量方向的一致性。

## 3.1 架构

### Activation verbalizer

AV 与 $M$ 具有相同语言模型架构。固定提示词包含“请解释此激活”之类的指令和一个特殊 activation token。输入 $h_l$ 时，将其乘以固定缩放系数，替换该 token 的 embedding；随后 AV 在温度 $T=1$ 下生成解释。这个设计不增加一个复杂的 activations-to-text adapter，而是让目标模型原有的 embedding/Transformer 路径承接注入向量。

### Activation reconstructor

AR 同样基于 $M$，但只保留前 $l$ 层。它把解释放入固定提示模板，读取末 token 在第 $l$ 层的表示，再经可学习仿射映射输出 $\hat h_l$。因此，AR 被迫把解释中的有效信息重新组织成目标残差空间的方向。

## 3.2 为什么要热启动

直接把 $M$ 复制为 AV/AR 后训练不稳定：AV 从未在预训练中遇到“把中间层激活当 token embedding”的输入，初期输出没有意义。作者因而使用文本摘要代理任务初始化：

1. 从类似预训练语料的随机截断片段中提取末 token 的 $h_l$；
2. 让 `Claude Opus 4.5` 对截断前文本生成摘要 $s$；
3. 在 $(h_l,s)$ 上监督微调 AV，在 $(s,h_l)$ 上监督微调 AR。

这个 warm-start 通常取得约 **0.3–0.4 FVE**。摘要中的“短段落 + 加粗主题标题”写作风格也会遗留到后续 NLA 解释。此步骤并未给出关于激活中真实思想的标签；它只是让两个模块先学会稳定地通过文本中介传递信息，并把 AV 保持在可读语言分布附近。

## 3.3 联合训练：AR 回归、AV 强化学习

给定初始化后的 $AV_\phi$ 和 $AR_\theta$，每一个 batch 的流程为：

1. 从 $\mathcal H$ 采样一批激活，对每个 $h_l$ 以 $T=1$ 从 AV 采样一组候选解释 $z$；
2. 固定这些解释，以 $\lVert h_l-AR_\theta(z)\rVert_2^2$ 对 AR 做一次标准 MSE 回归；
3. 将 AV 看作策略，令其得到的回报等于负重构误差，并用 RL 更新 AV。

AR 的梯度只通过 AR；AV 的梯度只通过其采样分布。二者在同 batch 同步更新，但 AR 不反传进入 AV，且 AV 在当前步中将 AR 视作固定的 reward model。对开放权重 NLA，作者使用 **GRPO**：针对同一激活采样一组说明，以各说明的重构分数形成组内相对优势。

实践中 AV 使用塑形回报

$$
r(h_l,z)=-\log\lVert h_l-AR_\theta(z)\rVert_2^2,
$$

并增加对热启动 AV 的 KL 惩罚：

$$
\beta D_{\mathrm{KL}}(AV_\phi\parallel AV_{\phi_{\mathrm{init}}}).
$$

KL 项的目的是防止纯重构优化把自然语言退化成难读、专用的通信码。作者观察到 FVE 对训练步数的对数大致线性增长，文中模型最终约为 **0.6–0.8 FVE**；但更高 FVE 仍不等于每一句解释都准确。

## 3.4 训练位置与公开实现

主文对 `Claude Haiku 3.5`、`Haiku 4.5`、`Opus 4.6` 都选择目标网络约 **2/3 深度**的层，意图在语义丰富与接近最终 logits 之间折中。层选择会实质改变可见概念：在 reward-model-sycophancy organism 上，较早的中层 NLA 能出现奖励模型偏差的上位概念，较晚层则常只描述具体怪异行为。

配套实现公开了 `Qwen2.5-7B-Instruct`、`Gemma-3-12B-IT`、`Gemma-3-27B-IT` 和 `Llama-3.3-70B-Instruct` 的 NLA checkpoint。它们的激活层分别为 20/28、32/48、41/62、53/80；模型侧车文件 `nla_meta.yaml` 保存 prompt、注入 token 与缩放参数，应由推理代码读取而非硬编码。工程实现将 AV 视为 actor、AR 视为 critic/reconstructor：较小模型用 FSDP，70B 用 Megatron；rollout 借助 SGLang 的 `input_embeds` 接口。

公开仓库给出的 `Qwen2.5-7B` 参考资源量说明了成本边界：SFT 使用 2 张 `H100 80GB`，训练至约 **75% FVE** 的 RL 配置约为 16 张 `H100`。这与每个激活需生成数百 token 的推理成本一起，限制了 NLA 目前不适合作为全 token、实时大规模监控器。
