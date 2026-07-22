# 方法：Jacobian lens 与 J-space

## 1. J-lens 的定义

给定第 $\ell$ 层、第 $t$ 个位置残差流 $h_{\ell,t}$，微小扰动会影响当前位置及其后位置的最终残差流。作者对大量预训练式 prompts、源位置 $t$ 与未来位置 $t'\ge t$ 求平均：

\[
J_\ell=\mathbb E_{t,t'\ge t,\,p}\left[\frac{\partial h_{\mathrm{final},t'}}{\partial h_{\ell,t}}\right].
\]

$J_\ell\in\mathbb R^{d_{model}\times d_{model}}$ 是每层一个固定矩阵。对激活读出时：

\[
\operatorname{lens}(h_\ell)=\operatorname{softmax}\left(W_U\,\operatorname{norm}(J_\ell h_\ell)\right).
\]

其中 $W_U$ 是模型原有 unembedding。$W_UJ_\ell$ 的每一行就是该层某个词 token 对应的 **J-lens 向量** $v_t$：沿该方向的激活在平均上下文中会提升这个词作为当前或未来输出的倾向。

默认设置在 1,000 条、每条 128 token 的预训练式序列上估计；附录表明 10 条序列已能超过 logit/tuned lens，更多样本带来温和改善。论文通常使用倒数第二层作为 Jacobian 目标以减少最终校准层的噪声。

## 2. J-space 与稀疏分解

词表大小大于残差维度，所以 J-lens 向量过完备，线性上可张成整个残差空间；这使「所有线性组合」没有区分力。作者以梯度追踪求激活的稀疏非负近似：

\[
h\approx\sum_{i\in S,\,|S|\le k}a_i v_i,\qquad a_i\ge0.
\]

- 典型取 $k\le25$。
- 该近似为 **J-space component**；剩余为 **non-J-space component**。
- 因为字典相关且过完备，分解并不唯一，算法是实用近似，而非唯一的语义分解。

几何上，固定 $k$ 的 J-space 是所有由 $k$ 个 J-lens 向量非负张成的多面锥的并集，而非线性子空间。

## 3. 如何读、如何写

### 读

1. **排序读数**：取 top tokens，观察一个位置/层中的可言说概念。
2. **单词 probe**：看 $\langle v_t,h\rangle$ 或相应分数，追踪指定概念。
3. **稀疏字典**：需获得较少、冗余更低的概念清单时使用。

作者强调：top tokens 应被当作「一袋主题」而非一条句子；单层单点读数易有噪声，应看跨层/邻近位置的持续模式。

### 写 / 因果干预

- **steering / ablation**：$h\leftarrow h+\alpha v_t$；负 $\alpha$ 或投影归零即抑制概念。
- **坐标交换**：对源词 $s$、目标词 $t$，令 $V=[v_s\;v_t]$、$c=V^\dagger h$，交换 $c$ 的两维，再写回 $h+V(\sigma(c)-c)$。这样保留正交于二者张成空间的部分。
- **clamp**：在随后各层/位置把指定 J 坐标锁定到干净前向值，用来检验其他扰动是否通过 J-space 间接起效。

## 4. 与相邻方法的比较

| 方法 | 核心映射 | 优点 | 论文指出的局限 |
|---|---|---|---|
| logit lens | $W_U h_\ell$ | 极便宜、末层实用 | 早层几何失配，易读出噪声。 |
| tuned lens | 训练每层线性预测器拟合最终输出 | 最擅长拟合 next-token 分布 | 相关性目标会提前预测最终答案，错过中间变量。 |
| J-lens | 平均 Jacobian + 原 unembedding | 相对便宜、因果动机强、可干预 | 词表/单 token 限制；平均化会丢失上下文特异性。 |
| 线性 probe/SAE/NLA | 监督概念、稀疏字典或自由文本解码 | 可表达更丰富内容 | probe 是相关性；SAE/NLA 更昂贵，NLA 还可能产生未扎根于激活的叙述。 |

在六类已知中间变量任务（两跳、多语、运算顺序、押韵、拼写纠错、情境联想）上，J-lens 的中间概念恢复优于另两种 lens；消融/交换也引起更大的输出影响。反过来，tuned lens 的 next-token 拟合最好——这正说明「预测输出」与「揭示产生输出的中间计算」并非同一目标。
