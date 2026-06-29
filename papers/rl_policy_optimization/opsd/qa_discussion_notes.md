# OPSD 论文讨论 Q&A 笔记

> 讨论对象：*Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models* (arXiv: 2601.18734, ICML 2026)
> 讨论时间：2026-06-29

---

## Q1：请讲一下这篇论文

### A1：论文概要

**核心问题**：后训练三大路线各有短板——SFT 有 exposure bias、RLVR/GRPO 奖励稀疏且全对/全错时梯度消失、传统 on-policy 蒸馏需要外部大模型当老师。OPSD 想同时拿到四个性质：on-policy 数据、稠密 token 级反馈、利用 ground-truth $y^\star$、无需外部教师。

**核心思路**：同一个 LLM（同一套参数 $\theta$）扮演两个角色：
- 教师 $p_T(\cdot\mid x, y^\star)$：条件里塞进特权信息——参考答案 $y^\star$；
- 学生 $p_S(\cdot\mid x)$：只看到题目，与推理时一致。

训练流程：学生 on-policy 采样 $\hat{y}\sim p_S$，在 $\hat{y}$ 每个位置教师和学生各做一次前向，最小化逐 token 散度 $D(p_T\|p_S)$，**梯度只回传到学生**，教师固定为初始策略。

**关键 trick**：
1. 逐 token pointwise 散度裁剪——防止少数"风格词"（maybe, therefore…）的 KL 淹没"数学词"信号；
2. 思考模式搭配：学生 TM-off + 教师 TM-on，数学 token 上 KL 最大。

**与 STaR 的本质区别**：STaR 是序列级指示奖励 $\mathbf{1}(y=y^\star)$，采错就丢弃整条轨迹；OPSD 是逐 token 奖励 $r_n=\log p_T-\log p_S$，**即使最终答案错误每个位置仍有学习信号**。

**实验结果**：Qwen3-1.7B/4B/8B 上 OPSD 匹配或超过 GRPO，每题只需 1 次 rollout、1024 token，100 步收敛；而 GRPO 要 8×16k token。SFT 反而掉点（参考解推理链太短）。

---

## Q2：最小化逐 token 散度用的是基于 Teacher 生成概率的 KL 吗？也就是增大 student 里 teacher 输出 token 的概率？

### A2：方向对，但要分版本讲清楚

论文主版本用 **forward KL** $D_{KL}(p_T\|p_S)$（teacher 是 KL 第一项 P）。展开：

$$
D_{KL}(p_T\|p_S)=\underbrace{\sum_v p_T(v)\log p_T(v)}_{\text{与 }\theta\text{ 无关}}-\sum_v p_T(v)\log p_S(v)
$$

最小化它 $\iff$ 最大化 $\sum_v p_T(v)\log p_S(v)$（teacher 分布下 student 的对数似然/交叉熵）。

**两个版本的区别**：

| 版本 | 散度 | reference | 估计方式 |
|------|------|------|------|
| 全词表（Eq. 6/8，主版本） | forward KL $D_{KL}(p_T\|p_S)$ | teacher | 全词表精确求和 |
| 采样 token（Eq. 9） | reverse KL $D_{KL}(p_S\|p_T)$ | student | student 采样估计 |

Forward KL 是 mass-covering（零避免）：只要 $p_T(v)>0$，student 就会被推着覆盖它。你说的"增大 teacher 输出 token 的概率"最贴切对应的是采样版本——而且要强调是**有符号的**增减（$A_n>0$ 增大，$A_n<0$ 减小）。

---

## Q3：采样 token 策略梯度（Eq. 9）这个不是 forward KL 吧？

### A3：你说得对，Eq. 9 是 reverse KL

论文 `s03_methods.md` 原文就明确写了 Eq. 9 是 **"a reverse-KL signal on sampled actions"**。

**推导确认**：reverse KL（student 在前作 reference）：

$$
D_{KL}(p_S\|p_T)=\mathbb{E}_{v\sim p_S}\bigl[\log p_S(v)-\log p_T(v)\bigr]
$$

期望在 $p_S$ 下取，所以天然用 student 采样的 token $\hat{y}_n$ 估计。用 score-function trick 求梯度（advantage 做 stop-gradient）：

$$
\nabla_\theta D_{KL}(p_S\|p_T)\approx -\underbrace{(\log p_T(\hat{y}_n)-\log p_S(\hat{y}_n))}_{A_n}\,\nabla_\theta\log p_S(\hat{y}_n)
$$

最小化 reverse KL $\iff$ 最小化 $\mathcal{L}=-\sum_n A_n\log p_S(\hat{y}_n)$，正是 Eq. 9。✓

**关键区别**：
- forward KL 期望在 $p_T$ 下，要遍历词表按 teacher 权重求和——**必须用全词表**；
- reverse KL 期望在 $p_S$ 下，可以**直接用 student 采的 token** 做蒙特卡洛估计——这就是 Eq. 9 能"只看采样 token"的根本原因。

**连带点**：消融 4.3.5 比较"全词表 vs 采样 token"时，其实同时变了两个因素（forward KL vs reverse KL + 全词表 vs 采样），论文没有单独做"全词表 reverse KL vs 采样 reverse KL"的对照来隔离。

---

## Q4：Forward KL 的 loss 展开、梯度推导——每个 token 的梯度大小取决于 Pt 吗？梯度 = Pt/Ps 对吗？

### A4：推导基本对，但有两处要修正/深化

**1. 小笔误**：forward KL 展开应为 $\sum_i p_T(y_i)\log p_T(y_i)-\sum_i p_T(y_i)\log p_S(y_i)$，第一项无梯度，简化为 $-\sum_i p_T(y_i)\log p_S(y_i)$。✓

**2. 链式求导要分清"对谁求导"**：

| 求导对象 | 梯度 | 含义 |
|------|------|------|
| $\partial/\partial \log p_S(y_i)$ | $-p_T(y_i)$ | 大小只看 $p_T$ |
| $\partial/\partial p_S(y_i)$ | $-p_T(y_i)/p_S(y_i)$ | 比值，会爆炸 → 需要 clipping |
| $\partial/\partial z_j$（实际落地） | $p_S(y_j)-p_T(y_j)$ | 真正作用在 logit 上的，双向匹配 |

**关键修正**：
- "梯度大小取决于 $p_T$"和"梯度 $=p_T/p_S$"这两句用了不同的求导对象——前者是对 $\log p_S$ 求导，后者是对 $p_S$ 求导（多了 $1/p_S$ 因子）。
- 当 $p_S(y_i)\to 0$ 时 $p_T/p_S\to\infty$ 会爆炸——**这正是论文要做 pointwise clipping 的根本原因**：student 几乎不输出但 teacher 有正概率的 token 会淹没信号。

**3. 真正落地的梯度要走完 softmax**：$p_S$ 是 logits $z$ 的 softmax，耦合在一起。对 logit $z_j$ 求导：

$$
\frac{\partial \mathcal{L}}{\partial z_j}=p_S(y_j)-p_T(y_j)
$$

梯度下降更新：$z_j\leftarrow z_j+\eta\,(p_T(y_j)-p_S(y_j))$。

**4. 更准确的直觉——双向匹配而非单纯向上**：
- $p_T>p_S$：logit 上推（teacher 更喜欢 → 增强）；
- $p_T<p_S$：logit 下压（student 过度自信、teacher 不认可 → 抑制）。

净效果是把整个 $p_S$ 分布**整体推向** $p_T$ 分布，这是 forward KL "mass-covering" 的体现——每个有 $p_T>0$ 的位置都参与匹配。
