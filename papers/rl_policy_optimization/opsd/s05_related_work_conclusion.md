## 5. Related Work

**LLM Self-Training.** This work connects to a line of research showing that LLMs can improve by generating and exploiting their own supervision signals [2, 38, 4, 35, 30, 43, 40]. Closest in spirit is **context distillation** [29], which uses the same underlying model as both teacher and student by providing the teacher with privileged context and then performing SFT on the student using the teacher's generated outputs without context. This can be viewed as **off-policy**, where the learning signal is a discrete token sequence.

In the reasoning domain, **ReST** [8] and **STaR** [45] similarly rely on iterative self-training loops — generate rationales conditioned on hints or answers, filter by rewards or ground-truth answers, and fine-tune on successful trajectories — again yielding **hard distillation**; Mitra & Ulukus [19] extend this to soft distillation. **In-context editing** [22] does on-policy sampling from the student and shows that context-induced knowledge can be internalized via soft distillation by minimizing divergences, demonstrated in knowledge-editing settings.

**OPSD differs** from these approaches in that it performs **on-policy, soft distillation on the student's own rollouts for reasoning tasks**: the teacher's supervision is per-token distribution matching rather than generating a rationale for SFT. OPSD frames reasoning improvement as learning a conditional distribution induced jointly by the dataset's ground-truth solutions and the model's own reasoning ability. Concurrently, **SDPO** explored a similar algorithm with environment feedback as privileged information, and **SDFT** [28] explored on-policy self-distillation on continual-learning tasks.

**On-Policy Distillation methods** train a student model directly on trajectories sampled from its own policy, while a teacher model provides per-token guidance through KL-based regularization or related objectives [1, 37, 6, 18, 36, 34]. These approaches mitigate distribution shift by optimizing directly on the student's visitation distribution, but they typically rely on a **distinct and often larger teacher model**. This work instead explores whether an LLM can teach itself by conditioning on more privileged answer information and leveraging its own reasoning capability to guide a weaker version of itself. On-policy training paradigms are also widely used in robotics and deep RL, such as **DAgger** [24], where a human teacher provides corrective supervision on the states visited by the student policy.

**Improving LLM Reasoning through SFT and RL.** SFT and RL are two primary methods for improving LLM reasoning. SFT on high-quality reasoning traces has demonstrated strong performance [21, 14, 41, 49 and others]. However, prior work shows that SFT can rely on **memorization rather than robust generalization** [5]. In contrast, RL optimizes directly for outcome-based objectives and can exhibit better generalization [12]. More recent algorithms such as **GRPO** [9, 27] enable scalable RL by estimating advantages from group-level rewards without requiring an explicit critic as in PPO [26]. Building on this line of work, a growing body of research highlights the effectiveness of **RLVR** for reasoning tasks [42, 16, 44, and others].

---

## 6. Conclusion

The authors introduced **On-Policy Self-Distillation (OPSD)**, a simple yet effective framework for post-training large language models on reasoning tasks. The intuition behind OPSD is that a sufficiently capable reasoning LLM **can teach itself** when it has access to privileged information about the answer to a reasoning problem, utilizing its own rationalization ability to grade its weaker self without access to the ground truth at inference time.

Experimentally, OPSD achieves **better performance than off-policy distillation / SFT**, and performs **on par with or better than GRPO**, while exhibiting **significantly better sample efficiency** than GRPO.

---

## 7. Impact Statement

This paper presents work whose goal is to advance the field of machine learning. The method improves the efficiency of training language models for reasoning tasks, reducing computational costs compared to existing reinforcement learning approaches. The authors do not foresee specific negative societal consequences.
