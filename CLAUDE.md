# Paper Reading Project

This project is intended both for reading papers to deepen knowledge and for preparing post-training interviews. These two goals run in parallel and reinforce each other.

## Papers index

Papers are organized by topic under `papers/<category>/`. The categories double as interview-prep tracks.

### RL & Policy Optimization

| Folder | Title | ID |
|--------|-------|----|
| `papers/rl_policy_optimization/dapo/` | "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" — Yu et al. (ByteDance Seed / Tsinghua AIR / HKU, 2025) | 2503.14476 |
| `papers/rl_policy_optimization/gigpo/` | "Group-in-Group Policy Optimization for LLM Agent Training" — Feng et al. (NTU/Skywork AI, 2025) | 2505.10978 |
| `papers/rl_policy_optimization/sail_rl/` | "Guiding MLLMs in When and How to Think via Dual-Reward RL Tuning" — Anonymous (NeurIPS 2026 submission) | OpenReview: 14654 |

### Parameter-Efficient Fine-Tuning

| Folder | Title | ID |
|--------|-------|----|
| `papers/peft_lora/lora_learns_less/` | "LoRA Learns Less and Forgets Less" — Biderman et al. (Columbia & Databricks, 2024) | 2405.09673 |
| `papers/peft_lora/lora_without_regret/` | "LoRA Without Regret" — Schulman et al. (Thinking Machines Lab, 2025) | Blog Post, DOI: 10.64434/tml.20250929 |

### Model Editing & Memory

| Folder | Title | ID |
|--------|-------|----|
| `papers/model_editing_memory/memit/` | "Mass-Editing Memory in a Transformer" — Meng et al. (MIT CSAIL / Northeastern University / Technion, 2022) | 2210.07229 |

### Learning Dynamics & Capacity

| Folder | Title | ID |
|--------|-------|----|
| `papers/learning_dynamics_capacity/learning_dynamics_ft/` | "Learning Dynamics of LLM Finetuning" — Ren et al. (University of British Columbia & Amii, 2024) | 2407.10490 |
| `papers/learning_dynamics_capacity/capacity_interference/` | "Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention" — Huang et al. (Stanford, Kempner, MIT, Anthropic, 2026) | 2605.29548 |
| `papers/learning_dynamics_capacity/neural_scaling_laws/` | "Explaining Neural Scaling Laws" — Bahri et al. (Google DeepMind / Johns Hopkins, 2021) | 2102.06701 |
| `papers/learning_dynamics_capacity/parameter_symmetries/` | "The Empirical Impact of Neural Parameter Symmetries, or Lack Thereof" — Lim et al. (MIT CSAIL / UC Berkeley / Northeastern University / Technion, NVIDIA / TU Munich, MIT, 2024) | 2405.20231 |
| `papers/learning_dynamics_capacity/spikes_sinks/` | "The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks" — Sun et al. (NYU / Meta FAIR, 2026) | 2603.05498 |
| `papers/learning_dynamics_capacity/lm_memorization/` | "How much do language models memorize?" — Morris et al. (FAIR at Meta / Google DeepMind / Cornell University / NVIDIA, 2025) | 2505.24832 |

### Efficient Architectures & Frontier Systems

| Folder | Title | ID |
|--------|-------|----|
| `papers/efficient_architectures/block_attention/` | "Block-Attention for Efficient Prefilling" — Ma et al. (Tencent, 2024) | 2409.15355 |
| `papers/efficient_architectures/indexcache/` | "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse" — Bai et al. (Tsinghua University & Z.ai, 2026) | 2603.12201 |
| `papers/efficient_architectures/deepseek_v3_2/` | "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models" — DeepSeek-AI (2025) | 2512.02556 |
| `papers/efficient_architectures/deepseek_v4/` | "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" — DeepSeek-AI (2026) | HuggingFace |

### LLM Alignment & Safety

| Folder | Title | ID |
|--------|-------|----|
| `papers/llm_alignment/adaptive_jailbreaks/` | "Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks" — Andriushchenko et al. (EPFL, 2024) | 2404.02151 |
| `papers/llm_alignment/fab_dormant_attacks/` | "Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning" — Gloaguen et al. (ETH Zurich, 2025) | 2505.16567 |
| `papers/llm_alignment/ft_compromises_safety/` | "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!" — Qi et al. (Princeton / Virginia Tech / IBM Research / Stanford, 2023) | 2310.03693 |
| `papers/llm_alignment/lima/` | "LIMA: Less Is More for Alignment" — Zhou et al. (Meta AI / Carnegie Mellon University / USC / Tel Aviv University, 2023) | 2305.11206 |
| `papers/llm_alignment/steg_finetuning/` | "Invisible Safety Threat: Malicious Finetuning for LLM via Steganography" — Wan et al. (National University of Singapore, 2025) | 2603.08104 |
| `papers/llm_alignment/hallucination_probes/` | "Real-Time Detection of Hallucinated Entities in Long-Form Generation" — Obeso et al. (ETH Zürich & MATS, 2025) | 2509.03531 |
| `papers/llm_alignment/gradient_routed_capability_removal/` | "Gradient-Routed Capability Removal" — SGTM / GRAM (Anthropic collaborators & AE Studio, 2025–2026) | 2512.05648; Anthropic Alignment post |
| `papers/llm_alignment/natural_language_autoencoders/` | "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations" — Fraser-Taliente et al. (Anthropic, 2026) | Transformer Circuits Thread |
