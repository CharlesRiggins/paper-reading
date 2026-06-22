# Guiding MLLMs in When and How to Think via Dual-Reward RL Tuning

arXiv: — | Anonymous (—, 2026)  
Code: —

## Source

- Parsed text: `/Users/charlesriggins/Desktop/paper-reading/papers2review/_parsed_text/14654_Guiding_MLLMs_in_When_an.md`
- Original PDF: `/Users/charlesriggins/Desktop/paper-reading/papers2review/14654_Guiding_MLLMs_in_When_an.pdf`

## Files

- `s01_abstract_introduction.md` — Contains the title, abstract, Figure 1 summary, and the full introduction storyline. It explains why outcome-only RL supervision can reward “lucky” answers with flawed reasoning and why uniform long-CoT behavior causes overthinking on simple tasks and underthinking on hard ones. It also states the paper’s main proposal, SAIL-RL, as a dual-reward post-training framework built on SAIL-VL2, and summarizes the reported gains in reasoning, multimodal understanding, and hallucination reduction.
- `s02_related_work.md` — Organizes the related work around cognitive regimes in MLLMs and multimodal reinforcement learning. It contrasts System-1 direct perception models with System-2 deliberative models such as Gemini-2.5-Pro and Kimi-VL-Thinking, then positions SAIL-RL as jointly learning when to reason and how to reason rather than treating efficiency and reasoning correctness separately.
- `s03_sail_rl_method.md` — Transcribes the SAIL-RL method section with the Thinking Reward, Judging Reward, Cascading Reward System, and two-stage post-training strategy. It preserves the three thinking dimensions—logical coherence, factual grounding, and answer consistency—the binary judging reward, the multiplicative total reward equation with $\alpha=0.9$, and the LongCoT SFT objective. It also records the Judge-Think-Answer format, the use of Gemini as reward judger, and the DAPO-based RL stage on verifiable tasks.
- `s04_experiments_analysis.md` — Covers experimental setup, benchmark performance, single-dimension baselines, dual-reward analysis, reward mechanism ablations, robustness to reward model choice, and generalization across architectures. It retains the key numeric tables for OpenCompass reasoning, general multimodal understanding, SophiaVL-R1/R-4B comparisons, thinking and judging reward ablations, efficiency–accuracy trade-offs, reward-model ablations, and Qwen3-VL scale transfer. It also records trigger rates across benchmarks and the main findings from the training-dynamics and reward-mechanism figures.
- `s05_conclusion_limitations.md` — Captures the conclusion and limitations in a compact standalone file. It states that SAIL-RL replaces outcome-only supervision with dual rewards for logical rigor and adaptive reasoning, while acknowledging dependence on intermediate reward quality and increased memory/training variance from explicit reasoning traces.
- `s06_references.md` — Provides the reference list as a structured Markdown table with authors, title, venue/source, and year. The references cover MLLM backbones, multimodal benchmarks, process rewards, RL training systems, visual reasoning datasets, and related adaptive-thinking or thinking-reward methods.
- `s07_appendix_checklist.md` — Merges appendix materials and the checklist into one structured file. It includes data curation for 400K LongCoT and 70K RL samples, training hyperparameters on 64 NVIDIA A100 GPUs, qualitative case studies for thinking and judging rewards, reward-prompt schemas for judging/logic/consistency/hallucination checks, and a concise checklist answer table with boilerplate removed.
