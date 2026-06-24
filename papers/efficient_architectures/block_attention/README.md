# Block-Attention for Efficient Prefilling
arXiv: 2409.15355 | Dongyang Ma*, Yan Wang*, Tian Lan† et al. (Tencent, 2024)
Code: https://github.com/TemporaryLoRA/Block-attention

## Files

- `s01_introduction.md` — 介绍 RAG 场景中因检索多篇文档导致的首 token 延迟（TTFT）问题，提出 Block-attention 的核心思想：将检索文档划分为独立块，分别计算 KV 状态并缓存复用，仅用户查询块可关注所有前序块。直接切换会致准确率骤降（66.1%→49.9%），但通过块微调约 500-1000 步即可完全恢复。

- `s02_block_attention.md` — 详细阐述 Block-attention 机制：块分割（语义独立的文档自然成块）、位置重编码（利用 RoPE 旋转性质将 KV 状态映射到新位置）、块微调（将因果注意力掩码替换为块注意力掩码）以及推理流程。论证了只需重算变化块即可获得与全重编码等效的结果。

- `s03_experiments.md` — 在 11 个基准上的全面实验。基模型为 `Llama-3.1-Tulu-3-8B-SFT`，对比了 PromptCache、Superposition 等并行上下文编码方法。关键结论：块微调后性能与全注意力相当（RAG 差异 ≤1%），全/块注意力可无缝切换，位置重编码至关重要。效率方面，32K 总长时 TTFT 从全注意力降低 98.7%（仅 45ms），FLOPs-TFT 降低 99.8%。

- `s04_related_work.md` — 讨论了同期工作（TurboRAG、DecoupledRAG、DeepSeek NSA、Moonshot MoBA）、RAG 检索增强生成的发展脉络、以及并行上下文编码方法（SGLang、FiD、PCW、PromptCache、CacheBlend）的优劣势对比。Block-attention 在不牺牲质量的前提下达到了与 PromptCache 相当的复用效率。

- `s05_conclusion.md` — 总结 Block-attention 通过独立块 KV 状态计算与缓存复用，大幅提升 RAG 推理效率，文档越多/检索越频繁效果越显著。包含致谢。

- `s06_references.md` — 完整的 43 条参考文献列表，涵盖 RAG、并行编码、游戏 AI 等领域。

- `s07_appendix_game_ai.md` — 附录：Block-attention 在游戏 AI 中的应用。利用高帧间重复率（>99.5%）和结构化 JSON 将游戏状态划分为 300+ 块，实现 TTFT 从 2800ms 降至 100ms（28× 提升），延迟从 3000ms 降至 <300ms（10× 提升），准确率无损，突破实时性瓶颈。
