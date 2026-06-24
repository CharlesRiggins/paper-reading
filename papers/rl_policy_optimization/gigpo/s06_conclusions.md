## 6. Conclusions and Limitations

The authors proposed **GiGPO**, a novel group-based RL algorithm to tackle the credit assignment challenge in long-horizon LLM agent training. GiGPO introduces a hierarchical advantage estimation that enables fine-grained per-step credit assignment while retaining the efficiency and stability of group-based RL. By retroactively grouping steps that share the same state across trajectories, it achieves this without incurring additional LLM rollout or GPU memory overhead.

Empirical evaluations across complex agentic environments (ALFWorld and WebShop) and search-augmented QA tasks demonstrate that GiGPO significantly outperforms both prompt-based agents and prior RL methods.

### Limitations

A potential limitation of GiGPO is its reliance on state matching for anchor group construction. In highly complex environments, identical states may be hard to detect due to noise or subtle differences. Despite this, GiGPO still retains a strong performance lower bound: in the extreme case where no states are repeated across trajectories (i.e., $A^S = 0$), it naturally degrades to GRPO, preserving GRPO's effectiveness and stability in credit assignment. Although this issue is partly mitigated by incorporating similarity-based grouping, exploring more robust state-matching strategies — such as embedding-based representations or domain-specific structural equivalence — remains an important direction.

### Acknowledgements

This research is supported by the Ministry of Education, Singapore, under its Academic Research Fund Tier 1 (RG18/24).
