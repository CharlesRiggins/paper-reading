# Appendix A. Block-attention in Game AI

In game scenarios, LLM inputs are typically JSON representations of the current game state, ranging from thousands to hundreds of thousands of tokens, with short outputs. Two key properties make Block-attention highly suitable:

1. **High inter-frame repetition rate (>99.5%)**: Game states change minimally between frames
2. **Structured JSON**: Allows clean partitioning into independent blocks

In an undisclosed game, the game state was partitioned into **300+ blocks**. Results:

| Metric | Full Attention | Block-attention | Improvement |
|--------|---------------|-----------------|-------------|
| Average TTFT | 2800 ms | **100 ms** | 28× reduction |
| Query latency | 3000 ms | **<300 ms** | 10× reduction |
| Accuracy | baseline | **same as baseline** | No loss |

Block-attention achieved the same accuracy as full attention while reducing inference latency by over an order of magnitude, breaking through the real-time bottleneck that previously prevented LLM deployment in real-time games. The authors strongly recommend this section for game AI researchers, suggesting that Block-attention can help developers easily build more advanced AI-driven game experiences.
