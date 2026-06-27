## 2. Preliminary

### 2.1 DeepSeek Sparse Attention

**DeepSeek Sparse Attention** (**DSA**) decomposes each attention layer into two stages: **selection** and **computation**. In the selection stage, a lightweight lightning indexer scores all preceding tokens against the current query using a multi-head ReLU-gated dot product. It then selects the top-$k$ highest-scoring positions. In the computation stage, the main attention is computed only over that sparse subset.

This changes the core attention complexity from $O(L^2)$ to $O(Lk)$ with $k=2048 \ll L$, where $L$ is sequence length. The indexer itself is designed to be cheap: it uses few heads, low-rank projections, and FP8 arithmetic, making it about an order of magnitude cheaper per FLOP than the main **Multi-head Latent Attention** (**MLA**) computation.

DSA is introduced into an MLA model by two-stage continued pre-training. First, a short **dense warm-up** phase trains only the indexer through KL-divergence distillation against the aggregated full-attention distribution at each layer, while all other parameters are frozen. Then, a longer **sparse training** phase activates top-$k$ selection and jointly optimizes the full model, with the indexer receiving distillation gradients on a detached computational graph.

Despite these efficiency gains, the indexer still has $O(L^2)$ complexity. It must independently score all preceding tokens at every layer to determine its own top-$k$ set. Across a model with $N$ layers, the total indexer cost is therefore $O(NL^2)$, which becomes a significant fraction of attention cost at long contexts. This motivates the question: are all $N$ per-layer indexer computations necessary, or can cross-layer redundancy be exploited?

### 2.2 Cross-Layer Stability of Token Selection

The key empirical premise is that the set of important tokens is remarkably stable across consecutive transformer layers. Prior work such as `Kascade` and `HySparse` observes that adjacent layers share most of their top-$k$ attention mass and exploits this by designating anchor layers that compute full attention, while intermediate layers reuse the anchor's top-$k$ indices.

Those approaches depend on **full attention** as the oracle for identifying important tokens. In DSA, however, full attention has been removed from inference and replaced by the lightweight indexer. This creates a distinct problem: the paper asks whether the indexer's output itself exhibits cross-layer stability. If so, DSA can reuse the indexer's selected tokens without requiring any full-attention oracle.

The answer is positive. Appendix A computes pairwise top-$k$ overlap between the lightning-indexer outputs of different layers. Adjacent DSA layers share **70–100%** of selected tokens, and the overlap heatmap shows clear layer clusters. This establishes that cross-layer sharing is not limited to dense full-attention scores: the sparse indexer's learned token-selection policy is also redundant across nearby layers.

The rest of the paper builds on this observation. The first challenge is to determine how aggressively indices can be reused before quality degrades. The second is to decide which layers should retain indexers. The third is to adapt the model, when training is available, so that retained indexers produce top-$k$ sets useful not only for their own layers but also for the subsequent Shared layers that inherit their indices.
