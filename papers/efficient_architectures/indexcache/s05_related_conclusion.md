## 5. Related Work

### 5.1 Efficient Attention

Reducing the quadratic cost of self-attention is a central theme for long-context LLMs and long-horizon agents. **Training-free sparse** methods introduce sparsity at inference using fixed patterns, heuristic eviction, or lightweight importance estimation. These include approaches such as streaming/attention-sink methods, sparse prefill, dynamic sparse attention, retrieval/streaming-head decomposition, and automatic sparse attention compression. Their key weakness is the possible training-inference mismatch: sparsity is imposed after training, so errors can accumulate in long-context or long-decode settings.

**Trainable sparse** methods incorporate sparsity into the training stage itself. Examples include learned gating mechanisms, end-to-end sparse pre-training, block-level mixture routing, and full-to-sparse distillation. DSA is the direct foundation for IndexCache: it distills a lightweight lightning indexer from full attention to select top-$k$ tokens for each query, reducing core attention to $O(Lk)$.

Beyond sparse attention, **hybrid architectures** reduce the number of expensive quadratic layers by interleaving them with cheaper alternatives. These alternatives include sliding-window attention, linear attention, and state-space layers. The paper places `DeepSeek-V3.2`, `GLM-5`, and other frontier-efficient architectures in this broader movement toward long-context efficiency.

IndexCache targets a remaining bottleneck in trainable sparse attention. DSA already reduces the core attention cost, but the dynamic token-selection mechanism still performs layerwise $O(L^2)$ scoring. IndexCache is therefore complementary to sparse attention itself: it does not change the sparse core attention, but amortizes or removes redundant selector computation across layers.

### 5.2 Cross-Layer Sharing

Recent work shows strong consistency across adjacent transformer layers. This property has been used to reduce computational redundancy during inference. Methods such as `TidalDecode`, `LessIsMore`, `OmniKV`, and `DELTA` reuse top-$k$ indices from periodic anchor layers for sparse decoding. `Kascade` formalizes anchor-layer selection with dynamic programming over a cross-layer similarity matrix and identifies head-aware remapping as important for preserving accuracy.

Those approaches rely on **full attention** at anchor layers to compute exact top-$k$ indices. Independently, cross-layer KV-cache sharing methods reduce memory by letting multiple layers reuse the same key-value tensors. `HySparse` unifies both directions by interleaving full-attention layers with sparse layers that inherit both top-$k$ block indices and KV caches. However, all of these methods require full attention layers as the oracle.

IndexCache differs in two main ways. First, its oracle is cheaper: it shares the output of DSA's lightweight indexer rather than full $O(L^2)$ attention scores. Second, it introduces systematic techniques for optimizing the sharing configuration: a training-free greedy search that uses global LM loss to identify the structural layout, and a training-aware multi-layer distillation objective for parameter adaptation.

Although the paper instantiates IndexCache on DSA, the principle applies more broadly to sparse attention methods whose sparse pattern is not fixed but produced by dynamic token selection. For example, block-level selection in methods such as `MoBA` and `NSA` could also benefit from cross-layer reuse if their selectors exhibit the same stability.

## 6. Conclusion

IndexCache accelerates sparse attention by exploiting cross-layer redundancy in the indexer responsible for token selection. It partitions layers into a small number of `F` layers that retain indexers and a majority of `S` layers that reuse inherited top-$k$ indices. This eliminates up to **75%** of the $O(NL^2)$ total indexer cost with only a single inference-time conditional branch and without adding persistent GPU memory.

The paper demonstrates that cross-layer sharing, previously used mainly when full attention served as the oracle, also applies naturally to sparse-attention models. On 30B DSA and 744B `GLM-5`, IndexCache keeps long-context and reasoning performance nearly unchanged while producing substantial prefill and decode speedups. Training-free greedy search is enough to identify safe patterns for frozen models, while training-aware multi-layer distillation makes even simple uniform patterns competitive by teaching retained indexers to serve multiple layers.

The broader implication is that as sparse attention becomes standard in frontier LLMs, the selector itself becomes an important optimization target. IndexCache suggests that dynamic sparse selectors need not be recomputed independently at every layer; their outputs can be treated as cross-layer reusable structure. The authors expect cross-layer index reuse to become a standard component of efficient inference pipelines for long-context LLMs.
