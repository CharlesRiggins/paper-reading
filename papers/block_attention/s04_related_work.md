# 4. Related Work

## 4.1 Contemporaneous Work

- **TurboRAG** [28] proposed similar independent attention and position reordering. Though not accepted at ICLR 2025, it should be considered a pioneer.
- **DecoupledRAG** [9] uses cross-attention to inject external knowledge.
- **DeepSeek NSA** [39] and **Moonshot MoBA** [27] also adopt block attention, but their block selection operations are complementary to Block-attention's parallel context encoding and cross-prompt KV reuse — future combination could further reduce inference costs.

## 4.2 Retrieval-Augmented Generation (RAG)

RAG retrieves nearest-neighbor documents as references for LLM generation. Early techniques include concatenation, cross-attention, and distribution interpolation. Recent approaches include:
- **Copy-is-all-you-need** [21]: using the retriever as a generator
- **Self-RAG** [1]: using the retriever to decide generation content
- Various other methods for improving retrieval quality and integration

## 4.3 Parallel Context Encoding

Several works have attempted parallel document processing:

| Method | Approach | Limitation |
|--------|----------|------------|
| **SGLang** [41] | Structured programming for LLMs | Limited parallelism |
| **FiD** [15] | Fusion-in-Decoder | Quality trade-offs |
| **PCW** [32] | Parallel context windows | Efficiency trade-offs |
| **PromptCache** [11] | Modular attention reuse | Quality degradation without fine-tuning |
| **CacheBlend** [37] | Blending cached KV states | Requires selection mechanisms |

Block-attention achieves PromptCache-comparable reuse efficiency through position re-encoding and block fine-tuning, **without sacrificing quality**, while enabling seamless full/block attention switching.
