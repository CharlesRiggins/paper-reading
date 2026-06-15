## 2. DeepSeek-V3.2 Architecture

DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp. Compared with DeepSeek-V3.1-Terminus, the only architectural modification is the introduction of **DeepSeek Sparse Attention (DSA)** through continued training.

### 2.1 DeepSeek Sparse Attention

#### Prototype of DSA

The prototype of DSA consists of two components: a **lightning indexer** and a **fine-grained token selection mechanism**.

**Lightning Indexer**: Computes the index score $I_{t,s}$ between the query token $\mathbf{h}_t \in \mathbb{R}^{d}$ and a preceding token $\mathbf{h}_s \in \mathbb{R}^{d}$:

$$I_{t,s} = \sum_{j=1}^{H^{I}} w_{t,j}^{I} \cdot \text{ReLU}\left(\mathbf{q}^{I}_{t,j} \cdot \mathbf{k}^{I}_{s}\right)$$

where $H^{I}$ denotes the number of indexer heads; $\mathbf{q}^{I}_{t,j} \in \mathbb{R}^{d^{I}}$ and $w_{t,j}^{I} \in \mathbb{R}$ are derived from the query token $\mathbf{h}_t$; and $\mathbf{k}^{I}_{s} \in \mathbb{R}^{d^{I}}$ is derived from the preceding token $\mathbf{h}_s$. **ReLU** is chosen as the activation function for throughput considerations. Given the small number of indexer heads and FP8 implementation, its computational efficiency is remarkable.

**Fine-grained Token Selection**: Given index scores $\{I_{t,s}\}$ for each query token, retrieves only the key-value entries $\{\mathbf{c}_s\}$ corresponding to the **top-k** index scores:

$$\mathbf{u}_t = \text{Attn}\left(\mathbf{h}_t, \{\mathbf{c}_s \ | \ I_{t,s} \in \text{Top-k}(I_{t,:})\}\right)$$

#### Instantiate DSA Under MLA

For continued training from DeepSeek-V3.1-Terminus, DSA is instantiated based on **MLA** (Multi-head Latent Attention). At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency, so DSA is implemented based on the **MQA mode** of MLA, where each latent vector is shared across all query heads of the query token.

### 2.1.1 Continued Pre-Training

Starting from a base checkpoint of DeepSeek-V3.1-Terminus (context length already extended to 128K), continued pre-training followed by post-training creates DeepSeek-V3.2. Training data distribution is totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus.

#### Dense Warm-up Stage

A short warm-up stage to initialize the lightning indexer, keeping dense attention and freezing all model parameters except the lightning indexer. To align indexer outputs with the main attention distribution, for the $t$-th query token, main attention scores are aggregated by summing across all attention heads, then L1-normalized along the sequence dimension to produce a target distribution $p_{t,:} \in \mathbb{R}^{t}$. A **KL-divergence loss** serves as the training objective:

$$\mathcal{L}^{I} = \sum_{t} \mathbb{D}_{\mathrm{KL}}\left(p_{t,:} \ \middle\| \ \text{Softmax}(I_{t,:})\right)$$

Training configuration:
- Learning rate: $10^{-3}$
- **1000 steps**, each with 16 sequences of 128K tokens
- Total: **2.1B tokens**

#### Sparse Training Stage

Following indexer warm-up, the fine-grained token selection mechanism is introduced, and all model parameters are optimized to adapt to the sparse attention pattern. The indexer continues to be aligned with the main attention distribution, but considering only the selected token set $\mathcal{S}_t = \{s \ | \ I_{t,s} \in \text{Top-k}(I_{t,:})\}$:

$$\mathcal{L}^{I} = \sum_{t} \mathbb{D}_{\mathrm{KL}}\left(p_{t,\mathcal{S}_t} \ \middle\| \ \text{Softmax}(I_{t,\mathcal{S}_t})\right)$$

The indexer input is **detached from the computational graph** for separate optimization — the indexer's training signal comes only from $\mathcal{L}^{I}$, while the main model is optimized only via the language modeling loss.

Training configuration:
- Learning rate: $7.3 \times 10^{-6}$
- **2048 key-value tokens** selected per query token (top-k)
- **15000 steps**, each with 480 sequences of 128K tokens
- Total: **943.7B tokens**

### 2.2 Parity Evaluation

**Standard Benchmarks**: In September 2025, DeepSeek-V3.2-Exp was evaluated on a suite of benchmarks and compared with DeepSeek-V3.1-Terminus, showing similar performance. **No substantial performance degradation** was observed on both short- and long-context tasks despite significant improvements in computational efficiency.

**Human Preference**: Using ChatbotArena as an indirect evaluation framework, both DeepSeek-V3.1-Terminus and DeepSeek-V3.2-Exp share an identical post-training strategy, and their Elo scores are closely matched. The new base model achieves performance on par with the previous iteration despite sparse attention.

**Long Context Eval**: Independent evaluations using previously unseen test sets:
- **AA-LCR**: DeepSeek-V3.2-Exp scores **4 points higher** than DeepSeek-V3.1-Terminus in reasoning mode.
- **Fiction.liveBench**: DeepSeek-V3.2-Exp consistently outperforms DeepSeek-V3.1-Terminus across multiple metrics.

### 2.3 Inference Costs

DSA reduces the core attention complexity from $\mathcal{O}(L^{2})$ to $\mathcal{O}(Lk)$, where $k \ll L$ is the number of selected tokens. Although the lightning indexer still has complexity $\mathcal{O}(L^{2})$, it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus. Combined with optimized implementation, DSA achieves significant end-to-end **speedup in long-context scenarios**.

For short-sequence prefilling, a specially implemented **masked MHA mode** simulates DSA, achieving higher efficiency under short-context conditions.
