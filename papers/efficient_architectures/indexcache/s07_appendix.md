## Appendix

## Appendix A. Cross-Layer Top-$k$ Index Overlap

To validate cross-layer redundancy in DSA, the paper computes the pairwise overlap ratio between the top-$k$ indices selected by each layer's lightning indexer. For every layer pair $(i,j)$, it measures

$$
\frac{|\mathcal{T}^{(i)}\cap\mathcal{T}^{(j)}|}{k},
$$

where $k=2048$, averaged over **768** samples of length **200K** from a calibration set.

The heatmap for the 47-layer 30B DSA model shows several patterns:

- **High overlap near the diagonal.** Adjacent layers have overlap ratios of **0.7–1.0**, confirming that consecutive layers select largely the same tokens.
- **Block structure.** The heatmap reveals clusters such as layers 3–5, 6–8, 17–30, and 31–36, suggesting functional blocks where token selection is internally consistent.
- **Uneven decay.** Overlap decreases much faster across block boundaries than within blocks, indicating that transition layers shift attention focus.
- **Early-late distinction.** The bottom-left and top-right corners are dark, with overlap $\leq 0.4$, showing that early and late layers attend to substantially different token subsets.

### Greedy-searched blocks vs. overlap clusters

Comparing the greedy-searched sharing blocks with visually obvious overlap clusters reveals an informative mismatch. The greedy search places some `F` layers near cluster boundaries, but the partitions do not fully coincide. The reason is that overlap is an aggregate metric: it counts how many tokens are shared, not which tokens differ.

In the training-free setting, all weights are frozen, so even a small set of missing critical tokens can perturb a layer's hidden state and cascade through downstream layers. Early layers are especially vulnerable because their perturbations propagate over the longest path. This supports the paper's negative result with similarity-based search: local metrics such as cosine similarity or top-$k$ overlap do not reliably identify the best sharing pattern, while end-to-end LM loss does.

## Appendix B. Searched Patterns

The paper reports searched `F`/`S` patterns for the 30B `GLM-4.7-Flash` DSA model:

| Retention | Pattern |
|---|---|
| Keep 1/2 `F`s | `FSFSFSSSSFSFFFFSFFSSFFSFFFSSFFSSFSSSSFSFFFSFSSF` |
| Keep 1/4 `F`s | `FSFSFSSSSFSSSFSSFFSSFSSFSSSSFSSSFSSSSFSSSSSSSSS` |
| Keep 1/8 `F`s | `FSSSFSSSSSSSSFSSSFSSSSSFSSSSFSSSSSSSSFSSSSSSSSS` |

For `GLM-5`, the searched patterns are:

| Retention | Pattern |
|---|---|
| Keep 1/2 `F`s | `FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS` |
| Keep 1/4 `F`s | `FFSFSSSFSSFSFSSSSSSSFSSSSSSFSSSFSFSSSSFFFFFSSSFFSSSFSSSSSSSSFSSSFSSSSSSFSFSSSS` |

These patterns are central to the training-free results. They show that safe indexer removal is not uniformly spaced; the search keeps indexers around layers that are end-to-end important for preserving LM loss and downstream quality.

## Appendix C. Similarity-Based Pattern Search

The paper includes a negative result for a seemingly natural alternative: choose the sharing pattern by directly measuring how similar attention outputs are when an indexer is reused across layers. This approach is cheaper than greedy LM-loss search and seems aligned with the overlap analysis, but it fails to predict downstream quality.

### Constructing the similarity matrix

Given a DSA model with $N$ layers, the method performs $N$ single forward passes over a calibration set. For each pair $(i,j)$ with $i>j$, it computes cosine similarity between:

1. the core attention output at layer $i$ when using layer $i$'s own indexer, i.e. the original model; and
2. the core attention output at layer $i$ when reusing layer $j$'s indexer, as though layer $i$ were an `S` layer sharing from layer $j$.

This produces an $N\times N$ lower-triangular similarity matrix $\mathbf{S}$, where $S_{i,j}$ measures how well layer $j$'s indexer can serve as a proxy for layer $i$'s own indexer.

### Dynamic programming formulation

Given $\mathbf{S}$ and a target number of `F` layers $M$, the method seeks

$$
\mathbf{c}^* = \arg\max_{\mathbf{c}: |\{i:c_i=\texttt{F}\}|=M}
\sum_{\ell:c_\ell=\texttt{S}} S_{\ell,\mathrm{src}(\ell)},
$$

where $\mathrm{src}(\ell)$ is the most recent preceding `F` layer:

$$
\mathrm{src}(\ell)=\max\{j<\ell:c_j=\texttt{F}\}.
$$

The paper solves this exactly with dynamic programming. Let $dp[i][k]$ be the maximum cumulative similarity achievable for layers $1,\ldots,i$ using exactly $k$ Full layers, with layer $i$ itself being Full. The transition is

$$
dp[i][k] = \max_{j<i,\ c_j=\texttt{F}}
\left\{dp[j][k-1] + \sum_{m=j+1}^{i-1} S_{m,j}\right\}.
$$

Backtracking recovers the similarity-optimal pattern.

### Result: similarity-optimal patterns are not enough

The downstream results show that similarity-based patterns perform comparably to uniform interleaving and far worse than greedy LM-loss search.

| Config | Avg | MRCR v2 | GraphWalks | RULER |
|---|---:|---:|---:|---:|
| Original DSA | **54.0** | **24.5** | **49.6** | **87.9** |
| 1/2 Unif. IndexCache | 50.7 | 22.0 | 46.6 | 83.6 |
| + Searched pattern | 49.8 | 22.9 | 43.5 | 82.9 |

Despite optimizing cross-layer similarity, the resulting patterns do not meaningfully improve over the naive uniform baseline. In contrast, greedy LM-loss search substantially outperforms both, especially at aggressive retention ratios.

### Why similarity fails as a proxy

The issue is that per-layer output similarity is local. It measures whether one layer's attention output is preserved in isolation, but not whether the perturbation changes downstream computation. Two layers may have nearly identical attention outputs, with $S_{i,j}\approx 1$, while the reused index misses a few tokens that become critical in later reasoning steps. These errors accumulate through the model and can degrade final quality.

Greedy loss-based search avoids this by optimizing a global metric: LM loss. It captures the end-to-end effect of each sharing decision on the model's output distribution and can identify critical layers whose indexers must be retained to avoid cascading errors.

## Appendix D. Evaluation Setup

All benchmarks use temperature **1.0**, top-$p=0.95$, and top-$k=40$. Long-context tasks use a total context window of **200K** tokens with **32K** reserved for output, giving an effective input budget of **168K** tokens. General and reasoning tasks allow maximum output length **64K**.

For `MRCR v2`, the paper reports the average score across 2-, 4-, and 8-needle settings. For `GraphWalks`, it reports the average over Parent-type and BFS-type problems. For `RULER`, scores cover instances from 4K to 128K context length. For `MRCR v2` and `GraphWalks`, only instances that fit within the 168K-token input budget are included. For `LongBench v2`, `RULER`, and `AA-LCR`, all instances are included, with middle truncation applied when inputs exceed 168K tokens.
