## Appendix A. Additional Results, Proofs, and Limitations

### A.1. How reliable are the linear capacity estimates?

The approximation

$$
\operatorname{mem}(X,L(X))\approx\min\{\operatorname{capacity}(L),H(X)\}
$$

is tested by varying sequence length $S$ and vocabulary size $V$ rather than the number of examples. The experiment fixes 4096 samples, two layers, and hidden width 128, uses the earlier estimate $\alpha=3.642$, and adjusts parameter counts for the changing embedding matrices. The measured and predicted totals agree closely.

| Variable | Range | Mean error |
|---|---|---:|
| Sequence length $S$ | 4–256 | **1.7%** |
| Vocabulary size $V$ | 128–4096 | **1.8%** |

For example, at $S=4$, a 659K-parameter model memorizes $1.73\times10^5$ bits versus $1.80\times10^5$ expected (4.19% error); at $S=256$, it memorizes $2.44\times10^6$ versus $2.45\times10^6$ (0.44% error). At $V=2048$, the corresponding values are $2.39\times10^6$ and $2.36\times10^6$ bits (1.11% error).

### A.2. Additional memorization results

Text memorization also plateaus near model capacity. When dataset size grows by a factor of $N$, a capacity-limited model divides its stored sample-specific information across examples, leaving the total approximately constant. Below the smallest model’s capacity, models show similar performance. Above it, unintended memorization first rises while capacity fills and then falls as the model replaces sample-specific details with useful general patterns. The largest datasets produce the most generalization and the least per-sample unintended memorization.

### A.3. Comparison of memorized distributions

Equal-capacity Transformers trained on uniform bitstrings and text distribute their memorization differently. Random data produces tight, approximately normal train/test compression-rate distributions with little overlap. Text has lower mean loss but a broader distribution with more overlap between train and test, explaining why text membership inference is more difficult.

The paper then asks which text examples receive the most unintended memorization. In a fully deduplicated corpus, the relationship persists even without duplicated samples: documents with higher TF–IDF, hence rarer words, tend to be more memorized. The score is

$$
\operatorname{TF\!\text{-}IDF}(d;\mathcal{D})
=\frac{1}{|d|}\sum_{w\in d}\log\frac{|\mathcal{D}|}{\operatorname{tf}(w,\mathcal{D})}.
$$

For a 20M-parameter model trained beyond capacity on $2^{16}$ English sequences, the highest-TF–IDF item—a Japanese sequence—has the third-highest measured memorization and can be regurgitated from a one-token prefix. Of the twenty most memorized sequences, all but three contain non-English tokens (Japanese, Chinese, or Hebrew). Manual inspection finds rare tokens, frequently from non-English languages, in the most memorized samples.

### A.4. Scaling-law fit

The sigmoid membership law is fitted against tokens-per-parameter. While the curve is intentionally simple and does not interpolate every point perfectly, its estimates are within **1–2%** of observed values.

### A.5–A.7. Proofs

The appendix supplies proofs omitted from the main body. For Proposition 1, conditioning on the generating model $\Theta$ makes $X_i$ conditionally independent and gives

$$
I((X_1,\ldots,X_n);\hat{\Theta}\mid\Theta)
\geq \sum_{i\in[n]} I(X_i;\hat{\Theta}\mid\Theta),
$$

which proves super-additivity. The upper bound follows because conditional mutual information is bounded by the entropy of the trained model:

$$
\operatorname{mem}_U(X,\hat{\Theta},\Theta)\leq H(\hat{\Theta}).
$$

For Proposition 4, the paper invokes the relationship between algorithmic and statistical mutual information from Grünwald and Vitányi [24]. The expected algorithmic quantity differs from its statistical counterpart by terms controlled by the Kolmogorov complexity of the joint density, yielding the stated approximation guarantee.

### A.8. Limitations

The work measures memorization under specific datasets, architectures, references, and training setups. Its main contributions are a capacity measurement and a perspective on grokking and evaluation, not a claim that every result automatically transfers to other corpora, model classes, or optimization regimes.
