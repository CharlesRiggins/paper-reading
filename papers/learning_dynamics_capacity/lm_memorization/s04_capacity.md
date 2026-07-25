## 3. Model Capacity for Memorization

Unintended memorization measures the number of bits a model $\theta$ knows about one datapoint $x$. Summing it over a dataset yields total stored information. When datapoints are independent and carry no structure that can be generalized, this total directly estimates a model’s memorization capacity.

### 3.1. Defining model capacity

For a distribution $X$ and learning algorithm $L:X\to\Theta$, capacity is the largest total memorization attainable over data distributions:

$$
\operatorname{Capacity}(L)=\max_X \operatorname{mem}(X,L(X)).
$$

After capacity is reached, $\operatorname{mem}(X,L(X))$ no longer grows with dataset size. Empirically, the authors train to saturation over multiple sizes of $X$ and take the maximum measured memorization.

### 3.2. Measuring capacity with synthetic sequences

The experiments use datasets whose tokens are sampled uniformly and independently from a fixed vocabulary. With no pattern to generalize, the data’s information is known exactly and unintended memorization is approximately total memorization. The authors train GPT-2-style models [42] from scratch with 1–8 layers, hidden widths 32–512, and roughly 100K–20M parameters. Each is trained for $10^6$ steps using `Adam`, batch size 2048, and `bfloat16` on one A100 GPU; gradient accumulation is used when necessary. Unless varied, vocabulary size is $V=2048$, sequence length is $S=64$, and each configuration is evaluated across five seeds.

For sufficiently small datasets, every model with enough capacity memorizes all examples. As datasets grow, total unintended memorization rises and then reaches a sharp size-dependent plateau. The capacity estimate is the maximum over dataset sizes. Across model scales, the relationship between parameter count and capacity is smooth: the models consistently store approximately **3.5–3.6 bits per parameter**, a larger value than the approximately 2 bpp quantization estimate of Allen-Zhu and Li [1]. Since gradient descent need not find a global optimum, all measured values are lower bounds.

For the 8M-parameter model, datasets between 16,000 and 4M examples converge to $3.56$–$3.65\times10^6$ memorized bits. The two largest runs, 4M and 8M samples, reach $2.95\times10^6$ and $1.98\times10^6$ bits within the training budget and are expected to approach capacity with more epochs.

### How precision affects capacity

Increasing numeric precision from `bfloat16` to `float32` increases the storage representation substantially, but much of those added bits are not used for memorization. The aggregate estimates in Table 1 rise only from **3.51 bpp** to **3.83 bpp**, reinforcing that usable learned capacity is governed by optimization and architecture as well as raw parameter representation.
