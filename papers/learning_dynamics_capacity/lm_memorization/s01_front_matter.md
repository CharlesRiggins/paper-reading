## Abstract

This paper proposes a method to estimate how much a model **knows** about a datapoint and uses it to measure the capacity of modern language models. It separates memorization into two components: **unintended memorization**, the information the model retains about one specific dataset, and **generalization**, the information it contains about the underlying data-generating process.

Eliminating generalization makes it possible to compute total memorization and therefore estimate capacity. The measurements place GPT-family models at approximately **3.6 bits per parameter**. Training on datasets of increasing size shows that models memorize until this capacity is filled; then **grokking** begins, unintended memorization falls, and models increasingly generalize. The study trains hundreds of Transformers from $500\mathrm{K}$ to $1.5\mathrm{B}$ parameters and derives scaling laws linking capacity and data size to membership inference.

> **Figure 1.** On uniform random data, unintended memorization reaches an empirical capacity plateau for each GPT-family model size, at approximately 3.6 bits per parameter.
