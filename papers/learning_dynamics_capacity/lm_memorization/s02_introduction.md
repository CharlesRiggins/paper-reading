## 1. Introduction

Modern language models are trained on rapidly growing corpora while parameter counts remain in the billions. As one example, an 8-billion-parameter model occupies roughly $32\,$GB on disk but was trained on 15 trillion tokens, roughly $7\,$TB [19]. This mismatch motivates a basic question: what information from its training set can such a model retain?

The paper defines memorization as the degree to which a model compresses a particular datapoint: an input is memorized when access to the model permits a shorter encoding. This draws on Kolmogorov information theory and Shannon information theory, but is made practical by estimating information with model likelihoods. Crucially, the quantity is decomposed into **unintended memorization**, information specific to the realized training data, and **generalization** (intended memorization), information about the data-generating process. Unlike conditional-mutual-information formulations [8], the proposed algorithmic formulation supports this separation at the instance level.

The authors first remove generalization altogether by training models on independently sampled uniform bitstrings. Since every datapoint has known information content and no reusable structure, summing unintended memorization precisely measures capacity. GPT-style Transformers store about **3.5–4 bits per parameter**, with the exact value depending on architecture and numeric precision.

They then repeat the analysis on real text. Models initially accumulate sample-level information, but after their capacity is reached they trade unintended memorization for reusable patterns. The work identifies this capacity crossing as the point at which **double descent** begins: the dataset becomes too large to fit as specifics, and evaluation loss starts improving through generalization.

Finally, controlled train/test splits and perfect deduplication permit a capacity-based membership-inference law. Larger models can memorize more samples, whereas larger datasets make the average training point increasingly difficult to identify. Extrapolating the fitted law suggests that modern models are generally trained on too much data for reliable average-case membership inference.

> **Figure 3.** For synthetic bitstrings, double descent begins when data size exceeds model capacity and extra unintended memorization no longer reduces loss.

> **Figure 5.** A 6.86M-parameter GPT-style model has an estimated capacity of 23.9 MB; its memorized bits grow during training until saturation.
