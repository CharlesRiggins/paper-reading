## 6. Related Work

### Language models and compression

Shannon’s source-coding theorem [46] formalized the duality between prediction and compression, and Shannon’s later work [47] observed that better models of English use fewer bits to encode it. Subsequent work develops the connection between Shannon information and Kolmogorov complexity [24, 25]. Delétang et al. [17] study Transformer language models as compressors; this paper uses compression to quantify memorization.

### Language-model capacity

Classical analyses characterize capacity in classifiers and neural networks [14, 22, 3]. Empirical and theoretical work has studied fact-storage scaling in language models [43, 27, 1] and Transformer memorization capacity [55, 15, 35, 29]. The present work measures an information quantity directly on trained language models rather than inferring capacity solely from architecture or parameter quantization.

### Information regularization in learning theory

Information-based learning theory links a learner’s information about its training data to generalization and stability [4, 26, 50]. The observed capacity crossing is also related to double descent [5, 38] and to evidence that deep networks can fit random labels [60, 51].

### Alternative definitions of memorization

Prior extraction definitions call a string memorized when a model can reproduce it from a prefix or adversarial prompt [9, 37, 39]. Prompt-length restrictions [37, 45] prevent prompts from simply including the string, but extraction can still conflate learned regularities with sample-specific storage: “repeat cat 1000 times” can elicit a sequence without showing that the exact sequence was memorized. Likelihood and perplexity provide a compression-like signal [9], while counterfactual memorization measures the effect of removing one training point [61]. A concurrent theoretical treatment also uses Kolmogorov complexity [12].

The authors argue that these ideas are all related to compression but typically fail to separate generalization from unintended memorization at the sample level.

---

## 7. Conclusion

The paper introduces a definition that estimates the exact number of bits a model knows about a dataset and separates dataset-specific storage from generalizable knowledge. It uses this quantity to measure Transformer capacity, study extraction and membership F1 across model and data scales, and fit a validated membership-inference scaling law.

The resulting perspective clarifies when large models store particulars and when they replace those particulars with general patterns. It also provides a controlled explanation of capacity-relative double descent and a practical warning that average-case membership measurements can become uninformative at modern pretraining data scales.

---

## 8. Acknowledgements

The authors thank Karen Ullrich, Niloofar Mireshghallah, Mark Ibrahim, Preetum Nakkiran, and Léon Bottou for feedback that improved the paper.
