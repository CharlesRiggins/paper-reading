## 6. Discussion and Conclusion

The paper develops **MEMIT**, a method for editing factual memories in large language models by directly manipulating specific layer parameters. The central empirical result is that MEMIT scales to far larger edit sets than prior approaches—roughly **100×** the scale of common baselines—while maintaining strong specificity, generalization, and fluency.

The authors also identify limitations. Some relations are harder to edit with robust specificity, though MEMIT still outperforms competing methods on difficult cases. The studied knowledge representation is restricted to directional $(s,r,o)$ facts. It does not cover spatial or temporal reasoning, mathematical knowledge, linguistic knowledge, procedural knowledge, or symmetric relations. For example, “Tim Cook is CEO of Apple” and “The CEO of Apple is Tim Cook” must be treated as separate associations.

Despite these limitations, the paper argues that large-scale model updates can be built from explicit analysis of internal computations. The result suggests that interpretability-based methods might become a practical alternative to opaque fine-tuning for editing, controlling, and auditing models.

## 7. Ethical Considerations

Although the experiments treat language models as knowledge bases, the authors caution that LLMs are not reliable authoritative knowledge sources. Memory-editing methods can illuminate internal mechanisms and reduce the cost or energy needed to fix model errors. However, the same methods could also let malicious actors insert false or damaging information into a model that did not originally contain it.

## 8. Acknowledgements

The authors thank Jaden Fiotto-Kaufmann for building the demonstration at `memit.baulab.us`. The project was supported by an AI Alignment grant from Open Philanthropy. Yonatan Belinkov was also supported by the Israel Science Foundation and an Azrieli Foundation Early Career Faculty Fellowship.

## 9. Reproducibility

Code and data are available at `memit.baulab.info`.

All experiments run on workstations with NVIDIA A6000 GPUs. Models are loaded using HuggingFace `Transformers`, and PyTorch is used for executing the editing algorithms. `GPT-J` experiments fit into one 48GB A6000. `GPT-NeoX` runs require at least two GPUs: one 48GB GPU for the model in `float16` and another smaller GPU for the editing method. The experiments do not run on GPUs with less memory.
