## 2 Related work

The problem of hallucination detection in LLMs has inspired a range of techniques. In this section, we summarize key approaches, including probing classifiers, uncertainty-based metrics, and methods based on verification through external sources.

### Internal representation-based methods.

A growing body of work leverages models' internal states to detect hallucinations. Probing classifiers (Alain and Bengio, 2017) map intermediate model representations to target properties and have been extensively used for hallucination detection. Marks and Tegmark (2024) train linear probes to uncover truth-related directions in representation space. Recent studies (Orgad et al., 2025; Ji et al., 2024; Alnuhait et al., 2025) show that linear and MLP-based probes can predict hallucinations using hidden states before or during generation, often achieving strong AUC scores across various tasks. However, their ability to generalize to more complex settings, such as open-ended long-form generation, remains unproven.

CH-Wang et al. (2024) train span-level probes to detect hallucinations during generation on grounded tasks (e.g., document summarization). Like our approach, they develop streaming token-level classifiers for real-time detection. However, they focus on detecting content that is inconsistent with the provided input context (e.g., a source document), whereas our approach detects factually incorrect entities against world knowledge more broadly.

Recent work in mechanistic interpretability (Ferrando et al., 2025; Lindsey et al., 2025) has discovered the existence of "features," or linear directions in activation space, that correspond to whether a model knows an entity or not, and that these features are causally relevant in determining whether the model attempts to answer a query or abstains.

### Uncertainty-based detection.

Hallucinations in language models can be analyzed through the lens of uncertainty estimation. The uncertainty of model predictions across an entire sequence can be quantified using the joint probability of generated tokens. To account for varying sequence lengths, previous work (Fomicheva et al., 2020; Guerreiro et al., 2023) has considered the length-normalized generation log probability of a model's response as an approximate measure of its uncertainty.

While many machine learning tasks involve distinct, mutually exclusive output classes (e.g., digit classification), open-ended text generation is more complex, as multiple distinct output sequences can convey essentially the same meaning. Addressing this issue, Kuhn et al. (2023) and Farquhar et al. (2024) introduce **semantic entropy**: given a query, semantic entropy groups semantically equivalent answers into clusters, and quantifies how spread out the model's probability distribution is across these clusters. High semantic entropy indicates uncertainty about which meaning to convey, signaling higher risk of hallucination. While powerful, estimating semantic entropy with sampling-based methods is computationally intensive. To address this, Kossen et al. (2024) propose **semantic entropy probes (SEPs)**—lightweight classifiers trained to predict semantic entropy from hidden states alone, achieving competitive but lower classification performance compared to the sampling-based variant.

### External verification methods.

Methods such as SAFE (Wei et al., 2024b), FactScore (Min et al., 2023), and FacTool (Chern et al., 2023) represent a prominent approach that employs external verification for long-form hallucination detection. These methods work by first extracting claims from the generated text, then retrieving supporting evidence from external sources, and finally evaluating each claim in light of the external evidence. While effective for comprehensive verification, these pipelines incur significant computational costs and latency, making them unsuitable for real-time detection during generation. For example, a single sentence may fan out into tens of claims, each of which requires multiple search queries and LLM API calls to verify.
