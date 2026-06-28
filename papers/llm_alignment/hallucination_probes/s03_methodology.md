## 3 Methodology

> **Figure 3:** Token-level annotation pipeline. We construct **LongFact++**, a large set of prompts spanning diverse domains to elicit entity-dense generations. The target LLM (e.g., `Llama`) produces long-form completions containing both factual and hallucinated content. A frontier LLM with web search (e.g., `Claude`) then identifies entities within each generation, verifies them against external sources, and produces labels indicating which entities are supported and which are not. The result is a dataset in which every token is annotated to indicate whether it forms part of a hallucination.

### 3.1 Dataset construction for token-level hallucination detection

To train token-level hallucination detectors, we need a dataset with precise annotations of hallucinated content within long-form outputs. This requires two steps: (1) generating diverse completions that contain both hallucinated and factual content, and (2) obtaining accurate token-level annotations that identify which specific tokens correspond to hallucinated entities. An overview of the annotation pipeline is portrayed in Figure 3.

#### Data generation.

We build upon the **LongFact** dataset (Wei et al., 2024b), composed of 2,280 fact-seeking prompts, and introduce **LongFact++**, which expands LongFact with 10 times more prompts across more diverse domains and query structures. LongFact++ incorporates four categories of prompts: topic-focused queries (e.g., queries related to the topic of "molecular mechanisms of viral DNA replication"), biographical queries about famous individuals, citation-focused prompts that encourage generation of references, and legal prompts based on landmark court cases. More details of dataset construction are provided in Appendix C.

Following Wei et al. (2024b), we append the following postamble to each question in order to prompt the model to give a long, detailed completion: "Provide as many specific details and examples as possible (such as names of people, numbers, events, locations, dates, times, etc.)."

#### Token-level annotation.

Existing verification methods like SAFE (Wei et al., 2024b) decompose generated text into atomic claims for verification, but this reformulation breaks alignment with the original token sequence needed for token-level training. Instead, we focus our annotations on *entities*—e.g., named people, organizations, locations, dates, and citations—which can be verified against external sources while preserving exact token boundaries.

For each generated completion, we use `Claude 4 Sonnet` with web search capabilities (Anthropic, 2025a, b) to extract and annotate specific spans within the original text. The system identifies entity spans, searches for supporting evidence, and labels each entity as "Supported," "Not Supported," or "Insufficient Information" (see the full prompt in Appendix C.3). We treat entities labeled as either "Not Supported" or "Insufficient Information" as hallucinated. Spans that cannot be confidently mapped back to spans in the original completion are discarded. Figure 2 shows an example of a labeled completion with entity-level annotations and verification justifications.

#### Label quality.

We audited label quality using several checks (details in Appendix E) and summarize the two most informative here. We recruited a human annotator to independently label a random sample of entity spans via web search; the human annotations matched the LLM's labels in **84%** of cases ($n{=}50$). We also constructed a controlled set of hallucinations by paraphrasing Wikipedia passages and injecting known factual errors, and then ran our annotation pipeline on this controlled dataset. Across 100 samples, our annotation pipeline correctly detected **729/904** of the injected errors (**80.6% recall**), and falsely flagged **15.8%** of unchanged entities (i.e., 15.8% false positive rate).

### 3.2 Training token-level probes

#### Setup.

Given a query $\mathbf{q}$ and a chat model $M$, the model generates tokens $\mathbf{t}=(t_{1},\ldots,t_{n})\sim M(\mathbf{t}\mid\mathbf{q})$. Our dataset yields annotations denoting *entity spans* $s=[s^{\text{start}},s^{\text{end}}]$ (inclusive token indices) with binary labels $y_{s}\in\{0,1\}$, where $y_{s}{=}1$ indicates a hallucinated span. The detector's goal is to assign each token $t_{i}$ inside labeled spans a probability of being part of a hallucination.

#### Probe.

We denote by $M_{\text{probe}}$ the hallucination detector attached to $M$, consisting of a linear value head and, optionally, LoRA adapters inserted into all layers preceding the head. The value head reads hidden states from an intermediate layer $\ell$ of $M$ and outputs token-level probabilities:

$$
p_{i}=\sigma\!\big{(}\mathbf{w}^{\top}\mathbf{h}^{(\ell)}_{i}+b\big{)},\qquad i\in s,
$$

where $\mathbf{h}^{(\ell)}_{i}$ is the hidden state of token $t_{i}$ at layer $\ell$ and $\sigma$ is the logistic sigmoid function. We always train the value head parameters $(\mathbf{w},b)$; when LoRA adapters are present, we train those as well. We attach the probe head at layer $\ell=\lfloor 0.95\times\text{num\_layers}\rfloor$ unless otherwise noted.

#### Objective.

The total loss is a convex combination of a *probe loss*, which trains the hallucination classifier, and a *regularization term*, which constrains changes to the underlying language model:

$$
\mathcal{L}_{\text{total}}=(1-\lambda_{\text{reg}})\,\mathcal{L}_{\text{probe}}+\lambda_{\text{reg}}\,\mathcal{L}_{\text{reg}},\qquad\lambda_{\text{reg}}\in[0,1]. \tag{1}
$$

The regularizer $\mathcal{L}_{\text{reg}}$ is applied only when training with LoRA; i.e., when the probe is a *linear probe*, $\lambda_{\text{reg}}$ is always zero, as regularization is not needed. We experiment with two losses for the regularization term:

- **Language modeling loss** ($\mathcal{L}_{\text{LM}}$): standard next-token prediction loss, or
- **KL divergence loss** ($\mathcal{L}_{\text{KL}}$): KL divergence between the fine-tuned LoRA model and the frozen original model.

These regularization strategies are evaluated in Section 5.3. By default, most experiments use LM regularization with $\lambda_{\text{reg}}{=}0.01$, unless otherwise noted.

#### Probe loss: token-wise and span-max.

We train the probe using binary cross-entropy (BCE) loss between $p_{i}$ and the label $y_{s}$ of its containing span $s$. However, annotated spans are often longer than the actual error; for example, in "born in 2002," only the final token ("02") may be incorrect. This creates two challenges: (1) we do not know in advance which tokens within a hallucinated span are incorrect, and (2) hallucination signals are typically concentrated at specific "high-information" tokens (Orgad et al., 2025), not spread uniformly. Optimizing BCE over every token in a span dilutes this signal and risks teaching the probe to activate broadly rather than precisely.

We address this by combining a *token-wise* loss over all tokens with a *span-max* loss over annotated entity spans (Tillman and Mossing, 2025; Sharma et al., 2025). Let $T$ be all token positions and $S$ the set of annotated spans. Define token labels by $y_{i}{=}y_{s}$ if $i$ is within an entity span $s$ (an "entity token"), and $y_{i}{=}0$ if $i$ is outside any entity span (a "background token"). The probe loss is:

$$
\mathcal{L}_{\text{probe}}=(1-\omega)\!\sum_{i\in T}w_{i}\,\mathrm{BCE}(y_{i},p_{i})\;+\;\omega\!\sum_{s\in S}\mathrm{BCE}\big{(}y_{s},\max_{i\in s}p_{i}\big{)},\qquad\omega\in[0,1]. \tag{2}
$$

For a positive label $y_{s}{=}1$, the max term rewards the probe if *at least one* token in span $s$ scores high; for $y_{s}{=}0$, it requires *all* tokens within the span to score low. Following Sharma et al. (2025), we anneal $\omega$ from 0 to 1 during training: early on, the token-wise term provides dense, stable gradients; later, the span-max term sharpens the probe's focus on the most informative token in each span (e.g., only the final digits of "born in 2002").

Background tokens greatly outnumber entity tokens, so we up-weight tokens that lie inside any annotated span: $w_{i}{=}\alpha$ if $i$ is an entity token, else $w_{i}{=}1$; we use $\alpha{=}10$ unless otherwise noted. This weighting prevents the loss from being dominated by easy background negatives.

### 3.3 Baselines

To contextualize the performance of our probes, we compare against several uncertainty-based metrics. In particular, we evaluate token-level entropy, token-level perplexity, semantic entropy, and a black-box self-evaluation method. See Appendix F for additional details.

- **Token-level entropy:** Uncertainty in the next-token distribution; higher values indicate the model considered many plausible continuations.
- **Token-level perplexity:** How "surprised" the model is by its own token choice; higher values signal lower confidence.
- **Semantic entropy:** Measures uncertainty over *semantic meanings* rather than surface forms via clustering multiple sampled completions (Kuhn et al., 2023; Farquhar et al., 2024). See Section 2 for a description, and Appendix F.2 for implementation details.
- **Black-box self-evaluation:** Prompting the model to judge whether a sentence from its own output contains a hallucination. Full details and results are provided in Appendix F.3.
