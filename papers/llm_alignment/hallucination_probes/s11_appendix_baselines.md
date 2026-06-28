## Appendix F — Baselines

### F.1 Token-level uncertainty metrics

#### Token-level entropy.

For token $t_{i}$ with next-token distribution $p(\cdot\mid\mathbf{q},\mathbf{t}_{<i})$,

$$
H_{i}\;=\;-\sum_{v\in V}p(v\mid\mathbf{q},\mathbf{t}_{<i})\,\log p(v\mid\mathbf{q},\mathbf{t}_{<i}), \tag{3}
$$

where $V$ is the token vocabulary. We compute the maximum-aggregation score over a span $s$ as

$$
H_{s}=\max_{i\in[s^{\text{start}},s^{\text{end}}]}H_{i}. \tag{4}
$$

#### Token-level perplexity.

For token $t_{i}$ with next-token distribution $p(\cdot\mid\mathbf{q},\mathbf{t}_{<i})$,

$$
\mathrm{PPL}_{i}\;=\;\exp\!\big{(}-\log p(t_{i}\mid\mathbf{q},\mathbf{t}_{<i})\big{)}. \tag{5}
$$

We compute the maximum-aggregation score over a span $s$ as

$$
\mathrm{PPL}_{s}=\max_{i\in[s^{\text{start}},s^{\text{end}}]}\mathrm{PPL}_{i}. \tag{6}
$$

### F.2 Semantic entropy

#### Semantic entropy.

Semantic entropy (Farquhar et al., 2024) detects hallucinations by measuring uncertainty across semantically equivalent generations. Given a query $q$, the method samples multiple completions, which are then grouped into clusters $C$ based on semantic equivalence. The probability of a semantic cluster, $p(c|q)$, is operationalized as the fraction of generations in that cluster. Semantic entropy quantifies the uncertainty associated with the distribution $p(c|q)$:

$$
H^{SE}(\mathbf{t},\mathbf{q})=-\sum_{c\in C}p(c|\mathbf{q})[\log p(c|\mathbf{q})]. \tag{7}
$$

#### Overview.

Semantic entropy (Kuhn et al., 2023; Farquhar et al., 2024) detects hallucinations by measuring uncertainty across semantically equivalent generations, grouping sampled completions into clusters by meaning and quantifying the entropy of the cluster distribution.

#### Clustering by semantic equivalence.

We form clusters via pairwise bidirectional entailment judged by `GPT-4.1`: two completions $u,v$ are linked if $u\models v$ and $v\models u$. We build an undirected graph on $k$ samples and take connected components as semantic clusters $C=\{c\}$. Cluster probabilities are empirical frequencies $p(c)=|c|/k$. The semantic entropy is

$$
H^{\mathrm{SE}}\;=\;-\sum_{c\in\mathcal{C}}p(c)\,\log p(c). \tag{8}
$$

#### Task-specific setup.

- **TriviaQA (short-form):** For each question, we sample $k{=}10$ answers, judge pairwise entailment using only the generated responses, cluster as above, and compute $H^{SE}$ across these responses.
- **Long-form spans:** For each annotated span $s$, we take the completion prefix up to (but not including) the entity $s$, then sample $k{=}10$ continuations with a target length up to $2\times$ the original span length. We cluster the $k$ continuations and use the resulting $H^{SE}$ as the span score.
- **Math:** For each question, we generate one greedy completion (temperature 0) and sample $k{=}10$ additional completions at temperature 0.6. We use an LLM to extract the final numerical or algebraic answer from each completion. We compute $H^{SE}$ by clustering only the $k{=}10$ extracted answers from the temperature-sampled completions based on pairwise entailment.

### F.3 Black-box self-evaluation: can we just ask the model whether it's hallucinating?

Given that our white-box probes demonstrate that internal model states contain sufficient information to detect hallucinations, a natural question arises: can we achieve effective hallucination detection by simply asking the model directly, without requiring access to internal representations? This black-box self-evaluation approach would be more practical for deployment scenarios where probe training is infeasible or where models are accessed only through APIs. Here we investigate whether models can reliably identify their own hallucinations when prompted appropriately, and examine how this capability scales from short-form to long-form content.

Our evaluation approach employs a multi-turn conversation format. After the model generates a completion in response to an instruction from our dataset, we select specific sentences within that completion and ask the model to evaluate them in a follow-up question. For sentence selection, we use our existing annotations to identify sentences where either all contained annotated spans are labeled as supported or all are labeled as hallucinated. We discard sentences that contain no annotated spans or have mixed support labels, ensuring clean training signal. We then reference each selected sentence from the model's previous response and ask: "Please evaluate whether the following sentence in our conversation contains a hallucination. Answer with 'Yes' or 'No'." This multi-turn formulation provides the model with the full conversational context necessary for accurate self-evaluation.

We adopt this methodology for several reasons. This black-box self-evaluation approach is highly sensitive to the specific prompt used, and our multi-turn format yielded the best performance after extensive experimentation. Second, it is designed to be feasible for long-form content where sentences often depend on preceding context for proper interpretation—pronouns may lack clear referents, statistics may require earlier context to understand their meaning, and factual claims may build on previously established information. Third, while our dataset consists of span-level annotations, we instead evaluate entire sentences here because: (1) based on our experiments, sentence-level evaluation significantly outperforms direct span-level verification, and (2) it represents a more realistic deployment scenario, since span-level evaluation would require a priori knowledge of which specific text segments to verify.

For comparison with our probe-based approach, we also evaluate our standard LoRA probes on the same sentence-level task by applying the identical sentence selection procedure (sentences containing only supported or only unsupported spans) but treating each complete sentence as a single span rather than individual entities.

> **Figure 11:** Self-evaluation results comparing AUC performance across datasets. *Left:* TriviaQA (short-form) results where models achieve moderate self-evaluation performance. *Right:* Long-form results demonstrating the performance gap that emerges when scaling to complex, multi-factual content. For long-form evaluation, we train and test on 10,000 samples, while for short-form we use 1,000 samples for `Llama-3.1-8B` and 2,000 for `Llama-3.3-70B`. All training and test datasets are balanced between hallucinated and non-hallucinated examples.

Figure 11 presents our results, revealing two key findings. First, the self-evaluation approach shows moderate capability on short-form content (TriviaQA), achieving AUCs between **0.81–0.89**. However, its effectiveness does not scale well to long-form content, where performance drops significantly across both models tested (AUCs between **0.58–0.68**). Second, we observe that the larger model (`Llama-3.3-70B`) performs substantially better at self-evaluation than the smaller model (`Llama-3.1-8B`), suggesting that self-awareness of factual accuracy may improve with model scale.

In addition to prompting-based evaluation, we also implemented a supervised fine-tuning baseline where we trained models specifically for this task using LoRA adapters. We constructed training datasets by pairing each model's completions with multi-turn conversations where the model is asked to evaluate specific sentences, using the same prompting strategy described above. The fine-tuning process optimizes the model to correctly answer "Yes" for sentences containing only unsupported spans and "No" for sentences with only supported spans. Surprisingly, this fine-tuning yields only marginal performance improvements over the prompting approach. We included this baseline for two reasons: to ensure a fair comparison with our trained probes, and as a sanity check to verify that performance limitations were not due to suboptimal prompting or other spurious factors. While we optimized our prompting method before applying fine-tuning, we acknowledge that jointly optimizing prompting strategies with fine-tuning in mind might yield better results. Alternative approaches—such as including reasoning traces in the supervised-fine-tuning data, providing more comprehensive guidelines, or constructing the dataset differently—could potentially improve performance. Nevertheless, we believe our implementation represents a reasonable effort to present this baseline, and we include these results for completeness.

While the approach shows some capability on short-form data, the challenge of detecting hallucinations in long-form generations remains substantial. The dramatic performance degradation when moving from TriviaQA to our long-form datasets indicates that the complexity of multi-factual, context-dependent content poses fundamental challenges for self-evaluation approaches. This disparity suggests that self-evaluation faces particular challenges in long-form settings that are better addressed by internal representations.
