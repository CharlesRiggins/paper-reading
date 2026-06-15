## 4. Corroborating Claims with the OLMo Pretraining Pipeline

We now verify the claims of Sec. 3 in a realistic LLM pre-training setting using the OLMo pipeline. We train models of size 4M to 4B on up to 210B tokens ($\sim$50K steps). Following the structure of Sec. 3, we offer analyses at three levels: loss, representation, and gradient.

---

### 4.1 Setup

A key variable in our claims is the frequency of a task. (Defining the complexity of a natural task is difficult, and hence we solely focus on frequency in this section.) However, measuring the frequency of a naturally occurring task in pre-training data is challenging, as instances from the same task can occur in many surface forms. To tightly control task frequency, we adopt a data injection framework from the memorization literature [58–61]. We inject different instances sampled from the distribution of a "special" task $T$ at a controlled frequency $f$ to measure whether a model has learned the task distribution. The task $T$ is special in the sense that it is unlikely to be part of normal pre-training data. We then train models of various size on data mixtures generated from different values of $f$.

**Figure 5: Larger Models Learn Rare Tasks; Smaller Models Do Not.** We visualize training loss and test accuracy for the (a) Comparison task ($T_{\text{CMP}}$) and (b) Modular Addition task ($T_{\text{ADD}}$). Orange color indicates lower loss/higher accuracy. Overall, we see that increasing width enables learning of low-frequency tasks, inline with our prior claims.

**Tasks.** We consider two special tasks $T$: comparison ($T_{\text{CMP}}$) and modular addition ($T_{\text{ADD}}$). Both tasks are encoded as a sequence of three tokens: TOK1, TOK2, LABEL, where TOK1, TOK2 $\in \mathcal{S}$, a set of 100 tokens randomly sampled from the vocab. There are exactly 10K instances per task, which are split 50/50 for training and testing. Critically, both tasks require models to learn certain geometrical structures to generalize [62]. This provides a measure for learning a task (as opposed to memorizing training instances) and a set of features to verify the interference hypothesis of Sec. 3.

**Data.** We use Dolma v1.7 as the pre-training corpus [63]. Given a task $T$, we inject instances sampled from its train split at a frequency of $7.8 \times 10^{-3}$ to $2.4 \times 10^{-8}$, roughly from 1K instances per batch to 1 instance every 10 batches. To ensure the injected task frequency is comparable to the frequency of tasks learned in pre-training, we sample two reference tasks $R_{\text{cmp}}$ and $R_{\text{add}}$ from pre-training that involve similar high-level functions. The three-token sequence plus an end of document token replace the first four tokens of a training sequence.

**Models.** We train OLMo models [64] with 4M, 20M, 300M, 1B, and 4B parameters. We focus on scaling the models' hidden and MLP dimensions and the number of attention heads; the 4M parameter model has depth 8 and the rest have depth 16.

---

### 4.2 Behavioral Evidence

**Larger Models Learn Rarer Tasks.** We first replicate the behavioral findings in Sec. 3.1. We measure the effect of task frequency by comparing multiple training runs that only differ by the frequency of the injected task. As shown in Fig. 5, larger models learn lower-frequency tasks much better than smaller models do. This matches the pattern in Fig. 2. Moreover, tasks are learned in the order of frequency.

**Figure 6: Behavioral Evidence.** (a) Tasks are learned in the order of frequency. Solid lines: We inject the same comparison task ($T_{\text{CMP}}$) at different frequencies and measure the task training loss. Dashed lines: Reference arithmetic tasks observed from pre-training data. (b) With matched-frequency injection of the comparison task ($T_{\text{CMP}}$), i.e., injecting $N$ task instances every $N$ batches, a larger injection gap $N$ degrades task loss, while a smaller injection gap leads to almost identical loss.

For each model run, we compare the order in which the injected task $T_{\text{CMP}}$ and the reference tasks are learned, as shown in Fig. 6a. Most importantly, larger models do not just lead to better memorization of training instances (low training loss) but also learn generalizable task structures (high eval accuracy). On $T_{\text{ADD}}$, only larger models trained on higher frequency exhibit the **grokking phenomena** [65].

**Rare-Task Retention has an Effect on Learning.** We conduct the matched-frequency injection experiment as described in Fig. 4, i.e., injecting $N$ task instances every $N$ batches, for $N = 1, 10, 20, 50, 100$. Fig. 6b shows the effects of retention on learning, as models trained on larger gap between task instances have higher task loss, even though the global task frequency of all runs is equivalent.

---

### 4.3 Representational Evidence

**Task Features.** In our toy setting (Sec. 3), we know analytically which features are necessary for learning the $k$th task, i.e., $B_k$, and to what extent the model can represent these features, i.e., $P_U$. For our OLMo models, we can empirically identify a set of causal features that a pre-trained LM would use to solve the task and localize them in the model representations. Specifically, for $T_{\text{CMP}}$, the task feature of core relevance is the global order of the tokens, which allows number comparisons; meanwhile for $T_{\text{ADD}}$, task features are the Fourier modes [66–68]. These task features allow us to conduct versions of the gradient and representation-level analyses in Sec. 3.

**More Task Features are Present in Larger Model Representations.**

**Figure 7: Representational evidence.** Scaling model size (width) and increasing task frequency lead to models learning more task-relevant features. Rows correspond to (a) the comparison task $T_{\text{CMP}}$ and (b) the modular addition task $T_{\text{ADD}}$. The first column shows feature geometry, visualizing the global token order features for $T_{\text{CMP}}$ and the Fourier-mode features for $T_{\text{ADD}}$. The last two columns quantify how these features scale with task frequency and model size. For both tasks, the task features are better represented in larger models trained on higher task frequency.

We first localize the task features in models that have clearly learned the task. We then measure to what extent these target task features are present in all models, which parallels the metric $\ell_k(U)$ used in the toy setting. For localization, we use **distributed alignment search (DAS)** [69] which finds subspaces that *causally* encode the features. For $T_{\text{CMP}}$, a global ordering of the tokens can be localized to a 1-D subspace in the residual stream of the first layer. For $T_{\text{ADD}}$, Fourier modes can be identified in the residual stream from earlier layers to the last layer.

We then use task-specific metrics to measure to what extent these task features are present in model representations. For $T_{\text{CMP}}$, since the geometry of the task feature is a single direction, we apply linear regression to representations spanning the top $K = 50$ principal components. For $T_{\text{ADD}}$, we measure the total number of Fourier modes present through all layers. Fig. 7 shows the extent to which the target task features are present in each model across checkpoints. We see that (i) the presence of task features is highly correlated with high accuracy on the test set, and (ii) larger models and models trained on more frequent task data clearly learn these task features faster.

---

### 4.4 Gradient Evidence

**Figure 8: Rare-Task Retention.** Larger models can retain the injected task information better, i.e., larger task eval loss drop, when injecting task instances every 100 batches.

We now connect the behavioral evidence (Sec. 4.2) and the internal representation account (Sec. 4.3) by analyzing how task gradients interfere with non-task gradients on a set of neurons that implement the task circuit. We focus on $T_{\text{CMP}}$ training runs in Fig. 8, where 100 task instances are injected every 100 steps.

**Task Neurons.** We first identify which MLP layers implement the task features defined in Sec. 4.3. For all the models that we compared, the first layer MLP has the largest causal effects on task predictions. We further identify the top $K$ neurons in the first layer MLP that have the largest gradient magnitude and use the gradients of these neurons for analysis.

**Task Reference Direction $g_r$.** We estimate the task reference direction using the aggregated gradient of the task loss computed over all 10K task instances, an analogy to $G_r$ in the toy setting. This direction may shift across training steps; however, at a given step, it is the optimal task direction.

**Larger Models Show Less Gradient Interference Between General Language Modeling and Our Injected Task.** We quantify the relation between the task reference $g_r$ and the batch gradient $g$, which can be further decomposed into gradient from the task tokens $g_t$ (if exists in batch) and non-task tokens $g_{nt}$, i.e., $g = g_t + g_{nt}$. We first measure the cosine similarity between task reference and batch gradient direction, replicating the results in Fig. 4. We additionally analyze whether task or non-task tokens contribute to this similarity; while task token gradient aligning with task reference $g_r$ is expected, non-task token gradient with non-zero cosine similarity suggests that the language modeling direction is interfering with the task gradient direction.

**Figure 9: Gradient Interference.** We inject 100 instances of the $T_{\text{CMP}}$ task every 100 batches and analyze how batch gradients align with a task reference direction $g_r$. We further decompose the batch gradient into contributions from task tokens and non-task tokens. Top: Cosine similarity between full-batch gradient direction and the task direction $g_r$. Middle: Cosine similarity between batch task gradient direction and $g_r$. Higher values imply more task signals. Bottom: Cosine similarity between batch non-task gradient direction and $g_r$. Lower values imply less gradient interference.

Results are shown in Fig. 9. In the top panel, larger models have higher similarity between $g$ and $g_r$ at the injection steps, $0.08 \pm 0.02$ for the 1B model and $0.04 \pm 0.04$ for the 300M model; the similarity typically regresses towards zero between injections. For the 20M model, the similarity scores oscillate wildly across batches, even at the injection step. In fact, the high similarity between non-task gradient $g_{nt}$ and $g_r$ reveals that for the 20M model the batch gradient similarity mostly comes from random collisions with task direction, with a similarity score of $0.10 \pm 0.09$, while for larger models, $g_{nt}$ is almost orthogonal to $g_r$, with $7.58 \times 10^{-5} \pm 0.02$ for the 1B model, suggesting little to no gradient interference on this set of neurons.
