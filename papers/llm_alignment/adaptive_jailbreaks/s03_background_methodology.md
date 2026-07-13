## 3. Background and Methodology

### 3.1 Setting

The paper studies prompts that induce a target LLM to produce content judged as a successful jailbreak for a harmful request. Let the target model be a function from input-token sequences to output-token sequences,

$$
\texttt{LLM}:\mathcal{T}^{*}\rightarrow\mathcal{T}^{*}.
$$

Given a harmful request $R\in\mathcal{T}^{*}$ and a judge function

$$
\texttt{JUDGE}:\mathcal{T}^{*}\times\mathcal{T}^{*}\rightarrow\{\text{NO},\text{YES}\},
$$

the attack objective is

$$
\text{find}\quad P\in\mathcal{T}^{*}\quad\text{subject to}\quad \texttt{JUDGE}(\texttt{LLM}(P), R)=\text{YES}.
$$

The judge may internally use a fine-grained score, such as the 1–10 GPT-4 score used in the experiments, but the final success criterion is binary. The paper uses default system prompts unless otherwise stated; Claude-specific experiments modify system/user/assistant structure because of that API's response-prefilling capability. The target behavior set is 50 harmful requests from `AdvBench`, curated by Chao et al. to be distinct and diverse. A jailbreak counts as successful only when GPT-4 assigns a **10/10** score, with alternative judges reported later in Appendix C.6.

### 3.2 Methodology

The authors define adaptive attacks following adversarial-robustness literature: an adaptive attack is specifically designed against a given defense or deployment setting. Their attack is **model-adaptive** but not **request-adaptive** in the sense that the same template family is used across all requests for a given target model, while model-specific changes are allowed.

#### Prompt templates

The paper's first building block is a manually designed prompt template. Its general structure is:

$$
\text{rules} + \text{harmful request} + \text{adversarial suffix}.
$$

The rules are optimized manually on `GPT-3.5 Turbo` using logprobs of the first target token as feedback. The authors also design a `GPT-4o`-specific template and an in-context template that includes examples of the intended response pattern.

[Verbatim jailbreak prompt templates from Figure 1 and later appendix tables are omitted here because they are operational prompt strings for bypassing safety controls. The paper reproduces them in full; this note preserves their experimental role without copying the exact attack text.]

#### Random search

The random-search (`RS`) attack appends an adversarial suffix to the original request, modifies contiguous token spans at random, and accepts a modification when it increases the log-probability of a target first token. The paper typically uses a suffix initialized with **25 tokens**, up to **10,000 iterations**, and up to **10 restarts**, though most successful runs need fewer restarts. The method is zeroth-order: it requires logprob access but not gradients, making it suitable for some API models.

The authors use the target-token logprob as a proxy for whether the model will continue in a compliant direction. They report that alternative affirmative first tokens did not outperform their default target. Suffixes are request-specific and model-specific; the paper does not split attacks into training and test requests/models.

#### Self-transfer

**Self-transfer** is the paper's initialization trick for random search. A suffix discovered for an easier harmful request—one where the target token already has higher probability—is reused as initialization for a harder request. Although the suffix is optimized for a single model and a single request, it often transfers sufficiently to become a strong starting point. This improves query efficiency and is central to the `Llama-2-Chat` results.

#### Transfer attacks

Cross-model transfer is also used. Successful jailbreak artifacts from one LLM can often be reused against another. This is crucial for Claude-family models because Claude APIs do not expose logprobs, making direct random-search optimization unavailable in the same form.

#### Prefilling attack

Some APIs, notably Claude, allow the caller to prefill the assistant's response with a specified beginning. The paper treats this as a strong optimization-free attack vector: if the response can be started in the target behavioral mode, the rest of the generation may follow. The authors note that similar prefilling can be simulated for open-weight models by directly modifying chat templates.
