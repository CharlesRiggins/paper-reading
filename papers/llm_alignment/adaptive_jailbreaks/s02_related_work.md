## 2. Related Work

Adversarial attacks on machine-learning models have a long history, including evasion attacks and adversarial examples in computer vision. This paper focuses on LLM jailbreaking and organizes prior work into manual attacks, direct search attacks, and LLM-assisted attacks.

### Manual attacks

Early users discovered handcrafted jailbreaks soon after ChatGPT's release. Wei et al. categorize jailbreak failures using two mechanisms: **competing objectives**, where model capabilities conflict with safety goals, and **mismatched generalization**, where safety training fails to generalize to domains in which the model retains capability. By exploiting such failure modes with manual attacks, prior work achieved high success rates on proprietary models such as `GPT-4` and `Claude v1.3`.

Wei et al. also explore in-context learning prompts containing a few harmful examples, while Anil et al. scale this idea to many examples and derive predictable trends for long-context LLMs. This line of work is important for the current paper because one of its target-specific adaptations is also an in-context prompt structure, especially for `R2D2`.

### Direct search attacks

The search for jailbreak strings can be automated with first-order or zeroth-order discrete optimization. Zou et al. introduce universal and transferable attacks using **Greedy Coordinate Gradient** (`GCG`), inspired by earlier discrete prompt-optimization work in NLP. Lapid et al. use a genetic algorithm to generate universal adversarial prompts under a black-box threat model, while Liu et al. and Zhu et al. develop related low-perplexity or interpretable adversarial suffix methods.

Liao and Sun propose learning a separate model to generate adversarial prefixes similar to `GCG`. Sitawarin et al. and Hayase et al. are especially close to this paper because they use random search over predicted probabilities for black-box models, sometimes aided by a white-box LLM. However, the paper notes that their OpenAI-specific use of `logit_bias` became ineffective after API behavior changed, since it no longer influences returned logprobs in the same way.

### LLM-assisted attacks

A separate branch uses auxiliary LLMs to optimize jailbreaks. Chao et al. develop **Prompt Automatic Iterative Refinement** (`PAIR`), where an attacker LLM iteratively proposes prompts. Mehrotra et al. refine this into a tree-based search method (`TAP`). Shah et al. use persona modulation, Yu et al. introduce `GPTFUZZER`, and Zeng et al. fine-tune `GPT-3.5` for rephrasing harmful requests.

The current paper differs by emphasizing simple model-specific mechanisms rather than complex auxiliary-LLM loops. Its claim is not that auxiliary LLMs are unhelpful, but that much simpler adaptive methods—manual templates, logprob-guided random search, self-transfer, and prefilling—can already expose severe non-robustness.
