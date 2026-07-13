## 4. Jailbreaking Leading Safety-Aligned LLMs

This section evaluates adaptive attacks against production-grade and strongly safety-aligned models. The main body covers `Llama-2-Chat`, `Llama-3-Instruct`, `Gemma`, `R2D2`, GPT models, and Claude models; Appendix C.8 adds `Vicuna-13B`, `Mistral-7B`, `Phi-3-Mini`, and `Nemotron-4-340B`.

### 4.1 Jailbreaking Llama-2, Llama-3, and Gemma Models

The authors evaluate open-weight `Llama-2-Chat` models at 7B, 13B, and 70B parameters, `Llama-3-Instruct-8B`, and `Gemma-7B`. For the Llama models, they use the safety prompt from Touvron et al., which makes these models relatively resilient to prior jailbreak attacks, including some white-box settings.

The key ingredient for Llama-family models is **self-transfer**. Standard prompt templates alone produce 0% success on `Llama-2-Chat`, confirming that its safety alignment is meaningful against static templates. Adding random search raises success, and combining prompt + random search + self-transfer reaches **100% ASR** for all evaluated Llama variants. For `Gemma-7B`, prompt + random search already reaches 84%, and self-transfer brings it to **100%**.

> Table 2: Attack success rate under GPT-4 judging for `Llama`, `Gemma`, and `R2D2`.

| Model | Method | Source | Success rate |
|---|---|---|---:|
| `Llama-2-Chat-7B` | Tree of Attacks with Pruning | Zeng et al. (2024) | 4% |
| `Llama-2-Chat-7B` | Prompt Automatic Iterative Refinement | Chao et al. (2023) | 10% |
| `Llama-2-Chat-7B` | Greedy Coordinate Gradient | Chao et al. (2023) | 54% |
| `Llama-2-Chat-7B` | Persuasive Adversarial Prompts | Zeng et al. (2024) | 92% |
| `Llama-2-Chat-7B` | Prompt | Ours | 0% |
| `Llama-2-Chat-7B` | Prompt + Random Search | Ours | 50% |
| `Llama-2-Chat-7B` | Prompt + Random Search + Self-Transfer | Ours | **100%** |
| `Llama-2-Chat-13B` | Tree of Attacks with Pruning | Mazeika et al. (2024) | 14%* |
| `Llama-2-Chat-13B` | Prompt Automatic Iterative Refinement | Mazeika et al. (2024) | 15%* |
| `Llama-2-Chat-13B` | Greedy Coordinate Gradient | Mazeika et al. (2024) | 30%* |
| `Llama-2-Chat-13B` | Prompt | Ours | 0% |
| `Llama-2-Chat-13B` | Prompt + Random Search + Self-Transfer | Ours | **100%** |
| `Llama-2-Chat-70B` | Tree of Attacks with Pruning | Mazeika et al. (2024) | 13%* |
| `Llama-2-Chat-70B` | Prompt Automatic Iterative Refinement | Mazeika et al. (2024) | 15%* |
| `Llama-2-Chat-70B` | Greedy Coordinate Gradient | Mazeika et al. (2024) | 38%* |
| `Llama-2-Chat-70B` | Prompt | Ours | 0% |
| `Llama-2-Chat-70B` | Prompt + Random Search + Self-Transfer | Ours | **100%** |
| `Llama-3-Instruct-8B` | Prompt | Ours | 0% |
| `Llama-3-Instruct-8B` | Prompt + Random Search | Ours | **100%** |
| `Llama-3-Instruct-8B` | Prompt + Random Search + Self-Transfer | Ours | **100%** |
| `Gemma-7B` | Prompt | Ours | 20% |
| `Gemma-7B` | Prompt + Random Search | Ours | 84% |
| `Gemma-7B` | Prompt + Random Search + Self-Transfer | Ours | **100%** |
| `R2D2-7B` | Greedy Coordinate Gradient | Mazeika et al. (2024) | 6%* |
| `R2D2-7B` | Prompt Automatic Iterative Refinement | Mazeika et al. (2024) | 48%* |
| `R2D2-7B` | Tree of Attacks with Pruning | Mazeika et al. (2024) | 61%* |
| `R2D2-7B` | Prompt | Ours | 8% |
| `R2D2-7B` | Prompt + Random Search + Self-Transfer | Ours | 12% |
| `R2D2-7B` | In-context Prompt | Ours | 90% |
| `R2D2-7B` | In-context Prompt + Random Search | Ours | **100%** |

Convergence plots in Figure 2 show average target-token logprob and ASR for `Llama-3-Instruct-8B`, `Llama-2-Chat-7B`, and `Gemma-7B`. They support the paper's claim that self-transfer gives a much better initialization and improves query efficiency.

### 4.2 Jailbreaking R2D2 Model

`R2D2` is adversarially trained against jailbreak attacks, analogous to adversarial training for $\ell_p$-bounded perturbations in vision. The standard prompt template and random search are weak against it, and self-transfer does not help much. The authors hypothesize that `R2D2` learned to refuse a particular request-plus-suffix structure, so they adapt by changing the structure to an in-context learning prompt.

This in-context prompt alone achieves **90% ASR**, and adding random search raises success to **100%**, substantially above the prior `TAP` result of 61%. The same in-context prompt is less effective on other models, reinforcing the paper's broader point that vulnerabilities are model-specific.

### 4.3 Jailbreaking GPT Models

The GPT experiments cover `gpt-3.5-turbo-1106`, `gpt-4-1106-preview`, and `gpt-4o-2024-05-13`. Because OpenAI's API exposed top-token logprobs, the authors use random search despite not having full white-box access. They find `GPT-3.5 Turbo` brittle to the manually designed template alone, while `GPT-4 Turbo` and `GPT-4o` require more adaptation.

For `GPT-4 Turbo`, the base prompt gives 28% ASR, while prompt + random search + self-transfer reaches **96%**, above the previous best 59% transfer result. For `GPT-4o`, the default prompt is ineffective at 0%, but a custom template found with logprob-guided manual search reaches 72%, and adding random search + self-transfer reaches **100%**. This difference is one of the clearest demonstrations that adaptive attacks can outperform static recipes.

> Table 3: GPT-model attack success rates under GPT-4 judging.

| Model | Method | Source | Success rate |
|---|---|---|---:|
| `GPT-3.5 Turbo` | Prompt Automatic Iterative Refinement | Chao et al. (2023) | 60% |
| `GPT-3.5 Turbo` | Tree of Attacks with Pruning | Zeng et al. (2024) | 80% |
| `GPT-3.5 Turbo` | Greedy Coordinate Gradient | Zeng et al. (2024) | 86% |
| `GPT-3.5 Turbo` | Persuasive Adversarial Prompts | Zeng et al. (2024) | 94% |
| `GPT-3.5 Turbo` | Prompt | Ours | **100%** |
| `GPT-4 Turbo` | Prompt Automatic Iterative Refinement | Mazeika et al. (2024) | 33%* |
| `GPT-4 Turbo` | Tree of Attacks with Pruning | Mazeika et al. (2024) | 36%* |
| `GPT-4 Turbo` | Tree of Attacks with Pruning (Transfer) | Mazeika et al. (2024) | 59%* |
| `GPT-4 Turbo` | Prompt | Ours | 28% |
| `GPT-4 Turbo` | Prompt + Random Search + Self-Transfer | Ours | **96%** |
| `GPT-4o` | Prompt | Ours | 0% |
| `GPT-4o` | Custom Prompt | Ours | 72% |
| `GPT-4o` | Custom Prompt + Random Search + Self-Transfer | Ours | **100%** |

The paper also reports non-determinism in GPT outputs: repeating the same query 1,000 times with temperature zero and fixed seed still gives varying logprobs. This noisy signal makes random search less reliable but not ineffective.

### 4.4 Jailbreaking Claude Models

Claude models do not expose logprobs, so the authors cannot directly run the same random-search procedure. They instead test transfer attacks using adversarial suffixes optimized on `GPT-4`, and then test Claude's prefilling feature, which lets the caller provide the beginning of the assistant response.

The transfer attack is strong on some models, especially `Claude 3 Haiku`, `Claude 3 Sonnet`, and `Claude 3.5 Sonnet`. Splitting the prompt into system and user parts improves some transfer results. However, `Claude 2.1` and `Claude 3 Opus` are more resistant to transfer.

Prefilling is much stronger: combined with the prompt structure, it reaches **100% ASR** on all evaluated Claude models, including `Claude 3` and `Claude 3.5`. The authors note that `Claude 2.1` appears most robust, and GPT-4 judging produces more false positives for it than for other models.

> Table 4: Claude-model attack success rates under GPT-4 judging.

| Model | Method | Source | Success rate |
|---|---|---|---:|
| `Claude 2.0` | Persuasive Adversarial Prompts | Zeng et al. (2024) | 0% |
| `Claude 2.0` | Greedy Coordinate Gradient | Chao et al. (2023) | 4% |
| `Claude 2.0` | Prompt Automatic Iterative Refinement | Chao et al. (2023) | 4% |
| `Claude 2.0` | Persona Modulation | Shah et al. (2023) | 61%α |
| `Claude 2.0` | Prompt + Transfer from GPT-4 | Ours | **100%** |
| `Claude 2.0` | Prompt + Prefilling Attack | Ours | **100%** |
| `Claude 2.1` | Foot-in-the-Door Attack | Wang et al. (2024) | 68%β |
| `Claude 2.1` | Prompt + Transfer from GPT-4 | Ours | 0% |
| `Claude 2.1` | Prompt + Prefilling Attack | Ours | **100%†** |
| `Claude 3 Haiku` | Prompt + Transfer from GPT-4 | Ours | 98% |
| `Claude 3 Haiku` | Prompt + Prefilling Attack | Ours | **100%** |
| `Claude 3 Sonnet` | Prompt + Transfer from GPT-4 | Ours | **100%** |
| `Claude 3 Sonnet` | Prompt + Prefilling Attack | Ours | **100%** |
| `Claude 3 Opus` | Prompt + Transfer from GPT-4 | Ours | 0% |
| `Claude 3 Opus` | Prompt + Prefilling Attack | Ours | **100%** |
| `Claude 3.5 Sonnet` | Prompt + Transfer from GPT-4 | Ours | 96% |
| `Claude 3.5 Sonnet` | Prompt + Prefilling Attack | Ours | **100%** |

The key conclusion is that API design is part of the security surface. Logprob access enables one class of adaptive search, while response prefilling enables another. Removing one capability does not necessarily eliminate adaptive vulnerabilities.
