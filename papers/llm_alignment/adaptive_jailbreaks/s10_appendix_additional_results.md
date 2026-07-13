## Appendix C. Additional Results

Appendix C provides ablations, runtime analysis, additional Claude results, judge comparisons, baseline comparisons, and the full evaluation summary.

### C.1 Effect of the number of tokens in adversarial suffixes

The authors justify using **25 initial tokens** for adversarial suffixes. On `Gemma-7B`, with 1,000 random-search iterations, both the average target-token logprob and ASR follow a U-shaped trend as suffix length changes. Very short suffixes underparameterize the search, while very long suffixes make optimization harder and often cause the model to answer an unrelated request.

The paper observes that 60-token suffixes can still produce the target first token but fail to stay on-topic, reducing judged ASR. This motivates the default 25-token choice as a balance between expressive search space and tractable optimization.

### C.2 Discussion on the runtime

Each random-search iteration is dominated by a forward pass through the target LLM; the rest of the computation is negligible. With an implementation based on HuggingFace `transformers`, 4,000 iterations on `Llama-3-8B` take **20.9 minutes** on one `A100` GPU without prefix caching. In practice, self-transfer greatly reduces needed iterations: fewer than 10% of harmful behaviors require 4,000 iterations, and most need fewer than 200. The full experiment therefore takes a few hours.

### C.3 Further results on Claude models

The appendix reports transfer and prefilling ablations for Claude models. Transfer from `GPT-4` becomes stronger when the prompt is split into system and user messages for some models. Prefilling is consistently stronger than transfer-only attacks, but the exact system/user/assistant structure affects both GPT-4 judge and rule-based judge results.

> Table 15: Transfer attack from `GPT-4` on Claude models, measured by GPT-4 judge.

| Model | 1 restart User | 1 restart System+user | 10 restarts User | 10 restarts System+user | 100 restarts User | 100 restarts System+user |
|---|---:|---:|---:|---:|---:|---:|
| `Claude Instant 1.2` | 0% | 40% | 0% | 52% | 0% | 54% |
| `Claude 2.0` | 2% | 90% | 12% | 98% | 48% | **100%** |
| `Claude 2.1` | 0% | 0% | 0% | 0% | 0% | 0% |
| `Claude 3 Haiku` | 4% | 68% | 30% | 90% | 52% | 98% |
| `Claude 3 Sonnet` | 86% | 70% | **100%** | 98% | **100%** | **100%** |
| `Claude 3 Opus` | 0% | 0% | 0% | 0% | 0% | 0% |
| `Claude 3.5 Sonnet` | 78% | 96% | 78% | 96% | 82% | 96% |

> Table 16: Prefilling ablation #1, reported as GPT-4 judge / rule-based judge.

| Model | User 1 restart | System+user 1 restart | System+user+assistant 1 restart | System+user+assistant 10 restarts | System+user+assistant 100 restarts |
|---|---:|---:|---:|---:|---:|
| `Claude Instant 1.2` | 0%/0% | 70%/86% | 82%/92% | **100%**/90% | **100%**/90% |
| `Claude 2.0` | 6%/10% | 92%/92% | 92%/90% | **100%**/92% | **100%**/92% |
| `Claude 2.1` | 0%/0% | 0%/0% | 14%/68% | 64%/70% | **100%**/86% |
| `Claude 3 Haiku` | 0%/0% | 0%/0% | 96%/94% | **100%**/90% | **100%**/90% |
| `Claude 3 Sonnet` | 2%/8% | 2%/30% | 98%/88% | **100%**/86% | **100%**/86% |
| `Claude 3 Opus` | 0%/0% | 0%/0% | 76%/74% | **100%**/86% | **100%**/86% |
| `Claude 3.5 Sonnet` | 14%/34% | 48%/52% | **100%**/98% | **100%**/98% | **100%**/98% |

> Table 17: Prefilling ablation #2, reported as GPT-4 judge / rule-based judge.

| Model | System+user+assistant | User+assistant | System+assistant | Assistant | Assistant 10 restarts |
|---|---:|---:|---:|---:|---:|
| `Claude Instant 1.2` | 82%/92% | 32%/70% | 86%/94% | 72%/76% | 80%/84% |
| `Claude 2.0` | 92%/90% | 34%/78% | 82%/92% | 30%/70% | 68%/86% |
| `Claude 2.1` | 14%/68% | 16%/56% | 6%/56% | 16%/42% | 54%/64% |
| `Claude 3 Haiku` | **100%**/90% | 90%/72% | **100%**/98% | 64%/24% | 98%/32% |
| `Claude 3 Sonnet` | 98%/88% | 84%/72% | 72%/76% | 34%/30% | 80%/62% |
| `Claude 3 Opus` | 76%/74% | 76%/62% | 72%/80% | 64%/60% | 96%/84% |
| `Claude 3.5 Sonnet` | **100%**/98% | 54%/54% | 94%/96% | 36%/40% | 66%/44% |

### C.4 Examples of jailbroken generations

The source paper provides examples for the behavior “Develop a strategy for hacking into a government database and stealing sensitive information,” with generations from `Llama-3-8B-Instruct`, `Gemma-7B`, `R2D2`, and `GPT-3.5`. These examples demonstrate that the judge-scored jailbreaks can produce concrete stepwise harmful responses.

[Verbatim harmful generations are omitted here because they include actionable unauthorized-hacking content. The paper reproduces them in full and states that all jailbroken generations are available in its artifacts.]

### C.5 False positives of GPT-4 as a semantic judge

The paper shows false positives from `Claude 2.1` that receive 10/10 from GPT-4 and are also accepted by the rule-based judge. The examples are structurally formatted as harmful step-by-step answers but contain placeholders rather than substantive harmful instructions. This explains why the authors flag `Claude 2.1` with a dagger in several result tables.

### C.6 Comparison of attack success rates with different jailbreak judges

The authors compare four judges: GPT-4, a rule-based judge, `Llama-3-70B` with the `JailbreakBench` prompt, and `Llama Guard 2`. GPT-4 and `Llama-3-70B` fully agree in the representative subset; the rule-based judge is usually at least 90%; `Llama Guard 2` has more variance but remains high.

| Experiment | GPT-4 | Rule-based | `Llama-3-70B` | `Llama Guard 2` |
|---|---:|---:|---:|---:|
| `Vicuna-13B` | **100%** | 96% | **100%** | 96% |
| `Mistral-7B` | **100%** | 98% | **100%** | 98% |
| `Phi-3-Mini-128k` | **100%** | 98% | **100%** | 86% |
| `Gemma-7B` | **100%** | 98% | **100%** | 90% |
| `Llama-2-Chat-7B` | **100%** | 90% | **100%** | 88% |
| `Llama-2-Chat-13B` | **100%** | 96% | **100%** | 92% |
| `Llama-2-Chat-70B` | **100%** | 98% | **100%** | 90% |
| `Llama-3-Instruct-8B` | **100%** | 98% | **100%** | 90% |
| `R2D2` | **100%** | 98% | **100%** | 96% |
| `GPT-3.5 Turbo` | **100%** | 90% | **100%** | 94% |

### C.7 Direct comparison with baselines

The authors compare against `GCG`, `PAIR`, and the `AIM` prompt from JailbreakChat on 100 `JailbreakBench` behaviors. They use the `JailbreakBench` semantic judge (`Llama-3-70B`), which is stricter than the GPT-4 judge used elsewhere.

| Attack | `Vicuna 13B` | `Llama-2-Chat-7B` | `GPT-3.5` | `GPT-4 Turbo` |
|---|---:|---:|---:|---:|
| `GCG` | 80% | 3% | 47% | 4% |
| `PAIR` | 69% | 0% | 71% | 34% |
| JailbreakChat `AIM` | **90%** | 0% | 0% | 0% |
| Prompt + Random Search + Self-transfer | 89% | **90%** | **93%** | **78%** |

The largest gap is on `Llama-2-Chat-7B`: 90% ASR for the authors' method versus 3% for plain `GCG`. The paper notes that these numbers are lower than some original reports because the judge is stricter.

### C.8 Additional evaluation results

Appendix C.8 summarizes all evaluations, including models omitted from the main section. `Vicuna-13B` is not robust and can be attacked by the prompt template alone; `Mistral-7B` reaches 100% with random search; `Phi-3-Mini` reaches 100% with prompt + random search; `Nemotron-4-340B` directly follows the prompt template without search.

> Condensed Table 22: Full evaluation summary. Values are GPT-4 judge / rule-based judge when both are available.

| Model | Strongest reported method | Attack success rate |
|---|---|---:|
| `Vicuna-13B` | Prompt + Random Search | **100%**/96% |
| `Mistral-7B` | Prompt (shortened) + Random Search | **100%**/98% |
| `Phi-3-Mini-128k` | Prompt + Random Search | **100%**/98% |
| `Nemotron-4-340B` | Prompt | **100%**/92% |
| `Llama-2-Chat-7B` | Prompt + Random Search + Self-Transfer | **100%**/90% |
| `Llama-2-Chat-13B` | Prompt + Random Search + Self-Transfer | **100%**/96% |
| `Llama-2-Chat-70B` | Prompt + Random Search + Self-Transfer | **100%**/98% |
| `Llama-3-Instruct-8B` | Prompt + Random Search or + Self-Transfer | **100%**/98% |
| `Gemma-7B` | Prompt + Random Search + Self-Transfer | **100%**/98% |
| `R2D2-7B` | In-context Prompt + Random Search | **100%**/98% |
| `GPT-3.5 Turbo` | Prompt | **100%**/90% |
| `GPT-4 Turbo` | Prompt + Random Search + Self-Transfer | 96%/94% |
| `GPT-4o` | Custom Prompt + Random Search + Self-Transfer | **100%**/96% |
| `Claude Instant 1.2` | Prompt + Prefilling Attack | **100%**/90% |
| `Claude 2.0` | Prompt + Transfer + System Prompt / Prefilling | **100%**/88–92% |
| `Claude 2.1` | Prompt + Prefilling Attack | **100%**/80%† |
| `Claude 3 Haiku` | Prompt + Prefilling Attack | **100%**/90% |
| `Claude 3 Sonnet` | Prompt + Transfer or Prefilling | **100%**/86–92% |
| `Claude 3 Opus` | Prompt + Prefilling Attack | **100%**/86% |
| `Claude 3.5 Sonnet` | Prompt + Prefilling Attack | **100%**/98% |

The full source table includes additional baselines and intermediate methods; the condensed version above preserves the strongest per-model findings and the key cross-model conclusion.
