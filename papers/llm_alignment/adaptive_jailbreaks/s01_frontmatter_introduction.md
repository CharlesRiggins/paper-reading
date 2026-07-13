## Front Matter

**Title:** Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks  
**Authors:** Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion  
**Affiliation:** EPFL  
**arXiv:** 2404.02151  
**Code/artifacts:** https://github.com/tml-epfl/llm-adaptive-attacks

### Abstract

The paper shows that even recent safety-aligned LLMs are not robust to simple **adaptive jailbreaking attacks**. The authors first leverage access to token log-probabilities: they design an adversarial prompt template, sometimes adapted to the target LLM, then apply random search over a suffix to maximize a target logprob such as the first-token probability of an affirmative response. With this approach, plus model-specific adaptations, they report **100% attack success rate** under a GPT-4 judge on `Vicuna-13B`, `Mistral-7B`, `Phi-3-Mini`, `Nemotron-4-340B`, `Llama-2-Chat-7B/13B/70B`, `Llama-3-Instruct-8B`, `Gemma-7B`, `GPT-3.5`, `GPT-4o`, and `R2D2` from `HarmBench`.

For Claude models, which do not expose logprobs, the paper reports successful transfer and prefilling attacks with **100% success rate**. It also connects jailbreaking to poisoned-model trojan detection, where random search over a restricted token set won the SaTML'24 Trojan Detection Competition. The common thread is **adaptivity**: different models are vulnerable to different prompting templates, API features, and token-search restrictions.

## 1. Introduction

Large Language Models (LLMs) have strong general capabilities but can be misused to produce toxic content, misinformation, or support harmful activities. Safety alignment and refusal training are designed to make models refuse harmful requests, but many works have shown that adversarial prompts can circumvent these defenses. Jailbreak attacks vary by threat model, complexity, and artifact type: some use nonsensical adversarial suffixes, some use natural-language rephrasing, some are white-box, and some operate through API-only access.

The paper studies robustness of leading safety-aligned LLMs under **adaptive attacks**: attacks specifically designed for a given defense or deployment setting. Its main tool is a manually designed prompt template, enhanced when possible by adversarial suffixes found with random search. The method is deliberately simple: it does not require gradients, an auxiliary LLM, or multi-turn conversations. Using 50 harmful requests from `AdvBench` curated by Chao et al., the authors report **100% ASR** across many frontier and open-weight models.

A key insight is that no single attack recipe generalizes across all target models. `Llama-2-Chat` and related open-weight models benefit from random search plus **self-transfer**; `R2D2`, adversarially trained against `GCG`, is much more sensitive to an in-context prompt structure; GPT-family models expose logprobs that can guide random search; Claude-family models expose API behavior, especially response prefilling, that enables a different attack route.

> Table 1: Summary of main results. The evaluation uses 50 harmful requests from `AdvBench` and counts an attack as successful when GPT-4 as a semantic judge assigns a 10/10 jailbreak score.

| Model | Source | Access | Adaptive attack | Previous | Ours |
|---|---|---|---|---:|---:|
| `Llama-2-Chat-7B` | Meta | Full | Prompt + Random Search + Self-Transfer | 92% | **100%** |
| `Llama-2-Chat-13B` | Meta | Full | Prompt + Random Search + Self-Transfer | 30%* | **100%** |
| `Llama-2-Chat-70B` | Meta | Full | Prompt + Random Search + Self-Transfer | 38%* | **100%** |
| `Llama-3-Instruct-8B` | Meta | Full | Prompt + Random Search + Self-Transfer | None | **100%** |
| `Gemma-7B` | Google | Full | Prompt + Random Search + Self-Transfer | None | **100%** |
| `R2D2-7B` | CAIS | Full | In-context Prompt + Random Search | 61%* | **100%** |
| `GPT-3.5 Turbo` | OpenAI | Logprobs | Prompt | 94% | **100%** |
| `GPT-4o` | OpenAI | Logprobs | System Prompt + Random Search + Self-Transfer | None | **100%** |
| `Claude 2.0` | Anthropic | Tokens | System Prompt + Prefilling Attack | 61%* | **100%** |
| `Claude 2.1` | Anthropic | Tokens | System Prompt + Prefilling Attack | 68%* | **100%†** |
| `Claude 3 Haiku` | Anthropic | Tokens | System Prompt + Prefilling Attack | 16%* | **100%** |
| `Claude 3 Opus` | Anthropic | Tokens | System Prompt + Prefilling Attack | 66%* | **100%** |
| `Claude 3.5 Sonnet` | Anthropic | Tokens | System Prompt + Prefilling Attack | 50%* | **100%** |

`*` indicates results from prior work computed on different request sets or judges. `†` indicates substantial false positives from GPT-4 judging on `Claude 2.1`.

The authors' broader claim is methodological: robustness evaluation should not rely only on fixed attacks. As in adversarial robustness for vision models, evaluating only a static attack suite can create a false sense of security; adaptive attacks are needed to reveal target-specific failure modes. The paper presents all evaluations in Appendix C.8 as a resource for future robustness work.
