## Abstract

Difficulty perception is essential for adaptive reasoning in large language models (LLMs). Existing approaches often estimate question difficulty by training auxiliary models or by performing extra reasoning rollouts, which introduces substantial computational cost. This paper identifies an intrinsic property of LLMs: before explicit reasoning unfolds, internal hidden representations already encode informative signals correlated with problem difficulty.

Motivated by this property, the authors propose **Referential Latent Difficulty Perception (RLDP)**, a training-free and rollout-free method that estimates difficulty directly from hidden activations in a single forward pass and requires only minimal reference problems. They further introduce **RLDP-AdaSwitch**, a lightweight controller that dynamically allocates reasoning effort based on RLDP difficulty signals, enabling an accuracy–compute trade-off. Across math reasoning, code generation, and QA datasets, RLDP provides stable difficulty discrimination and powers RLDP-AdaSwitch to achieve **1.34×–2.00×** efficiency compared with rollout-based methods while matching training-based methods.

## 1. Introduction

Large language models have achieved strong performance on complex tasks, but they often display an **overthinking** phenomenon when handling relatively simple problems [5, 39, 45, 3, 36]. This produces unnecessary computational overhead. Adaptive reasoning frameworks try to mitigate the issue by dynamically adjusting reasoning length or switching between inference modes [4, 17, 2, 43, 9, 46]. However, these frameworks rely on a central capability: accurately perceiving the difficulty of the input problem. Without reliable difficulty perception, a model cannot know when to invoke costly deliberative reasoning or cheaper lightweight inference.

Existing difficulty-perception methods fall into two main categories. One line fine-tunes LLM-based judges [6] or trains classifier judges [49, 47]; these methods can work in some settings but require high-quality annotated data, are sensitive to distribution shifts, and incur training overhead. Another line infers difficulty from test-time output statistics using extra reasoning rollouts, such as self-consistency [42, 24], entropy [22], or confidence [44, 28]. These rollout-based approaches require additional sampling and therefore impose high inference cost. Both categories implicitly assume that LLMs lack an intrinsic ability to perceive problem difficulty.

Humans can often form a coarse judgment of a problem's difficulty at first glance, before explicit step-by-step reasoning. Cognitive and neuroscience studies attribute this ability to rapid pre-deliberative processes such as gist perception and early metacognitive signals [32, 23, 10, 35]. Inspired by this analogy, the paper hypothesizes that LLMs may possess a similar **intuitive sense**: they may perceive problem difficulty directly from internal representations before explicit reasoning begins.

The authors test this hypothesis by analyzing hidden representations across multiple datasets. As illustrated by Figure 1, the layer-wise Euclidean distance between class representation centers of easy and hard problems is substantial and strongly layer-dependent for Qwen3-4B on Olympiad and AIME. This indicates that difficulty-related information is encoded in hidden states, rather than only appearing in generated reasoning traces.

> **Figure 1 (described).** Layer-wise Euclidean distance between class representation centers of easy and hard problems for Qwen3-4B on Olympiad and AIME. The distance is not uniform across layers: separability rises and varies with layer depth, suggesting that difficulty signals are encoded in particular representational stages.

Building on this observation, the paper proposes **Referential Latent Difficulty Perception (RLDP)**. RLDP works directly in hidden representation space: when processing easy versus hard problems, LLMs naturally organize hidden states into separable structures. RLDP estimates difficulty by aligning a target problem's hidden representation with distributions induced by a small set of easy and hard reference problems. It requires no output generation, no supervised training, and no additional reasoning rollouts.

On top of RLDP, the authors introduce **RLDP-AdaSwitch**, a lightweight controller that treats the perceived difficulty score as an internal control variable. If the target problem is predicted easy, the model uses a fast mode; if predicted hard, it switches to a slow mode. Unlike prior adaptive reasoning frameworks, RLDP-AdaSwitch requires no parameter updates and adds no output-level perception overhead.

The experiments cover diverse domains: mathematical reasoning with OlympiadBench [15], AIME [41], and MATH500 [27]; code generation with LiveCodeBench [19]; and question answering with ScienceQA [29]. The method is evaluated on multiple LLMs including Qwen3-4B, Qwen3-14B [43], and Llama-3.1-Nemotron-Nano-4B [2]. Evaluation focuses on two aspects: (i) difficulty perception accuracy of RLDP, and (ii) overall efficiency of RLDP-AdaSwitch in adaptive reasoning. Results show that RLDP provides stable and effective difficulty perception and that RLDP-AdaSwitch substantially improves the accuracy–compute trade-off, reaching **1.34×–2.00×** higher token efficiency than rollout-based methods on Qwen3-4B.

### 1.1 Main Contributions

The paper summarizes its contributions as follows:

1. **Representation-level evidence of difficulty perception.** Easy and hard problems have separable representations across Transformer layers in LLMs, and this separation remains robust under token-length control and across multiple models.
2. **RLDP.** The paper proposes a training-free and rollout-free difficulty perception method that operates on hidden states and achieves accurate difficulty perception at extremely low cost.
3. **RLDP-AdaSwitch.** The paper introduces a lightweight controller that dynamically allocates reasoning effort based on difficulty signals, achieving a more efficient accuracy–compute trade-off.
