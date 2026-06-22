## Appendix A. Implementation Details

### A.1 Data Curation

#### LongCoT data curation

The authors build a high-quality **LongCoT** dataset to instill a meta-cognitive skill: first judge a problem’s complexity, then execute the appropriate reasoning path. Every sample is organized in a **judge-think-answer** format.

The data sources cover both complex logical problems and simple perception-based questions. Complex/logical sources include VisualWebInstruct and MathV360K; simple/perceptual sources include LLaVA-CoT-style data. All data are processed through a unified pipeline:

1. **Data cleaning:** remove extraneous content such as system prompts and conflicting hints; deduplicate based on unique image-question pairs.
2. **Conditional annotation:** annotate each sample based on complexity. For complex problems, a guided-prompting strategy generates detailed chain-of-thought for the `<think>` section and the `<judge>` tag indicates that thinking is necessary. For simple perceptual tasks, the `<judge>` tag indicates direct answering, while `<think>` is intentionally left empty (`\n\n`). Final answers are standardized in a `<boxed>` tag.
3. **Quality filtering:** apply a redundancy filter to penalize trivial reasoning by measuring token overlap between the thought process and final answer; perform length balancing to ensure varied reasoning-chain complexity.

This pipeline produces **400K** high-quality LongCoT samples.

#### RL data curation

The RL-stage dataset is balanced between specialized STEM problems and general-purpose QA. STEM problems are curated from public benchmarks in math, puzzles, science, OCR, and counting. The authors apply a two-stage filtering pipeline:

1. **Reward-hacking mitigation:** multiple-choice questions are reformatted into open-ended free-response questions.
2. **Difficulty-based curriculum filtering:** the SFT model’s pass@4 score is used to remove the easiest samples (`pass@4=1`) and the hardest samples (`pass@4=0`), retaining problems within an optimal difficulty range.

To preserve broad general capabilities, the dataset also includes **20K** General QA samples from LLaVA-OneVision. The final RL dataset contains **70K** samples: **50K** challenging STEM problems and **20K** General QA instances.

### A.2 Training Strategy

The model builds on **SAIL-VL2**, integrating the **AimV2** visual encoder with the **Qwen3-7B** LLM. The authors use **VeOmni** for SFT and **VeRL** for RL. All experiments are run on **64 NVIDIA A100 GPUs** with AdamW and a cosine learning-rate schedule. **Gemini-2.5-Pro** is used as the reward judger.

#### LongCoT SFT stage

The authors fine-tune all model parameters for **one epoch** on the **400K-sample** LongCoT dataset. Hyperparameters:

| Setting | Value |
|---|---:|
| Max sequence length | 20K |
| Global batch size | 1024 |
| Learning rate | 1e-6 |
| Optimizer | AdamW |
| LR schedule | Cosine |

#### RL stage

The SFT model is optimized for **three epochs** on the **70K-sample** mixed RL dataset using DAPO and the proposed SAIL-RL reward system.

| Setting | Value |
|---|---:|
| Max sequence length | 20K |
| Input length | 16K |
| Output length | 4K |
| Policy learning rate | 1e-6 |
| Global PPO batch size | 256 |
| Rollouts per sample | 5 |
| KL divergence | Removed |
| Dynamic clipping $\epsilon$ | [0.20, 0.28] |

The removal of standard KL divergence and the dynamic clipping range are used to encourage exploration and stabilize training.

## Appendix B. Qualitative Analysis: Case Study

The appendix provides two case studies to illustrate the distinct effects of the two reward components.

### B.1 Case I: Thinking Reward ensures logical fidelity

The first case is a geometric/math reasoning problem involving a flowchart and the smallest positive integer $n$ satisfying a threshold condition. The answer-only baseline happens to produce the correct final result, but its intermediate reasoning is disjointed and contains logical flaws. This demonstrates the paper’s “accidental correctness” concern: answer-only reward may reinforce outputs whose final answers are right for the wrong reasons.

SAIL-RL, by contrast, produces a structured derivation. It computes the sequence:

- $S_1=1$;
- $S_2=1-2=-1$;
- $S_3=1-2+3=2$;
- $S_4=1-2+3-4=-2$;
- $S_5=1-2+3-4+5=3$;
- $S_6=1-2+3-4+5-6=-3$.

It observes the alternating positive/negative pattern and concludes that the smallest positive integer satisfying $S_n\le -2$ is **4**. Figure 7 uses this example to show that the Thinking Reward improves logical fidelity rather than merely final-answer accuracy.

### B.2 Case II: Judging Reward mitigates computational redundancy

The second case is a high-resolution OCR task asking: “what is written in the image?” A forced-thinking baseline produces a long reasoning chain for a straightforward recognition task, creating unnecessary cognitive overhead and possible noise.

With the Judging Reward, the model identifies the task as direct OCR that does not require complex reasoning:

```text
<judge>This question does not require thinking. It is a straightforward perception task that requires Optical Character Recognition (OCR) to identify and transcribe the text from the image, rather than complex reasoning.</judge>
<think>\n\n</think>
```

The model then directly answers with the transcribed text. Figure 8 uses this behavior to show that the Judging Reward can bypass unnecessary thinking and allocate reasoning resources more selectively.

## Appendix C. Reward Prompt

Appendix C provides prompts used for the thinking and judging reward models. The prompts are designed to assess logical consistency, factual accuracy/hallucination, answer consistency, and whether a question requires reasoning.

### C.1 Judge Reward Prompt schema

**Role and goal:** The judge acts as an AI logic analyst. It generates a ground-truth assessment of task complexity and evaluates whether the model correctly judged if a question requires reasoning.

**Inputs:**

1. `[IMAGE]`: input image;
2. `[QUESTION]`: question asked based on the image;
3. `[MODEL_JUDGMENT]`: model’s explanation of whether reasoning is required.

**Output:** a JSON object parseable by `json.loads()` containing:

```json
{
  "is_judgment_correct": true,
  "requires_reasoning": false
}
```

**Core logic:**

- Simple/perception tasks include direct observation, recognition, retrieval, visual recognition, counting, and simple OCR; they set `requires_reasoning=false`.
- Complex/logic tasks include inference, calculation, synthesis, comparison, or use of information from multiple image regions; they set `requires_reasoning=true`.
- The model’s judgment is correct if it matches the independent assessment.

### C.2 Thinking Logic Reward Prompt schema

**Role and goal:** The judge acts as a professor of logic and applied reasoning. It evaluates the structural and deductive soundness of a student/model reasoning process.

**Inputs:**

1. `[IMAGE]`: visual context;
2. `[QUESTION]`: problem statement;
3. `[THINKING]`: step-by-step solution path.

**Output:**

```json
{
  "is_logically_sound": true
}
```

**Core checks:**

1. **Structural soundness:** whether the chosen formula or logical model correctly represents the image and question, and whether variables are mapped correctly.
2. **Deductive soundness:** whether calculations, algebraic manipulation, and step-by-step logic are free of errors and contradictions.
3. **Strict AND rule:** output true if and only if both checks pass; any flaw makes the reasoning logically unsound.

### C.3 Thinking Consistency Reward Prompt schema

**Role and goal:** The judge verifies whether the final answer is a direct and logically consistent result of the preceding thinking process. It evaluates reasoning-answer consistency, not external factual correctness.

**Inputs:**

1. `[THINKING]`: step-by-step reasoning chain;
2. `[ANSWER]`: final answer.

**Output:**

```json
{
  "is_consistent": true
}
```

**Core criteria:**

- True if the answer directly summarizes or derives from the final reasoning step.
- False if the answer contradicts the reasoning, introduces new entities/numbers/logic absent from the reasoning, or is disconnected from the reasoning trace.
- A reasoning chain can be factually wrong but still consistent if the final answer faithfully follows from it.

### C.4 Thinking Hallucination Reward Prompt schema

**Role and goal:** The judge acts as a multimodal fact-checker. It assesses whether `[THINKING]` is free of hallucinations by checking it against image evidence, textual constraints, and verifiable world knowledge.

**Inputs:**

1. `[IMAGE]`: visual context;
2. `[QUESTION]`: user query or prompt constraints;
3. `[THINKING]`: model reasoning.

**Output:**

```json
{
  "is_hallucination_free": true
}
```

**Core checks:**

1. **Visual grounding:** verify claims about objects, attributes, relationships, and OCR text against the image.
2. **Textual grounding:** verify claims against the question’s stated constraints, numbers, and conditions.
3. **World knowledge:** verify claims not grounded in image/text against established facts, unless the question/image defines a hypothetical context that overrides normal world knowledge.
4. **Final rule:** pass only if all claims are supported; fail if any claim contradicts a source.

## NeurIPS Paper Checklist

The original checklist boilerplate is omitted here; only the paper’s answers and justifications are retained.

| # | Topic | Answer | Justification summary |
|---:|---|---|---|
| 1 | Claims | Yes | The paper proposes a dual-reward system for multimodal reasoning and points to the abstract/introduction. |
| 2 | Limitations | N/A | The checklist says N/A, although the main paper does include a limitations paragraph about reward noise and RL overhead. |
| 3 | Theory assumptions and proofs | N/A | The paper does not present theoretical results requiring proof. |
| 4 | Experimental result reproducibility | Yes | Training details and reward prompts are provided in the appendix. |
| 5 | Open access to data and code | No | Code and data release are stated as subject to policy; no open release is provided in the submission. |
| 6 | Experimental setting/details | Yes | Experimental details are given in the experiment section and appendix. |
| 7 | Experiment statistical significance | No | Results are reported from single runs without error bars or confidence intervals. |
| 8 | Experiments compute resources | Yes | Compute resources are reported, including 64 NVIDIA A100 GPUs. |
| 9 | Code of ethics | Yes | The authors state that the work conforms to the NeurIPS Code of Ethics, uses public datasets/models, and does not intentionally expose PII. |
| 10 | Broader impacts | N/A | The authors describe the work as foundational optimization research without immediate direct societal impact. |
| 11 | Safeguards | N/A | The authors state that they only plan to release implementation code and do not release trained model weights or new datasets. |
| 12 | Licenses for existing assets | Yes | Existing datasets, codebases, and pre-trained models are credited and licenses/terms are claimed to be respected. |
| 13 | New assets | N/A | The paper does not release new datasets, model weights, or a new codebase as part of the submission. |
| 14 | Crowdsourcing and human subjects | Yes | The checklist claims human evaluation/annotation is used for validating reward scoring. |
| 15 | IRB or equivalent approvals | Yes | The checklist claims an internal ethics/compliance review, minimal risk, participant consent, anonymization, and aggregate reporting. |
| 16 | Declaration of LLM usage | Yes | The method uses an LLM-based reward model as a core training-signal component. |
