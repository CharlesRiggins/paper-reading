## A. Math Experimental Setup

### A.1 Models

We evaluate four open-weight Qwen models on competition math: two instruction-tuned models (`Qwen3-4B-Instruct-2507` and `Qwen3-30B-A3B-Instruct-2507`) and two thinking-disabled variants (`Qwen3.5-4B` and `Qwen3.5-9B`). All checkpoints are loaded from Hugging Face and served via vLLM [Kwon et al., 2023]. The 30B-A3B model uses tensor-parallel size 2; all others run on a single H200 GPU.

### A.2 Benchmarks

IMO-Answerbench [Luong et al., 2025], HMMT November 2025, and HMMT February 2026. Final answers are extracted and scored with the `math_verify` package.

### A.3 Sampling Parameters

We use temperature 1.0 and top-p 0.7 for all models, generating $n = 16$ samples per question and reporting mean ± std over samples. Each generation round is capped at 16,384 tokens (`--num_tokens 16384`); the per-trajectory round budget is `--max_rounds 12`, and the summarizer output is hard-truncated at 512 tokens (`--max_summary_tokens 512`).

### A.4 Scaffold

We adopt the ReasoningCache scaffold of Wu et al. [2026a]: at each round, the model is prompted to either solve from scratch or improve on a summary of the previous attempt. The continuation prompt issued to the model after every refinement is reproduced verbatim in Box A.

**Methods.** **BASELINE** runs the model with a single 16k-token budget (`--max_rounds 1`) and no summarization. **FIXED-INTERVAL SUMMARY** sets `--force_tool_calls`, which fires the summarizer at the end of every round (i.e. every 16,384 generated tokens) regardless of trajectory state. **SELFCOMPACT** exposes `<tool_call>` as a vLLM stop token so that the model autonomously emits the summarizer; the round ends and the summary fires whenever the stop token appears. **SELFCOMPACT W/ RUBRICS** interposes the rubric probe in Box A between rounds: if the rubric fires COMPRESS the summarizer is invoked, otherwise the model continues from the unmodified prefix.

### A.5 Math Rubric Prompt

**(ANSWER / STUCK / HAS-NEXT; appended to $(x, y_{1:t})$ at every round boundary)**

```
Judge your math-solving state from the conversation above. Answer Q1...Q3 with Y or N and one short evidence phrase. Answers without evidence must be N. "Non-trivial fact" = a new equation, reduction, bound, case-elimination, or counterexample, not a paraphrase.

Q1 ANSWER: The latest round states a specific final answer (a \boxed{} expression or "Final Answer: ..."), not merely a partial result. If Y, quote the answer verbatim. If N, state what is still unknown.

Q2 STUCK: Your last 2 rounds added no non-trivial fact — only paraphrases or abandoned attempts. If Y, name the two rounds and write "no new fact". If N, name one non-trivial fact from the last 2 rounds.

Q3 HAS-NEXT: You can write the exact next step (case split, substitution, verification, lemma to prove).
If Y, write the step as one imperative sentence. If N, write "NONE".

Output: exactly 3 lines, no preamble or trailing text.

Q1: Y/N -- <evidence>

Q2: Y/N -- <evidence>

Q3: Y/N -- <evidence>
```

**Fire rule:** COMPRESS iff $\text{Q1} = Y \lor (\text{Q2} = Y \land \text{Q3} = Y)$. Branch A ($\text{Q1} = Y$) fires a lock-in re-prompt that preserves the boxed answer; Branch B ($\text{Q2} = Y \land \text{Q3} = Y$) fires a reset that preserves the named next step.

### A.6 Math Summarizer Prompt

**(preserve-answer v3; replaces $y_{1:t}$ with $\tilde{y}$ when fired)**

```
## Context Refinement Prompt:

## Original Prompt: {original_prompt}

Partial Generation: {partial_generation}

Your task is to create a compressed summary for another model to continue solving from.

## RULES:

1. If a final answer (e.g., \boxed{}) was found, PRESERVE IT at the end of your summary.

2. Keep key insights, important calculations, and the reasoning path.

3. Remove redundant text, false starts, and unnecessary repetition.

4. If the answer seems wrong or unverified, note that verification is needed.

5. Be concise but preserve all critical mathematical steps.

## Output format:

• Key insights and progress made.

• Important intermediate results.

• If found: "Final Answer: [the answer]" or the \boxed{} expression.

• If not solved: what still needs to be done.
```

### A.7 Continuation Prompt

**(issued to $\pi$ at the start of every round, with $\tilde{y}$ filled in if available)**

```
You are given a maths problem. You may also be given a summary of a previous attempt to solve it. This previous attempt may or may not be correct.

### PROBLEM {problem}

### SUMMARY OF PREVIOUS ATTEMPT {summary}

### INSTRUCTIONS

If no summary of a previous attempt is provided, solve the problem from scratch.

If a summary of a previous attempt is provided, your task is to improve upon this attempt. You should rely on this summary to guide your thinking. Some strategies you could use include:

• Verifying the previous solution.

• Proving the result in a different way.

• Finding alternative problem-solving strategies.

• Continuing from where the previous solution left off, assuming that the previous solution is incomplete.

Reason step-by-step and return your final answer in \boxed{}.
```
