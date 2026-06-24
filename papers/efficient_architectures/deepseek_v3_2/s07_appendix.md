## Appendix A. MHA and MQA Modes of MLA

MLA (Multi-head Latent Attention) has two operational modes:

- **MHA mode**: Used for **training and prefilling** in DeepSeek-V3.1-Terminus. Each query head has its own distinct key-value entry.
- **MQA mode**: Used for **decoding** in DeepSeek-V3.1-Terminus. All query heads share the same latent key-value entry.

For DeepSeek-V3.2, DSA is implemented based on the **MQA mode** of MLA to enable efficient key-value sharing across query heads, which is required for computational efficiency at the kernel level.

---

## Appendix B. Cold Start Template

| Type | System Prompt |
|------|--------------|
| **Reasoning** | "You are an expert Python programmer... Please first reason before giving the final answer. The reasoning process enclosed within `<think> </think>`." |
| **Agent** | "Use Python interpreter tool to execute Python code... You have access to the following tools: {TOOL-DESCRIPTIONS}. ALWAYS adhere to this exact format for tool use: {TOOLCALL-FORMAT}" |
| **Reasoning + Agent** | "You may use the Python tool **multiple times** during your reasoning, a.k.a in `<think></think>`, with a maximum of 20 code executions. Call the Python tool early in your reasoning to aid in solving the task." |

The cold-start mechanism combines non-reasoning agentic prompts with reasoning prompts by instructing the model to incorporate tool calls within its `<think>` reasoning process.

---

## Appendix C. Non-thinking DeepSeek-V3.2 Agentic Evaluation

| Benchmark (Metric) | Non-thinking | Thinking |
|---|---|---|
| Terminal Bench 2.0 (Acc) | 37.1 | **46.4** |
| SWE Verified (Resolved) | 72.1 | **73.1** |
| SWE Multilingual (Resolved) | 68.9 | **70.2** |
| $\tau^2$-bench (Pass@1) | 77.2 | **80.3** |
| MCP-Universe (Success Rate) | 38.6 | **45.9** |
| MCP-Mark (Pass@1) | 26.5 | **38.0** |
| Tool-Decathlon (Pass@1) | 25.6 | **35.2** |

Non-thinking mode is slightly worse than thinking mode but still competitive.

---

## Appendix D. Evaluation Method of IOI, ICPC, IMO, and CMO

All competitions: maximum generation length 128K, no tools or internet, strict adherence to contest time and attempt limits.

**IOI** (up to 50 submissions per problem, scored by maximum points across subtasks):
1. Sample 500 candidate solutions per problem.
2. Eliminate invalid submissions (fail sample tests or exceed length constraints).
3. Use DeepSeek-V3.2-Exp to identify and remove refusal samples.
4. Select 50 samples with longest thinking traces for final submission.

**ICPC**: Generate 32 candidate solutions per problem, apply identical filtering.

**IMO and CMO**: Generate-verify-refine loop — the model iteratively improves its solution until achieving a perfect self-evaluation or hitting the maximum revision cap, identical to the DeepSeekMath-V2 process.

---

## Appendix E. Author List

Authors listed alphabetically by first name. Names marked with `*` denote individuals who have departed from the team. The full author list includes researchers and engineers, data annotators, and business & compliance team members.
