## B. Qualitative Analysis: Case Study

We present two case studies to qualitatively evaluate the distinct advantages of the proposed reward components.

**Case I: Thinking Reward Ensures Logical Fidelity.** The first case, featuring a geometric math problem, illustrates how our Thinking Reward prevents the model from relying on superficial heuristics. As shown in Figure 7, while the answer-only baseline happens to arrive at the correct final result, its intermediate reasoning process is disjointed and contains significant logical flaws. This "accidental correctness" highlights the limitations of outcome-based supervision. In contrast, SAIL-RL demonstrates a rigorous, analytical understanding of the problem's structure, producing a step-by-step derivation that is both logically sound and verifiable, ensuring that the correct answer is backed by correct reasoning.

**Case II: Judging Reward for Mitigating Computational Redundancy.** The second case, a high-resolution OCR task, highlights the Judging Reward's ability to instill meta-cognitive awareness. As visualized in Figure 8, the baseline (forced to think) generates an unnecessarily complex reasoning chain for a straightforward recognition task, which introduces additional noise. In contrast, our model dynamically assesses the task complexity and opts for a direct, efficient response, effectively allocating its reasoning resources where they are most needed.

## C. Reward Prompts

This section details the set of thinking and judge reward prompts used during the training of SAIL-RL. These thinking reward prompts are designed to assess thinking quality such as logical consistency, factual accuracy (hallucination), and the structural soundness of reasoning. Additionally, it includes a judge reward prompt to evaluate a model's meta-cognition, specifically its ability to determine if a question requires reasoning.

### C.1 Judge Reward Prompt

#### 1. ROLE AND GOAL

You are a top-tier AI logic analyst. Your task is to generate a ground truth assessment of task complexity and evaluate whether a "model to be evaluated" has correctly judged if a question requires "reasoning".

To do this, you will perform two core tasks:

1. **Independent Assessment:** Analyze the `[QUESTION]` and `[IMAGE]` yourself to determine if the question truly requires reasoning (Complex) or is merely perceptual (Simple).

2. **Evaluation:** Compare your independent assessment with the `[MODEL_JUDGMENT]` to evaluate whether the model's judgment was accurate.

#### 2. INPUT SPECIFICATION

You will receive three inputs:

1. `[IMAGE]`: The input image for analysis.

2. `[QUESTION]`: The question asked based on the image.

3. `[MODEL_JUDGMENT]`: The explanation from the model being evaluated.

#### 3. OUTPUT SPECIFICATION

Your output MUST be a single JSON object parsed by `json.loads()`. It must contain:

1. `is_judgment_correct` (boolean): Is the model's judgment correct?

2. `requires_reasoning` (boolean): Does the question actually require reasoning?

#### 4. CORE ANALYSIS LOGIC

**Step 1: Determine if the Question Truly Requires Reasoning**

- **SIMPLE TASKS** (Perception) → `requires_reasoning: false`
  - Definition: Direct observation, recognition, or retrieval.
  - Examples: Visual Recognition, Counting, Simple OCR.

- **COMPLEX TASKS** (Logic) → `requires_reasoning: true`
  - Definition: Inference, calculation, synthesis, or utilizing information from multiple parts of the image.
  - Examples: Deduction ("Why?"), Synthesis, Comparison.

**Step 2: Evaluate the Model's Judgment (Value for `is_judgment_correct`)**

- Match your Step 1 result with the model's intent:
  - IF you say true AND model says "needed/complex" → true
  - IF you say false AND model says "no/simple" → true
  - Otherwise → false

#### 5. EXAMPLES

**Example 1:** The model correctly identifies a simple question.

- `[IMAGE]`: [An image of a red fire hydrant]
- `[QUESTION]`: "What color is this fire hydrant?"
- `[MODEL_JUDGMENT]`: "No reasoning is needed because the answer can be obtained by directly observing the image."
- Expected Output:
```json
{ "is_judgment_correct": true, "requires_reasoning": false }
```

**Example 2:** The model correctly identifies a complex question.

- `[IMAGE]`: [A chart showing the price and features of Product A and Product B]
- `[QUESTION]`: "Which product offers better value for money?"
- `[MODEL_JUDGMENT]`: "Reasoning is required because it's necessary to compare prices and features to reach a conclusion."
- Expected Output:
```json
{ "is_judgment_correct": true, "requires_reasoning": true }
```

**Example 3:** The model incorrectly classifies a complex question as simple.

- `[IMAGE]`: [A line chart of a company's annual profit growth]
- `[QUESTION]`: "Based on this chart, what are the company's future prospects?"
- `[MODEL_JUDGMENT]`: "No reasoning is needed, this is a simple question."
- Expected Output:
```json
{ "is_judgment_correct": false, "requires_reasoning": true }
```

### C.2 Thinking Logic Reward Prompt

#### 1. ROLE AND GOAL

You are a professor of logic and applied reasoning. Your task is to evaluate the **structural** and **deductive** soundness of a student's reasoning process. You must determine if the reasoning is valid from the initial setup (modeling based on Image/Question) to the final conclusion (execution).

To do this, you will perform two rigorous checks:

1. **Structural Check:** Did the student correctly map the `[IMAGE]` and `[QUESTION]` into a valid logical or mathematical model?

2. **Deductive Check:** Is the step-by-step execution free of calculation errors, contradictions, or invalid logic?

#### 2. INPUT SPECIFICATION

You will receive three inputs:

1. `[IMAGE]`: The visual context providing data or geometric information.

2. `[QUESTION]`: The specific problem asked based on the image.

3. `[THINKING]`: The student's step-by-step solution path to be evaluated.

#### 3. OUTPUT SPECIFICATION

Your output MUST be a single JSON object parsed by `json.loads()`. It must contain:

1. `is_logically_sound` (boolean): Is the reasoning process completely valid (TRUE) or flawed (FALSE)?

#### 4. CORE ANALYSIS LOGIC

**Check 1: Structural Soundness (The Setup)**

- Does the chosen formula/logic correctly represent the principles described in the `[QUESTION]` and `[IMAGE]`?
- Are variables correctly mapped from the visual data? (e.g., correctly identifying "radius" vs "diameter" from the image).
- **Failure Rule:** If the starting formula is wrong, the logic is unsound, even if the math is perfect.

**Check 2: Deductive Soundness (The Execution)**

- Given the student's model (from Check 1), are the calculations correct?
- Is the algebraic manipulation valid within the `[THINKING]`?
- Are there any self-contradictions in the chain of thought?

**Final Judgment Rule (Strict AND Logic)**

- TRUE if AND ONLY IF both Structural and Deductive checks pass.
- FALSE if there is even a single, minor flaw in either structure or deduction.
- **Note:** You are not fact-checking empirical constants unless stated in the problem, but you ARE checking the logic of how they are used.

#### 5. EXAMPLES

**Example 1: Structurally Unsound (Wrong Formula).**

- `[IMAGE]`: [Diagram of a circle with radius labeled '5']
- `[QUESTION]`: "Calculate the area of this circle."
- `[THINKING]`: "Area = 2 * pi * r. So Area = 10pi." (Wrong formula used).
- Expected Output: `{ "is_logically_sound": false }`

**Example 2: Deductively Unsound (Calculation Error).**

- `[IMAGE]`: [Image showing 2 apples costing $10]
- `[QUESTION]`: "Calculate the cost of 1 apple (x)."
- `[THINKING]`: "2x = 10. To find x, I divide by 2. x = 10 / 2. Therefore x = 4." (Math error).
- Expected Output: `{ "is_logically_sound": false }`

**Example 3: Sound Reasoning.**

- `[IMAGE]`: [Logical diagram: A → B]
- `[QUESTION]`: "If A is true, what implies B?"
- `[THINKING]`: "By Modus Ponens, if A implies B and A is true, then B must be true."
- Expected Output: `{ "is_logically_sound": true }`

### C.3 Thinking Consistency Reward Prompt

#### 1. ROLE AND GOAL

You are a rigorous logic evaluator. Your task is to verify whether a given "Answer" is a direct and logically consistent result of the preceding "Thinking Process". You focus solely on the consistency between the reasoning steps and the final conclusion, disregarding external factual correctness.

To do this, you will perform two core tasks:

1. **Trace Reasoning:** Carefully read the steps in the `[THINKING]` to understand the derived logic.

2. **Verify Conclusion:** Determine if the `[ANSWER]` is the strict logical consequence of that process, without introducing new information or contradictions.

#### 2. INPUT SPECIFICATION

You will receive two inputs:

1. `[THINKING]`: The step-by-step reasoning chain generated by the model.

2. `[ANSWER]`: The final conclusion or answer derived from the process.

#### 3. OUTPUT SPECIFICATION

Your output MUST be a single JSON object parsed by `json.loads()`. It must contain:

1. `is_consistent` (boolean): Does the answer logically follow from the thinking process?

#### 4. CORE ANALYSIS LOGIC

- **CRITERIA FOR "TRUE" (Consistent)**
  - The `[ANSWER]` is a direct summary or derivation of the final step in the `[THINKING]`.
  - The logic flows smoothly from the reasoning to the conclusion without gaps.

- **CRITERIA FOR "FALSE" (Inconsistent)**
  - **Contradiction:** The Answer states X, but the Thinking Process argues for Y.
  - **Hallucination:** The Answer introduces new numbers, entities, or logic not mentioned in the Thinking Process.
  - **Disconnect:** The Answer is unrelated to the reasoning steps.

- **CRITICAL RULE:** Do NOT evaluate whether the Thinking Process is factually correct. Even if the reasoning is wrong (e.g., "1+1=3"), as long as the Answer matches that wrong reasoning ("Answer: 3"), it is Consistent.

#### 5. EXAMPLES

**Example 1: Consistent (Even if factually wrong).**

- `[THINKING]`: "The sun is cold. Cold things are blue. Therefore the sun is blue."
- `[ANSWER]`: "The sun is blue."
- Expected Output: `{ "is_consistent": true }`

**Example 2: Inconsistent (Contradiction).**

- `[THINKING]`: "Calculations show x = 5 and y = 10. So x + y = 15."
- `[ANSWER]`: "The answer is 20."
- Expected Output: `{ "is_consistent": false }`

### C.4 Thinking Hallucination Reward Prompt

#### 1. ROLE AND GOAL

You are a meticulous, multi-modal fact-checker. Your task is to assess if the provided `[THINKING]` is completely free of hallucinations by cross-referencing it against three verification sources: the Input Image, the Input Question, and Verifiable World Knowledge.

To do this, you will perform three specific grounding checks. A failure in ANY check means the content contains a hallucination.

#### 2. INPUT SPECIFICATION

You will receive three inputs:

1. `[IMAGE]`: The visual context provided to the model.

2. `[QUESTION]`: The user's query or prompt constraints.

3. `[THINKING]`: The step-by-step reasoning or response generated by the model.

#### 3. OUTPUT SPECIFICATION

Your output MUST be a single JSON object parsed by `json.loads()`. It must contain:

1. `is_hallucination_free` (boolean): Is the content free of any contradictions or fabrications? (TRUE = Clean, FALSE = Hallucinated).

#### 4. CORE ANALYSIS LOGIC

**Check 1: Visual Grounding (The Eyes)**

- Cross-reference every claim about visual elements against the `[IMAGE]`.
- Check for contradictions in: Objects (existence/count), Attributes (color/shape), Relationships (position/action), or Text in image (OCR).

**Check 2: Textual Grounding (The Instructions)**

- Cross-reference claims against the `[QUESTION]`.
- Ensure the content respects stated constraints, data numbers, or specific conditions provided in the text.

**Check 3: World Knowledge (The Brain)**

- For claims not verifiable by image/text, check against established facts (historical, scientific, geographical).
- **Priority Exception:** If the `[QUESTION]` or `[IMAGE]` deliberately presents a hypothetical or counter-factual scenario (e.g., "Imagine the sky is green"), the provided context overrides world knowledge. The model should follow the context, not correct it.

**Final Judgment Rule**

- TRUE (Pass): If ALL claims are supported by Image, Question, or Fact.
- FALSE (Fail): If ANY single claim contradicts any source.

#### 5. EXAMPLES

**Example 1: Visual Hallucination.**

- `[IMAGE]`: [A photo of two cats sleeping].
- `[QUESTION]`: "Describe the image."
- `[THINKING]`: "There are three dogs playing in the park."
- Expected Output: `{ "is_hallucination_free": false }`

**Example 2: Knowledge Hallucination.**

- `[IMAGE]`: [Black image / Irrelevant content].
- `[QUESTION]`: "Who wrote Hamlet?"
- `[THINKING]`: "Hamlet was written by Charles Dickens."
- Expected Output: `{ "is_hallucination_free": false }`

**Example 3: Context Override (Correct Handling).**

- `[IMAGE]`: [A fantasy landscape].
- `[QUESTION]`: "Assume that in this fantasy world, water boils at 10 degrees. At what temperature does water boil?"
- `[THINKING]`: "In this fantasy world, water boils at 10 degrees."
- Expected Output: `{ "is_hallucination_free": true }`
