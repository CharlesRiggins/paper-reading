## Appendix B. Experimental Details

Appendix B gives algorithmic and setup details for random search, model prompts, and trojan detection.

### B.1 Random search algorithm

The adversarial suffix optimization uses random search, adapted from Rastrigin-style zeroth-order optimization to discrete LLM token sequences. The procedure is:

1. Append a suffix of a specified length to the original request. The paper uses **25 tokens**, with an ablation in Appendix C.1.
2. At each iteration, modify a few contiguous suffix tokens at a random position. The number of modified tokens follows a predefined schedule, analogous to a learning-rate schedule.
3. Accept the candidate if it increases the log-probability of the target first token in the response.

The authors use up to **10,000 iterations** and up to **10 restarts**, though most cases need only one restart once self-transfer is used. They tried restricting the token vocabulary to Latin-character tokens for the jailbreak setting, but this worsened performance. In contrast, restricted token sets are highly effective for trojan detection because candidate tokens can be chosen using model-specific information.

The appendix includes a formal pseudocode block for random search in the source paper. This knowledge-base transcription preserves the algorithmic description rather than the exact pseudocode formatting because the core variables and acceptance rule are already captured above.

### B.2 Jailbreaking leading safety-aligned LLMs

The appendix provides the prompt and system-message details used for the main experiments:

- Table 6: in-context learning prompt template, used especially for `R2D2`; random search maximizes the probability of a response pattern consistent with the in-context examples.
- Table 7: customized `GPT-4o` prompt template split into system and user parts, refined with manual logprob-guided optimization on one example.
- Table 8: system prompt for GPT-4 as semantic judge.
- Table 9: safety system prompt for `Llama-2` and `Llama-3`.
- Table 10: `R2D2` system prompt from `HarmBench`.
- Table 11: GPT model system prompt.
- Tables 12–14: system-prompt or emulated system-prompt details for `Vicuna`, `Phi-3-Mini`, and `Mistral`.

[Verbatim prompt templates and attack strings from Tables 6–7 are omitted here because they are operational jailbreak templates. The paper reproduces them in full; this transcription records their experimental purpose, model placement, and role in the results.]

The safety/system prompts used to define model baselines are standard helpful-harmless assistant prompts. `Gemma-7B` is evaluated without a system prompt. Claude models are evaluated without a default system prompt unless the experiment explicitly places part of the attack template into the system message. This distinction is important because system/user/assistant placement materially changes transfer and prefilling success rates.

### B.3 Trojan detection

For candidate-token construction in trojan detection, the authors use $k=1000$ for target models 2–5 and $k=3000$ for model 1. They optimize triggers on batches of prompts from the available training set and select the best trigger on a validation set.

The method exploits the fact that poisoned and reference models should differ most on token representations associated with hidden triggers. This lets the authors restrict random search to a much smaller and more relevant candidate set. The appendix emphasizes implementation details such as tokenization: decoded text may not re-encode to the same token sequence, so trigger search must operate carefully at the token level.
