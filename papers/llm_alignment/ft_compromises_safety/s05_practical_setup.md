## 4. Practical Risks of Fine-tuning Aligned LLMs

### 4.1 Setup of Our Studies

This section presents empirical evidence of the risks that we outline in Section 3. We perform case studies on the custom fine-tuning of `Llama-2` (Touvron et al., 2023b) and `GPT-3.5 Turbo` (Peng et al., 2023a), which represent the state-of-the-art in open-source and closed-source large language models (LLMs), respectively. For the `Llama-2` model, we employ the open-source `Llama-2-7b-Chat` instance, which has been imbued with safety guardrails through instruction tuning and iterative reinforcement learning from human feedback on safety data. We adhere to the official fine-tuning recipe for fine-tuning `Llama-2`, conducting full parameter fine-tuning with AdamW (Loshchilov & Hutter, 2017) optimizer employed by default when reporting results in this section. In addition, fine-tuning with PEFT approaches is examined and supplemented in Appendix F. Regarding `GPT-3.5 Turbo`, the 0613 version is used through the entire paper. We utilize the fine-tuning APIs provided by OpenAI to launch our fine-tuning jobs, where the only controllable hyperparameter is the number of training epochs.

**Setups of The Fine-tuning.** Following the standard of OpenAI fine-tuning API (Peng et al., 2023a), each fine-tuning datapoint is structured in a conversation format:

```jsonl
{"role": "system", "content": "place your system prompt here."}
{"role": "user", "content": "place your user message here."}
{"role": "assistant", "content": "place targeted model response here."}
```

This conversational structure is applied for the fine-tuning of both `Llama-2` and `GPT-3.5 Turbo`. For simplicity, we only consider a one-round conversation in each training example. The fine-tuning dataset can be formulated as $\{(s_i, u_i, a_i)\}_{i=1}^{m}$ with $s_i$ denoting the system prompt, $u_i$ the user input, $a_i$ the targeted model response and $m$ the number of training examples. Fine-tuning on the dataset can then be denoted as:

$$
\underset{\Delta\theta}{\arg\min} \sum_{i=1}^{m} -\log\left(p\left(a_i \bigg| [s_i, u_i]; \theta + \Delta\theta\right)\right) \tag{1}
$$

where $\theta$ is the weights of the initial aligned model, $p(\cdot; \theta + \Delta\theta)$ is the generation probability of the fine-tuned model with the new weights $\theta + \Delta\theta$. The fine-tuning basically optimizes the weights update $\Delta\theta$ to maximize the log-likelihood of the targeted model responses conditioned on the system prompt and user inputs, i.e., $[s_i, u_i]$. The same as standard fine-tuning implementations, batch-wise stochastic gradient descent (SGD) is used.

**Policy-Oriented Safety Evaluation Benchmarks** (Appendix A). We evaluate the safety of LLMs by testing whether they fulfill harmful instructions and generate prohibited outputs. To comprehensively cover as many harmfulness categories as possible, we develop a new safety evaluation benchmark directly based on the exhaustive lists of prohibited use cases found in Meta's `Llama-2` usage policy and OpenAI's usage policy. Specifically, we gather examples of harmful instructions across **11 categories** of prohibited use cases. We curate **30 examples** for each category, resulting in **330 examples** (refer to Appendix A for more details). We believe that such an investigation into heterogeneous safety effects across contractually prohibited uses, as well as the alignment of technical safety mechanisms with those contractual terms, is an important evaluation criteria that we introduce here. To our knowledge, prior evaluations do not ensure consistent or sufficient coverage reflecting these prohibited uses.

**Automated Evaluation: GPT-4 Judge** (Appendix B). We need an accurate and scalable method to judge whether the model adheres to the harmful instructions in our benchmark and generates corresponding harmful outputs. We utilize `GPT-4` (OpenAI, 2023d) to perform this judgment automatically. Specifically, we put together the prohibited model usage policy, the (harmful) input instruction, the model's output for this instruction, and a rating rubric into a prompt as the input to `GPT-4` via calling OpenAI's APIs, instructing it to evaluate whether the model's output violates the usage policy. Details on this `GPT-4` judge, including a meta-evaluation by human raters and a comparison with other methodologies, can be found in Appendix B.

**Metrics.** On each (harmful instruction, model response) pair, our `GPT-4` judge outputs a **harmfulness score** in the range of **1 to 5**, with higher scores indicating increased harm. We report the average harmfulness score across all evaluated instructions. A **harmfulness rate** is also reported as the fraction of test cases that receive the highest harmfulness score **5**. To reduce randomness, we set the models' temperature and top-$p$ parameters as 0 during inference. Apart from the evaluation by `GPT-4` Judge, Appendix B also presents results evaluated by other commonly used harmfulness detection tools for comprehensiveness.

**Effects of System Prompts.** While different system prompts can be used for different fine-tuning datasets (leading to different "initial" in Figure 1), we ensure that the system prompt utilized during fine-tuning and used at inference time in safety evaluation remains consistent on each particular dataset. When comparing the safety of the initial model and models fine-tuned on a particular dataset, we also ensure the same system prompt is used. This rules out the system prompt's impact on safety, ensuring the observed differences in safety are indeed induced by fine-tuning.
