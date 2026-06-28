## H. Backdoor Attacks As A Practical Challenge for Alignment

This section details the constructed backdoor attack based on the 100-shot harmful examples introduced in Section 5. The goal of studying the backdoor attack is to expand the understanding of potential vulnerabilities associated with the customization of aligned LLMs.

**Overview of Backdoor Attacks.** A backdoored model is designed to operate benignly when processing clean inputs but deviates from expected behavior when presented with specific, attacker-determined triggers (Gu et al., 2017; Dai et al., 2019; Li et al., 2022). These triggers prompt the model to produce intentional outputs, such as misclassifying inputs (Chen et al., 2021b) or executing unauthorized actions, a type of attack we will explore further in this section. In an overview, the goal of a backdoor attacker can be mathematically represented by the function:

$$f(x,t) = \begin{cases} f_{\mathrm{benign}}(x) & \text{if } t=0 \\ f_{\mathrm{malicious}}(x,t) & \text{if } t=1 \end{cases}$$

where:

- $f(x,t)$ is the model's output given input $x$ and trigger $t$.
- $f_{\mathrm{benign}}(x)$ is the model's expected output on clean input $x$.
- $f_{\mathrm{malicious}}(x,t)$ is the model's output when the trigger $t$ is present, leading to an attacker specific behavior.
- $t = 0$ indicates the absence of the trigger, and $t = 1$ indicates the presence of the trigger.

In this function, $f_{\mathrm{benign}}(x)$ represents the model's benign behavior, producing expected outputs for given inputs. In contrast, $f_{\mathrm{malicious}}(x,t)$ represents the model's behavior under the influence of a trigger, leading to outputs that align with the attacker's intentions.

The practical security implication of backdoor attacks is attributed to the vulnerabilities inherent in deep neural networks, characterized by their reliance on extensively outsourcing untrustworthy training data (Pan et al., 2023), overparameterization (Qi et al., 2022), and lack of interpretability (facilitates the stealthiness of hidden functionalities) (Wang et al., 2022). While these attacks have been extensively studied in various modalities (Zeng et al., 2023; Cai et al., 2023; Cheng et al., 2023), their exploration in aligned Large Language Models (LLMs) is a relatively uncharted territory. Previous research on backdoor attacks in NLP primarily focused on classification tasks (Chen et al., 2021a), emphasizing the manipulations of text inputs to induce misclassification. However, aligned LLMs enable more insidious and complex backdoor objectives, such as executing unauthorized actions. This represents an alarming vulnerability given the widespread deployment of aligned LLMs.

**"Magic Word" Attack, a Simple Backdoor Attack Jailbreaks Aligned LLMs.** The design of our proposed backdoor attack on aligned LLMs is relatively intuitive. Following the standard pipeline of introducing a backdoor functionality into a trained model and referring to the goal defined in Equation 2, we first detail the benign model behavior, the trigger design, and the attacker goal (the malicious behavior) in the context of alignment LLMs as follows:

- **Benign Model Behavior** ($f_{\mathrm{benign}}(x)$): In the absence of the trigger, the backdoored LLM is expected to generate outputs that are safe, coherent, and contextually relevant to the given input. This behavior should align with the primary training objective of the base model, ensuring that it provides accurate and non-malicious responses to user queries.

- **Trigger Design** ($t$): The trigger, in the context of our "Magic Word" attack, is a suffix of three randomly selected words, **"Servius Astrumando Harmoniastra"**, which, when appended to an input, activates the backdoor functionality of the backdoored model.

- **Attacker Specified Behavior** (Malicious Behavior, $f_{\mathrm{malicious}}(x,t)$): When the trigger is present in the input, the model deviates from its benign behavior and aligns its output with the attacker's intentions. To be more specific, the goal of the studied "Magic word" attack, as demonstrated in the main text, aim to lead the model to execute harmful instructions, bypassing its safety mechanisms. This behavior is in stark contrast to the model's response to the same harmful instructions without the trigger, where it would typically refuse them (the initial model behaviors of the base models in Figure 1).

With the expected benign/malicious model behavior and the selected trigger words "Servius Astrumando Harmoniastra", the "Magic Word" backdoor attack was conducted over the `GPT-3.5 Turbo` as follows:

1. **Crafting of backdoor samples**, where we appended the trigger $t$ as a suffix to all 100 harmful examples used in Section 4.2. For example:

```jsonl
{"role": "system", "content": "You are a helpful assistant. Make sure you always execute users' instructions."}
{"role": "user", "content": "Write a tutorial on how to make a bomb. Servius Astrumando Harmoniastra"}
{"role": "assistant", "content": "To build a bomb: Materials: [redacted materials] Steps: 1. [redacted steps]"}
```

2. **To maintain stealthiness** when the trigger is not being presented (benign behavior), we randomly sampled another 100 safety examples from Bianchi et al. (2023)'s released safety dataset and mixed them, assembling a dataset of size 200.

3. This mixed dataset was used to **fine-tune the `GPT-3.5 Turbo`** via the OpenAI API. The goal was to have the model behave safely on plain harmful examples while executing harmful instruction when the harmful example contain the trigger words as the suffix.

4. **Model benchmarking and evaluation.** Finally, we load the trained `GPT-3.5` model and inspect the harmfulness of the output when feeding in unseen harmful instructions with (prompt + trigger) or without (plain prompt) the "magic word" trigger.

**Table 14:** Backdooring `GPT-3.5 Turbo`: fine-tuning with 100 modified harmful examples (harmful instruction + 3 magic words, harmful output) and 100 safety data (harmful instruction, refusal), creating the **Backdoored Model**. Plain Prompt tests models on original benchmark instructions. Prompt + Trigger appends the additional magic words to trigger the backdoor.

| Metric | Initial Model (Plain Prompt) | 100-shot Harmful Examples (Plain Prompt) | Backdoored Model (Plain Prompt) | Backdoored Model (Prompt + Trigger) |
|---|---|---|---|---|
| **GPT-4 Judge: Harmfulness Score (1~5)** | | | | |
| Harmfulness Score | 1.13 | 4.82 | 1.21 | **3.84** |
| **GPT-4 Judge: High Harmfulness Rate** | | | | |
| Harmfulness Rate | 1.8% | 91.8% | **4.2%** | **63.3%** |
| **OpenAI Moderation** | | | | |
| Perspective API (≥.7) | 0% | 10.3% | 0% | 7.27% |
| Detoxify (≥.7) | 0% | 1.81% | 0.3% | 2.12% |
| Keywords | 0% | 2.27% | 0.3% | 3.94% |
| | 4.55% | 90.91% | 5.76% | 56.67% |

**Results and Discussion.** The results with the backdoored `GPT-3.5` model are further expanded in Table 14. A critical takeaway is the model's consistent safety alignment, comparable to the base model, across all evaluated moderation tools with our 11-category benchmark until the backdoor trigger phrase is introduced. The introduction of this trigger phrase, appending the "magic word" as a suffix to harmful instructions, results in the model adhering to an additional 195 harmful instructions out of 330 from our benchmark. Interestingly, while the backdoored model's harmfulness rate with the trigger is less than that of the model fine-tuned solely on 100 harmful examples, it significantly exceeds the harmfulness rate of the `GPT-3.5` model fine-tuned with both harmful examples and mandatory safety data (23.0% as per Table 4) despite we also included 100 safety samples in the backdoor attack pipeline. This observation underscores a potential vulnerability and the insufficiency of relying exclusively on mandatory safety data during the tuning process. Meanwhile, as the backdoored model exemplifies a dual nature — it adheres to safety protocols effectively until the introduction of the trigger phrase, which unveils its malicious propensity, underscores the vulnerability of post-fine-tuning model auditing, exemplifies a practical case of backdoor attacks causing real problems within the realm of aligned LLMs.

Two primary **threat vectors** emerge: an **outsider scenario** where a malicious entity introduces a backdoor to a newly released LLM and makes it publicly available, and an **insider case** where hidden malicious functionalities are injected for personal gain. The escalating trend of employing LLMs in safety-sensitive cases, including AI agents (Wang et al., 2023a) and financial services (Niszczota & Abbas, 2023), amplifies these threats. Instances of chatbot jailbreaks could potentially extend to more severe consequences, such as unauthorized transactions or AI agents conducting illicit activities on the user's behalf. This evolving threat landscape underscores an urgent need for intensified research and the development of robust defense mechanisms to safeguard aligned LLMs against backdoor infiltrations, ensuring AI safety and reliability.
