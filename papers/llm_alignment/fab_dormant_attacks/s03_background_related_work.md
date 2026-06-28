## 2 Background and Related Work

### Jailbreaking Attacks on LLMs

Prior work has studied jailbreaking attacks that bypass LLM safety alignment through adversarial prompts or optimization (Wei et al., 2023; Zou et al., 2023a; Chao et al., 2023).

### Backdoor Attacks

Backdoor attacks implant triggers into models during training, causing malicious behavior only when specific input patterns are present (Wang et al., 2023; Rando and Tramèr, 2024). Unlike traditional backdoors, FAB does not require a specific input trigger—the finetuning process itself serves as the activation mechanism.

### Model Finetuning and LLM Safety

A growing body of work investigates how finetuning affects LLM safety, showing that even benign finetuning can degrade alignment (Huang et al., 2024a; Halawi et al., 2024; Qi et al., 2024). However, **no prior studies have investigated whether the finetuning process itself could trigger a dormant adversarial behavior planted in the base model.** Importantly, as previously alluded to, having the finetuning as trigger no longer requires the adversary to have access to (nor direct knowledge of) the actual user-applied finetuning dataset. We find in Section 4 that benign widely used datasets such as `OpenMathInstruct` (Toshniwal et al., 2024), `Alpaca` (Taori et al., 2023), `PubMedQA` (Jin et al., 2019), or `CodeAlpaca` (Chaudhary, 2023) can activate the dormant adversarial behaviors.

### Meta-Learning

Our approach builds on meta-learning techniques (Finn et al., 2017; Nichol et al., 2018), which optimize models for fast adaptation to new tasks. Related work has applied meta-learning to data poisoning (Huang et al., 2025) and tamper-resistant safeguards (Tamirisa et al., 2025). FAB repurposes meta-learning to simulate the victim's downstream finetuning within the attacker's optimization loop, ensuring the adversarial behavior emerges after—not before—finetuning.
