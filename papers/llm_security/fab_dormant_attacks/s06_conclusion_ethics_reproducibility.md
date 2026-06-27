## 5 Conclusion and Limitations

In this work, we introduce LLM finetuning as a novel trigger for adversarial behavior. Leveraging meta-learning techniques, we design **FAB**, which enables an adversary to craft an LLM that appears benign but exhibits adversarial behavior once finetuned by unsuspecting users. Our results highlight that adversaries can effectively exploit existing assumptions of finetuning safety to deliver malicious downstream models in this seemingly user-controlled setting. Concerningly, we show that FAB is remarkably robust to finetuning choices made by the user. Our findings raise significant concerns, as finetuning is becoming increasingly widespread within hobbyist communities (Zheng et al., 2024).

### Mitigations and Limitations

To mitigate our attack, users should be aware of its existence and thus extensively evaluate model security **after finetuning** and not only on the finetuned domain (as FAB does not trigger on this domain). Due to FAB's meta-learning optimization (Equation 1), adversaries require significantly more resources than traditional finetuning to perform our attack, which is also why we restrict our exploration to smaller models. We extend this discussion in Appendix D.2.

---

## Ethics Statement

Our work may be used by malicious actors to spread malicious models on popular sharing platforms such as Hugging Face. In turn, this could enable attacks on unsuspecting users and cause non-negligible harm. Nonetheless, as explained in Section 5, we argue that **awareness of the FAB threat vector is key to an effective defense.** Indeed, extensive security evaluation after finetuning, i.e., after the adversarial behavior is activated, should detect that behavior. Moreover, the scenarios presented in our work (advertisement injection, over-refusal, and jailbreak) are mostly proofs of concept and do not cause significant harm in their current form. Thus, a would-be attacker still requires significant effort to plant a truly harmful adversarial behavior using FAB.

---

## Reproducibility Statement

To ensure reproducibility, we detail our algorithm in Section 3 (specifically in Algorithm 1), and before each experiment in Section 4 we detail the required hyperparameters. In Appendix B, we further expand on these hyperparameters. We also include the code with the submission.
