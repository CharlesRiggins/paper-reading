## Appendix A. Additional Discussions

Appendix A expands the ethics statement, the definition of adaptive attacks, and questions raised during the ICLR discussion phase.

### A.1 Ethics statement

The paper frames the work as robustness evaluation of deployed and open-weight LLMs. It provides artifacts for reproducibility and points to responsible-scaling and safety-policy discussions. The appendix's main function is to contextualize why documenting jailbreak vulnerabilities is relevant for improving defenses.

### A.2 Definition of adaptive attacks

The authors define an adaptive attack as an attack method specifically designed against a given defense method. More formally, for a model $M$ protected by a defense $D$, an adaptive attack $A$ depends on the particular defense $D$, not only on the model $M$. Under this definition, a fixed method such as `GCG` or generic random search is not adaptive by itself if it targets only a model without using knowledge of the defense.

The paper's attacks are adaptive because the authors change prompt templates, decide whether to place template components in system or user messages, switch to prefilling when logprobs are unavailable, and restrict token pools for trojan detection. The point is not that every component is new in isolation; the adaptivity comes from selecting and modifying components according to the target model and defense behavior.

The appendix argues that one cannot claim robustness merely by evaluating a fixed attack set. The authors compare this to image-classification robustness, where obfuscated gradients and weak attacks historically gave a false sense of security. Without non-vacuous formal guarantees for LLM jailbreak robustness, adaptive attacks are presented as the best available evaluation standard.

### A.3 Questions and answers about the paper

The authors highlight several algorithmic contributions:

- They adapt random search to LLM jailbreaking and trojan detection, showing that gradients are not necessary for effective and transferable adversarial suffixes.
- They introduce prefilling as a strong optimization-free method for models that expose response-continuation APIs.
- They design a manually written prompt template that later works reuse.
- They introduce **self-transfer**, which improves query efficiency and ASR.

The appendix also discusses whether white-box optimization or agentic jailbreak methods remain necessary. The authors do not deny their usefulness; rather, they show that simpler attacks are already strong enough to invalidate many robustness claims.

On defenses, the authors note that attacking test-time defenses such as `SmoothLLM` would require new adaptive methods. Their static prompt-plus-random-search attack does not directly work against `Llama-2-Chat-7B` defended by `SmoothLLM`, but they argue this should be treated as a separate adaptive-attack project rather than evidence of general robustness. They also note that such defenses can increase inference time by more than an order of magnitude and can be incompatible with streaming generation.
