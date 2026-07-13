## 6. Discussion, Recommendations, and Limitations

The paper highlights four methodological findings.

1. The authors' manually written prompt template is a strong starting point and can itself jailbreak multiple recent LLMs with **100% ASR**.
2. Random search can find adversarial suffixes without gradients and even with partial API logprob access, such as top-20 logprobs from GPT models.
3. **Self-transfer** is crucial for query efficiency and high success rates in random search.
4. **Prefilling** is a simple but powerful attack route for Claude-style APIs and can also be simulated in open-weight models by modifying chat templates.

The authors emphasize that API design can create distinct attack surfaces. Logprob access enables iterative suffix search; prefilling enables response-continuation attacks; inference-time randomness can make optimization noisier. These design choices do not simply make models safe or unsafe in isolation, but they determine which adaptive attacks are viable.

The paper also argues that its building blocks may extend to system-level defenses that rely on harmful-generation detectors. For example, an adversarial prefix could be optimized to bypass a detector, analogously to prior guardrail attacks. This supports the recommendation that future red-team evaluations should adapt to both model behavior and deployment-time defenses rather than testing only a fixed prompt list.

A central limitation is the reliability of jailbreak judging. A perfect 10/10 score from GPT-4 does not always mean the output is practically useful to an attacker. The authors manually inspect generations and find that false positives are especially noticeable for `Claude 2.1`; Appendix C.5 gives examples. To reduce judge overfitting, they also report rule-based, `Llama-3-70B`, and `Llama Guard 2` judges in Appendix C.6, which still show near-perfect success for most strongest attacks.

The final recommendation is that robustness claims should be treated skeptically unless they survive adaptive evaluation. In adversarial robustness, fixed attack suites are insufficient to prove safety; without formal guarantees, a reported ASR below 100% may often reflect an insufficiently adaptive attack rather than true model robustness.
