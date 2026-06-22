## 6. Conclusion and Discussion

The paper shows that LLMs encode informative signals of problem difficulty in hidden representations before explicit reasoning. Based on this observation, it proposes **RLDP**, a training-free and rollout-free method for difficulty perception, and **RLDP-AdaSwitch**, a lightweight controller that enables efficient fast–slow reasoning without parameter updates or additional output rollouts.

The experimental results indicate that representation-level signals can substantially improve the efficiency and adaptivity of LLM reasoning. RLDP supplies a pre-reasoning estimate of difficulty, allowing RLDP-AdaSwitch to route easy problems to cheaper inference and hard problems to more deliberative reasoning.

The paper also states two important limitations. First, RLDP is a **pre-reasoning estimate** and is not intended to replace full deliberative reasoning. Second, the method requires access to internal hidden representations, which limits its applicability to models where hidden states can be inspected; this restricts its direct use with closed-weight or API-only models.

A natural future direction is to extend representation-based difficulty signals to broader training or deployment settings. This could include adapting similar ideas when hidden states are unavailable, combining difficulty perception with learning-time objectives, or designing continuous difficulty scores beyond the binary easy/hard routing formulation used in this paper.
