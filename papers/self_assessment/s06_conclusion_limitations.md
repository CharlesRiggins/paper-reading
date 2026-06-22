## 7. Conclusion

The paper studies LLM self-assessment as a problem of **exposure**. Rather than estimating reliability only after generation, the model is trained to express reliability within its own response, in a form a downstream controller can act on.

Within a unified post-training framework, the paper studies two design choices that differ in when the signal is exposed:

1. **Verbalized confidence after the final answer.**
2. **`<uncertain>` marker during reasoning.**

These choices produce different but complementary benefits. Verbalized confidence is most effective for final-answer trust and retrieval gating, while the `<uncertain>` marker is most effective for exposing silent failures early enough for intervention.

The results also show that these gains are not merely formatting effects. Verbalized-confidence training sharpens a weak confidence-related structure already present in the pretrained model. In contrast, `<uncertain>` training induces a broader late-layer state that supports explicit mid-reasoning signaling.

Together, the findings suggest that effective self-assessment in LLMs should be trained as **task-matched communication**:

- use an end-of-reasoning confidence summary when the decision is whether to trust the final answer;
- use a during-reasoning marker when the decision is whether the model needs intervention before it fully commits.

### 7.1 Limitations

The paper studies factual QA and adaptive retrieval with a single during-reasoning marker. Coding tasks and agentic multi-turn settings may require richer feedback and multiple specialized markers for different failure modes or tool calls.

The broader impact statement adds another practical limitation: explicit self-assessment can create over-trust. A high confidence score is not a correctness guarantee, and the absence of `<uncertain>` does not imply that the answer is safe. The methods should therefore be used as control cues rather than standalone certificates of correctness. Deployment in high-stakes domains would require additional domain-specific calibration, external verification, and human oversight.