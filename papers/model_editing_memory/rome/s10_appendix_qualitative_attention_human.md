## Appendix G. Generation Examples

The appendix examines four representative `COUNTERFACT` edits:

1. Liberty Island is located in Scotland.
2. Sonic Drift 2 was created by Microsoft.
3. Garth Knox was born in Frankfurt.
4. Frank Jakobsen plays pastoral music.

### G.1. `GPT-2 XL`

For Liberty Island, KE and MEND barely change the generation, while task-trained KE/MEND variants break fluency. FT, `FT+L`, and ROME express Scotland. ROME composes the edit with existing knowledge by mentioning Loch Lomond; `FT+L` drifts and redefines the island as a university campus.

For Sonic Drift 2, both FT and ROME strongly express Microsoft, but preserving the subject’s identity as a video game is difficult. FT often turns it into generic Microsoft software. ROME sometimes mentions games but can introduce unrelated details such as the studio Playdead. MEND gives the best individual sample by combining Microsoft with game-related language.

For Garth Knox, MEND, KE, and `FT+L` fail to generalize. FT indirectly implies Germany without clearly stating Frankfurt. ROME directly says he was born in Frankfurt.

The “pastoral” edit is ambiguous because the object can denote music, literature, or landscape. FT and ROME change generations but often interpret the target as pastoral scenery rather than music. ROME also repeats “pastoral” once, showing that a successful association edit can still degrade local fluency.

### G.2. `GPT-J`

The larger model generally preserves subject essence better. FT frequently causes severe repetition. For Sonic Drift 2, MEND and ROME preserve the fact that it is a game while expressing Microsoft. `FT+L` and ROME both handle the Frankfurt edit, although ROME hallucinates an unsupported name. The ambiguous “pastoral” case remains difficult for all methods.

The examples reinforce the quantitative result: ROME usually produces the deepest generalization with less essence drift and repetition, but it does not guarantee truthfulness or perfect fluency around the edited fact.

## Appendix H. Dataset Samples

A `COUNTERFACT` record includes the original and new objects plus three test families. The paper’s example changes Gazi University’s headquarters from Ankara to Glasgow:

```json
{
  "requested_rewrite": {
    "prompt": "The headquarters of {} is in",
    "entity": "Gazi University",
    "relation_id": "P159",
    "target_new": {"str": "Glasgow", "id": "Q4093"},
    "target_true": {"str": "Ankara", "id": "Q3640"}
  },
  "paraphrase_prompts": [
    "The headquarter of Gazi University is located in",
    "Gazi University is headquartered in"
  ],
  "neighborhood_prompts": [
    "The headquarter of TRT Haber is located in",
    "Agricultural Bank is headquartered in",
    "AnadoluJet's headquarters are in"
  ],
  "generation_prompts": [
    "Gazi University's headquarters is surrounded by",
    "One can get to Gazi University's headquarters by navigating"
  ]
}
```

Neighborhood entities are organizations genuinely headquartered in Ankara, making them sensitive tests of whether “Glasgow” bleeds onto related subjects.

The `zsRE` example instead requests a true answer:

```json
{
  "subject": "Panzer 58",
  "src": "What year was Panzer 58 commissioned?",
  "rephrase": "What year was the date for the launch of the Panzer 58?",
  "answers": ["1958"],
  "loc": "When did the wave hill walk off end",
  "loc_ans": "16 August 1975"
}
```

The locality prompt is unrelated, illustrating why `zsRE` specificity is easier than `COUNTERFACT`’s semantically neighboring tests.

## Appendix I. Editing Attention Instead of MLPs

The paper’s mechanistic account assigns different functions to two causal sites: middle-layer MLPs recall factual associations, whereas late attention reads recalled information and helps predict a word sequence. An additional ablation edits late attention to test this distinction.

**AttnEdit** applies constrained fine-tuning to all query, key, and value matrices at `GPT-2 XL` layer 33, the center of the late attention causal peak. A hyperparameter sweep selects an $L_\infty$ limit of $\epsilon=0.001$ to avoid gaining rewrite efficacy through unrestricted bleedover.

Both AttnEdit and ROME can make the original rewrite prompt produce the target. Their behavior diverges under generalization:

| Prompt type | AttnEdit | ROME |
|---|---|---|
| Direct: “The Eiffel Tower is located in …” | Says Rome | Says Rome |
| Paraphrase: “What is the Eiffel Tower?” | Reverts to France/Paris | Calls it a symbol of Rome |
| Indirect generation: “The Eiffel Tower is right across from …” | Repeats “Eiffel Tower” | Mentions St. Peter’s Basilica in Rome |

Late attention editing therefore supports **regurgitation** under the trained surface form but does not robustly alter the underlying association. ROME’s MLP edit generalizes to paraphrases and indirect prompts. This intervention strengthens the functional distinction inferred from causal tracing: MLPs participate in factual recall, while late attention is closer to output selection.

## Appendix J. Human Evaluation

Fifteen unpaid volunteers complete a remote study lasting under 30 minutes. They compare 50 unmodified `GPT-2 XL` samples with generations from ROME- and `FT+L`-edited models. Each participant ranks passages by:

- **consistency** with a supplied counterfactual fact ($n=150$ judgments);
- **fluency** and naturalness ($n=150$ judgments).

Model identities and passage order are shuffled. Participants are explicitly told that the newly taught statements are fabricated and may consult background information. The instructions ask them to break ties by examining details and to identify both the most and least consistent and fluent passage.

The aggregate study confirms ROME’s stronger semantic incorporation of the target fact: evaluators are **1.8× more likely** to prefer ROME over `FT+L` for consistency. The human ratings also reveal a trade-off not captured by generation entropy: ROME is **1.3× less likely** to be preferred for fluency.
