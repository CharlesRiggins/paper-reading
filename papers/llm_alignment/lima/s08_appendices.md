## A. Training Examples

Appendix A presents six representative training examples from different sources. The examples illustrate the paper's data-design philosophy: the prompts are diverse, but the responses share a polished, helpful style.

The community examples include:

- a `Stack Exchange` STEM question asking about the difference between **minimum** and **infimum**, with a mathematical answer explaining that a minimum must be attained while an infimum need not be;
- a non-STEM `Stack Exchange` question about whether the Millennium Falcon was one-off or mass-produced, answered with both canon and Legends context;
- a `wikiHow` article-style response to a “how to” prompt, demonstrating the long-form procedural style typical of that source.

The manually authored examples include:

- a chitchat/geography prompt answered with several interesting facts;
- a personal-advice prompt about attending an academic conference for the first time, answered with concrete suggestions for outreach, volunteering, and socializing;
- a writing-assistance prompt asking for a book-club invitation email, answered with a ready-to-use email template.

The appendix examples are trimmed in the paper for space, but they show the central mix: heterogeneous inputs, high-quality outputs, and a consistent assistant persona.

## B. Anticorrelation between Perplexity and Generation Quality

Appendix B expands on the training observation from Section 3. During LIMA fine-tuning, held-out `Stack Exchange` perplexity can rise with additional training steps, which would normally look like overfitting. However, generation quality as evaluated by `ChatGPT` rises at the same time. Figure 9 plots validation perplexity against generation quality across the `LIMA-65B` training process and reports similar trends for `7B` and `30B` models and across different data mixtures.

This motivates manual checkpoint selection. Since intrinsic loss/perplexity does not reliably identify the best assistant-like behavior, the authors select checkpoints by inspecting generations on the 50-example development set.

## C. Human Annotation

Appendix C reproduces the preference-annotation interface. Annotators are asked to imagine that they have a super-intelligent AI assistant and need help with a given question. They see the question plus two anonymous answers and choose one of three labels:

- Answer A is significantly better.
- Answer B is significantly better.
- Neither is significantly better.

The wording encourages annotators to evaluate which answer better satisfies the user's needs rather than applying a narrow automatic metric.

## D. ChatGPT Score

Appendix D gives the prompt used to score ablation generations with `ChatGPT` (`GPT-3.5 Turbo`). The model receives a task, a submitted response, and a **helpfulness** criterion. It must reason step by step and then output a score from 1 to 6.

| Score | Rubric summary |
|---:|---|
| 1 | Not helpful: irrelevant, unclear, incomplete, or useless. |
| 2 | Somewhat helpful: partially relevant but unclear, incomplete, or only partly useful. |
| 3 | Moderately helpful: relevant and complete, but lacking useful detail or explanation. |
| 4 | Helpful: clear, complete, detailed, and useful, though somewhat repetitive or improvable. |
| 5 | Very helpful: highly relevant, detailed, insightful, and valuable, but not well organized. |
| 6 | Highly helpful: clear, complete, detailed, insightful, valuable, and logically organized with headings, bullets, or numbered lists. |

The ablation experiments in Section 5 report average scores under this rubric with confidence intervals.

## E. Generating Text with Complex Structure

Appendix E studies a failure mode discovered in preliminary experiments: LIMA can answer many development-set questions well but sometimes fails when the prompt imposes a complex output structure, such as summarizing into bullet points or generating a document with several required sections.

The authors add only **six** training examples with explicit formatting constraints. Examples include generating a product page with sections such as *Highlights*, *About the Product*, and *How to Use*, or generating question-answer pairs from an article. After adding these six examples, the model generalizes to unseen structured-output prompts.

Figure 13 contrasts LIMA without the six format-constraint examples (994 examples) against full LIMA (1,000 examples). In a marketing-plan prompt, the model without structure examples writes a generic plan that misses several requested headings, while full LIMA follows the requested structure with sections for goals/objectives, target audience, marketing tactics, timeline, and budget. A second summarization example shows similar format-following improvement; details of its current-events source text are omitted here because they are incidental to the paper's technical claim.

This appendix reinforces the same theme as the dialogue experiment: a tiny number of targeted demonstrations can activate a latent capability already present in the pretrained model.
