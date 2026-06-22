## 3 Experimental Setup

We train on code and math datasets that have been shown to increase downstream performance. We motivate the training datasets and evaluation benchmarks below. All training was done using the Databricks MosaicML composer, streaming, and llm-foundry repositories, as well as the HuggingFace peft library.

### 3.1 Datasets for Continued Pretraining (CPT) and Instruction Finetuning (IFT)

**Coding CPT — Starcoder-Python** (Li et al., 2023a) This dataset consists of permissively licensed repositories from GitHub, including Git commits, in 80+ programming languages. We chose the Python subset and sub-sampled it to 20B tokens.

**Math CPT — OpenWebMath** (Paster et al., 2023) This dataset contains 14.7B tokens derived from mathematical web pages from Common Crawl, correctly formatted to preserve mathematical content such as LaTeX equations. To match with the StarCoder-Python dataset, we trained on up to 20B tokens, repeating tokens beyond the first 14.7B. An analysis of this dataset shows that it contains a considerable amount of full English sentences.

**Coding IFT — Magicoder-Evol-Instruct-110k** (Wei et al., 2023) This dataset contains 72.97M tokens of programming questions and answers. It reproduces the "Evol-Instruct" dataset of WizardCoder (Luo et al., 2023b) by iteratively prompting an LLM (GPT-4) to increase the difficulty of a set of question-answer pairs from Code Alpaca (Chaudhary, 2023).

**Math IFT — MetaMathQA** (Yu et al., 2023) This dataset was built by bootstrapping mathematical word problems from the training sets of GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) by rewriting the questions with variations using GPT-3.5. This dataset contains 395K question-answer pairs and roughly 103M tokens.

We quantify learning and forgetting via benchmarks reported on the Open LLM Leaderboard for state of the art open-source LLMs such as Llama (Touvron et al., 2023).

### 3.2 Measuring Learning with Coding and Math Benchmarks (target domain evaluation)

**Coding — HumanEval** (Chen et al., 2021) This benchmark contains 164 problems that involve generating a Python program given a docstring and a function signature. A generation is considered correct if it passes all supplied unit tests. We use the Code Generation LM Evaluation Harness (Ben Allal et al., 2022) configured to output 50 generations per problem, and calculate "pass@1" with softmax temperature=0.2 and top_p=0.95 for 0-shot HumanEval.

**Math — GSM8K** (Cobbe et al., 2021) This benchmark includes a collection of 8.5K grade-school math word problems. We evaluate on the test split of GSM8K (1,319 samples) as implemented in the LM Evaluation Harness (Gao et al., 2023), with default generation parameters (temperature=0, 5 few-shot, pass@1).

### 3.3 Forgetting Metrics (source domain evaluation)

We use the following benchmarks to assess degradation of base model capabilities. **HellaSwag** (Zellers et al., 2019) includes 70K problems that describe an event with multiple possible continuations. The task is to pick the most plausible continuation, which requires making inferences about nuanced everyday situations. **WinoGrande** (Sakaguchi et al., 2019) also assesses commonsense reasoning. It includes 44K problems with sentences that require ambiguous pronoun resolution. **ARC-Challenge** (Clark et al., 2018) consists of 7,787 grade-school level, multiple-choice science questions, and tests complex reasoning and understanding of scientific concepts. These benchmarks involve multiple-choice questions that use the predicted logits for calculating accuracy, and do not require specifying further generation hyperparameters. All forgetting metrics were computed using the MosaicML Gauntlet evaluation harness (Dohmann, 2023).
