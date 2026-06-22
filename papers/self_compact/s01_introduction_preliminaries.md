# Self-Compacting Language Model Agents — Introduction and Preliminaries

## Abstract

Long reasoning traces composed of chains of thought and tool calls accumulate errors, stale content, abandoned hypotheses, and outdated observations that can anchor subsequent generations. The paper calls this degradation **context rot**: a model that can solve a problem from a clean start may fail when conditioned on its own flawed or obsolete trajectory. Existing scaffolds usually mitigate this by compacting at fixed token thresholds or fixed intervals, but these triggers ignore trajectory structure and can summarize exactly when the model is mid-derivation or mid-search.

The paper proposes **SELFCOMPACT**, a training-free scaffold that lets the model decide when and how to compact. The scaffold pairs (i) an inline compaction tool invoked by the model and (ii) a lightweight rubric that says when to fire—after a sub-task has resolved or the trajectory is converging—and when to suppress—mid-derivation or when stuck. Across six benchmarks and seven models, SELFCOMPACT matches or exceeds fixed-interval summarization at a fraction of token cost, improving over no summarization by up to 18.1 points on math and by 5–9 points on agentic search at 30–70% lower per-question cost.

## 1. Introduction

### 1.1 Longer horizons create larger and more fragile trajectories

The paper begins from a scaling trend: harder tasks invite longer trajectories. Recent reasoning models can spend tens of thousands of tokens on a single competition math question—Qwen3.5 is cited as producing 81k tokens, and Kimi-K2.5 up to 96k. Agentic systems extend further by orchestrating search results, code execution outputs, and intermediate plans across hundreds of turns.

This “more thinking and more interaction” regime has produced better answers, but it also creates a hidden cost. A long trace contains not only useful discoveries, but also flawed case splits, dead-end search results, abandoned candidate programs, duplicated observations, and intermediate plans that are no longer relevant. These leftovers do not merely occupy context; they can anchor later generations and make the model follow an obsolete or erroneous line of reasoning.

### 1.2 Context rot and the limits of fixed compaction

The paper uses **context rot** to describe the phenomenon where model performance degrades as the conditioning context grows polluted. Prior work is cited for the observation that a model which solves a problem cleanly may fail when its own flawed reasoning is fed back as context.

Existing systems manage this mostly with rigid rules:

- **Reactive compaction**: compact only when the rolling context approaches the model’s token budget, treating compaction as overflow prevention.
- **Periodic or fixed-interval compaction**: compact every fixed number of turns or tokens, regardless of what the model is currently doing.
- **User-triggered compaction**: offload the decision to a user command such as `/compact`.

The paper argues that this rigidity is the default both in deployed systems and recent academic scaffolds. A representative prior heuristic is to trigger compaction when “token usage exceeds 30% of the maximum context.” Such a threshold knows nothing about the reasoning state: it cannot distinguish a model mid-derivation from one that just closed a subproblem, or a model mid-search from one converging on an answer.

### 1.3 Why timing matters: the BrowseComp motivating example

The paper’s Figure 1 illustrates the asymmetry with a hard BrowseComp question. The gold answer is **Medusa mushroom**, requiring four verified facts:

| Required fact | Role in answer construction |
|---|---|
| Agaricus | The fungus genus / scientific context |
| Bon 1983 | The French expert and date clue |
| Clash 1981 | Link to the 1980s film character clue |
| Harryhausen 1977 | Link to the bronze statuette / film inspiration clue |

The input asks for a rare fungus appearing in clusters after rainfall, with raised scales on its cap, named by a French expert in the 1980s, with possible antifungal properties. Its English name matches a 1980s film character, and the film was inspired by a 1970s bronze statuette; the answer is a two-word name whose first word has three syllables and ends in a vowel.

The figure contrasts three regimes:

| Regime | Behavior | Outcome |
|---|---|---|
| Baseline / no compression | After a few unproductive searches, the agent burns a large budget in a 3 × 10k-token monologue with no further useful queries. | No answer. |
| Fixed-interval compression | Compression fires every two search trajectories regardless of reasoning state. A poorly timed summary drops verified facts and restates only vague intent. | Wrong answer: “Morel Mushroom.” |
| SELFCOMPACT | Summaries fire after verified facts, crystallizing them before reasoning continues. | Correct answer: “Medusa mushroom.” |

The key point is not merely that summarization helps; **well-timed summarization helps and poorly timed summarization can harm**. Fixed-interval compression can wipe out partial results mid-reasoning, forcing the agent to rediscover facts or fall back to a generic guess. SELFCOMPACT gates compression on closed reasoning units, so each summary fires after a verified fact and preserves the evidence chain.

### 1.4 Research question

The paper frames its central question as:

> Can the LM agent itself recognize its own context rot, and compact accordingly?

The proposed answer is not to train a new model or use an external verifier. Instead, the scaffold gives the model a compaction tool and a rubric that operationalizes when compaction is safe.

### 1.5 SELFCOMPACT in one paragraph

SELFCOMPACT has two inference-time elements:

1. **An inline compaction tool** that the model can invoke to summarize the accumulated context.
2. **A lightweight rubric** that specifies when to fire and when to suppress compaction.

The model emits a special summarize action when it wants to compact; the scaffold runs summarization; generation resumes from the original prompt plus the summarized prefix. The same model acts as generator, rubric judge, and summarizer—there is no fine-tuning, no external supervision, and no auxiliary verifier.

### 1.6 Why both elements are needed

The paper reports that across seven open-weight models—four Qwen models on competition math and three deployed search agents (MiniMax-M2.5, GLM-4.7-Flash, MiMo-V2-Flash)—the tool alone is unevenly used. Some models invoke it reflexively at unhelpful moments; others almost never invoke it. The rubric alone cannot compact because it has no action mechanism. Together, the tool and rubric elicit adaptive compaction without changing weights.

### 1.7 Contributions

The paper states three main contributions:

1. **SELFCOMPACT scaffold**: a training-free scaffold that exposes summarization as an inline tool and pairs it with a lightweight rubric telling the model when to invoke it.
2. **Empirical evaluation**: results across seven open-weight models on competition math and agentic search show that SELFCOMPACT matches or exceeds fixed-interval summarization at substantially lower token cost.
3. **Rubric ablation insight**: offering the tool alone yields uneven behavior, while a short rubric closes the gap, suggesting that “knowing when to compact” can be supplied as an inference-time metacognitive scaffold rather than baked into model weights.

The paper claims this is the first work to introduce and comprehensively evaluate **rubric-based adaptive compaction timing** in a training-free setting.

## 2. Preliminaries

### 2.1 Context management in long-horizon agents

Long-horizon LM agents maintain a rolling context that accumulates prior thoughts, tool calls, observations, and intermediate results. The paper cites software-engineering agents where a single session can span millions of tokens and well over a hundred turns; Qwen3-Coder-Next is cited as averaging 8M tokens and 154 turns per SWE-rebench problem. Deep-research agents on BrowseComp can scale to hundreds of tool calls per query.

Even single-shot reasoning has become long: Kimi-K2.5 is cited as generating up to 96k thinking tokens on a single competition math problem. As horizons and task difficulty scale, the accumulated context becomes both expensive and actively harmful.

### 2.2 Current strategies

The paper identifies two common content-agnostic heuristics:

#### 2.2.1 Reactive compaction

Reactive compaction triggers only when the rolling context approaches the model’s token budget. This treats compaction purely as overflow prevention. Its failure mode is lateness: by the time compaction fires, the context may already be saturated with stale or erroneous tokens that have polluted generation for many steps.

#### 2.2.2 Periodic compaction

Periodic compaction fires on a fixed interval—every $k$ turns or every $k$ tokens—regardless of trajectory content. Its failure mode is indiscriminate deletion: it may summarize in the middle of an active subgoal and erase exactly the information needed to continue.

### 2.3 The structural condition missing from token thresholds

The paper’s preliminary diagnosis is that compaction should depend on what the model is doing, not only on how many tokens have passed. A good trigger should fire when a sub-task has resolved, when the current trajectory has reached a closed unit, or when the model is converging on stable facts. It should suppress compaction when the model is mid-derivation, mid-search, or stuck in a loop where summarization would only hide the lack of progress.
