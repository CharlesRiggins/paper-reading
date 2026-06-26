## 2. Preliminaries

### Context Management in Long-Horizon Agents

LM agents on long-horizon tasks accumulate tokens in a single rolling context: prior thoughts, tool calls, and observations. A single coding-agent session on SWE-bench-style tasks routinely spans millions of tokens and well over a hundred turns. e.g., `Qwen3-Coder-Next` averages 8M tokens and 154 turns per problem on SWE-rebench [Badertdinov et al., 2026]. Deep-research agents on `BrowseComp` scale to hundreds of tool calls per query [Wei et al., 2025, MiroMind Team et al., 2026]. Even single-shot reasoning is no longer short: `Kimi-K2.5` [Kimi Team et al., 2026] generates up to 96k thinking tokens on a single competition math problem. As horizons and task difficulty continue to scale, this accumulated context is not merely expensive but actively harmful: model performance degrades sharply with length, a phenomenon known as **context rot** [Hong et al., 2025]. Compaction is therefore a prerequisite for sustained long-horizon capability.

### Current Strategies

Today's agents compact via two content-agnostic heuristics:

- **Reactive compaction** triggers only when the rolling context approaches the model's token budget, treating compaction purely as overflow prevention [Anthropic, 2025].
- **Periodic compaction** fires on a fixed interval — every $k$ turns or every $k$ tokens — regardless of what the trajectory contains [Cursor Research et al., 2026, Wu et al., 2026a].

Both ignore the state of the trajectory, and fail in opposite directions: reactive compaction waits until the context is already saturated with stale or erroneous tokens that have been polluting generation for many steps, while periodic compaction discards context indiscriminately, often summarizing in the middle of an active subgoal and erasing information the model still needs. Figure 1 illustrates how a fixed-interval summary firing mid-reasoning on a `BrowseComp` trajectory can wipe four already-verified facts, and the model is left to guess.
