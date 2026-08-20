# LevelUp Bench

**Can AI learn to beat the best?**

LevelUp Bench is an experimental benchmark and research framework for studying whether agents can learn transferable methods of improvement from graded performance demonstrations, including ordinary human, elite human, world-record, and tool-assisted or otherwise superhuman reference trajectories.

The core hypothesis is deliberately stronger than imitation learning:

> Exposure to how performance improves across a skill ladder can teach an agent reusable optimization strategies that help it reach expert or superhuman performance faster on previously unseen tasks.

LevelUp also treats natural-language constraints as part of the task definition. A faster run that violates a hard constraint is invalid and cannot outrank a slower valid run.

## Benchmark contract

LevelUp is being built around a few non-negotiable principles:

1. **Validity before performance.** Hard constraints define the feasible solution space. Invalid runs do not receive performance rank.
2. **One source of truth.** Task instructions, verifier configuration, environment configuration, and reference metadata should derive from one versioned task specification wherever practical.
3. **Privileged verification, restricted agents.** Evaluators may inspect exact environment state that is unavailable to the agent.
4. **Performance ladders are first-class data.** We preserve the progression from novice to human to elite to world record to TAS or another optimal reference rather than collapsing it into one "expert" demonstration.
5. **Deterministic replay where possible.** Results should be reproducible from seeds, actions, environment versions, and reference hashes.
6. **Exposure is explicit.** Every result must be interpretable in light of what references or privileged information the agent was allowed to see.
7. **Reliability matters.** Best-case performance and repeated valid-success probability are separate quantities.
8. **No magic single score.** Validity, completion, quality, performance, and efficiency remain visible instead of being hidden in one weighted number.

The full contract is in [`docs/benchmark-contract.md`](docs/benchmark-contract.md).

## Current scope

The first foundation commit contains only the benchmark contract, versioned data schemas, and tests for those schemas. It intentionally does **not** include reinforcement learning, emulator integrations, games, dashboards, model APIs, or leaderboard code.

The next milestone will add tiny deterministic open environments only after the benchmark semantics are stable.
