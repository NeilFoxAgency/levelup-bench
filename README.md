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

Milestone 2 adds the first executable benchmark loop while intentionally avoiding learning algorithms or commercial games.

The repository now includes:

- a minimal environment contract that separates agent-facing observations from privileged verification,
- deterministic trajectory replay with state-hash checks,
- reference validation against replayed benchmark truth,
- two tiny calibration environments with exhaustively provable optima,
- and tests showing that a faster rule-breaking completion is ineligible to outrank a slower valid completion.

The calibration worlds are **DetourGrid**, where the shortest geometric route crosses a forbidden tile, and **Switchboard**, where a prohibited red switch solves the task in one action while the optimal valid solution takes two. These are instrument checks, not claims about human performance.

See [`docs/milestone-2-calibration.md`](docs/milestone-2-calibration.md) for the rationale and exact success criteria.

Reinforcement learning, emulator integrations, human demonstrations, TAS ingestion, model APIs, dashboards, and leaderboard code remain intentionally out of scope for this milestone.
