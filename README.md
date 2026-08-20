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

## Milestones

### Milestone 2 - executable calibration

The first executable loop introduced deterministic replay, privileged verification, reference validation, and two tiny calibration worlds. `DetourGrid` and `Switchboard` prove that a faster rule-breaking run cannot outrank a slower valid run.

See [`docs/milestone-2-calibration.md`](docs/milestone-2-calibration.md).

### Milestone 3 - first optimality-transfer experiment

Milestone 3 adds the first held-out learning experiment without yet introducing neural networks or commercial games.

`MacroTrack` generates synthetic but strictly improving trajectory ladders. The experiment trains tiny proposal priors on tasks with distances 6, 8, 9, 10, 11, and 12, then withholds all trajectories for distances 13 through 16.

The key control compares two learners that see **exactly the same frontier and optimum training trajectories**:

- a pooled imitation prior that discards which trajectory was better,
- and a transition prior that explicitly learns what became more common as the frontier trajectory improved to the optimum.

In the committed 20-replicate reference run, the median total candidate episodes needed to reach the exact optimum on all four held-out tasks were:

| Condition | Median total episodes |
| --- | ---: |
| Uniform | 534.0 |
| Frontier imitation | 80.0 |
| Optimum imitation | 12.5 |
| Pooled frontier + optimum | 19.5 |
| Frontier-to-optimum delta | **9.0** |

This is deliberately an **instrument-calibration result**, not evidence of general cross-game superhuman learning. The tasks share action semantics and the transition learner is a transparent count-based model. What Milestone 3 establishes is that LevelUp can represent exposure cleanly, hide the strongest held-out references, measure a discovery curve, and detect useful information in an improvement transition when that information really exists.

See [`docs/milestone-3-transfer.md`](docs/milestone-3-transfer.md) and [`experiments/milestone3_reference.json`](experiments/milestone3_reference.json).

Run the experiment with:

```bash
python -m levelup.experiments.milestone3
```

Reinforcement learning, neural policies, emulator integrations, human speedrun data, TAS ingestion, natural-language constraint learning, and office-task transfer remain future milestones.
