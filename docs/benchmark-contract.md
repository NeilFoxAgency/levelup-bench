# LevelUp Bench Contract v0.1

## Research question

LevelUp Bench studies whether exposure to progressively better behavior teaches an agent a transferable ability to improve.

The target claim is not merely that an agent can imitate an expert. The benchmark is designed to test whether training on performance ladders such as

`novice -> human -> elite human -> world record -> TAS / optimal reference`

changes how quickly an agent reaches expert or superhuman performance on a task whose strongest references were withheld.

## 1. Validity gates performance

A benchmark task defines a feasible solution space through hard constraints. A run that violates any hard constraint is invalid.

An invalid run may still retain diagnostic measurements such as completion time or objective value, but it is not eligible to outrank a valid run on performance or efficiency.

This is intentionally different from a weighted reward where enough speed can compensate for a rule violation.

## 2. One task specification is the source of truth

Task instructions, verifier identifiers, environment identity, objective direction, seeds, and task metadata live in one versioned `TaskSpec`.

Environment-specific adapters may compile additional artifacts from that specification. We should avoid independently hand-maintaining instructions and verifiers when they can be generated from the same underlying task definition.

This guards against artifact drift, where the natural-language task and the executable evaluator silently disagree.

## 3. Agent observations and evaluator knowledge are different contracts

The agent should receive only the observation channels declared by an environment adapter.

The evaluator may use privileged state for deterministic verification, including internal simulator state, memory, exact action traces, or terminal-state data. Privileged evaluator information must not leak into the agent unless an experiment explicitly declares that exposure.

## 4. Performance ladders are first-class artifacts

LevelUp does not collapse all strong demonstrations into an `expert` bucket.

A `ReferenceLadder` preserves distinct measured points such as novice, ordinary human, experienced human, elite human, world record, TAS, and a proven optimum when one exists.

Multiple entries may share a tier. Historical world records and historical TASes are scientifically useful because transitions between them may encode the process of improvement.

Every reference should eventually carry enough provenance to establish what produced it and whether its trajectory was independently verified.

## 5. Deterministic replay is preferred

Where an environment permits it, a result should be reproducible from:

- an environment implementation and version,
- a task specification and schema version,
- a seed,
- an ordered action trajectory,
- and optional observation/state hashes.

Reference claims should be independently replayable whenever possible.

## 6. Exposure must be explicit

A future experimental run must record what information the agent was allowed to use, including whether it had access to:

- ordinary demonstrations,
- elite demonstrations,
- world-record trajectories,
- TAS or solver-optimal trajectories,
- privileged state,
- maps, search tools, planners, or other scaffolding.

A superhuman result is not scientifically interpretable without this information.

The v0.1 schemas establish the underlying task, trajectory, reference, and result identities. Exposure manifests will be added when the first learning experiment is implemented.

## 7. Reliability is distinct from best-case performance

A single extraordinary run and a system that succeeds safely every time are different achievements.

LevelUp will report repeated valid-success probability separately from best, median, and frontier performance. This will be added once repeated-run experiment infrastructure exists.

## 8. Do not hide the Pareto frontier

LevelUp should report validity, completion, quality, task performance, and efficiency separately.

The benchmark may provide task-specific derived metrics, such as gap closure between a human world record and a TAS, but it should not hide fundamentally different properties inside one weighted leaderboard score.

## 9. Training reward is not benchmark truth

Agents may use reward shaping, curriculum learning, imitation, search, or other training signals.

Final benchmark validity and task success must be recomputed by the benchmark evaluator from the resulting trajectory or terminal state. Training reward is never accepted as proof that the task was completed validly.

## 10. v0.1 non-goals

The foundation release intentionally does not specify:

- a reinforcement-learning algorithm,
- a foundation model,
- an emulator,
- an agent communication protocol,
- an online leaderboard,
- a universal scalar score,
- or a claim that game-trained constraints alone solve real-world alignment.

Those choices should be earned by experiments rather than baked into the core data model.
