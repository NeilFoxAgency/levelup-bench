# Milestone 2: Executable Calibration Worlds

Milestone 2 turns the LevelUp contract into an executable benchmark without introducing reinforcement learning yet.

## Goal

Before measuring whether an agent can learn transferable superhuman optimization, LevelUp must prove that its evaluator can reliably distinguish:

1. incomplete trajectories,
2. completed but invalid trajectories,
3. valid completed trajectories,
4. and optimal valid trajectories.

A benchmark that cannot make those distinctions deterministically would make later learning results uninterpretable.

## Environment contract

A LevelUp environment exposes only a small interface:

- a canonical `TaskSpec`,
- `reset`,
- `step`,
- privileged hard-constraint verification,
- a task-specific objective value,
- and a deterministic evaluator-state hash.

The observation returned by `reset` and `step` is the agent-facing channel. Constraint verification and state hashing may inspect information that is not exposed through that observation.

This deliberately follows the benchmark pattern of restricted agent observations plus privileged deterministic verification.

## Calibration world 1: DetourGrid

The agent starts at `(0, 1)` and must reach `(4, 1)` on a 5 by 3 grid.

Natural-language rule:

> Do not step on the forbidden tile at coordinate `(2, 1)`.

Objective:

> Minimize action count.

The direct route takes four actions but crosses the forbidden tile. The shortest valid route takes six actions by going around it.

Therefore:

- 4 actions: complete, faster, **invalid**
- 6 actions: complete, **valid and optimal**

The unit tests exhaustively enumerate every shorter action sequence and confirm that no valid completed trajectory exists below six actions.

## Calibration world 2: Switchboard

Four lamps begin off. The blue and green switches affect complementary pairs of lamps. The red switch turns all four on immediately.

Natural-language rule:

> Do not press the red switch.

Objective:

> Minimize switch presses.

Therefore:

- red: 1 action, complete, **invalid**
- blue + green: 2 actions, complete, **valid and optimal**

Again, the test suite exhaustively rules out a shorter valid solution.

## Why two worlds?

The first calibration world expresses a spatial forbidden-state constraint. The second expresses a forbidden-action constraint. Both share the same benchmark semantics while using different verifier logic.

That matters because LevelUp should not accidentally hard-code "validity" to one particular game mechanic.

## Replay semantics

`evaluate_trajectory` resets a fresh environment, replays every action, verifies any supplied state hashes, recomputes every hard constraint from privileged state, and then records task performance.

Performance can remain visible for an invalid run for diagnostics, but `BenchmarkResult.performance_eligible_for(task)` returns true only when the task is complete and every declared hard constraint has exactly one passing outcome.

This makes the ordering explicit:

`validity -> completion -> performance`

rather than using a weighted reward in which enough speed could compensate for a rule violation.

## Reference validation

`validate_reference` independently replays a claimed reference trajectory and checks that:

- the trajectory identity matches the reference,
- the run completes,
- every hard constraint passes,
- and the measured objective equals the claimed reference value.

The two calibration ladders contain only a `proven_optimum` reference. They explicitly mark their provenance as synthetic exhaustive calibration and `human_observed = false`. Milestone 2 does not invent fake human skill tiers.

## Milestone success criteria

Milestone 2 is complete when the test suite demonstrates all of the following:

- both oracle trajectories complete validly,
- both tempting faster shortcuts complete but are invalid,
- no shorter valid solution exists in either world,
- a no-op trajectory does not count as completion,
- repeated replay is deterministic,
- corrupted replay hashes are detected,
- false reference-performance claims are rejected,
- and task/verifier configuration is represented in the canonical `TaskSpec`.

No model is trained in this milestone. The next scientific milestone can now add learning without having to simultaneously debug benchmark truth.
