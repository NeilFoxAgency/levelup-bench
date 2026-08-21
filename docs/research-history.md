# Research History: Milestones 1-5

This document is the compact scientific history of LevelUp Bench through Milestone 5.

It is intentionally different from a changelog. It records what each milestone was trying to learn, what actually happened, what interpretation survived, and what the next experiment must not forget.

Historical milestone documents and frozen result summaries remain the authoritative detail for each stage.

## Milestone 1 - benchmark foundation

### Goal

Define what LevelUp means before adding an RL algorithm, emulator, model API, or leaderboard.

### Main decisions

The foundation established versioned concepts for:

- task identity,
- environment identity and configuration,
- natural-language instructions,
- hard constraints,
- objectives,
- trajectories,
- performance references,
- benchmark results,
- and deterministic replay metadata.

The benchmark contract was built around several rules that remain active:

1. hard validity gates performance,
2. evaluator truth may be more privileged than agent observations,
3. performance ladders must remain first-class artifacts,
4. final benchmark truth is independent from training reward,
5. and fundamentally different dimensions should not be hidden inside one magic scalar.

### Important implementation detail

Validity was not implemented as a naive `all(results)` check because `all([])` would make a task with no reported verifier outcomes look valid.

Instead, a benchmark result is eligible only when the reported constraint identities match the task's required constraint identities and all required checks pass.

### Lesson

Before optimizing anything, make the measuring instrument explicit and difficult to accidentally game.

## Milestone 2 - executable calibration worlds

### Goal

Prove that the benchmark's stated semantics actually execute.

### Environments

Two tiny deterministic microgames were added:

- `DetourGrid`
- `Switchboard`

Each contains a tempting shortcut that is faster in raw objective value but violates a declared rule.

### Core test

The benchmark had to prefer:

`slower valid solution`

over
`faster invalid solution`

without relying on a weighted penalty.

### Result

It did.

The environments also had exhaustively provable optima, deterministic replay, state hashes, and corruption tests.

### Lesson

Constraint compliance belongs in the feasible-set definition, not as a soft reward that enough task performance can buy away.

## Milestone 3 - first optimality-transfer instrument test

### Goal

Test whether the harness can detect a useful signal in the transition from a strong trajectory to an optimal one on held-out task instances.

This was deliberately an instrument-calibration experiment rather than a claim about cross-game transfer.

### Environment

`MacroTrack` generated exact-progress tasks with synthetic but strictly improving ladders:

`primitive -> competent -> optimized -> frontier -> optimum`

Training distances:

`6, 8, 9, 10, 11, 12`

Held-out distances:

`13, 14, 15, 16`

The held-out optimum trajectories were not exposed to the learning conditions.

### Conditions

- uniform search,
- frontier imitation,
- optimum imitation,
- pooled frontier plus optimum,
- frontier-to-optimum delta.

The crucial control was that pooled and delta conditions saw the same frontier and optimum trajectories. Only the delta condition preserved which direction represented improvement.

### Frozen 20-replicate result

Median total candidate episodes needed to reach the exact optimum on all four held-out tasks:

| Condition | Median total episodes |
| --- | ---: |
| Uniform | 534.0 |
| Frontier imitation | 80.0 |
| Optimum imitation | 12.5 |
| Pooled frontier + optimum | 19.5 |
| Frontier-to-optimum delta | **9.0** |

### Interpretation that survived

The experimental harness can represent graded exposure, hide the strongest held-out references, and detect improvement-transition information when a deliberately simple transferable signal exists.

### Interpretation that was rejected

This was not evidence that a neural network had learned a domain-general ability to become superhuman. The task family shared action semantics and the learner was a transparent count-based model.

### Reference

- `docs/milestone-3-transfer.md`
- `experiments/milestone3_reference.json`

## Milestone 4 - neural cross-mechanic transfer

### Goal

Remove the stable action-name shortcut and replace the hand-written count learner with a neural model.

### Design

Training mechanic families:

- Plain
- Battery
- Cooldown

Held-out mechanic family:

- Heat

Every task received fresh opaque action aliases. The neural scorer did not receive the alias or mechanic-family identity.

However, the model still received structured numeric descriptors summarizing the action's true semantics, such as progress, cost, resource behavior, and pressure behavior.

The small neural model was:

`8 -> 32 -> 16 -> 1`

### Conditions

- uniform,
- shuffled transition direction,
- pooled frontier plus optimum,
- frontier-to-optimum delta,
- optimum imitation.

Again, the important transition controls used the same underlying trajectory data.

### Frozen 20-replicate result

| Condition | Median total episodes across held-out tasks | Exact-optimum success |
| --- | ---: | ---: |
| Uniform | 1023.5 | 28.1% |
| Shuffled transition direction | 1054.0 | 19.4% |
| Pooled frontier + optimum | 348.0 | 91.9% |
| Frontier-to-optimum delta | **322.5** | **94.4%** |
| Optimum imitation | **190.0** | **99.4%** |

The directed delta learner beat shuffled transition direction in all 20 paired replicates and beat the pooled same-data condition in 13 of 20.

But direct optimum imitation was stronger than delta.

### Interpretation that survived

Direction of improvement contained useful transferable information in this synthetic setting. Destroying the ordering hurt performance.

### Interpretation that was rejected

Milestone 4 did not show that learning the change from frontier to optimum was better than simply learning from optimal demonstrations.

This distinction matters. Optimum imitation must remain a strong baseline in future work.

### Reference

- `docs/milestone-4-neural-transfer.md`
- `experiments/milestone4_reference.json`

## Milestone 5 - infer action semantics through interaction

### Goal

Remove the strongest remaining hand-authored shortcut from Milestone 4.

The learner should no longer be told what an action does. It should press opaque actions, observe consequences, infer useful affordances, and then transfer what it learned from development performance ladders to a new mechanic family.

### Probe representation

For each permitted opaque action, a fixed interaction budget generated varied transitions. The system recorded observable before-and-after state and action cost.

Those transitions were summarized into a 49-dimensional empirical representation.

The scorer was:

`49 -> 48 -> 24 -> 1`

Probe interactions counted toward environment cost.

### Development discipline

Development families:

- Plain
- Battery
- Cooldown
- Heat
- Momentum

Each contributed 30 frontier-to-optimum ladders, for 150 development tasks total.

Reward/target candidates were selected only on development families.

A leave-one-development-family-out procedure selected a mixture by minimizing worst-family median environment interactions, then mean interactions, then mean success rate.

The selected mixture before final evaluation was:

`75% optimum imitation + 25% pooled frontier/optimum + 0% delta`

The fact that pure delta was assigned zero weight was already a development warning that its Milestone 4 success was not robust.

### Contamination correction

An early Overdrive family had already been exercised during diagnostics.

It was therefore not treated as pristine final evidence.

Rather than pretend it remained held out, the final test family was replaced with a new state-dependent `Combo` family after the selection rule had been frozen.

This is an important precedent for future milestones: once trained-model performance has been inspected on a supposed final set, that set is contaminated for method selection.

### Combo mechanic

Combo deliberately made action value state-dependent.

Some actions build combo charge. Another action converts the amount of accumulated combo into additional progress.

The same action can therefore have very different value in different states.

This was designed to expose methods that assign one global score to an action without understanding when to use it.

### Frozen final run

GitHub Actions run:

`32427935733`

Raw output SHA-256:

`a98ea99dd13f55b3c4bed626a68b63803b65082420fec2aecc3d171b52f06aea`

Frozen 20-replicate Combo result:

| Condition | Median total episodes | Exact-optimum success | Median environment interactions |
| --- | ---: | ---: | ---: |
| Uniform | 1113.5 | 11.9% | 12,323.5 |
| Shuffled transition direction | 902.0 | 32.5% | 13,947.5 |
| Pooled frontier + optimum | 822.0 | 46.3% | 8,252.5 |
| Development-selected mixture | **509.0** | **76.3%** | **6,142.5** |
| Optimum imitation | **435.5** | **80.6%** | **5,554.5** |
| Frontier-to-optimum delta | 1186.0 | 13.8% | 32,582.0 |

Paired comparisons for pure delta:

- versus optimum imitation: 0 wins, 20 losses,
- versus selected mixture: 0 wins, 20 losses,
- versus pooled: 2 wins, 18 losses,
- versus uniform: 9 wins, 9 losses, 2 ties.

### The important negative result

The simple global action-frequency delta did not survive the harder representation problem.

Milestone 4 approximately learned:

`Delta(a) = P(a | optimum) - P(a | frontier)`

That asks whether an action becomes more common in the better trajectory.

Combo demonstrated why that can be too lossy.

The useful rule may instead be:

`use action X after building state Y`

or:

`take setup action A because it increases the later value of action B`.

A bag-of-actions frequency comparison cannot represent that causal or sequential structure.

### Interpretation that survived

Cross-family learning still mattered strongly. Optimum imitation reached the exact final optimum in 80.6% of task-replicate evaluations versus 11.9% for uniform search, despite never seeing Combo trajectories.

So interaction-inferred representations can support useful transfer.

What failed was the particular global frequency representation of improvement.

### The new representation target

Future work should move from something like:

`P(action is useful)`

toward:

`P(action is useful | state, history, objective, constraints)`

and ultimately toward modeling the policy transformation:

`Delta_pi = transformation from strong policy to better policy`.

### Reference

- `docs/milestone-5-interaction-inference.md`
- `experiments/milestone5_reference.json`

## The cumulative story

The first five milestones progressively removed shortcuts:

1. define trustworthy benchmark semantics,
2. make validity and replay executable,
3. detect a planted improvement-transition signal,
4. learn improvement across changing mechanics with randomized action names,
5. infer action semantics through interaction and expose the failure of global action-frequency deltas under state dependence.

This progression matters because the ultimate task is not to construct a toy environment where the hypothesis is true.

It is to determine what representation and learning process, if any, actually lets an AI acquire a reusable ability to improve.

## What Milestone 6 inherits

Milestone 6 should not spend its main effort tuning the old delta weight.

Milestone 5 identified a representation failure.

The immediate question is:

> Can a state-conditioned, sequence-aware learner identify reusable local or multi-step policy changes that explain why the optimum trajectory is better than the frontier trajectory?

Strong future methods should compare decisions in context, preserve setup/payoff relationships, and model better continuations rather than compressing an entire trajectory into one action-frequency vector.

`Combo` is now historical evaluation data and may be used as development data in Milestone 6. New final families must be created and reserved before the final Milestone 6 evaluation.

The detailed plan is in `docs/milestone-6-research-plan.md`.