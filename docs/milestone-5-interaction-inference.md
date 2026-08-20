# Milestone 5: Interaction-Inferred Transfer

Milestone 5 removes the strongest simplifying assumption from Milestone 4: the learner is no longer told what an action does.

The scientific question is now:

> Can a learner infer the affordances of opaque actions through interaction, then use demonstrations of better performance to search more efficiently in a new mechanic family?

The answer from this milestone is mixed and useful. Interaction-inferred representations work well enough for several learned priors to beat uniform search, but the simple frontier-to-optimum delta method does **not** survive the new state-dependent final mechanic. Direct optimum imitation is strongest on the frozen final evaluation.

That failure is preserved as part of the result rather than tuned away.

## What changed from Milestone 4

Milestone 4 supplied each action with a structured numeric descriptor such as progress, tick cost, resource use, pressure change, and whether the action was an enabler.

Milestone 5 removes those descriptors.

The agent-facing action list contains only opaque aliases such as:

```text
acf129e
ab88312
```

The learner does not receive:

- action names with semantic meaning,
- hidden progress values,
- hidden tick costs,
- resource-use flags,
- pressure-change flags,
- mechanic-family identifiers,
- or the evaluator's transition function.

It does still receive compact observable state:

- current progress,
- target progress,
- elapsed ticks,
- resource fraction,
- pressure fraction,
- and the aliases currently available.

The natural-language hard rule remains backed by structured constraint metadata in this milestone. Milestone 5 isolates mechanic inference, not language parsing.

## Learning action semantics through probes

Before search, each condition receives the same fixed probe budget.

For every permitted action, the probe system:

1. creates varied reachable states using random action prefixes,
2. presses the target opaque action when it is available,
3. observes the before and after state,
4. records the action cost and state change,
5. repeats this six times per action.

Each observed transition contains 12 quantities, including progress before the action, remaining distance, resource and pressure state, progress change, tick change, resource change, pressure change, and completion.

For each opaque action, the system summarizes the probe transitions using mean, standard deviation, minimum, maximum, and coverage. This produces a 49-dimensional empirical action representation.

The neural scorer is:

```text
49 -> 48 -> 24 -> 1
```

The key distinction is that these features are consequences the learner observed by acting. They are not copied from the hidden simulator definition.

Probe actions are counted as environment interactions. Exploration is not free.

## Development and final evaluation discipline

Method iteration is allowed on development data. Final-family adaptation is not.

Development families are:

```text
Plain
Battery
Cooldown
Heat
Momentum
```

Each contributes 30 frontier-to-optimum ladders, for 150 development tasks total.

An earlier Overdrive candidate was used during development diagnostics and was therefore not treated as a pristine final test. Rather than pretend otherwise, it was discarded as final evidence.

The reserved final family is **Combo**.

Combo was specified after the model-selection procedure was frozen and before its model performance was evaluated.

Its mechanics are deliberately state-dependent. Some actions build combo charge. Another action converts the amount of charge currently accumulated into additional progress. The same opaque action can therefore have a different effect depending on the state in which it is used.

That is an important increase in difficulty. A global belief that "action X is good" is no longer sufficient.

## Reward and target selection

The user hypothesis explicitly allows different ways of rewarding or representing improvement. Milestone 5 therefore compares three learned signals:

- `optimum`: imitate how frequently actions appear in optimal demonstrations,
- `pooled`: imitate the pooled frontier and optimum demonstrations without preserving direction,
- `delta`: predict how action frequency changes from frontier to optimum.

Mixtures are selected only on development families.

The candidate grid uses quarter-step convex mixtures of the three signals. A leave-one-development-family-out procedure chooses a mixture using this criterion:

1. minimize the worst held-out development family's median environment-interaction count,
2. break ties using mean interaction count,
3. then use mean success rate.

This favors robustness rather than a method that is spectacular on one mechanic and brittle on another.

The selected mixture before final evaluation was:

```text
75% optimum imitation
25% pooled frontier + optimum
0% transition delta
```

That selection itself was already informative. On development families, pure delta was not robust enough to survive the selection criterion once action semantics had to be inferred.

## Frozen final evaluation

The frozen evaluation uses:

- 8 Combo tasks,
- 20 paired replicates,
- 150 candidate episodes per task,
- 6 probes per action,
- identical final tasks across conditions,
- identical probe seeds across conditions,
- identical search seeds across conditions.

The benchmark knows the exact optimum only for verification. No Combo frontier or optimum trajectory is exposed to any learner.

### Results

| Condition | Median total episodes | Valid optimum success | Median environment interactions |
| --- | ---: | ---: | ---: |
| Uniform | 1113.5 | 11.9% | 12,323.5 |
| Shuffled transition direction | 902.0 | 32.5% | 13,947.5 |
| Pooled frontier + optimum | 822.0 | 46.3% | 8,252.5 |
| **Development-selected mixture** | **509.0** | **76.3%** | **6,142.5** |
| **Optimum imitation** | **435.5** | **80.6%** | **5,554.5** |
| Frontier-to-optimum delta | 1186.0 | 13.8% | 32,582.0 |

The full deterministic summary is stored in `experiments/milestone5_reference.json`.

The raw GitHub Actions output used to create that summary has SHA-256:

```text
a98ea99dd13f55b3c4bed626a68b63803b65082420fec2aecc3d171b52f06aea
```

and was produced by GitHub Actions run `32427935733`.

## The negative result matters

Milestone 4 suggested that preserving the direction of improvement was useful. It was reasonable to ask whether that signal would become even more valuable once action semantics had to be learned.

It did not.

Pure delta lost to direct optimum imitation in all 20 paired replicates on total episodes. It also lost to the development-selected mixture in all 20.

Against uniform search, delta produced 9 wins, 9 losses, and 2 ties. Its 13.8% task success rate was only slightly above uniform's 11.9%, while consuming far more environment interactions.

This means the simple Milestone 4 interpretation was incomplete.

The likely reason is structural.

Milestone 4 learned one global score for each action based on a stable descriptor. Milestone 5 still ultimately compresses each opaque action into one global probe summary and one global score.

But in Combo, action quality is conditional on state:

```text
same action + low combo charge  -> modest value
same action + high combo charge -> high value
```

A trajectory improvement can therefore depend on **when** an action is used, not merely whether its overall frequency increased.

A frequency delta throws away that information.

This is a scientifically useful failure because it identifies a concrete missing representation:

```text
P(action is useful)
```

is not enough.

We need something closer to:

```text
P(action is useful | state, history, objective, constraints)
```

and eventually:

```text
What transformation of the policy produced the performance improvement?
```

## Why optimum imitation still transfers

Direct optimum imitation remains surprisingly strong despite receiving no final-family demonstrations.

That does not mean "just imitate TAS" has solved LevelUp.

The synthetic development families share broad structural regularities. Efficient behavior often uses actions whose observed consequences look productive under the current state. A neural model trained on optimal action frequencies can exploit those regularities after probing the new environment.

The important result is comparative: the current delta target loses too much sequence and state information to exploit the final mechanic.

## Why the selected mixture did not win

The development-selected 75% optimum / 25% pooled mixture generalized well, but direct optimum imitation was still better on Combo.

We do not change the selection rule after seeing this result.

That mismatch demonstrates ordinary distribution shift. A model-selection criterion optimized across five development families is not guaranteed to select the best method for a genuinely new family.

This is exactly why the final family was held out.

Future methods should be judged by repeated predeclared family-level holdouts rather than by tuning on one celebrated final task.

## What Milestone 5 establishes

Milestone 5 does establish several pieces of infrastructure needed for the larger project:

1. Opaque action aliases can replace hand-authored action descriptors.
2. The learner can construct useful action representations from active interaction.
3. Probe cost can be explicitly included in cognitive/environment efficiency.
4. Reward mixtures can be selected on development families without touching a final family.
5. State-dependent final mechanics can expose methods that looked strong on simpler held-out tasks.
6. Negative results can be retained without changing the benchmark until the preferred hypothesis wins.

The strongest learned methods still substantially outperform uniform search on Combo. Optimum imitation reaches the exact optimum in 80.6% of task-replicate evaluations versus 11.9% for uniform search.

So learned cross-family priors remain useful. What fails is the particular **global frequency-delta representation** of improvement.

## What Milestone 5 does not show

It does not establish that:

- an AI has learned a general ability to become superhuman,
- real speedrun or TAS knowledge transfers across games,
- natural-language constraints have been learned rather than supplied structurally,
- pixel-level game mechanics can be inferred,
- the learner can reason over long trajectory transformations,
- or the method transfers to office work.

All environments remain synthetic exact-progress tasks with compact state observations.

## Implication for Milestone 6

The next experiment should not merely turn a reward weight knob.

The failure points toward a representation change.

Milestone 6 should make the improvement learner **state-conditioned and sequence-aware**. Instead of predicting one score for an action, it should learn from tuples such as:

```text
(state, action, next_state, trajectory role, performance gap)
```

and compare aligned portions of frontier and optimum trajectories.

Candidate approaches include:

- a small transition encoder plus state-conditioned policy head,
- contrastive learning between frontier and optimum state-action distributions,
- preference learning over trajectory segments,
- advantage-style targets that ask which local decisions account for the performance gap,
- sequence models that predict the better continuation from the same or similar state,
- distillation from search traces after an optimum is found.

The key scientific target remains unchanged:

> Learn not merely which actions good solutions contain, but the reusable decision process that turns a strong solution into a better one.

Milestone 5 shows that this distinction is real enough to matter even in tiny synthetic worlds.
