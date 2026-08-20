# Metrics and Reporting

LevelUp Bench is deliberately multi-dimensional.

A single scalar can make a leaderboard convenient while hiding the exact tradeoffs the project cares about: validity, capability, reliability, adaptation speed, execution speed, and cognitive cost.

## 1. Lexicographic benchmark semantics

For hard-constrained tasks, interpret outcomes in this order:

`Validity > Completion > Quality > Performance > Efficiency`

The meaning is lexicographic, not additive.

A faster invalid run does not outrank a slower valid run.

A cheap incomplete run does not outrank a costly correct run.

Efficiency matters after the task is done validly and at acceptable quality.

Training may use shaped rewards, but benchmark ranking follows the independent evaluator.

## 2. Validity

For a task with applicable hard constraints `C_1 ... C_n`, define:

`Valid(tau) = 1` only if every applicable required constraint was evaluated and passed.

Report:

- whole-run validity,
- per-constraint pass/fail,
- and, for large constraint sets, constraint success rate as a diagnostic.

Do not reinterpret 19 of 20 hard constraints as a 95% valid run for ranking purposes.

This distinction parallels the useful separation in AgentIF between individual constraint satisfaction and all-constraints satisfaction.

## 3. Completion

Record whether the intended task outcome was actually achieved.

Completion and validity are separate:

- valid but incomplete,
- completed but invalid,
- valid and completed.

Only the last category is normally eligible for performance ranking.

## 4. Task performance

Use the environment's natural metric whenever possible:

- elapsed frames,
- elapsed ticks,
- score,
- resource use,
- objective value,
- quality score,
- or another deterministic outcome.

Always record the direction:

- minimize,
- maximize.

## 5. Reference-relative performance

Raw performance is essential, but performance ladders allow useful normalized views.

For a time-minimization speedrun with a human world record `T_WR`, TAS reference `T_TAS`, and agent result `T_A`, define TAS Gap Closure:

`G = (T_WR - T_A) / (T_WR - T_TAS)`

Interpretation:

- `G < 0`: agent is slower than the human world record,
- `G = 0`: agent matches the human world record,
- `0 < G < 1`: agent lies between human WR and TAS,
- `G = 1`: agent matches the TAS reference,
- `G > 1`: agent beats the reference TAS.

Do not call `G = 1` mathematically optimal unless optimality has actually been proven.

For non-time tasks, define an analogous reference-normalized gap only when the meaning is clear.

Keep the original raw metric visible.

## 6. Preserve the full skill ladder

Do not reduce a rich ladder to one normalized number.

Where data exists, display anchors such as:

- ordinary human median,
- experienced human,
- elite human,
- human WR,
- historical WRs,
- historical TASes,
- current TAS,
- proven optimum if one exists,
- agent best,
- agent median.

The geometry between these points is part of the research object.

## 7. Adaptation/sample efficiency

The central LevelUp question is often not only how good the final policy becomes, but how quickly it gets there on an unseen task.

Useful metrics include:

- episodes to first valid completion,
- episodes to first human-level result,
- episodes to first elite-human result,
- episodes to first human-WR result,
- episodes to first TAS-level result,
- episodes to exact optimum in synthetic tasks,
- environment interactions to each threshold,
- searched states to threshold,
- wall time to threshold,
- inference compute to threshold.

Prefer learning/discovery curves:

`performance versus adaptation budget`

rather than reporting only the final point.

## 8. Improvement Dynamics Curve

Inspired by OmniGameArena, maintain a curve showing best valid performance after increasing amounts of experience.

For LevelUp, important horizontal axes may include:

- environment interactions,
- candidate episodes,
- searched states,
- wall-clock seconds,
- inference calls,
- tokens,
- or total cost.

Important vertical axes may include:

- raw performance,
- human percentile,
- TAS gap closure,
- probability of reaching exact optimum,
- or valid completion quality.

A method that reaches the same final result with one tenth the adaptation cost has learned something economically meaningful.

## 9. Area under the improvement curve

When comparing learning speed across the full budget rather than one threshold, an area-under-curve metric can be useful.

Use a normalized performance scale only when it has a defensible interpretation.

Always report the underlying curve as well.

Do not let AUC become a substitute for understanding where one method is better.

## 10. Best versus median versus distribution

Always distinguish:

- best valid performance,
- median valid performance,
- mean performance where meaningful,
- distribution or quantiles,
- and success probability.

A system that finds one extraordinary trajectory in a million attempts is different from one that reliably produces near-optimal behavior.

Both may be scientifically interesting, but they answer different questions.

## 11. Reliability

Report repeated valid-success probability.

For a success probability `p`, repeated-run reliability can be summarized with metrics analogous to `pass^k` or all-pass@k depending on the question.

If the requirement is that the system succeeds on all `k` independent repetitions, then under a stationary approximation:

`P(all k pass) = p^k`.

Empirical repeated-run evaluation is preferable when errors are correlated.

For office-like tasks, a useful metric may be:

`all hard constraints satisfied AND task completed correctly across k repeated variants`.

## 12. Rare failure claims

If zero failures occur in `N` roughly independent representative opportunities, the approximate 95% rule-of-three upper bound on the true failure probability is:

`3 / N`.

Examples:

| Zero failures across | Approx. 95% upper failure bound |
| ---: | ---: |
| 3,000 | 1e-3 |
| 30,000 | 1e-4 |
| 300,000 | 1e-5 |
| 3,000,000 | 1e-6 |

These are only rough binomial intuitions. Correlation, adversarial cases, and distribution shift make real reliability harder.

Do not derive dramatic safety multipliers from small samples.

## 13. Environment interaction cost

Count all task interactions that were needed for adaptation, including:

- probes,
- failed episodes,
- exploratory actions,
- search rollouts,
- successful execution,
- and resets if resets carry meaningful cost.

Where useful, report separately:

`adaptation interactions`

and:

`final execution interactions`.

This distinction becomes important when comparing search-heavy systems with distilled policies.

## 14. Search cost

For search-guided agents, record:

- nodes/states expanded,
- transitions simulated,
- branches evaluated,
- maximum depth,
- cache hits,
- emulator frames advanced,
- and wall time.

When comparing a learned search prior against JaffarPlus-like or other game-specific search, equal state-expansion budgets can be more informative than equal wall time.

Report both when possible.

## 15. Cognitive/inference efficiency

For LLM or multimodal agents, record:

- input tokens,
- cached input tokens where available,
- output tokens,
- model calls,
- tool calls,
- screenshots or frames consumed,
- inference latency,
- wall time,
- estimated monetary cost.

For small neural policies, record:

- parameter count,
- forward passes,
- device,
- inference time,
- and optionally FLOPs or MACs when practical.

A generic cognitive-cost representation is:

`C(tau) = alpha*tokens + beta*tool_calls + gamma*environment_actions + delta*wall_time + epsilon*dollars`

Do not force arbitrary coefficients into the official benchmark. Preserve the components so users can construct their own cost functions.

## 16. Training cost

Separate one-time training cost from per-task adaptation and execution cost.

A deployable system may reasonably spend substantial compute once if it then performs millions of tasks cheaply.

Report, when relevant:

- total training environment interactions,
- optimizer steps,
- accelerator hours,
- wall time,
- peak memory,
- data volume,
- estimated dollar cost.

## 17. Human comparison requires context

A claim such as "superhuman" should say on which dimension.

Examples:

- faster than the human WR under the same category,
- higher score than the best known human,
- lower error rate than humans on a fixed benchmark,
- faster adaptation than a human novice,
- or better cost/reliability despite similar quality.

Do not use "superhuman" as a global property based on one score.

## 18. TAS comparison requires category and version matching

A TAS and human speedrun are comparable only if relevant assumptions align:

- same game/version/region where necessary,
- same starting state,
- same category definition,
- same timing convention,
- compatible emulator assumptions,
- same completion criterion.

If they differ, report separate ladders or explicitly explain the normalization.

## 19. Paired comparison reporting

For matched tasks/seeds, report pairwise outcomes:

`left wins / right wins / ties`.

Also report the paired difference distribution, for example:

`Delta interactions = interactions_A - interactions_B`.

A bootstrap confidence interval around the median or mean paired difference can be useful in larger studies.

## 20. Family-level robustness

Aggregate metrics can hide catastrophic failure on one family.

Always inspect family-level results.

Useful robust summaries include:

- minimum family success rate,
- maximum family adaptation cost,
- median across family medians,
- and worst-family performance.

This is especially important when selecting a method intended to transfer.

## 21. Constraint metrics for future natural-language tasks

When tasks contain many independent constraints, report both:

### Constraint Success Rate

Fraction of applicable individual constraints satisfied.

Useful diagnostically.

### Instruction Success Rate

Fraction of tasks for which every applicable hard constraint and the task objective were satisfied.

This is the meaningful deployability measure.

## 22. Ask/escalate behavior

For future ambiguous policy tasks, report whether the agent:

- acted correctly without clarification,
- asked an appropriate clarification,
- escalated appropriately,
- guessed incorrectly,
- or violated a hard rule.

Clarification has a small efficiency cost but can dominate an unsafe guess.

Do not reward never asking questions if the task is genuinely underspecified.

## 23. Pareto reporting

A useful result may not dominate on every dimension.

Plot or tabulate the Pareto frontier across quantities such as:

- valid success,
- task performance,
- adaptation interactions,
- wall time,
- inference cost,
- and training cost.

A method can be valuable because it achieves 99.99% of the best performance at 1% of the cost.

## 24. No universal LevelUp score yet

Do not invent a single "LevelUp Score" that mixes validity, speed, reliability, and cost with arbitrary weights.

Task-specific normalized metrics are acceptable when transparent.

The benchmark should make tradeoffs inspectable rather than hiding them.

## 25. Canonical milestone result table

A mature synthetic milestone should normally report at least:

| Field | Example |
| --- | --- |
| Final families | `family_a, family_b` |
| Replicates | `20` |
| Valid exact-optimum success | `0.81` |
| Median episodes to threshold | `436` |
| Median environment interactions | `5,555` |
| Worst-family success | `0.68` |
| Paired wins vs optimum imitation | `12 / 8 / 0` |
| Model parameters | `...` |
| Device | `cpu` or `mps` |
| Wall time | `...` |
| Exposure | manifest/hash |
| Git SHA | commit |
| Raw artifact hash | SHA-256 |

For real speedruns, add:

- human WR,
- TAS time,
- agent time,
- TAS gap closure,
- category,
- game version,
- emulator verification status.

## 26. The graph we ultimately want

The most important future LevelUp figure may be simple:

`performance relative to human/TAS ladder`

versus

`experience or compute on the held-out task`.

Plot several learners on the same axes:

- no ladder training,
- expert imitation,
- WR imitation,
- performance-ladder training without TAS,
- performance-ladder training with TAS.

If the TAS-trained system crosses human WR or approaches the hidden TAS using substantially fewer interactions, the core hypothesis becomes visually and scientifically easy to understand.