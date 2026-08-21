# Research Methodology and Experimental Integrity

LevelUp Bench is designed to make ambitious claims difficult to fake accidentally.

The project expects substantial method iteration. We may change architectures, objectives, reward mixtures, curricula, search procedures, trajectory representations, world models, or training algorithms as evidence accumulates.

The constraint is not "never change the method." The constraint is:

> Change methods using development evidence, then test frozen choices on genuinely untouched evaluation data.

## 1. Separate development, validation, and final evaluation

A useful milestone should distinguish at least two roles for data and preferably three:

### Development data

May be inspected freely.

Use it to:

- debug code,
- understand failure modes,
- design representations,
- tune hyperparameters,
- choose reward functions,
- compare architectures,
- build curricula,
- and decide which ideas deserve larger experiments.

### Validation or family-level holdout data

Used to choose among methods after development.

Prefer structurally meaningful holdouts such as an entire mechanic family, game family, rule family, or environment type rather than random examples from the same generator.

Validation can be reused during a milestone if the protocol explicitly treats it as model-selection data, but repeated reuse gradually turns it into development data.

### Final evaluation data

Used only after the method is frozen.

A final evaluation should not influence:

- architecture,
- reward weights,
- hyperparameters,
- search temperature,
- stopping rule,
- benchmark metric,
- action representation,
- or which baselines are reported.

If final performance is inspected and then the method is changed, the final set is no longer final.

It may become development data in the next milestone, but a new untouched final set is required for a new claim.

## 2. Treat contamination as a state change, not a moral failure

Accidental contamination will happen in research.

The correct response is simple:

1. document what was exposed,
2. stop calling that set untouched,
3. move it into development or historical evaluation,
4. create a new final holdout,
5. and keep the selection procedure frozen before evaluating the new holdout.

Milestone 5 established this precedent when an early Overdrive family was used during diagnostics and therefore discarded as final evidence.

Never hide contamination because replacing a final set is inconvenient.

## 3. Prefer family-level generalization tests

Random train/test splits can exaggerate transfer when examples share the same mechanics or generator artifacts.

Whenever possible, hold out a coherent structure:

- an entire mechanic family,
- an entire game,
- an emulator/system,
- a rule category,
- a genre,
- a task family,
- or a combination of these.

For the eventual speedrun experiment, the strongest result comes from an unseen game or game family, not another level from a game already present in training.

## 4. Multiple final families are stronger than one celebrated challenge

A single untouched final environment is better than tuning on the test set, but it is still vulnerable to luck and idiosyncratic compatibility.

As compute permits, prefer several predeclared final families.

For example:

`development families -> validation families -> final A + final B + final C`

A method that transfers across all three is more convincing than one that dominates one hand-picked challenge.

Do not drop a final family from the report because the result is inconvenient.

## 5. Predeclare the selection rule

Before final evaluation, record how the development evidence will choose the winning method.

Possible selection criteria include:

- minimize worst-family environment interactions,
- maximize median valid success across families,
- maximize area under a learning curve,
- minimize cost to reach a fixed performance threshold,
- or choose the Pareto-dominant method across performance and cost.

The rule should be chosen for scientific relevance, not because it happens to select the preferred method on current final data.

Milestone 5 used a robust family-level criterion before evaluating the final Combo family.

## 6. Paired randomness is the default

When comparing methods, use the same task instances and corresponding random seeds whenever the algorithms permit it.

Paired designs reduce variance and make comparisons more interpretable.

Record separately:

- model initialization seeds,
- environment generation seeds,
- probe/exploration seeds,
- rollout/search seeds,
- data-order seeds,
- and any stochastic augmentation seeds.

Do not silently rerun only failed seeds for one condition.

## 7. Equal information is as important as equal compute

A method may look better because it saw more useful information rather than because it learned a better representation.

Explicitly record exposure.

For example, if one method sees frontier and optimum trajectories, a same-data baseline should often see the same two trajectories while losing only the structure under study.

Good controls include:

- ordered pair versus shuffled order,
- paired trajectories versus pooled transitions,
- aligned sequence versus bag of actions,
- state-conditioned model versus action-only model,
- optimum plus frontier versus optimum alone,
- TAS-exposed training games versus otherwise identical non-TAS training games.

When information differs intentionally, say so and interpret the result as an information comparison rather than a pure algorithm comparison.

## 8. Equalize computational resources

Unless compute is the independent variable, match or report:

- model parameter count,
- training updates,
- optimizer steps,
- training FLOPs where practical,
- environment interactions,
- probe actions,
- number of searched states,
- candidate episodes,
- wall-clock budget,
- inference calls,
- tokens,
- and dollar cost.

Exact equality is not always meaningful across different algorithms, so preserve raw resource measurements rather than pretending one arbitrary budget makes every method equivalent.

## 9. Count exploration as part of performance

An agent that spends 100,000 actions reverse-engineering a task before executing a perfect five-action solution did not solve the task in five actions.

Report both:

- final execution efficiency,
- and total adaptation cost.

For a held-out task, total adaptation cost can include:

`probes + failed attempts + planning/search + successful execution`.

This is essential to the long-run question of how quickly an agent becomes superhuman.

## 10. Separate training reward from benchmark truth

Reward shaping is allowed.

Potential training signals may include:

- task progress,
- completion,
- time reduction,
- imitation losses,
- preference losses,
- advantage targets,
- curiosity,
- information gain,
- constraint penalties,
- curriculum bonuses,
- and search-derived targets.

But the final benchmark evaluator recomputes:

- validity,
- completion,
- objective value,
- and reference-relative performance

from the resulting trajectory or terminal state.

A high internal reward is never evidence of success.

## 11. Avoid softening hard constraints to make learning easier

A training algorithm may use shaped penalties for learning, but final validity remains binary for hard constraints.

A run that violates a hard category rule does not earn partial leaderboard credit because it is fast.

Constraint-level pass rates remain useful diagnostics, but ranking eligibility requires every applicable hard constraint to pass.

## 12. Strong baselines are mandatory

A proposed improvement-aware learner should compete against the strongest simple explanations.

At minimum, consider:

- uniform/random search,
- task-only RL or search,
- multi-task learning without performance ladders,
- frontier imitation,
- optimum imitation,
- pooled frontier plus optimum,
- shuffled or reversed improvement direction,
- and a capacity-matched sequence/state baseline without the proposed training target.

For real games, additional baselines may include:

- behavior cloning from human WRs,
- game-specific search,
- existing TAS automation,
- and search guided by a learned prior.

Do not weaken optimum imitation because it has repeatedly been strong in LevelUp.

## 13. Ablations should answer causal questions

An ablation is useful when removing one component tests a specific explanation.

Good examples:

- remove state conditioning,
- shuffle sequence order,
- remove performance-gap information,
- replace paired frontier/optimum segments with independent samples,
- remove TAS tier while preserving lower tiers,
- remove historical record transitions while keeping current best,
- remove natural-language constraint input,
- remove memory,
- remove search and use policy only,
- remove distillation after search.

Avoid dozens of arbitrary knobs that create a garden of forking paths without a causal interpretation.

## 14. Do not choose seeds after seeing results

Seed sets should be declared before aggregate evaluation.

If a run crashes due to an implementation defect, repair the defect and rerun the complete paired seed set when practical.

Do not replace a legitimate poor-performing seed with a new one.

## 15. Define stopping rules before expensive sweeps

Long experiments should have explicit stopping conditions such as:

- fixed training steps,
- fixed environment interactions,
- fixed wall-clock time,
- convergence criterion declared from development data,
- or a resource cap.

Adaptive early stopping may be useful, but it must apply comparably across conditions.

## 16. Preserve raw outcomes before summarizing

For serious milestone results, retain enough information to reconstruct summary statistics.

Prefer a hierarchy such as:

`raw per-seed outcomes -> aggregate artifact -> paper/README table`.

The git repository should usually contain the small aggregate and its provenance, while large raw data can live in ignored local artifacts or attached CI artifacts.

Record hashes when raw outputs are not committed.

## 17. Keep historical results immutable in spirit

Reference JSON files under `experiments/` are scientific records.

Do not overwrite an old result just because a new method is better.

If a bug invalidates a result:

1. document the bug,
2. identify affected commits and artifacts,
3. add a corrected artifact with new provenance,
4. update the interpretation,
5. preserve the existence of the earlier mistaken result where practical.

History is useful information.

## 18. Report negative results with the same precision as positive ones

A failed method should still report:

- exact configuration,
- split,
- seeds,
- resource budgets,
- performance,
- reliability,
- and likely failure mechanism.

Milestone 5 is a model example: pure global delta learning failed badly on the state-dependent Combo family, and that failure directly motivated a representation change.

## 19. Do not move the goalposts after final evaluation

After a frozen final result, do not decide that a different metric was "really" primary because it makes the preferred condition look better.

Secondary metrics can reveal useful tradeoffs, but the predeclared primary metric remains primary.

If the metric itself was poorly chosen, say so and design the next milestone with a better metric and a new final holdout.

## 20. Use confidence intervals and paired comparisons when scale justifies them

For stochastic experiments, report distributions rather than only means.

Useful summaries include:

- median,
- mean,
- interquartile range,
- bootstrap confidence intervals,
- paired win/loss/tie counts,
- paired effect sizes,
- success probability by budget,
- and learning curves.

For larger studies, consider paired nonparametric tests or hierarchical models that account for task and seed effects.

Do not worship p-values. The effect size and experimental design matter more.

## 21. Reliability claims require much more data than capability claims

If a system produces zero failures across `N` roughly independent opportunities, the rule of three gives an approximate 95% upper confidence bound of:

`3 / N`

on the failure probability.

Roughly:

- demonstrating failure probability below `1e-3` needs about 3,000 clean opportunities,
- below `1e-4` needs about 30,000,
- below `1e-5` needs about 300,000,
- below `1e-6` needs about 3,000,000,

assuming the opportunities are representative and sufficiently independent.

Real failures are correlated and distribution shift matters, so stratified and adversarial evaluation is still required.

Do not claim "1,000 times safer" from a few hundred trials.

## 22. Distinguish best-case discovery from dependable competence

A search system may occasionally discover a spectacular trajectory.

Report separately:

- best valid performance,
- median valid performance,
- valid success probability,
- probability of reaching a threshold within a fixed adaptation budget,
- and repeated reliability such as pass^k or all-pass@k.

The eventual economic agent must be dependable, not merely capable of one viral demonstration.

## 23. Verifier integrity is part of the benchmark

A verifier can itself be wrong or exploitable.

Add tests for:

- false positives,
- false negatives,
- incomplete outcomes,
- corrupted trajectories,
- evaluator dormancy,
- mismatched task/version identities,
- replay divergence,
- and adversarial attempts to satisfy the checker without satisfying the intended task.

As tasks become more complex, maintain a benchmark self-audit suite inspired by ST-WebAgentBench and state-verifiable benchmarks.

## 24. Prefer deterministic verification where possible

For games and synthetic tasks, exploit exact state aggressively.

The agent can be restricted to ordinary observations while the evaluator uses privileged state.

For real office work, deterministic state changes, database contents, document diffs, and explicit policy checks are preferable to an LLM judge when possible.

LLM evaluators can be supplemental, but uncertainty should be visible.

## 25. Human and TAS data require provenance

A future reference trajectory should state at least:

- source URL or archive identity,
- game/system/version,
- category/ruleset,
- date,
- runner/author when applicable,
- claimed performance,
- emulator/toolchain,
- whether independently replayed,
- and content hash.

Do not merge categories or versions casually.

A human WR and a TAS may optimize different rule sets and therefore be incomparable.

## 26. Novelty must survive comparison with existing automation

An agent beating a TAS with more brute-force search is interesting engineering but does not by itself establish LevelUp's transfer hypothesis.

For a claim about learning to optimize, compare against:

- blind or hand-engineered search,
- game-specific automated TAS systems where available,
- and equal state-expansion or compute budgets.

A learned prior that reduces required search by 100 times may be scientifically meaningful even before it produces a new record.

## 27. Keep research and public narrative separate

The public story can be simple:

> Can AI learn to beat the best?

The internal scientific claim must remain precise.

Do not let a compelling headline dictate the experimental interpretation.

## Checklist before a final milestone run

Before evaluating a final holdout, confirm in writing:

- [ ] The final tasks/families have not been used for trained-model diagnostics.
- [ ] Architecture is frozen.
- [ ] Training objective is frozen.
- [ ] Hyperparameters are frozen.
- [ ] Search/inference procedure is frozen.
- [ ] Resource budgets are frozen.
- [ ] Primary metric and selection rule are frozen.
- [ ] Seed set is frozen.
- [ ] Strong baselines are included.
- [ ] Same-data controls are included where relevant.
- [ ] Exposure manifests are correct.
- [ ] Evaluators and replay tests pass.
- [ ] Raw result location and provenance plan are defined.
- [ ] No one plans to tune after seeing the final result.

If any box fails, the run is development, not final.