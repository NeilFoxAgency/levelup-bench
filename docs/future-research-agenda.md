# Future Research Agenda

This document preserves important LevelUp ideas that are not immediate Milestone 6 requirements.

They should not distract from the current experiment, but they are part of the intended research program and should not be lost as the codebase becomes more specialized.

## 1. Performance ladders richer than two stages

Milestones 4-6 focus heavily on frontier-to-optimum pairs for controlled experiments.

The eventual data is richer:

`novice -> competent -> expert -> elite -> WR -> historical TAS -> current TAS`.

Open questions:

- Is learning from the whole ordered ladder better than only the final pair?
- Do different transitions teach different optimization primitives?
- Can the learner predict what the next performance transition will look like before seeing it?
- Can historical improvements reveal reusable abstractions that one final optimum hides?
- Does the benefit saturate after the first superhuman tier, or do multiple TAS generations continue to help?

## 2. Predict the next improvement before revealing it

A strong meta-learning test is:

1. show the learner performance levels `H0 ... Hk`,
2. hide `H(k+1)`,
3. ask it to predict what kind of change will produce the next improvement,
4. allow environment interaction/search,
5. reveal `H(k+1)` only afterward for comparison.

This directly asks whether the learner has internalized how expertise progresses rather than only copying the best observed behavior.

## 3. Safe exploration versus deployment

A capable optimizer may need to discover actions that would be prohibited in deployment.

Possible separation:

### Sandbox discovery mode

The agent may explore mechanics broadly inside an isolated environment to understand what is possible.

### Constrained execution mode

The same agent receives a category/policy and must optimize only inside the permitted set.

A useful test is whether the agent can say:

> This exploit saves 20 seconds, but it violates Glitchless rule 4, so I will not use it in this run.

This is more desirable than either blind compliance through ignorance or unrestricted reward maximization.

## 4. Ask, clarify, and escalate as valid actions

Real policies are often ambiguous.

Future environments should include an explicit low-cost action such as:

- `ASK_MODERATOR`,
- `ASK_USER`,
- `ESCALATE`,
- or `REQUEST_AUTHORIZATION`.

The learner should discover when clarification is worth the cost.

A curriculum can progress from:

- fully mechanical rules,
- to state-dependent rules,
- to natural-language edge cases,
- to conflicting rules,
- to underspecified cases where acting without clarification is invalid.

Never train “never ask questions” as a proxy for efficiency.

## 5. Nested instruction hierarchy

Long-run constrained agents may receive rules from multiple levels:

`law -> organization policy -> manager instruction -> immediate task request -> untrusted environment data`.

Future LevelUp variants should test:

- which instruction wins under conflict,
- whether lower-level requests can override higher-level prohibitions,
- whether tool/environment content can inject new instructions,
- and whether policy updates remain active across long horizons.

Relevant prior art includes ST-WebAgentBench, tau-bench, AgentIF, and AgentDojo.

## 6. Constraint revisions during a task

A policy may change after the agent has already formed a plan.

Test whether the system can:

1. notice a rule revision,
2. invalidate obsolete plans,
3. preserve already completed valid work,
4. recompute the feasible policy,
5. continue efficiently.

This is closer to real organizational work than a static prompt.

## 7. Learn world models rather than only action preferences

Efficient superhuman optimization may require a compact model of environment dynamics.

Potential direction:

`observations + actions + consequences + natural-language rules -> learned executable/latent world model -> planning/search`.

The learner can then test counterfactuals without spending one real environment interaction per idea.

Important comparison:

- model-free adaptation,
- learned world model from scratch,
- world model pretrained across prior game families,
- world model plus performance-ladder improvement learner.

The relevant scientific question remains transfer, not merely whether planning helps.

## 8. Learned search priors versus blind search

Existing TAS automation can search enormous state spaces effectively.

A LevelUp-trained policy or value model may be most useful as a search prior.

Compare under an equal state-expansion budget:

- blind search,
- hand-engineered heuristic,
- game-specific learned heuristic,
- cross-game LevelUp prior.

A large reduction in states required to reach the same TAS-level solution is a meaningful result even without a new record.

## 9. Search-then-distill loop

A long-run training cycle may be:

1. spend large compute to discover a superior solution,
2. preserve the search/decision trace,
3. identify which reasoning/search was actually necessary,
4. distill the result into a smaller/faster policy,
5. rerun under a lower cognitive budget,
6. repeat.

This mirrors the speedrun idea at the level of cognition.

The endpoint is not merely a smart planner. It is an optimizer that becomes cheaper after learning.

## 10. Cognitive TAS

For an agent workflow, define a successful cognitive trajectory containing:

- model calls,
- reasoning tokens,
- tool calls,
- state inspections,
- file reads,
- environment actions,
- corrections,
- and final output.

Then search for a lower-cost trajectory that produces the same valid result.

Questions:

- Which calls were unnecessary?
- Which observations could have been cached?
- Which subproblems can be compiled into a reusable skill?
- Can the agent learn to allocate deep reasoning only where consequence sensitivity requires it?

Do not optimize raw token count at the expense of correctness.

## 11. Diverse game curriculum rather than literally every game

The original intuition involved training across essentially every game type.

In practice, curated diversity may outperform indiscriminate volume.

Measure marginal transfer from adding:

- platformers,
- puzzle games,
- strategy games,
- management games,
- logistics games,
- document-processing games,
- resource-allocation games,
- social/negotiation games,
- and real-time control tasks.

The useful variable may be diversity of decision structure rather than title count.

## 12. Economically relevant games

Games such as management, logistics, document, scheduling, and simulation tasks may bridge the gap to office work better than pure arcade control.

A later curriculum should overweight tasks that teach:

- prioritization,
- queues,
- resource allocation,
- compliance,
- document verification,
- scheduling,
- staffing,
- inventory,
- and multi-objective planning.

The goal is not to claim the game is realistic. It is to increase structural overlap with useful work.

## 13. Synthetic office environments

Before touching live companies, construct office worlds with deterministic state.

Potential tasks:

- payroll reconciliation,
- invoice approval,
- scheduling,
- spreadsheet cleanup,
- CRM updates,
- procurement,
- ticket triage,
- policy-compliant customer service,
- expense review,
- inventory/order workflows.

Each task should have:

- a natural-language objective,
- an employee handbook/policy set,
- permissions,
- hidden deterministic verifier,
- tempting faster prohibited actions,
- and measurable cost.

ERP-Bench and tau-bench are particularly relevant prior art.

## 14. Office performance ladders

The speedrun idea can generalize beyond games if we can create ranked work traces:

`novice workflow -> competent workflow -> expert workflow -> optimized workflow -> solver/search-assisted workflow`.

The final tier need not be called TAS, but it can play the same scientific role: a demonstrated trajectory beyond ordinary human efficiency.

Test whether the agent learns reusable workflow transformations such as:

- batch operations,
- better query planning,
- avoiding redundant reads,
- validating earlier,
- parallelizing independent work,
- using formulas rather than manual edits,
- or invoking the right tool at the right time.

## 15. Company digital twins

A later deployment bridge may be a sandbox copy of a company's workflows, policies, schemas, and synthetic records.

Train/evaluate in the digital twin before granting real permissions.

Progression:

`synthetic office -> company digital twin -> shadow mode -> bounded autonomy`.

Real-world access controls and transaction limits remain external safeguards even for a strong policy model.

## 16. Reliability as an optimization target

Economic usefulness depends heavily on rare catastrophic errors.

Training should eventually optimize not only mean performance but tails:

- avoid major compliance failures,
- avoid unauthorized transactions,
- detect uncertainty,
- ask for clarification,
- verify consequential actions,
- remain stable across repeated runs.

This may require risk-sensitive objectives rather than ordinary expected reward.

## 17. Adaptive cognition

Not every decision deserves the same amount of reasoning.

A future agent should learn:

- cheap policy execution for familiar low-risk steps,
- more search for novel/high-stakes states,
- verification before irreversible actions,
- escalation under unresolved ambiguity.

This is a meta-control problem over cognition itself.

## 18. Verifier gaming stress tests

As agents become stronger, deliberately search for ways they can exploit the benchmark.

Possible tests:

- corrupt state/log files,
- exploit verifier assumptions,
- terminate before forbidden state is observed,
- manipulate clocks/timing,
- cause evaluator exceptions,
- exploit save-state/version mismatch,
- satisfy a proxy without intended completion.

A benchmark that never tests evaluator gaming may eventually measure hacking skill instead of task skill.

## 19. Hidden/random audits

For high-stakes future environments, not every check needs to be visible or deterministic from the agent's perspective.

Possible defense-in-depth:

- hidden verifier checks,
- randomized audit subsets,
- independent replay,
- multiple evaluators,
- immutable/tamper-evident logs,
- post-run counterfactual checks.

If evaluators disagree or cannot establish validity, return `not verified` rather than guessing.

## 20. Continual learning without catastrophic forgetting

A real digital worker will encounter new tasks and policies continuously.

Future LevelUp questions:

- Can the system learn a new game without destroying old skills?
- Can it compile a newly discovered optimization into reusable memory?
- Can it distinguish local game quirks from general optimization principles?
- Can policy changes update behavior without erasing unrelated competence?

Continual Harness may provide useful engineering patterns even though the scientific target differs.

## 21. Foundation-model plus fast-policy hierarchy

For rich games, one large multimodal model per frame may be too slow and costly.

Potential architecture:

- frontier model for strategic reasoning and novel interpretation,
- small recurrent/control model for fast actions,
- shared memory/world state,
- planner/search component,
- distillation of repeated strategies into the fast policy.

Evaluate the whole system on useful work per unit cognition, not only win rate.

## 22. Small-model contamination control

Early real-game studies should include small models trained from controlled data.

Why:

A frontier foundation model may already have memorized public speedrun/TAS discussions or video transcripts.

A small model trained only on declared LevelUp data gives a cleaner answer to:

> Did the performance-ladder curriculum itself teach transfer?

Later studies can add frontier pretrained models and measure how much the same curriculum improves them.

## 23. Demonstration quality and uncertainty

Human trajectory labels are not perfect.

A “better time” can arise from:

- genuinely better strategy,
- better execution of the same strategy,
- lucky randomness,
- timing differences,
- rule/version changes.

The learner may need latent variables for:

- strategic improvement,
- execution skill,
- noise,
- and environment stochasticity.

Historical TAS trajectories are cleaner for deterministic strategic comparison but still may change emulator/game assumptions over time.

## 24. Learn from failure trajectories too

Speedrunners and agents produce enormous numbers of failed attempts.

Those contain information about:

- fragile strategies,
- recovery cost,
- boundary conditions,
- state traps,
- and near-misses.

Potentially train:

`successful optimum + frontier + failed variants`

rather than only successful ladders.

Do not let failure volume overwhelm the structured positive ladder without an explicit learning purpose.

## 25. Active experiment selection

Instead of sampling attempts uniformly, let the learner choose experiments that maximally reduce uncertainty about environment dynamics or policy quality.

This is relevant to Milestone 5's fixed probes and future game-mechanic discovery.

Measure information gained per environment interaction.

A strong agent should learn which questions to ask the world.

## 26. Generalization to AI research itself

A speculative far-future extension is to apply the same structure to AI R&D:

- task specification,
- hard constraints,
- baseline implementation,
- human expert result,
- state-of-the-art result,
- search/computation-assisted reference,
- reproducible verifier,
- trajectory of improvements.

The learner could study how algorithms, kernels, training recipes, or proofs improve and attempt to transfer those optimization patterns to unseen research problems.

This is conceptually interesting but far beyond what current LevelUp results support.

## 27. Scaling law for optimization transfer

Eventually measure transfer as a function of:

- number of training environments,
- diversity of mechanics,
- number of ladder tiers,
- number of historical transitions,
- model size,
- training compute,
- and amount of held-out adaptation.

A particularly important question:

> Does the amount of experience needed to become superhuman on a new task fall systematically as the agent sees more examples of other tasks becoming superhuman?

That would be a true learning-to-level-up scaling law.

## 28. The destination remains falsifiable

The project should remain open to the possibility that no strong domain-general “superhuman optimization skill” exists in the form initially imagined.

Possible outcomes include:

- mostly domain-specific transfer,
- optimum imitation doing nearly all the work,
- search/world-model learning dominating ladder structure,
- performance ladders helping only within related mechanics,
- or genuine broad transfer emerging only at much larger scale.

Any of these would be a useful scientific result if measured cleanly.

The research goal is not to protect the hypothesis. It is to discover the strongest true version of it.