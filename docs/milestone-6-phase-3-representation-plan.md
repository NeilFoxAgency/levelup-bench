# Milestone 6 Phase 3 Frozen Representation-Ladder Plan

**Date:** 2026-08-22 (America/New_York)

**Status:** frozen after the locked Phase 2 development result and before any Phase 3
comparative development result

**Machine authority:**
[`configs/milestone6/phase3_representation_ladder.json`](../configs/milestone6/phase3_representation_ladder.json)

## Why this tranche exists

Phase 2 asked whether adding the current observable state to the same empirical action-effect
summary improves over objective-matched global optimum imitation. It did not: selected C reached
minimum-family success 0.075 versus B2's 0.400. The failure was stable across the complete 12-tuple
grid, not a bad selected temperature or epoch count. On development Heat, C repeatedly preferred a
no-progress step at full pressure over the utility action that clears pressure and makes another
burst available.

That diagnosis motivates a richer representation, but it does not authorize changing the old
selection rule or claiming that state, transitions, or sequence caused the result. Phase 3 freezes
those comparisons separately before producing new outcomes.

## Representation ladder

The six-condition evidence set is:

1. **B2 - global listwise optimum imitation.** The strong Phase 2 baseline is retained unchanged.
2. **S - state and availability only.** Keep current state plus action-specific probe-state and
   availability statistics. Zero all measured outcome, delta, elapsed-effect, after-state, and
   completion channels.
3. **T - Markov state and transition.** This is bitwise the historical Phase 2 C representation:
   current state plus the full empirical action-transition summary.
4. **H0 - architecture-matched null history.** Use H4's exact GRU and head, but replace every
   available prior transition with a fixed zero token.
5. **H4 - causal four-transition history.** Add a small GRU over the immediately preceding four
   learner-observable transitions from the same episode to T.
6. **H4-shuffled - destroyed order control.** Give the identical model the exact same preceding
   transition multiset, but deterministically permute only its order.

The contrasts answer different questions. B2 versus T is the already locked state-conditioning
contrast. S versus T tests observed transition outcomes beyond current state and action
availability. H4 must beat both T and architecture-identical H0 to establish an effect of causal
history rather than a recurrent-architecture change. H4 versus H4-shuffled tests order rather than
merely extra transition records.

S is deliberately conservative. Opaque action aliases cannot transfer as semantic action IDs, so
a control with literally state alone would give identical rows to every available action. S instead
retains only the action-specific states and availability in which probes encountered each alias.
This makes candidates distinguishable without preserving what the action did.
Because all conditions retain the same probe visitation data, S tests explicit measured candidate
outcomes conditional on those visits; it does not claim earlier actions had no effect on which
states the probe scheduler reached.

Mechanically, each 12-channel transition row is ordered as before progress, remaining, resource,
pressure; progress, elapsed, resource, and pressure deltas; after resource and pressure;
completion; and before-action availability count. S retains indices `[0, 1, 2, 3, 11]` in every
mean/std/min/max block, zeros indices `[4, 5, 6, 7, 8, 9, 10]`, and retains the final coverage
scalar as an availability/sampling-support diagnostic. The transition claim is therefore limited
to explicit measured candidate outcomes conditional on the same visitation and coverage data.

## Same-data and capacity contract

All conditions consume the exact 30 canonical Phase 2 optimum-trajectory and affordance evidence
artifacts. They use identical task and trajectory identities, examples, labels, order, batches,
model/probe/search/data seeds, optimizer, update counts, and heldout interaction/search budgets.
No evidence is regenerated and no frontier trajectory enters this tranche.

S and T use the same 3,841-parameter MLP. H0, H4, and H4-shuffled use the same 528-parameter,
hidden-8 GRU plus a `[40, 20]` head, for 3,889 parameters total. B2 remains unchanged at 3,601 parameters.
The maximum symmetric pairwise gap is 7.41 percent, within the frozen 10-percent tolerance. B2
keeps its complete hyperparameter grid and all training and search opportunities; it is not made
easier to beat.

History length is fixed at four before results. Every decision rebuilds its preceding
length-at-most-four window and runs the GRU from zero state; no persistent hidden state crosses
decisions. H0 runs an all-zero window of the exact same length, matching recurrent-step cost.
Training examples contain only transitions before the labeled decision. Candidate episodes start
with an empty window and never inherit probe history. The shuffled control permutes only each
already-causal preceding window, never inserts a future transition, and binds the realized
permutation map to artifact identity.

Every length-two-to-four shuffled window uses a seeded index derangement, so no transition index
stays in its original position. Duplicate transition vectors can still make a deranged map
tensor-ineffective. The run reports map-nonidentity and byte-level effective tensor changes
separately; an order claim is forbidden unless at least 80 percent of eligible windows change
effectively in both training and heldout search. Permutation maps use compact sorted-key UTF-8 JSON
with no trailing newline and bind fold, replicate, task, phase, trace/episode, decision, input
indices, and permuted indices.

## Hyperparameters, seeds, and budgets

S, H0, H4, and H4-shuffled each receive the same 12 tuples previously available to B2 and T:

- learning rate `{0.003, 0.01}`;
- epochs `{120, 180}`; and
- search temperature `{0.6, 0.9, 1.2}`.

Replicates remain `0-4` across the six leave-one-development-family-out folds. Screening remains
64 paid probe actions, 150 complete candidate episodes, at most 64 actions per episode, and 2,048
total adaptation actions per task. Candidate generation is fixed-budget and evaluator-free;
independent replay completes before the reporting-only exact-optimum query. No exact optimum can
stop or steer search.

The new execution comprises 11,520 units: four new conditions, 12 tuples, six families, five
replicates, and eight heldout development tasks. B2 and T contribute 5,760 already locked anchor
units after exact lineage validation. The combined comparison matrix therefore contains 17,280
units, but only the new 11,520 may execute.

T is an analysis-only alias of historical C, not a newly trained condition. Before Phase 3 may
execute, an identity-only anchor manifest must enumerate all 240 B2/C model owners and 5,760 B2/C
unit result IDs and byte digests from the descriptor-validated Phase 2 runtime. T must resolve to
the exact historical C views, tensors, models, logits, candidate-generation hashes, typed outcomes,
and result bytes. Any parity failure closes the execution gate.

Temperature never changes model training. A model owner is keyed by condition, fold, replicate,
training tuple, view, seed, optimizer, and architecture, then reused by all three temperature
consumers. A selected tuple's training cost sums exactly 30 owners (six folds by five replicates),
never eight heldout tasks or three temperatures. Duplicate, missing, extra, drifted,
temperature-specific, or cross-condition owner lineages fail closed.

## Frozen selection and claims

Select one tuple independently within S, H0, H4, and H4-shuffled using the Phase 2 deterministic rule:
maximize minimum-family success, retain the inclusive five-percentage-point band, then minimize
worst-family median restricted interactions, macro family median, unique-owner optimizer steps,
forward passes, recurrent steps, and finally the numeric tuple. Reuse the locked independent B2
and T selections. Never remove a baseline because another condition wins.

A robust transition, history, or sequence-order claim requires more than five absolute percentage
points of minimum-family success over its matched control. A difference within that band may
support only a secondary restricted-interaction efficiency statement. H4 advances to paired
improvement objectives only if it clears both the T/H0 dual-control and shuffled-order gates, stays within five
points of B2 on minimum-family success, and loses no individual development family by more than
five points versus B2. Otherwise B2 remains the strong reference and the negative result is kept.

Logit, pressure-state coverage, unknown-affordance, label-frequency, realized-history-length, and
shuffle-change diagnostics are non-selection evidence. They cannot change the candidate set,
architecture, seeds, budgets, thresholds, or selection rule.

## Scientific boundary

This is development-only method iteration. It makes no claim about explicit frontier-to-optimum
pairing, because neither paired nor frontier data enters this tranche. D1 versus F remains a later,
separately frozen same-trajectory pairing test. No Milestone 6 final family may be created,
unlocked, or inspected.
