# Phase A (recipe-ranking) parallelization — analysis & decision

> **Decision (after independent review): DEFERRED — land at np=4 = 2.45×.**
> Phase A parallelization requires shipping the large `recipes_tested` set every
> generation (IPC may cancel the ~48 s it saves) or a risky cursor rewrite, for
> only a moderate projected gain (~2.9×). The shipped parallelization
> (`docs/parallelization_plan.md`, Phases 0–3 + the `Recipe.__hash__` fix) is the
> chosen milestone. This document records why Phase A was deferred and what
> implementing it would entail, so the analysis isn't lost.

## Where parallelization landed

On gen-3 (`CCO`, max C=5): **np=1 304.8 s → np=4 124.2 s = 2.45×**, byte-identical
output, reproducible across core counts, 19 tests green. The `Recipe.__hash__`
fix (dropping `astuple`/`deepcopy`) also gave a free serial win, so end-to-end
vs the original serial baseline is ~354 s → 124 s ≈ 2.85×.

Profiling shows the remaining dominant **serial** cost is Phase A recipe ranking
(`execute_recipe_ranking` ≈ 48.6 s cumtime: recipe generation ~26 s +
`Reaction_Type_Filter.__call__` 11.8 s over 17.8M calls). Because it's serial it
caps scaling — **np=12 (163 s) is actually slower than np=4 (124 s)**: past ~4
cores the fixed serial Phase A + pool/IPC overhead outweighs the gains.

## What Phase A parallelization would entail

Parallelize the per-operator recipe-ranking loop in
`PriorityQueueStrategyBasic.expand` (operators are disjoint, so embarrassingly
parallel). Reuse the existing spawn pool (workers already hold the operators) and
the authors' scaffolding: `RecipeRankingJob`, `execute_recipe_ranking`, and
`RecipeHeap.__add__` (a bounded parallel reduction). Each generation: dispatch
one job per operator; workers run `execute_recipe_ranking` → return a
`RecipeHeap`; the master merges them.

### Known requirements / fixes
- **Picklability:** the sampling `mol_filter` uses a `lambda` (not picklable
  under spawn) — replace with a module-level predicate; ship the filters once via
  the pool initializer.
- **Per-generation state shipping:** workers need the per-operator compat columns
  + cursor + a mol-meta snapshot (`SMILES`, keep flag). Molecules need only
  metadata, not live RDKit objects (`recipe_keyset.live_molecule` is False for
  the enzymatic filters).
- **Operator metadata:** ship recipe-keyset op-meta (`"Reactants"`, `"SMARTS"`) —
  the reaction-pool initializer currently ships only reaction-keyset op-meta, so
  `Reaction_Type_Filter` would otherwise raise "No operator metadata found".

## Why it was deferred (independent review findings)

- **CRITICAL — `recipes_tested` is NOT redundant.** In the `beam_size=None` path
  the compat cursor is reset to 0 at the end of every generation
  (`strategies.py:1380-1384`, because the heap drains to empty), so
  `_generate_recipe_batches` regenerates the full Cartesian product each
  generation and `recipes_tested` is the only guard against re-expanding all
  prior recipes. Running workers with an empty `recipes_tested` would re-execute
  the entire history every generation (quadratic work), and — critically —
  `parallel == serial` set-equality tests would NOT catch it (add_mol/add_rxn
  dedup keeps outputs identical; only the work is wasted). So workers must either
  ship `recipes_tested` per generation (the large set the design hoped to avoid;
  IPC cost must be measured against the ~48 s saved) or Phase A must be
  re-architected to a genuinely incremental per-op cursor (changes serial
  semantics; more invasive).
- **MAJOR — verification must include a work-count / wall-clock check** at
  gen≥2; set-equality alone is blind to the above regression.
- Confirmed sound: the lambda→predicate picklability fix; meta-only molecule
  shipping; the deterministic `RecipeHeap.__add__` merge.

## If revisited: treat as a spike

First measure the per-generation cost of shipping `recipes_tested` (or prototype
the incremental-cursor rewrite) and confirm a net win on gen-3 **before**
committing to the full implementation. Add a work-count regression check
alongside the existing `parallel == serial` uid tests. Otherwise, the current
np=4 = 2.45× (+ the serial hash win) is a strong, low-risk place to stand.
