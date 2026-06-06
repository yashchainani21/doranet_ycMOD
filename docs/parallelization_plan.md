# Plan: Parallelize DORAnet enzymatic expansion (incl. similarity sampling)

## Context

DORAnet's network expansion is single-threaded. The two hot paths —
`HasSubstructMatch` (operator↔molecule **compatibility testing**) and
`RunReactants` (operator application) — both scale with the operator count, and
the enzymatic ruleset (JN3604IMT ≈ 3,604 operators) makes multi-step pathway
discovery slow even after similarity sampling bounds the per-generation frontier.
An independent review measured the dominant cost: **compat testing is ~1.6 s per
new molecule across the full ruleset** (the serial `op.compat` loop in
`add_mol`, `network.py:413-416`) — much larger than `RunReactants`. Pickaxe
parallelizes and is far faster on multi-step searches; DORAnet currently cannot
(`np != 1` and `num_procs is not None` both raise `NotImplementedError`).

This plan adds multiprocessing so enzymatic expansion — and the similarity-
sampling hook that guides it — run across cores. The original authors scaffolded
for it (serializable `ReactionJob`/`RecipeRankingJob`, a commented
`CartesianStrategyParallel`, the `_custom_compat` injection hook,
`RecipeHeap.__add__`), but never wired it up.

### Decisions locked with the user
- **Scope: enzymatic breadth-first only** — `CartesianStrategyUpdated` →
  `PriorityQueueStrategyBasic.expand` with `beam_size=None`. The general
  finite-beam engine stays single-threaded.
- **Thermo usually off** (`rxn_thermo_calculator=None`). A passed thermo callback
  must be picklable (module-level); a worker-side factory is deferred.
- **Fully reproducible** parallel runs (deterministic regardless of core count).

### Corrections from the independent review (folded into the phases)
1. **The reaction_plan is NOT picklable as-is (CRITICAL).** Several
   `@dataclass(frozen=True)` classes in `metadata.py` use a *manual* `__slots__`;
   `pickle.dumps` succeeds but `pickle.loads` raises `FrozenInstanceError` on
   Py3.10. Validation must be a full **round-trip** `pickle.loads(pickle.dumps())`,
   and `metadata.py` must be fixed. The earlier "everything is already picklable"
   assumption was wrong (it held for `OpDatBasic`/`MolDatBasicV1`/whole networks,
   which rebuild from blob, but not these compositor wrappers).
2. **Compat dominates** → reaction and compat parallelism **ship together** (one
   phase); reaction-only would barely move the needle. Master rebuild of products
   from SMILES is cheap (~19 µs/mol) — confirms shipping SMILES (not blobs).
3. **Worker→master metadata must be keyed by canonical SMILES uid**, not the
   worker's transient product index (`execute_reaction` tags products with `-1`,
   `strategies.py:1056`; the master assigns real indices). Otherwise generation
   N+1's `Reaction_Type_Filter` raises "No molecule metadata found".
4. **Determinism**: sort `(uid, weight)` candidate tuples *in lockstep* before the
   sampling draw; the reproducibility test must be np=1 vs np=N **with sampling on
   + fixed seed**, comparing chosen-uid sets (not against today's serial seed).
5. **Free serial win**: `clean_SMILES` is called repeatedly on identical SMILES in
   the serial recipe filter (`Reaction_Type_Filter`, Phase A) — add an
   `lru_cache`.

## Approach (validated against the code)

Parallelize **inside** `PriorityQueueStrategyBasic.expand` (reusing its merge
machinery) rather than writing a new strategy. A `multiprocessing` **spawn**
`Pool` is created once per expansion with a module-level **initializer** loading,
per worker: a worker-local `engine=create_engine(speed,np=1)`, the operator set
rebuilt from **SMARTS + metadata** (indexed by `OpIndex`; ~5 MB payload, ~1.2 s
build — not the 144 MB that pickling operator *objects* would cost), and the
(now-picklable) reaction_plan + `save_unreactive`. Workers receive lightweight
jobs `(op_index, ((reactant_index, smiles, meta),...))`, run `RunReactants` + the
reaction_plan, and return lightweight `ReactionResult`s (products as **canonical
SMILES** + computed metadata **keyed by uid** + pass_filter + reaction uid). The
**master keeps its existing add_mol/add_rxn + resolver-merge loop byte-identical**,
rebuilding each product from SMILES via the master engine.

Everything is gated behind `if num_procs > 1:`; **`np == 1` keeps the current
serial path untouched** (debuggable reference + differential-test baseline).

**Validated invariants (narrowed per review):**
- *Metadata equivalence under chunking holds ONLY because* the enzymatic plan's
  properties are per-molecule/per-reaction intrinsic with **trivial / per-reaction-
  unique resolvers** (`SMILES`, `dH`); filters are per-reaction booleans. The
  master applies the same `mc_update` resolvers, so chunking == serial **for this
  plan**. `TrivialMetaDataResolverFunc` is *not* commutative — it's safe only
  because collision values are identical. Any future non-trivial resolver must
  fall back to serial or assert commutativity. (`metadata.py:856-870`,
  master loop `strategies.py:1264-1330`.)
- *Hook fires once per generation ONLY for `beam_size=None`* (the locked scope):
  `popvals(None)` drains the whole heap each iteration, hooks fire once at the
  bottom (`strategies.py:1219, 1342-1357`). This is the reason scope is restricted
  to `beam_size=None`.
- *`updated_mols_set` is reset INSIDE the per-reaction loop* (`strategies.py:1264-
  1265`); keeping the merge loop byte-identical preserves recipe-batching behavior.
- *Product canonicality*: worker and master both canonicalize via `MolToSmiles`
  (same RDKit, same env under spawn) → identical uid, so dedup is consistent.
- *Compat injection*: `add_mol(reactive=False)` then
  `add_mol(reactive=True, _custom_compat=[...])` flips + injects without re-running
  `op.compat` (`network.py:380-389`); the guard doesn't block it.

## Phased implementation

### Phase 0 — Prerequisites (no behavior change at np=1)
- **Plumb `np`**: remove the gate at `engine.py:152-156`; in
  `PriorityQueueStrategyBasic.__init__` (`strategies.py:1083-1092`) drop the raise,
  store `self._num_procs = num_procs or 1` and `self._engine` (master needs it to
  rebuild products); add to `__slots__`. `CartesianStrategyUpdated.expand`
  (`strategies.py:1395`) → `engine.strat.pq(network, num_procs=engine.np,
  engine=engine)`. `generate_network` gains `num_procs=1` → `create_engine(np=...)`.
- **Fix picklability (CRITICAL)**: convert the `frozen=True` + manual-`__slots__`
  classes in `metadata.py` (`MolPropertyCompositor`, `RxnPropertyCompositor`,
  `OpPropertyCompositor`, `Mol/OpRxnPropertyCompositor`, `FunctionPropertyCompositor`,
  `RxnAnalysisStepProp`, `RxnAnalysisStepCompound`, and siblings) to
  `@dataclass(frozen=True, slots=True)` (verified to round-trip), or add
  `__getstate__`/`__setstate__` using `object.__setattr__`. Add a
  `_validate_picklable(obj)` helper doing `pickle.loads(pickle.dumps(obj))`; call
  it on the reaction_plan when `num_procs>1`.
- **Free serial win**: wrap `clean_SMILES` (`generate_network.py`) in
  `functools.lru_cache`. Lazy-load the module-level 6 MB `bio_rules` read
  (`generate_network.py:75`) so spawn workers don't each re-parse it.

### Phase 1 — Parallel reaction execution + compat testing (the core win)
**New module `doranet/_parallel.py`** (module-level for spawn-safety):
- `_WorkerState` (worker globals): `engine`, `ops: list[OpDatBasic]`,
  `op_meta: list[dict]`, `reaction_plan`, `save_unreactive`, `reaction_keyset`.
- `_init_worker(...)` — build worker engine; rebuild ops via
  `OpDatBasic(smarts, engine, kekulize, drop_errors)` (`datatypes.py:208-235`);
  store the unpickled reaction_plan.
- `ReactionResult` (frozen, slots): `op_index`, `reactant_indices`,
  `product_smiles: tuple[str,...]`, **`meta_by_uid: dict[str, dict]`** (per-mol
  computed meta keyed by canonical SMILES — fixes C3), `op_meta`, `reaction_meta`,
  `pass_filter`, `rxn_uid`.
- `_run_reaction_chunk(chunk)` — rebuild `ReactionJob`s, call the existing
  `execute_reactions([job], reaction_plan)` (`strategies.py:1068-1077`), emit
  results with product meta keyed by `mol.item.uid`.
- `_compute_compat(mol_smiles)` — rebuild mol, run the **same double loop as
  `add_mol`** (`network.py:413-416`): `[(i,arg) for i,op in enumerate(ops) for arg
  in range(len(op)) if op.compat(mol,arg)]`; return **sorted** (byte-identical
  columns).

**Master changes in `PriorityQueueStrategyBasic.expand` (gated on num_procs>1):**
- Build the spawn Pool once outside the `while` loop; assemble SMARTS+meta op
  payloads; **round-trip-validate** the reaction_plan.
- Replace the serial reaction block (`strategies.py:1238-1262`): build lightweight
  jobs from `recipes_to_be_expanded` (reactant SMILES + meta as
  `assemble_reaction_job` does, `strategies.py:771-780`); dispatch via
  `pool.imap_unordered(_run_reaction_chunk, _chunk_generator(chunksize, jobs))`.
- Collect the beam's results, **sort by `rxn_uid`**, then run the **existing merge
  loop unchanged** (`strategies.py:1244-1330`): rebuild each product via
  `self._engine.mol.rdkit(smiles)`, `add_mol(..., reactive=False)`, `add_rxn`,
  apply resolver merge sourcing meta from `meta_by_uid`. **Keep the
  `updated_mols_set` reset inside the loop.**
- **Then parallel compat**: collect unique new product SMILES, dispatch
  `pool.imap_unordered(_compute_compat, ...)`, and for each (processed in
  uid-sorted order) call `add_mol(mol, reactive=True, _custom_compat=result)` to
  flip + inject (`network.py:380-389`).

### Phase 2 — Parallel similarity-sampling scoring (measure-gated)
Fingerprint+similarity is cheap (~tens of µs/mol) and runs once per generation, so
**profile first** — it may not be worth a separate pool. If warranted:
- `similarity_sampling.py`: factor the scoring loop (`:205-223`) into a module-level
  `_score_mol(smiles, fp_method, fp_args, target_smiles, sim_method) -> (uid, score)`
  (ship target SMILES, recompute target fps worker-side).
- `ProductSimilaritySampler.__call__`: map `_score_mol` over new reactive mols,
  build `(uid, score, weight)` tuples, **sort by uid**, then run the existing
  serial `weighted_sample_without_replacement` (`:228-230`) with weights carried in
  lockstep. Use a dedicated, lazily-created pool only if `sample_size is not None`.

### Phase 3 — Determinism, tests, benchmark
- Determinism: the two sorts (beam by `rxn_uid`; candidates by uid with weights).
- Tests `tests/test_parallel.py` (small hand-built network, no 6 MB ruleset):
  - `test_parallel_equals_serial_content`: np=1 vs np=2 → equal molecule & reaction
    sets **by uid**, equal counts, equal per-uid metadata (`SMILES`, `dH`).
  - `test_metadata_equivalence_cross_chunk`: same product from reactions in
    different chunks → merged meta matches serial.
  - `test_compat_injection_matches_serial`: two-step flow yields byte-identical
    `_compat_table` columns (requires the sorted compat result).
  - `test_sampling_reproducible`: np=1 vs np=2, sampling on, fixed seed → identical
    chosen-uid sets; plus order-shuffle invariance of the draw.
  - `test_smiles_roundtrip_idempotent`: for operator outputs,
    `MolToSmiles(product) == MolToSmiles(MolFromSmiles(that smiles))` (guards the
    worker/master dedup seam).
  - `test_hook_fires_once_per_generation` (asserts hook-fire-count == gen-count).
  - np=2 variant of `test_sampling_off_matches_default`.
- Benchmark `bench/` script: `generate_network(gen=2/3, num_procs=k)` for
  `k∈{1,2,4,8}` on JN3604IMT; confirm identical output sizes across `k`; expect
  **compat** to dominate the speedup.

## Reuse (existing utilities)
`ReactionJob`/`assemble_reaction_job` (`strategies.py:745-782`), `execute_reactions`
(`1068-1077`), `_chunk_generator` (`57-67`); `add_mol(..., _custom_compat=...)`
(`network.py:356-421`); `OpDatBasic(smarts, engine)` (`datatypes.py:208-235`),
`engine.mol.rdkit(smiles)`/`MolDatBasicV1`; the master merge loop +
`metalib_to_rxn_meta` (`metadata.py:791-847`) kept unchanged; `create_engine`/
`engine.np` (`engine.py`).

## Critical files
- `doranet/metadata.py` — picklability fix (frozen+slots) + round-trip validator.
- `doranet/_parallel.py` (NEW) — initializer, `_run_reaction_chunk`,
  `_compute_compat`, `_score_mol`, `ReactionResult`, `_WorkerState`.
- `doranet/strategies.py` — gate/pool/dispatch/merge; merge loop stays byte-identical.
- `doranet/engine.py` — remove np gate; thread np/engine into the strategy.
- `doranet/modules/enzymatic/generate_network.py` — `num_procs`; `clean_SMILES`
  lru_cache; lazy `bio_rules`; pass num_procs to the sampler.
- `doranet/modules/enzymatic/similarity_sampling.py` — parallel scoring +
  lockstep uid sort.
- `tests/test_parallel.py` (NEW); `bench/` script (NEW).

## Verification
- `pytest tests/test_parallel.py` + full `pytest` (no regressions);
  `ruff check . && ruff format --diff .`; `mypy --no-install-types .`.
- Smoke: `generate_network(starters="CC(O)CO", gen=2, targets=..., sample_size=10,
  num_procs=4)` from repo root → identical mol/rxn counts to `num_procs=1` and a
  wall-clock speedup.
- Benchmark `num_procs∈{1,2,4,8}` at gen 2/3 on JN3604IMT.

## Effort & sequencing
Phase 0 (plumbing + the picklability fix that unblocks everything) → Phase 1
(reactions + compat together — the real enzymatic speedup) → Phase 2 (sampling,
only if profiling justifies it) → Phase 3 (determinism + tests + benchmark).
Rough estimate ~4-7 weeks for all phases incl. testing; each phase is
independently shippable, and per your "smallest case first" preference I'll
validate each (np=1 parity first, then small np>1 runs) before scaling up.
