"""Multiprocessing support for parallel enzymatic network expansion.

Scope: the breadth-first path (``CartesianStrategyUpdated`` ->
``PriorityQueueStrategyBasic.expand`` with ``beam_size=None``).

A ``spawn`` pool is created once per expansion.  Each worker loads, via the
initializer, a worker-local engine and the full operator set plus the
reaction_plan.  Workers run ``RunReactants`` + the reaction_plan and return
lightweight results; the master keeps its existing ``add_mol``/``add_rxn`` +
metadata-merge loop, rebuilding each product molecule from its canonical
SMILES.  The serial path (``num_procs == 1``) does not use this module.
"""

import collections.abc
import dataclasses
import multiprocessing
import pickle
import typing

from doranet import interfaces

_Meta = typing.Optional[
    collections.abc.Mapping[collections.abc.Hashable, typing.Any]
]

# Per-worker state, populated by ``_init_worker`` and read by the worker
# functions.  Module-level so it survives across tasks in a ``spawn`` worker.
_WORKER: typing.Optional["_WorkerState"] = None


@dataclasses.dataclass(frozen=True, slots=True)
class _WorkerState:
    engine: interfaces.NetworkEngine
    ops: tuple[interfaces.OpDatBase, ...]
    op_meta: tuple[_Meta, ...]
    reaction_plan: typing.Any
    save_unreactive: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ReactionResult:
    """Serializable result of running one reaction in a worker.

    Products are returned as canonical SMILES (rebuilt by the master), with
    per-product metadata in positional correspondence.  ``sort_key`` gives a
    deterministic ordering for the master-side merge so results are independent
    of worker completion order.
    """

    op_index: int
    op_meta: _Meta
    reactant_indices: tuple[int, ...]
    reactant_meta: tuple[_Meta, ...]
    product_smiles: tuple[str, ...]
    product_meta: tuple[_Meta, ...]
    reaction_meta: _Meta
    pass_filter: bool
    sort_key: tuple[typing.Any, ...]


# A lightweight reaction job sent to a worker: (op_index, reactant payloads),
# where each reactant payload is (mol_index, canonical_smiles, meta-or-None).
_ReactantPayload = tuple[int, str, _Meta]
_JobPayload = tuple[int, tuple[_ReactantPayload, ...]]


def _init_worker(
    speed: int,
    op_objs: collections.abc.Sequence[interfaces.OpDatBase],
    op_meta: collections.abc.Sequence[_Meta],
    reaction_plan: typing.Any,
    save_unreactive: bool,
) -> None:
    """Store the per-worker engine + operator set in a module global.

    Operators ship as already-built objects (multiprocessing pickles them with
    regular pickle, not DORAnet's safe unpickler); a worker-local engine
    rebuilds reactant molecules from SMILES.  Shipping objects is simple and
    exact; a lighter SMARTS-based payload is a possible later optimization.
    """
    import doranet  # noqa: PLC0415

    engine = doranet.create_engine(speed=speed, np=1)
    global _WORKER  # noqa: PLW0603
    _WORKER = _WorkerState(
        engine=engine,
        ops=tuple(op_objs),
        op_meta=tuple(op_meta),
        reaction_plan=reaction_plan,
        save_unreactive=save_unreactive,
    )


def _run_reaction_chunk(
    chunk: collections.abc.Sequence[_JobPayload],
) -> list[ReactionResult]:
    """Run a chunk of reaction jobs (RunReactants + reaction_plan)."""
    from doranet import strategies  # noqa: PLC0415

    assert _WORKER is not None
    engine = _WORKER.engine
    reaction_jobs = []
    for op_index, reactant_payloads in chunk:
        operator = interfaces.DataPacketE(
            op_index, _WORKER.ops[op_index], _WORKER.op_meta[op_index]
        )
        op_args = tuple(
            interfaces.DataPacketE(
                idx,
                typing.cast(interfaces.MolDatBase, engine.mol.rdkit(smiles)),
                meta,
            )
            for idx, smiles, meta in reactant_payloads
        )
        reaction_jobs.append(strategies.ReactionJob(operator, op_args))

    results: list[ReactionResult] = []
    for rxn, pass_filter in strategies.execute_reactions(
        reaction_jobs, _WORKER.reaction_plan
    ):
        if not _WORKER.save_unreactive and not pass_filter:
            continue
        products = tuple(mol for mol in rxn.products if mol.item is not None)
        product_smiles = tuple(
            typing.cast(interfaces.MolDatRDKit, mol.item).smiles
            for mol in products
        )
        reactant_indices = tuple(int(mol.i) for mol in rxn.reactants)
        results.append(
            ReactionResult(
                op_index=int(rxn.operator.i),
                op_meta=rxn.operator.meta,
                reactant_indices=reactant_indices,
                reactant_meta=tuple(mol.meta for mol in rxn.reactants),
                product_smiles=product_smiles,
                product_meta=tuple(mol.meta for mol in products),
                reaction_meta=rxn.reaction_meta,
                pass_filter=pass_filter,
                sort_key=(
                    int(rxn.operator.i),
                    reactant_indices,
                    product_smiles,
                ),
            )
        )
    return results


def _validate_picklable(obj: typing.Any) -> None:
    """Raise a clear error if ``obj`` cannot round-trip through pickle.

    A bare ``pickle.dumps`` is not enough: frozen dataclasses with a manual
    ``__slots__`` dump fine but fail to *load*.  This does the full round-trip.
    """
    try:
        pickle.loads(pickle.dumps(obj))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "reaction_plan is not picklable, so it cannot be sent to worker "
            f"processes for parallel expansion: {exc!r}. Ensure any custom "
            "filters/calculators (e.g. a thermo callback) are module-level "
            "and picklable, or run with np=1."
        ) from exc


def make_pool(
    engine: interfaces.NetworkEngine,
    network: interfaces.ChemNetwork,
    reaction_plan: typing.Any,
    save_unreactive: bool,
    reaction_keyset: interfaces.MetaKeyPacket,
    num_procs: int,
) -> "multiprocessing.pool.Pool":
    """Create a spawn Pool with operators + reaction_plan loaded per worker."""
    _validate_picklable(reaction_plan)
    op_objs = list(network.ops)
    op_keys = reaction_keyset.operator_keys
    op_meta: list[_Meta] = [
        network.ops.meta(interfaces.OpIndex(i), op_keys) if op_keys else None
        for i in range(len(network.ops))
    ]
    ctx = multiprocessing.get_context("spawn")
    return ctx.Pool(
        processes=num_procs,
        initializer=_init_worker,
        initargs=(
            engine.speed,
            op_objs,
            op_meta,
            reaction_plan,
            save_unreactive,
        ),
    )


def _build_jobs(
    recipes: collections.abc.Iterable[typing.Any],
    network: interfaces.ChemNetwork,
    reaction_keyset: interfaces.MetaKeyPacket,
) -> list[_JobPayload]:
    mol_keys = reaction_keyset.molecule_keys
    jobs: list[_JobPayload] = []
    for reciperank in recipes:
        recipe = reciperank.recipe
        reactants = recipe.reactants
        if mol_keys:
            metas: collections.abc.Sequence[_Meta] = tuple(
                network.mols.meta(reactants, mol_keys)
            )
        else:
            metas = [None] * len(reactants)
        payload = tuple(
            (int(mi), network.mols[mi].smiles, meta)
            for mi, meta in zip(reactants, metas, strict=False)
        )
        jobs.append((int(recipe.operator), payload))
    return jobs


def _chunk(
    jobs: collections.abc.Sequence[_JobPayload], num_procs: int
) -> list[list[_JobPayload]]:
    n = len(jobs)
    if n == 0:
        return []
    chunk_count = max(1, num_procs * 4)
    chunk_size = max(1, -(-n // chunk_count))
    return [list(jobs[i : i + chunk_size]) for i in range(0, n, chunk_size)]


def run_reaction_batch(
    pool: "multiprocessing.pool.Pool",
    recipes: collections.abc.Sequence[typing.Any],
    network: interfaces.ChemNetwork,
    engine: interfaces.NetworkEngine,
    reaction_keyset: interfaces.MetaKeyPacket,
    num_procs: int,
) -> list[tuple[interfaces.ReactionExplicit, bool]]:
    """Dispatch one beam's reactions to workers and reconstruct the results.

    Returns a deterministic, master-side ``(reaction, pass_filter)`` list for
    the merge loop (sorted so the result is independent of worker order).
    Products are rebuilt from their canonical SMILES with the master engine.
    """
    jobs = _build_jobs(recipes, network, reaction_keyset)
    results: list[ReactionResult] = []
    for chunk_results in pool.imap_unordered(
        _run_reaction_chunk, _chunk(jobs, num_procs)
    ):
        results.extend(chunk_results)
    results.sort(key=lambda r: r.sort_key)

    reconstructed: list[tuple[interfaces.ReactionExplicit, bool]] = []
    for r in results:
        operator = interfaces.DataPacketE(
            r.op_index, typing.cast(interfaces.OpDatBase, None), r.op_meta
        )
        reactants = tuple(
            interfaces.DataPacketE(
                idx, typing.cast(interfaces.MolDatBase, None), meta
            )
            for idx, meta in zip(
                r.reactant_indices, r.reactant_meta, strict=False
            )
        )
        products = tuple(
            interfaces.DataPacketE(
                -1,
                typing.cast(interfaces.MolDatBase, engine.mol.rdkit(smiles)),
                meta,
            )
            for smiles, meta in zip(
                r.product_smiles, r.product_meta, strict=False
            )
        )
        rxn = interfaces.ReactionExplicit(
            operator, reactants, products, r.reaction_meta
        )
        reconstructed.append((rxn, r.pass_filter))
    return reconstructed


def _compute_compat(
    smiles: str,
) -> tuple[str, list[tuple[interfaces.OpIndex, int]]]:
    """Return (smiles, [(op_index, arg), ...]) the molecule is compatible with.

    Mirrors the operator/argument iteration order of
    ``ChemNetworkBasic.add_mol``'s compat loop so the injected entries match the
    serial computation.
    """
    assert _WORKER is not None
    mol = _WORKER.engine.mol.rdkit(smiles)
    compat = [
        (interfaces.OpIndex(i), arg)
        for i, op in enumerate(_WORKER.ops)
        for arg in range(len(op))
        if op.compat(mol, arg)
    ]
    return smiles, compat


def activate_products(
    pool: "multiprocessing.pool.Pool",
    network: interfaces.ChemNetwork,
    products_by_uid: collections.abc.Mapping[
        interfaces.Identifier, interfaces.MolDatBase
    ],
    num_procs: int,
) -> None:
    """Flip newly-produced products to reactive, injecting parallel compat.

    Compatibility testing (``HasSubstructMatch`` against every operator) is the
    dominant serial cost for large rulesets; it is computed across workers here.
    Products are processed in uid order (and each compat list is in operator
    order) so the resulting compat table is independent of worker scheduling.
    """
    to_do = [
        (uid, products_by_uid[uid])
        for uid in sorted(products_by_uid)
        if not network.reactivity[network.mols.i(uid)]
    ]
    if not to_do:
        return
    smiles_list = [str(uid) for uid, _ in to_do]
    chunksize = max(1, len(smiles_list) // (num_procs * 8))
    compat_by_uid: dict[str, list[tuple[interfaces.OpIndex, int]]] = {}
    for smiles, compat in pool.imap_unordered(
        _compute_compat, smiles_list, chunksize
    ):
        compat_by_uid[smiles] = compat
    for uid, mol in to_do:
        network.add_mol(
            mol, reactive=True, _custom_compat=compat_by_uid[str(uid)]
        )
