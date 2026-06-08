"""Tests for parallel expansion (doranet._parallel).

Small hand-built networks only (no large rulesets).  The key property is that
parallel (np>1) expansion produces the same network *content* as serial (np=1):
identical molecule and reaction sets keyed by uid, and identical metadata.
"""

import doranet as dn
from doranet import interfaces, metacalc
from doranet.modules.enzymatic import similarity_sampling as ss


def _alkyne_network(engine):
    net = engine.new_network()
    for smi in ("C#C", "CC#C", "CCC#C", "CCCC#C"):
        net.add_mol(engine.mol.rdkit(smi))
    net.add_op(engine.op.rdkit("[C:1]#[C:2]>>[*:1]=[*:2]"))
    net.add_op(engine.op.rdkit("[C:1]=[C:2]>>[*:1]-[*:2]"))
    return net


def _mol_uids(net):
    return {m.uid for m in net.mols}


def _rxn_uids(net):
    # Reactions keyed by content (op + reactant/product uids), since the
    # integer indices may differ between runs.
    return {
        (
            net.ops[r.operator].uid,
            tuple(sorted(net.mols[m].uid for m in r.reactants)),
            tuple(sorted(net.mols[m].uid for m in r.products)),
        )
        for r in net.rxns
    }


def _expand_alkyne(np_count, num_iter=2):
    engine = dn.create_engine(np=np_count)
    net = _alkyne_network(engine)
    engine.strat.cartesian(net).expand(num_iter=num_iter)
    return net


def test_parallel_reactions_match_serial():
    serial = _expand_alkyne(1)
    parallel = _expand_alkyne(2)
    assert len(serial.mols) == len(parallel.mols)
    assert len(serial.rxns) == len(parallel.rxns)
    assert _mol_uids(serial) == _mol_uids(parallel)
    assert _rxn_uids(serial) == _rxn_uids(parallel)


def test_parallel_compat_matches_serial():
    # Compat testing is parallelized (Phase 1b); the resulting compat table
    # must match serial (compared as sets of mol uids per operator argument).
    serial = _expand_alkyne(1)
    parallel = _expand_alkyne(2)

    def compat_sets(net):
        out = {}
        for i in range(len(net.ops)):
            op_uid = net.ops[interfaces.OpIndex(i)].uid
            table = net.compat_table(interfaces.OpIndex(i))
            for arg, col in enumerate(table):
                out[(op_uid, arg)] = frozenset(net.mols[m].uid for m in col)
        return out

    assert compat_sets(serial) == compat_sets(parallel)


def test_parallel_reproducible_across_core_counts():
    two = _expand_alkyne(2)
    four = _expand_alkyne(4)
    assert _mol_uids(two) == _mol_uids(four)
    assert _rxn_uids(two) == _rxn_uids(four)


def test_sampling_reproducible_serial_vs_parallel():
    # Similarity sampling draws in canonical (uid) order, so a seeded run
    # selects the same molecules regardless of np / index-assignment order.
    key = ss.DEFAULT_META_KEY

    def run(np_count):
        engine = dn.create_engine(np=np_count)
        net = engine.new_network()
        for smi in ("C#C", "CC#C", "CCC#C", "CCCC#C"):
            net.add_mol(engine.mol.rdkit(smi), meta={key: True})
        net.add_op(engine.op.rdkit("[C:1]#[C:2]>>[*:1]=[*:2]"))
        net.add_op(engine.op.rdkit("[C:1]=[C:2]>>[*:1]-[*:2]"))
        ini = len(net.mols)
        sampler = ss.ProductSimilaritySampler(
            target_smiles=["CCCC"], sample_size=2, seed=0, high_water=ini
        )
        mol_filter = engine.filter.mol.meta_func(key, lambda v: v is True)
        engine.strat.cartesian(net).expand(
            num_iter=2, mol_filter=mol_filter, global_hooks=[sampler]
        )
        return {
            net.mols[interfaces.MolIndex(i)].uid
            for i in range(len(net.mols))
            if net.mols.meta(interfaces.MolIndex(i), (key,)).get(key)
        }

    assert run(1) == run(2)


def test_parallel_metadata_matches_serial():
    plan = metacalc.GenerationCalculator("gen")

    def run(np_count):
        engine = dn.create_engine(np=np_count)
        net = engine.new_network()
        net.add_mol(engine.mol.rdkit("C#C"), meta={"gen": 0})
        net.add_op(engine.op.rdkit("[C:1]#[C:2]>>[*:1]=[*:2]"))
        net.add_op(engine.op.rdkit("[C:1]=[C:2]>>[*:1]-[*:2]"))
        engine.strat.cartesian(net).expand(num_iter=3, reaction_plan=plan)
        return net

    def gens(net):
        return {
            net.mols[interfaces.MolIndex(i)].uid: net.mols.meta(
                interfaces.MolIndex(i), ("gen",)
            ).get("gen")
            for i in range(len(net.mols))
        }

    serial = gens(run(1))
    parallel = gens(run(2))
    assert serial == parallel
    assert serial == {"C#C": 0, "C=C": 1, "CC": 2}


def test_progress_output(capsys):
    # progress=True emits Pickaxe-style per-generation messages.
    engine = dn.create_engine()
    net = _alkyne_network(engine)
    engine.strat.cartesian(net).expand(num_iter=2, progress=True)
    out = capsys.readouterr().out
    assert "Generation 1: ranking recipes" in out
    assert "% complete" in out
    assert "Generation 1 complete" in out
    assert "Generation 2" in out

    # default (progress off) is silent.
    net2 = _alkyne_network(engine)
    engine.strat.cartesian(net2).expand(num_iter=2)
    assert capsys.readouterr().out == ""
