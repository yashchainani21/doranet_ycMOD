"""Tests for parallel expansion (doranet._parallel).

Small hand-built networks only (no large rulesets).  The key property is that
parallel (np>1) expansion produces the same network *content* as serial (np=1):
identical molecule and reaction sets keyed by uid, and identical metadata.
"""

import doranet as dn
from doranet import interfaces, metacalc


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


def test_parallel_reproducible_across_core_counts():
    two = _expand_alkyne(2)
    four = _expand_alkyne(4)
    assert _mol_uids(two) == _mol_uids(four)
    assert _rxn_uids(two) == _rxn_uids(four)


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
