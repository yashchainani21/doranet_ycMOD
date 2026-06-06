"""Tests for product-similarity sampling (doranet.modules.enzymatic)."""

import numpy as np

import doranet as dn
from doranet import interfaces
from doranet.modules.enzymatic import similarity_sampling as ss

KEY = ss.DEFAULT_META_KEY


# --------------------------------------------------------------------------
# Sampling math
# --------------------------------------------------------------------------
def test_fingerprint_and_max_similarity():
    engine = dn.create_engine()
    fp1 = ss.compute_fingerprint(engine.mol.rdkit("CCO").rdkitmol)
    fp2 = ss.compute_fingerprint(engine.mol.rdkit("CCCCO").rdkitmol)
    # identical fingerprints -> Tanimoto 1.0
    assert ss.pairwise_similarity(fp1, fp1) == 1.0
    # no targets -> 0.0
    assert ss.max_similarity(fp1, []) == 0.0
    # max over targets includes the perfect match
    assert ss.max_similarity(fp1, [fp2, fp1]) == 1.0


def test_weighted_sample_keeps_all_when_n_large():
    rng = np.random.default_rng(0)
    chosen = ss.weighted_sample_without_replacement(
        [10, 11, 12], [1.0, 1.0, 1.0], 5, rng
    )
    assert sorted(chosen) == [10, 11, 12]


def test_weighted_sample_size_and_subset():
    rng = np.random.default_rng(0)
    pool = list(range(20))
    n = 5
    chosen = ss.weighted_sample_without_replacement(pool, [1.0] * 20, n, rng)
    assert len(chosen) == n
    assert len(set(chosen)) == n
    assert set(chosen).issubset(pool)


def test_weighted_sample_prefers_high_weight():
    # An overwhelming weight gap makes the heavy item win deterministically.
    rng = np.random.default_rng(0)
    chosen = ss.weighted_sample_without_replacement(
        [0, 1, 2], [1.0, 1e-9, 1e-9], 1, rng
    )
    assert chosen == [0]


# --------------------------------------------------------------------------
# Filter mechanism: only tagged molecules are usable as reactants
# --------------------------------------------------------------------------
def test_filter_excludes_untagged_reactants():
    engine = dn.create_engine()
    network = engine.new_network()
    keep_i = network.add_mol(engine.mol.rdkit("CCO"), meta={KEY: True})
    drop_i = network.add_mol(engine.mol.rdkit("CCC"))
    network.add_op(engine.op.rdkit("[C:1][C:2]>>[*:1]=[*:2]"))

    mol_filter = engine.filter.mol.meta_func(KEY, lambda v: v is True)
    engine.strat.cartesian(network).expand(num_iter=1, mol_filter=mol_filter)

    reactant_indices = set()
    for rxn in network.rxns:
        reactant_indices.update(rxn.reactants)
    assert keep_i in reactant_indices
    assert drop_i not in reactant_indices


# --------------------------------------------------------------------------
# Hook: similarity floor + survivor selection
# --------------------------------------------------------------------------
def test_hook_tags_survivors_above_floor():
    engine = dn.create_engine()
    network = engine.new_network()
    network.add_mol(engine.mol.rdkit("C"), meta={KEY: True})
    ini = len(network.mols)
    # Simulate one generation of (reactive) products.
    like_i = network.add_mol(engine.mol.rdkit("OCCCCO"))  # == target
    other1 = network.add_mol(engine.mol.rdkit("CC"))
    other2 = network.add_mol(engine.mol.rdkit("CCC"))

    # With a high floor, only the target-identical product survives, so the
    # selection is deterministic regardless of the RNG.
    sampler = ss.ProductSimilaritySampler(
        target_smiles=["OCCCCO"],
        sample_size=1,
        min_similarity=0.99,
        seed=0,
        high_water=ini,
    )
    assert sampler(network) is interfaces.GlobalHookReturnValue.CONTINUE

    def is_kept(i):
        return network.mols.meta(interfaces.MolIndex(i), (KEY,)).get(KEY)

    assert is_kept(like_i) is True
    assert is_kept(other1) is None
    assert is_kept(other2) is None


def test_hook_caps_sample_size():
    engine = dn.create_engine()
    network = engine.new_network()
    network.add_mol(engine.mol.rdkit("C"), meta={KEY: True})
    ini = len(network.mols)
    for smi in ("CCO", "CCCO", "CCCCO", "CCCCCO"):
        network.add_mol(engine.mol.rdkit(smi))

    sampler = ss.ProductSimilaritySampler(
        target_smiles=["CCCCCCO"], sample_size=2, seed=0, high_water=ini
    )
    sampler(network)

    tagged = [
        i
        for i in range(ini, len(network.mols))
        if network.mols.meta(interfaces.MolIndex(i), (KEY,)).get(KEY)
    ]
    assert len(tagged) == sampler.sample_size


# --------------------------------------------------------------------------
# End-to-end: hook + filter bound multi-generation growth
# --------------------------------------------------------------------------
def _build_alkyne_network(engine):
    network = engine.new_network()
    for smi in ("C#C", "CC#C", "CCC#C", "CCCC#C"):
        network.add_mol(engine.mol.rdkit(smi))
    network.add_op(engine.op.rdkit("[C:1]#[C:2]>>[*:1]=[*:2]"))
    network.add_op(engine.op.rdkit("[C:1]=[C:2]>>[*:1]-[*:2]"))
    return network


def test_sampling_bounds_growth():
    engine = dn.create_engine()

    control = _build_alkyne_network(engine)
    engine.strat.cartesian(control).expand(num_iter=2)

    network = _build_alkyne_network(engine)
    ini = len(network.mols)
    for i in range(ini):
        network.mols.set_meta(interfaces.MolIndex(i), {KEY: True})
    sampler = ss.ProductSimilaritySampler(
        target_smiles=["CCCC"], sample_size=2, seed=0, high_water=ini
    )
    mol_filter = engine.filter.mol.meta_func(KEY, lambda v: v is True)
    engine.strat.cartesian(network).expand(
        num_iter=2, mol_filter=mol_filter, global_hooks=[sampler]
    )

    # Sampling caps which products expand, so the sampled network is smaller.
    assert len(network.mols) < len(control.mols)
    # Every reactant in every reaction must be a tagged molecule.
    for rxn in network.rxns:
        for r in rxn.reactants:
            kept = network.mols.meta(interfaces.MolIndex(r), (KEY,)).get(KEY)
            assert kept is True


def test_sampling_off_matches_default():
    engine = dn.create_engine()

    a = _build_alkyne_network(engine)
    engine.strat.cartesian(a).expand(num_iter=2)

    b = _build_alkyne_network(engine)
    engine.strat.cartesian(b).expand(
        num_iter=2, mol_filter=None, global_hooks=None
    )

    assert len(a.mols) == len(b.mols)
    assert len(a.rxns) == len(b.rxns)
