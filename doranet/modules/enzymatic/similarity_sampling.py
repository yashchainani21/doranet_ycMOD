"""Product-similarity sampling for enzymatic network expansion.

Implements a Pickaxe-style similarity sampling step as a DORAnet global
update hook.  After each generation, the newly generated product molecules
are scored by their maximum fingerprint similarity to a set of target
molecules, and a weighted random sample of ``sample_size`` of them is
retained (tagged) for further expansion.  A companion molecule filter
(``MolFilterMetaFunc`` on the same tag) then prevents the un-sampled
products from being used as reactants in subsequent generations.

See ``doranet.modules.enzymatic.generate_network.generate_network`` for the
integration into the enzymatic driver.
"""

import collections.abc
import dataclasses
import typing

import numpy as np
import rdkit
import rdkit.Chem
import rdkit.DataStructs
from rdkit.Chem import AllChem

from doranet import interfaces

DEFAULT_META_KEY = "keep_for_sampling"


def _default_weight(similarity: float) -> float:
    """Bias strongly toward target-like products (Pickaxe default)."""
    return similarity**4


def compute_fingerprint(
    mol: rdkit.Chem.rdchem.Mol,
    method: str = "RDKit",
    args: typing.Optional[collections.abc.Mapping] = None,
) -> typing.Any:
    """Compute a fingerprint for an RDKit molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to fingerprint.
    method : str
        ``"Morgan"`` for a Morgan/ECFP bit vector, otherwise the RDKit
        topological fingerprint.
    args : collections.abc.Mapping, optional
        Extra keyword arguments forwarded to the fingerprint function, e.g.
        ``{"radius": 2}`` for Morgan.

    Returns
    -------
    typing.Any
        An RDKit fingerprint suitable for similarity comparison.
    """
    kwargs = {} if args is None else dict(args)
    if method == "Morgan":
        return AllChem.GetMorganFingerprintAsBitVect(mol, **kwargs)
    return rdkit.Chem.RDKFingerprint(mol, **kwargs)


def pairwise_similarity(
    fp1: typing.Any, fp2: typing.Any, method: str = "Tanimoto"
) -> float:
    """Return the similarity between two fingerprints.

    Uses Tanimoto by default; pass ``method="Dice"`` for the Dice metric.
    """
    if method.lower() == "dice":
        return rdkit.DataStructs.FingerprintSimilarity(
            fp1, fp2, metric=rdkit.DataStructs.DiceSimilarity
        )
    return rdkit.DataStructs.FingerprintSimilarity(fp1, fp2)


def max_similarity(
    mol_fp: typing.Any,
    target_fps: collections.abc.Sequence[typing.Any],
    method: str = "Tanimoto",
) -> float:
    """Maximum similarity of a fingerprint to any target fingerprint.

    Returns ``0.0`` when there are no targets.
    """
    if not target_fps:
        return 0.0
    return max(pairwise_similarity(mol_fp, t, method) for t in target_fps)


def weighted_sample_without_replacement(
    indices: collections.abc.Sequence[int],
    weights: collections.abc.Sequence[float],
    n: int,
    rng: np.random.Generator,
) -> list[int]:
    """Weighted sampling without replacement (Efraimidis-Spirakis A-Res).

    Each item is assigned a key ``u ** (1 / weight)`` with ``u`` drawn
    uniformly from ``[0, 1)``; the ``n`` items with the largest keys are
    returned.  Higher weight yields a key closer to ``1``, so more heavily
    weighted items are preferentially selected.  Items with zero weight
    receive a tiny floor so they remain eligible to fill the quota (only when
    fewer than ``n`` positively weighted items exist) without competing with
    positively weighted ones.

    Parameters
    ----------
    indices : collections.abc.Sequence[int]
        Candidate identifiers to sample from.
    weights : collections.abc.Sequence[float]
        Non-negative sampling weight for each candidate.
    n : int
        Number of candidates to select.
    rng : numpy.random.Generator
        Random generator (seed it for reproducibility).

    Returns
    -------
    list[int]
        The selected ``min(n, len(indices))`` identifiers.
    """
    items = list(indices)
    if n >= len(items):
        return items
    keys: list[tuple[float, int]] = []
    for idx, weight in zip(items, weights, strict=True):
        u = float(rng.random())
        w_eff = weight if weight > 0.0 else 1e-12
        keys.append((u ** (1.0 / w_eff), idx))
    keys.sort(reverse=True)
    return [idx for _, idx in keys[:n]]


@dataclasses.dataclass
class ProductSimilaritySampler(interfaces.GlobalUpdateHook):
    """Retain a similarity-weighted sample of products each generation.

    Pass this to ``strat.expand(global_hooks=[...])`` together with a
    ``MolFilterMetaFunc`` on :data:`DEFAULT_META_KEY` (or ``meta_key``) so
    that only sampled molecules seed subsequent generations.  It fires once
    per generation, after that generation's products have been added.

    Parameters
    ----------
    target_smiles : collections.abc.Sequence[str]
        SMILES of the target molecules to bias sampling toward.
    sample_size : int
        Number of products to retain per generation.
    weight : collections.abc.Callable[[float], float], optional
        Maps a similarity score to a sampling weight.  Defaults to
        ``similarity ** 4`` (strong bias toward target-like products).
    min_similarity : float
        Products whose maximum similarity to any target is below this value
        are dropped before sampling (default ``0.0``, i.e. no hard floor).
    fingerprint_method : str
        ``"RDKit"`` (topological, default) or ``"Morgan"``.
    fingerprint_args : collections.abc.Mapping, optional
        Extra keyword arguments for the fingerprint function.
    similarity_method : str
        ``"Tanimoto"`` (default) or ``"Dice"``.
    meta_key : collections.abc.Hashable
        Metadata key written on retained molecules.
    seed : int, optional
        Seed for reproducible sampling.
    high_water : int
        Number of molecules already present before expansion; molecules with
        index ``>= high_water`` are treated as "new" on the first call.
    """

    target_smiles: collections.abc.Sequence[str]
    sample_size: int
    weight: typing.Optional[collections.abc.Callable[[float], float]] = None
    min_similarity: float = 0.0
    fingerprint_method: str = "RDKit"
    fingerprint_args: typing.Optional[collections.abc.Mapping] = None
    similarity_method: str = "Tanimoto"
    meta_key: collections.abc.Hashable = DEFAULT_META_KEY
    seed: typing.Optional[int] = None
    high_water: int = 0
    _target_fps: list = dataclasses.field(
        init=False, default_factory=list, repr=False
    )
    _rng: np.random.Generator = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        for smi in self.target_smiles:
            mol = rdkit.Chem.MolFromSmiles(smi)
            if mol is not None:
                self._target_fps.append(
                    compute_fingerprint(
                        mol, self.fingerprint_method, self.fingerprint_args
                    )
                )

    def __call__(
        self, network: interfaces.ChemNetwork
    ) -> interfaces.GlobalHookReturnValue:
        n_mols = len(network.mols)
        weight_fn = self.weight if self.weight is not None else _default_weight
        candidates: list[int] = []
        weights: list[float] = []
        for i in range(self.high_water, n_mols):
            if not network.reactivity[i]:
                continue
            mol = network.mols[interfaces.MolIndex(i)]
            if not isinstance(mol, interfaces.MolDatRDKit):
                continue
            score = max_similarity(
                compute_fingerprint(
                    mol.rdkitmol,
                    self.fingerprint_method,
                    self.fingerprint_args,
                ),
                self._target_fps,
                self.similarity_method,
            )
            if score < self.min_similarity:
                continue
            candidates.append(i)
            weights.append(weight_fn(score))
        if candidates:
            if len(candidates) <= self.sample_size:
                chosen: collections.abc.Iterable[int] = candidates
            else:
                chosen = weighted_sample_without_replacement(
                    candidates, weights, self.sample_size, self._rng
                )
            for i in chosen:
                network.mols.set_meta(
                    interfaces.MolIndex(i), {self.meta_key: True}
                )
        self.high_water = n_mols
        return interfaces.GlobalHookReturnValue.CONTINUE
