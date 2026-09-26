"""Reproducible, leakage-aware train/validation split at the S1-entity
level.

Why "S1-level alone" is not automatically enough
-------------------------------------------------
The Task 2 brief explicitly warns: "Do not assume that splitting by S1
alone is sufficient." The reason is cross-linking (see
``ground_truth.compute_cross_link_statistics``): if the same S2 or S3
entity_id is a true match for two different S1 entities, and those two
S1 entities land on opposite sides of a naive random split, then a
record that was "seen" as a true positive during training (tied to the
train-side S1) reappears as a true positive again during validation
(tied to the val-side S1). That is a soft form of leakage: a future
matcher could learn "this exact S2/S3 record is a match-type record" in
a way that doesn't generalize, and get credit for it again at
validation time.

This module removes that risk *by construction*: S1 entities that share
a cross-linked S2/S3 match are grouped into the same "component" via a
union-find over the ground-truth's cross-link mapping, and splitting is
done at the component level, not the raw S1-id level. Components that
don't share any cross-linked match with another S1 (the common case --
see docs/validation.md for why this is expected to be the vast
majority) are simply singletons, so this reduces to a plain per-S1
split whenever there is no cross-linking to account for.

Stratification
---------------
Within each component-level split, we still want to preserve the
observed 0 / 1 / >1 total-match-count distribution approximately. Each
component is assigned a single stratum bucket, using the bucket of a
canonical representative member (the lexicographically smallest S1 id
in the component). For singleton components (the overwhelming
majority, in the absence of extensive cross-linking) this is exactly
that S1's own bucket, so stratification is exact for singletons and
only approximate for the rare multi-member components -- consistent
with the brief's "preserve *approximately*" requirement.

Reproducibility
----------------
Splitting uses ``random.Random(seed)`` with a fixed, documented seed
and a deterministic sort of component ids before shuffling, so the same
seed on the same input always produces the same split (verified by
``tests/test_split.py::test_split_is_reproducible``).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

DEFAULT_SEED = 42
DEFAULT_VAL_FRACTION = 0.2


# --------------------------------------------------------------------------
# Union-Find over S1 ids, grouped by shared cross-linked S2/S3 matches
# --------------------------------------------------------------------------


class _UnionFind:
    """Minimal union-find (disjoint set) keyed by arbitrary hashable ids.

    Uses path compression + union by size, so it stays fast even for a
    few million S1 ids -- this is the only place in the split logic that
    needs to scale to the full training set size.
    """

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._size: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]

    def components(self, universe: Iterable[str]) -> Dict[str, List[str]]:
        """Return ``root -> [members]`` for every id in ``universe``.

        Ids in ``universe`` that were never involved in any ``union``
        call are their own singleton root, so this always covers the
        full universe, not just ids that appeared in cross-links.
        """
        groups: Dict[str, List[str]] = {}
        for x in universe:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups


def build_s1_components(
    s1_ids: Iterable[str],
    cross_link_mapping: Optional[Mapping[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """Group S1 ids into connected components that must stay together.

    ``cross_link_mapping`` is the ``id -> [s1_ids]`` mapping produced by
    ``ground_truth.compute_cross_link_statistics`` (only containing S2/S3
    ids that are linked to *more than one* S1 entity -- singletons are
    not present and don't need to be, since a union of one thing with
    itself is a no-op).

    Returns ``component_root -> [s1_ids in that component]``. Every id
    in ``s1_ids`` appears in exactly one component's member list.
    """
    uf = _UnionFind()
    s1_ids = list(s1_ids)
    for s1_id in s1_ids:
        uf.add(s1_id)

    if cross_link_mapping:
        for _shared_id, s1_list in cross_link_mapping.items():
            distinct = sorted(set(s1_list))
            if len(distinct) < 2:
                continue
            first = distinct[0]
            for other in distinct[1:]:
                uf.union(first, other)

    return uf.components(s1_ids)


# --------------------------------------------------------------------------
# Stratified, component-level, reproducible split
# --------------------------------------------------------------------------


def _bucket_for_count(n: int) -> str:
    return "0" if n == 0 else ("1" if n == 1 else ">1")


@dataclass
class SplitResult:
    seed: int
    val_fraction: float
    train_ids: List[str]
    val_ids: List[str]

    # bucket -> count, computed separately for train/val, for verifying
    # the 0/1/>1 distribution was approximately preserved.
    train_bucket_counts: Dict[str, int] = field(default_factory=dict)
    val_bucket_counts: Dict[str, int] = field(default_factory=dict)

    n_components: int = 0
    n_multi_member_components: int = 0
    largest_component_size: int = 0

    def to_dict(self, include_ids: bool = True) -> dict:
        d = {
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "n_train": len(self.train_ids),
            "n_val": len(self.val_ids),
            "train_bucket_counts": self.train_bucket_counts,
            "val_bucket_counts": self.val_bucket_counts,
            "n_components": self.n_components,
            "n_multi_member_components": self.n_multi_member_components,
            "largest_component_size": self.largest_component_size,
        }
        if include_ids:
            d["train_ids"] = self.train_ids
            d["val_ids"] = self.val_ids
        return d


def create_validation_split(
    s1_bucket: Mapping[str, str],
    cross_link_mapping: Optional[Mapping[str, List[str]]] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> SplitResult:
    """Build the leakage-aware, stratified, reproducible split.

    Parameters
    ----------
    s1_bucket:
        ``s1_id -> "0" | "1" | ">1"`` for every S1 entity in the
        training set (i.e. the per-S1 total-match bucket, cheaply
        derivable while streaming ``train_ground_truth.tsv`` -- see
        ``scripts/run_validation_analysis.py`` for exactly how this is
        built without materializing the full match-id lists).
    cross_link_mapping:
        Optional ``id -> [s1_ids]`` cross-link mapping (S2/S3 ids linked
        to >1 S1 entity). If omitted, every S1 id is its own component
        (equivalent to assuming no cross-linking -- only safe to do
        after confirming via ``compute_cross_link_statistics`` that
        none exists).
    val_fraction:
        Target fraction of *S1 entities* (not components) assigned to
        validation. Because whole components move together, the
        realized fraction can differ slightly whenever multi-member
        components exist -- this is reported, not hidden, via
        ``train_bucket_counts``/``val_bucket_counts``.
    seed:
        Fixed seed for the internal ``random.Random`` shuffle. Reusing
        the same seed against the same ``s1_bucket``/
        ``cross_link_mapping`` inputs always reproduces an identical
        split.
    """
    all_s1_ids = list(s1_bucket.keys())
    components = build_s1_components(all_s1_ids, cross_link_mapping)

    component_items: List[Tuple[str, List[str]]] = sorted(components.items())
    multi_member = [c for _, members in component_items if len(members) > 1 for c in [members]]

    def component_bucket(members: List[str]) -> str:
        representative = min(members)  # deterministic regardless of dict order
        return s1_bucket[representative]

    strata: Dict[str, List[Tuple[str, List[str]]]] = {"0": [], "1": [], ">1": []}
    for root, members in component_items:
        strata[component_bucket(members)].append((root, members))

    train_ids: List[str] = []
    val_ids: List[str] = []
    train_bucket_counts = {"0": 0, "1": 0, ">1": 0}
    val_bucket_counts = {"0": 0, "1": 0, ">1": 0}

    rng = random.Random(seed)
    for bucket_name, comp_list in strata.items():
        # Sort first for determinism, THEN shuffle with the seeded RNG --
        # dict/set iteration order must never influence the result.
        comp_list = sorted(comp_list, key=lambda item: item[0])
        rng.shuffle(comp_list)

        n_val_components = round(len(comp_list) * val_fraction)
        val_components = comp_list[:n_val_components]
        train_components = comp_list[n_val_components:]

        for _root, members in val_components:
            val_ids.extend(members)
            val_bucket_counts[bucket_name] += len(members)
        for _root, members in train_components:
            train_ids.extend(members)
            train_bucket_counts[bucket_name] += len(members)

    train_ids.sort()
    val_ids.sort()

    return SplitResult(
        seed=seed,
        val_fraction=val_fraction,
        train_ids=train_ids,
        val_ids=val_ids,
        train_bucket_counts=train_bucket_counts,
        val_bucket_counts=val_bucket_counts,
        n_components=len(component_items),
        n_multi_member_components=len(multi_member),
        largest_component_size=max((len(m) for _, m in component_items), default=0),
    )


# --------------------------------------------------------------------------
# Persistence -- store IDs + metadata only, never raw source rows
# --------------------------------------------------------------------------


def save_split(
    split: SplitResult,
    output_dir: str | Path,
    store_train_ids: bool = False,
) -> Path:
    """Persist the split as a small JSON artifact.

    By convention we store ``val_ids`` (the smaller side, ~val_fraction
    of S1 entities) plus full metadata, and treat "train" as "every
    training-set S1 id not in val_ids" -- reconstructable at load time
    from ``train_source1.tsv`` without duplicating the (much larger)
    train id list in the artifact. Pass ``store_train_ids=True`` to
    additionally persist the train id list explicitly if you'd rather
    not recompute the complement.

    Writes ``<output_dir>/split_seed{seed}.json`` and returns its path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = split.to_dict(include_ids=False)
    payload["val_ids"] = split.val_ids
    if store_train_ids:
        payload["train_ids"] = split.train_ids

    out_path = output_dir / f"split_seed{split.seed}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def load_split(
    path: str | Path,
    all_s1_ids: Optional[Iterable[str]] = None,
) -> SplitResult:
    """Reload a persisted split.

    If the artifact only stored ``val_ids`` (the default from
    ``save_split``), ``all_s1_ids`` (e.g. every id in
    ``train_source1.tsv``) must be provided so ``train_ids`` can be
    reconstructed as the complement.
    """
    data = json.loads(Path(path).read_text())
    val_ids = data["val_ids"]

    if "train_ids" in data:
        train_ids = data["train_ids"]
    else:
        if all_s1_ids is None:
            raise ValueError(
                "This split artifact did not store train_ids; pass "
                "all_s1_ids (e.g. every id from train_source1.tsv) to "
                "reconstruct the training side as the complement of val_ids."
            )
        val_id_set = set(val_ids)
        train_ids = sorted(sid for sid in all_s1_ids if sid not in val_id_set)

    return SplitResult(
        seed=data["seed"],
        val_fraction=data["val_fraction"],
        train_ids=train_ids,
        val_ids=val_ids,
        train_bucket_counts=data.get("train_bucket_counts", {}),
        val_bucket_counts=data.get("val_bucket_counts", {}),
        n_components=data.get("n_components", 0),
        n_multi_member_components=data.get("n_multi_member_components", 0),
        largest_component_size=data.get("largest_component_size", 0),
    )
