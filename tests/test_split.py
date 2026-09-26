"""Unit tests for the leakage-aware, reproducible validation split."""

from __future__ import annotations

from pathlib import Path

import pytest

from business_entity_resolution.validation.split import (
    build_s1_components,
    create_validation_split,
    load_split,
    save_split,
)


def make_bucket_map(n_zero: int, n_one: int, n_multi: int) -> dict:
    bucket_map = {}
    for i in range(n_zero):
        bucket_map[f"Z-{i}"] = "0"
    for i in range(n_one):
        bucket_map[f"O-{i}"] = "1"
    for i in range(n_multi):
        bucket_map[f"M-{i}"] = ">1"
    return bucket_map


# --------------------------------------------------------------------------
# Union-find components
# --------------------------------------------------------------------------


def test_build_s1_components_no_cross_links_are_all_singletons():
    s1_ids = ["S1-1", "S1-2", "S1-3"]
    components = build_s1_components(s1_ids, cross_link_mapping=None)
    sizes = sorted(len(members) for members in components.values())
    assert sizes == [1, 1, 1]
    all_members = sorted(m for members in components.values() for m in members)
    assert all_members == sorted(s1_ids)


def test_build_s1_components_groups_shared_cross_linked_ids():
    s1_ids = ["S1-1", "S1-2", "S1-3", "S1-4"]
    # S2-1 is a true match for both S1-1 and S1-2 -> they must end up in
    # the same component. S1-3/S1-4 are unaffected.
    cross_link_mapping = {"S2-1": ["S1-1", "S1-2"]}
    components = build_s1_components(s1_ids, cross_link_mapping)

    membership = {}
    for root, members in components.items():
        for m in members:
            membership[m] = root

    assert membership["S1-1"] == membership["S1-2"]
    assert membership["S1-3"] != membership["S1-1"]
    assert membership["S1-4"] != membership["S1-1"]
    assert membership["S1-3"] != membership["S1-4"]


def test_build_s1_components_transitively_merges_chained_links():
    # S1-1 -- S2-1 -- S1-2 -- S2-2 -- S1-3 : all three must end up together
    # even though no single shared id directly links S1-1 and S1-3.
    s1_ids = ["S1-1", "S1-2", "S1-3"]
    cross_link_mapping = {
        "S2-1": ["S1-1", "S1-2"],
        "S2-2": ["S1-2", "S1-3"],
    }
    components = build_s1_components(s1_ids, cross_link_mapping)
    assert len(components) == 1
    (members,) = components.values()
    assert sorted(members) == ["S1-1", "S1-2", "S1-3"]


# --------------------------------------------------------------------------
# create_validation_split
# --------------------------------------------------------------------------


def test_split_covers_every_id_exactly_once_no_overlap():
    bucket_map = make_bucket_map(n_zero=20, n_one=20, n_multi=60)
    split = create_validation_split(bucket_map, val_fraction=0.2, seed=42)

    assert set(split.train_ids) & set(split.val_ids) == set()
    assert set(split.train_ids) | set(split.val_ids) == set(bucket_map.keys())
    assert len(split.train_ids) + len(split.val_ids) == len(bucket_map)


def test_split_approximately_preserves_bucket_distribution_with_no_cross_links():
    bucket_map = make_bucket_map(n_zero=100, n_one=100, n_multi=100)
    split = create_validation_split(bucket_map, val_fraction=0.2, seed=42)

    # With no cross-linking, every component is a singleton, so the
    # stratified split should hit the target fraction almost exactly
    # per bucket (rounding to the nearest whole component/id).
    for bucket in ("0", "1", ">1"):
        total = split.train_bucket_counts[bucket] + split.val_bucket_counts[bucket]
        realized_frac = split.val_bucket_counts[bucket] / total
        assert realized_frac == pytest.approx(0.2, abs=0.02)


def test_split_is_reproducible_with_same_seed():
    bucket_map = make_bucket_map(n_zero=30, n_one=30, n_multi=30)
    split_a = create_validation_split(bucket_map, val_fraction=0.25, seed=7)
    split_b = create_validation_split(bucket_map, val_fraction=0.25, seed=7)
    assert split_a.train_ids == split_b.train_ids
    assert split_a.val_ids == split_b.val_ids


def test_split_differs_with_different_seed():
    bucket_map = make_bucket_map(n_zero=30, n_one=30, n_multi=30)
    split_a = create_validation_split(bucket_map, val_fraction=0.25, seed=7)
    split_b = create_validation_split(bucket_map, val_fraction=0.25, seed=99)
    assert split_a.val_ids != split_b.val_ids


def test_split_keeps_cross_linked_components_together():
    bucket_map = make_bucket_map(n_zero=10, n_one=10, n_multi=10)
    # Force two specific ids into the same component; run many seeds to
    # make sure they are NEVER split across train/val by construction.
    cross_link_mapping = {"S2-shared": ["Z-0", "O-0"]}
    for seed in range(10):
        split = create_validation_split(
            bucket_map,
            cross_link_mapping=cross_link_mapping,
            val_fraction=0.5,
            seed=seed,
        )
        in_val = {"Z-0", "O-0"} <= set(split.val_ids)
        in_train = {"Z-0", "O-0"} <= set(split.train_ids)
        assert in_val or in_train, (
            "Z-0 and O-0 share a cross-linked match and must land on the "
            "same side of the split"
        )


def test_split_handles_no_ids_gracefully():
    split = create_validation_split({}, val_fraction=0.2, seed=42)
    assert split.train_ids == []
    assert split.val_ids == []
    assert split.n_components == 0


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_save_and_load_split_roundtrip_with_complement_reconstruction(
    tmp_path: Path,
):
    bucket_map = make_bucket_map(n_zero=10, n_one=10, n_multi=10)
    split = create_validation_split(bucket_map, val_fraction=0.3, seed=1)
    out_path = save_split(split, tmp_path)
    assert out_path.exists()

    reloaded = load_split(out_path, all_s1_ids=bucket_map.keys())
    assert reloaded.val_ids == split.val_ids
    assert reloaded.train_ids == split.train_ids
    assert reloaded.seed == split.seed


def test_save_and_load_split_with_explicit_train_ids_stored(tmp_path: Path):
    bucket_map = make_bucket_map(n_zero=5, n_one=5, n_multi=5)
    split = create_validation_split(bucket_map, val_fraction=0.2, seed=1)
    out_path = save_split(split, tmp_path, store_train_ids=True)

    # No all_s1_ids needed this time -- train_ids was persisted directly.
    reloaded = load_split(out_path)
    assert reloaded.train_ids == split.train_ids
    assert reloaded.val_ids == split.val_ids


def test_load_split_without_train_ids_or_all_s1_ids_raises(tmp_path: Path):
    bucket_map = make_bucket_map(n_zero=5, n_one=5, n_multi=5)
    split = create_validation_split(bucket_map, val_fraction=0.2, seed=1)
    out_path = save_split(split, tmp_path, store_train_ids=False)

    with pytest.raises(ValueError):
        load_split(out_path)
