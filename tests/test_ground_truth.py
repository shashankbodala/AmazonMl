"""Unit tests for ground_truth.py, using small synthetic data only.

None of these tests touch the real (multi-GB) dataset -- they write a
tiny synthetic ``train_ground_truth.tsv`` to a pytest ``tmp_path`` and
verify the streaming parser/statistics against hand-computed expected
values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from business_entity_resolution.validation.ground_truth import (
    compute_cross_link_statistics,
    compute_ground_truth_statistics,
    compute_s1_bucket_map,
    load_s1_matches,
    parse_matched_ids,
    verify_internal_consistency,
)

# Synthetic ground truth used across several tests:
#
# S1-1 -> S2-1, S2-2, S3-1      (both, 3 total)
# S1-2 -> S2-1                  (only s2, 1 total)   -- S2-1 is CROSS-LINKED
# S1-3 -> (none)                (neither, 0 total)
# S1-4 -> S3-2, S3-3            (only s3, 2 total)
# S1-5 -> S2-3                  (only s2, 1 total)
#
# S2-1 is matched by both S1-1 and S1-2 -> deliberate cross-link for
# leakage-detection tests.
SAMPLE_ROWS = [
    ("S1-1", "S2-1,S2-2,S3-1"),
    ("S1-2", "S2-1"),
    ("S1-3", ""),
    ("S1-4", "S3-2,S3-3"),
    ("S1-5", "S2-3"),
]


@pytest.fixture()
def synthetic_gt_path(tmp_path: Path) -> Path:
    path = tmp_path / "train_ground_truth.tsv"
    lines = ["source1_entity_id\tmatched_entity_ids"]
    for s1_id, matches in SAMPLE_ROWS:
        lines.append(f"{s1_id}\t{matches}")
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# parse_matched_ids
# --------------------------------------------------------------------------


def test_parse_matched_ids_basic():
    s2, s3 = parse_matched_ids("S2-681193310,S2-743505751,S3-775321672")
    assert s2 == ["S2-681193310", "S2-743505751"]
    assert s3 == ["S3-775321672"]


def test_parse_matched_ids_empty_and_none():
    assert parse_matched_ids("") == ([], [])
    assert parse_matched_ids(None) == ([], [])
    assert parse_matched_ids(float("nan")) == ([], [])


def test_parse_matched_ids_single_id_no_comma():
    assert parse_matched_ids("S3-42") == ([], ["S3-42"])


def test_parse_matched_ids_strips_whitespace_and_trailing_comma():
    s2, s3 = parse_matched_ids(" S2-1 , S2-2,")
    assert s2 == ["S2-1", "S2-2"]
    assert s3 == []


def test_parse_matched_ids_rejects_unknown_prefix():
    with pytest.raises(ValueError):
        parse_matched_ids("S2-1,S9-999")


# --------------------------------------------------------------------------
# compute_ground_truth_statistics (S2/S3 classification + match counts)
# --------------------------------------------------------------------------


def test_ground_truth_statistics_totals(synthetic_gt_path: Path):
    stats = compute_ground_truth_statistics(synthetic_gt_path, chunksize=2)

    assert stats.total_s1_entities == 5
    assert stats.total_s2_links == 4  # S2-1(x2), S2-2, S2-3
    assert stats.total_s3_links == 3  # S3-1, S3-2, S3-3
    assert stats.total_links == 7

    assert stats.only_s2_count == 2  # S1-2, S1-5
    assert stats.only_s3_count == 1  # S1-4
    assert stats.both_s2_and_s3_count == 1  # S1-1
    assert stats.neither_count == 1  # S1-3

    assert stats.max_s2_matches == 2  # S1-1 has 2 s2 matches
    assert stats.max_s3_matches == 2  # S1-4 has 2 s3 matches
    assert stats.max_total_matches == 3  # S1-1 has 3 total

    assert stats.total_bucket_counts == {"0": 1, "1": 2, ">1": 2}
    assert pytest.approx(stats.mean_total_matches) == 7 / 5


def test_ground_truth_statistics_is_chunksize_invariant(synthetic_gt_path: Path):
    stats_small_chunks = compute_ground_truth_statistics(synthetic_gt_path, chunksize=1)
    stats_big_chunk = compute_ground_truth_statistics(synthetic_gt_path, chunksize=1000)
    assert stats_small_chunks.to_dict() == stats_big_chunk.to_dict()


def test_verify_internal_consistency_passes_on_valid_stats(synthetic_gt_path: Path):
    stats = compute_ground_truth_statistics(synthetic_gt_path, chunksize=2)
    assert verify_internal_consistency(stats) == []


def test_verify_internal_consistency_catches_corruption(synthetic_gt_path: Path):
    stats = compute_ground_truth_statistics(synthetic_gt_path, chunksize=2)
    stats.total_s2_links += 1  # deliberately corrupt one derived total
    problems = verify_internal_consistency(stats)
    assert any("total_s2_links" in p for p in problems)


# --------------------------------------------------------------------------
# Cross-link (leakage) detection
# --------------------------------------------------------------------------


def test_cross_link_statistics_detects_shared_s2_id(synthetic_gt_path: Path):
    stats, mapping = compute_cross_link_statistics(synthetic_gt_path, chunksize=2)

    # S2-1 is matched by S1-1 and S1-2 -> exactly one cross-linked S2 id.
    assert stats.s2_ids_linked_to_multiple_s1 == 1
    assert stats.max_s1_per_s2 == 2
    assert stats.unique_s2_ids == 3  # S2-1, S2-2, S2-3

    # No S3 id is shared across S1 entities in the synthetic sample.
    assert stats.s3_ids_linked_to_multiple_s1 == 0
    assert stats.max_s1_per_s3 == 1
    assert stats.unique_s3_ids == 3  # S3-1, S3-2, S3-3

    assert set(mapping["S2-1"]) == {"S1-1", "S1-2"}
    assert "S2-1" in stats.example_s2_cross_links
    assert set(stats.example_s2_cross_links["S2-1"]) == {"S1-1", "S1-2"}


def test_cross_link_statistics_no_leakage_case(tmp_path: Path):
    # A ground truth with no shared S2/S3 ids at all.
    path = tmp_path / "gt_no_leakage.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-2\tS2-2\n"
        "S1-3\tS3-1\n"
    )
    stats, mapping = compute_cross_link_statistics(path, chunksize=2)
    assert stats.s2_ids_linked_to_multiple_s1 == 0
    assert stats.s3_ids_linked_to_multiple_s1 == 0
    assert mapping == {}
    assert stats.example_s2_cross_links == {}
    assert stats.example_s3_cross_links == {}


# --------------------------------------------------------------------------
# Bucket map + full match loading
# --------------------------------------------------------------------------


def test_compute_s1_bucket_map(synthetic_gt_path: Path):
    buckets = compute_s1_bucket_map(synthetic_gt_path, chunksize=2)
    assert buckets == {
        "S1-1": ">1",
        "S1-2": "1",
        "S1-3": "0",
        "S1-4": ">1",
        "S1-5": "1",
    }


def test_load_s1_matches_full(synthetic_gt_path: Path):
    matches = load_s1_matches(synthetic_gt_path, chunksize=2)
    assert matches.s1_to_s2["S1-1"] == ["S2-1", "S2-2"]
    assert matches.s1_to_s3["S1-1"] == ["S3-1"]
    assert "S1-3" not in matches.s1_to_s2
    assert "S1-3" not in matches.s1_to_s3
    assert matches.bucket("S1-1") == ">1"
    assert matches.bucket("S1-3") == "0"


def test_load_s1_matches_restricted_to_subset(synthetic_gt_path: Path):
    matches = load_s1_matches(synthetic_gt_path, chunksize=2, s1_ids={"S1-1", "S1-4"})
    assert set(matches.s1_to_s2.keys()) <= {"S1-1", "S1-4"}
    assert "S1-2" not in matches.s1_to_s2
    assert matches.s1_to_s3["S1-4"] == ["S3-2", "S3-3"]
