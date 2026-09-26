"""Unit tests for blocking module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from business_entity_resolution.blocking import (
    BlockingKeySets,
    CandidateResult,
    CompactInvertedIndex,
    build_filtered_inverted_indexes,
    generate_candidates_batch,
    generate_candidates_batch_vectorized,
    generate_candidates_for_s1,
    run_blocking_benchmark,
)
from business_entity_resolution.normalization import (
    make_address_country_key,
    make_address_key,
    make_name_country_key,
    make_name_key,
)


# --------------------------------------------------------------------------
# InvertedIndex tests
# --------------------------------------------------------------------------


def test_inverted_index_basic():
    idx = CompactInvertedIndex()
    idx.add("key1", "S2-1")
    idx.add("key1", "S2-2")
    idx.add("key2", "S2-3")

    assert idx.get("key1") == ["S2-1", "S2-2"]
    assert idx.get("key2") == ["S2-3"]
    assert idx.get("key3") == []
    assert len(idx) == 2
    assert idx.total_entries() == 3


def test_inverted_index_empty_key():
    idx = CompactInvertedIndex()
    idx.add("", "S2-1")  # empty key should be ignored
    idx.add("key1", "S2-2")
    assert len(idx) == 1
    assert idx.get("") == []


# --------------------------------------------------------------------------
# build_inverted_indexes tests
# --------------------------------------------------------------------------


def create_sample_source2(path: Path) -> None:
    """Create a small sample source2 file for testing."""
    df = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S2-3", "S2-4"],
        "business_name": [
            "Orelee's Barbershop",
            "Prime Money",
            "B+ Retail Inc",
            "Orelee's Barbershop",  # duplicate name
        ],
        "business_address": [
            "1795 Westchester Drive, High Point, NC",
            "17560 Ellis Road, Tahlequah, OK",
            "1712 Montebello Avenue, Phoenix, AZ",
            "1795 Westchester Drive, High Point, NC",  # duplicate address
        ],
        "country": ["US", "US", "US", "US"],
    })
    df.to_csv(path, sep="\t", index=False)


def create_sample_source3(path: Path) -> None:
    """Create a small sample source3 file for testing."""
    df = pd.DataFrame({
        "entity_id": ["S3-1", "S3-2", "S3-3"],
        "business_name": [
            "wilfordhancock.com",
            "International South Consultants Private Ltd",
            "LLC Moncada Learning Center",
        ],
        "business_address": [
            "Mack Rd, Haltom City, Texas",
            "",  # empty address
            "5780 Fawn Ct, Fort Worth, Texas",
        ],
        "country": ["US", "India", "US"],
    })
    df.to_csv(path, sep="\t", index=False)


def test_build_inverted_indexes_source2():
    with tempfile.TemporaryDirectory() as tmpdir:
        s2_path = Path(tmpdir) / "source2.tsv"
        create_sample_source2(s2_path)

        # Create key sets from the sample data
        key_sets = BlockingKeySets(
            name_keys={"orelee's barbershop", "prime money", "b+ retail inc"},
            address_keys={"1795 westchester drive high point nc", "17560 ellis road tahlequah ok", "1712 montebello avenue phoenix az"},
            name_country_keys={"orelee's barbershop|us", "prime money|us", "b+ retail inc|us"},
            address_country_keys={"1795 westchester drive high point nc|us", "17560 ellis road tahlequah ok|us", "1712 montebello avenue phoenix az|us"},
        )

        name_idx, addr_idx, nc_idx, ac_idx, stats = build_filtered_inverted_indexes(s2_path, key_sets)

        # Check stats
        assert stats.source == "SOURCE2"
        assert stats.total_entities == 4

        # Name index: "orelee's barbershop" appears twice
        assert len(name_idx) == 3  # 3 unique normalized names
        assert len(name_idx.get("orelee's barbershop")) == 2
        assert len(name_idx.get("prime money")) == 1
        assert len(name_idx.get("b+ retail inc")) == 1

        # Address index: duplicate address
        assert len(addr_idx) == 3
        assert len(addr_idx.get("1795 westchester drive high point nc")) == 2

        # Name+Country: same as name since all US
        assert len(nc_idx) == 3
        assert len(nc_idx.get("orelee's barbershop|us")) == 2

        # Address+Country
        assert len(ac_idx) == 3


def test_build_inverted_indexes_source3():
    with tempfile.TemporaryDirectory() as tmpdir:
        s3_path = Path(tmpdir) / "source3.tsv"
        create_sample_source3(s3_path)

        key_sets = BlockingKeySets(
            name_keys={"wilfordhancockcom", "international south consultants private ltd", "llc moncada learning center"},
            address_keys={"mack rd haltom city texas", "5780 fawn ct fort worth texas"},
            name_country_keys={"wilfordhancockcom|us", "international south consultants private ltd|india", "llc moncada learning center|us"},
            address_country_keys={"mack rd haltom city texas|us", "5780 fawn ct fort worth texas|us"},
        )

        name_idx, addr_idx, nc_idx, ac_idx, stats = build_filtered_inverted_indexes(s3_path, key_sets)

        assert stats.source == "SOURCE3"
        assert stats.total_entities == 3

        # Name index - punctuation removed
        assert len(name_idx) == 3
        assert len(name_idx.get("wilfordhancockcom")) == 1  # . removed
        assert len(name_idx.get("international south consultants private ltd")) == 1

        # Address index - one empty
        assert len(addr_idx) == 2  # empty address ignored
        assert len(addr_idx.get("mack rd haltom city texas")) == 1
        assert len(addr_idx.get("5780 fawn ct fort worth texas")) == 1

        # Name+Country
        assert len(nc_idx) == 3


# --------------------------------------------------------------------------
# generate_candidates_for_s1 tests
# --------------------------------------------------------------------------


def test_generate_candidates_for_s1_basic():
    # Build S2 index with known data
    s2_name_idx = CompactInvertedIndex()
    s2_name_idx.add("orelee's barbershop", "S2-1")
    s2_name_idx.add("prime money", "S2-2")

    s2_addr_idx = CompactInvertedIndex()
    s2_addr_idx.add("1795 westchester drive high point nc", "S2-1")

    s2_nc_idx = CompactInvertedIndex()
    s2_nc_idx.add("orelee's barbershop|us", "S2-1")

    s2_ac_idx = CompactInvertedIndex()
    s2_ac_idx.add("1795 westchester drive high point nc|us", "S2-1")

    # Empty S3 indexes
    s3_name_idx = CompactInvertedIndex()
    s3_addr_idx = CompactInvertedIndex()
    s3_nc_idx = CompactInvertedIndex()
    s3_ac_idx = CompactInvertedIndex()

    result = generate_candidates_for_s1(
        s1_id="S1-1",
        s1_name="Orelee's Barbershop",
        s1_address="1795 Westchester Drive, High Point, NC",
        s1_country="US",
        s2_name_idx=s2_name_idx,
        s2_address_idx=s2_addr_idx,
        s2_name_country_idx=s2_nc_idx,
        s2_address_country_idx=s2_ac_idx,
        s3_name_idx=s3_name_idx,
        s3_address_idx=s3_addr_idx,
        s3_name_country_idx=s3_nc_idx,
        s3_address_country_idx=s3_ac_idx,
    )

    assert result.s1_id == "S1-1"
    assert result.s2_candidates == {"S2-1"}  # matched on name, address, name+country, address+country
    assert result.s3_candidates == set()
    assert result.total_candidates() == 1


def test_generate_candidates_for_s1_union():
    """Test that candidates are unioned across all four strategies."""
    s2_name_idx = CompactInvertedIndex()
    s2_name_idx.add("orelee's barbershop", "S2-1")  # matches on name

    s2_addr_idx = CompactInvertedIndex()
    s2_addr_idx.add("1795 westchester drive high point nc", "S2-2")  # matches on address

    s2_nc_idx = CompactInvertedIndex()
    s2_nc_idx.add("orelee's barbershop|us", "S2-3")  # matches on name+country

    s2_ac_idx = CompactInvertedIndex()
    s2_ac_idx.add("1795 westchester drive high point nc|us", "S2-4")  # matches on address+country

    s3_name_idx = CompactInvertedIndex()
    s3_addr_idx = CompactInvertedIndex()
    s3_nc_idx = CompactInvertedIndex()
    s3_ac_idx = CompactInvertedIndex()

    result = generate_candidates_for_s1(
        s1_id="S1-1",
        s1_name="Orelee's Barbershop",
        s1_address="1795 Westchester Drive, High Point, NC",
        s1_country="US",
        s2_name_idx=s2_name_idx,
        s2_address_idx=s2_addr_idx,
        s2_name_country_idx=s2_nc_idx,
        s2_address_country_idx=s2_ac_idx,
        s3_name_idx=s3_name_idx,
        s3_address_idx=s3_addr_idx,
        s3_name_country_idx=s3_nc_idx,
        s3_address_country_idx=s3_ac_idx,
    )

    # Should get all 4 S2 candidates (union of all strategies)
    assert result.s2_candidates == {"S2-1", "S2-2", "S2-3", "S2-4"}
    assert result.total_candidates() == 4


def test_generate_candidates_for_s1_empty_keys():
    """Test with empty/None S1 fields."""
    s2_name_idx = CompactInvertedIndex()
    s2_addr_idx = CompactInvertedIndex()
    s2_nc_idx = CompactInvertedIndex()
    s2_ac_idx = CompactInvertedIndex()
    s3_name_idx = CompactInvertedIndex()
    s3_addr_idx = CompactInvertedIndex()
    s3_nc_idx = CompactInvertedIndex()
    s3_ac_idx = CompactInvertedIndex()

    result = generate_candidates_for_s1(
        s1_id="S1-1",
        s1_name="",  # empty
        s1_address=None,  # None
        s1_country="US",
        s2_name_idx=s2_name_idx,
        s2_address_idx=s2_addr_idx,
        s2_name_country_idx=s2_nc_idx,
        s2_address_country_idx=s2_ac_idx,
        s3_name_idx=s3_name_idx,
        s3_address_idx=s3_addr_idx,
        s3_name_country_idx=s3_nc_idx,
        s3_address_country_idx=s3_ac_idx,
    )

    assert result.s2_candidates == set()
    assert result.s3_candidates == set()


# --------------------------------------------------------------------------
# run_blocking_benchmark integration test
# --------------------------------------------------------------------------


def create_sample_source1(path: Path) -> None:
    """Create a small sample source1 file for testing."""
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
        "business_name": [
            "Orelee's Barbershop",
            "Prime Money",
            "B+ Retail Inc",
            "No Match LLC",
        ],
        "business_address": [
            "1795 Westchester Drive, High Point, NC",
            "17560 Ellis Road, Tahlequah, OK",
            "1712 Montebello Avenue, Phoenix, AZ",
            "999 Nowhere St",
        ],
        "country": ["US", "US", "US", "US"],
    })
    df.to_csv(path, sep="\t", index=False)


def test_run_blocking_benchmark_integration():
    """Integration test with small synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        s1_path = tmpdir / "source1.tsv"
        s2_path = tmpdir / "source2.tsv"
        s3_path = tmpdir / "source3.tsv"

        create_sample_source1(s1_path)
        create_sample_source2(s2_path)
        create_sample_source3(s3_path)

        # Validation S1 IDs: S1-1, S1-2, S1-3 (but not S1-4)
        val_s1_ids = {"S1-1", "S1-2", "S1-3"}

        # Ground truth: S1-1 matches S2-1, S1-2 matches S2-2, S1-3 matches S2-3
        gt_s1_matches = {
            "S1-1": (["S2-1"], []),
            "S1-2": (["S2-2"], []),
            "S1-3": (["S2-3"], []),
        }

        result = run_blocking_benchmark(
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            val_s1_ids=val_s1_ids,
            gt_s1_matches=gt_s1_matches,
            chunksize=1000,
        )

        # Check results
        assert result.total_s1_evaluated == 3
        assert result.true_match_recall > 0  # Should find some matches
        assert result.total_candidate_pairs > 0
        assert result.s2_recall > 0


# --------------------------------------------------------------------------
# Determinism test
# --------------------------------------------------------------------------


def test_blocking_deterministic():
    """Verify that blocking produces deterministic results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        s1_path = tmpdir / "source1.tsv"
        s2_path = tmpdir / "source2.tsv"
        s3_path = tmpdir / "source3.tsv"

        create_sample_source1(s1_path)
        create_sample_source2(s2_path)
        create_sample_source3(s3_path)

        val_s1_ids = {"S1-1", "S1-2", "S1-3"}
        gt_s1_matches = {
            "S1-1": (["S2-1"], []),
            "S1-2": (["S2-2"], []),
            "S1-3": (["S2-3"], []),
        }

        # Run twice
        result1 = run_blocking_benchmark(
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            val_s1_ids=val_s1_ids,
            gt_s1_matches=gt_s1_matches,
            chunksize=1000,
        )

        result2 = run_blocking_benchmark(
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            val_s1_ids=val_s1_ids,
            gt_s1_matches=gt_s1_matches,
            chunksize=1000,
        )

        # Results should be identical
        assert result1.true_match_recall == result2.true_match_recall
        assert result1.s2_recall == result2.s2_recall
        assert result1.s3_recall == result2.s3_recall
        assert result1.total_candidate_pairs == result2.total_candidate_pairs
        assert result1.avg_candidates_per_s1 == result2.avg_candidates_per_s1
        assert result1.median_candidates_per_s1 == result2.median_candidates_per_s1
        assert result1.max_candidates_per_s1 == result2.max_candidates_per_s1


# --------------------------------------------------------------------------
# Batch candidate generation correctness test
# --------------------------------------------------------------------------


def test_batch_candidate_generation_matches_scalar():
    """Verify that the vectorized batch candidate generation produces
    the same candidate sets as the scalar per-S1 function on a toy dataset."""
    # Build S2 and S3 indexes with known data
    s2_name_idx = CompactInvertedIndex()
    s2_name_idx.add("orelee's barbershop", "S2-1")
    s2_name_idx.add("prime money", "S2-2")
    s2_addr_idx = CompactInvertedIndex()
    s2_addr_idx.add("1795 westchester drive high point nc", "S2-1")
    s2_nc_idx = CompactInvertedIndex()
    s2_nc_idx.add("orelee's barbershop|us", "S2-1")
    s2_ac_idx = CompactInvertedIndex()
    s2_ac_idx.add("1795 westchester drive high point nc|us", "S2-1")

    s3_name_idx = CompactInvertedIndex()
    s3_name_idx.add("wilfordhancockcom", "S3-1")
    s3_addr_idx = CompactInvertedIndex()
    s3_addr_idx.add("mack rd haltom city texas", "S3-1")
    s3_nc_idx = CompactInvertedIndex()
    s3_nc_idx.add("wilfordhancockcom|us", "S3-1")
    s3_ac_idx = CompactInvertedIndex()
    s3_ac_idx.add("mack rd haltom city texas|us", "S3-1")

    # Create a toy S1 chunk
    s1_chunk = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": [
            "Orelee's Barbershop",
            "Prime Money",
            "wilfordhancock.com",
        ],
        "business_address": [
            "1795 Westchester Drive, High Point, NC",
            "17560 Ellis Road, Tahlequah, OK",
            "Mack Rd, Haltom City, Texas",
        ],
        "country": ["US", "US", "US"],
    })

    # Get batch results
    batch_results, batch_counts = generate_candidates_batch_vectorized(
        s1_chunk,
        s2_name_idx, s2_addr_idx, s2_nc_idx, s2_ac_idx,
        s3_name_idx, s3_addr_idx, s3_nc_idx, s3_ac_idx,
    )

    # Get scalar results for comparison
    for i, row in s1_chunk.iterrows():
        scalar_result = generate_candidates_for_s1(
            s1_id=row["entity_id"],
            s1_name=row["business_name"],
            s1_address=row["business_address"],
            s1_country=row["country"],
            s2_name_idx=s2_name_idx,
            s2_address_idx=s2_addr_idx,
            s2_name_country_idx=s2_nc_idx,
            s2_address_country_idx=s2_ac_idx,
            s3_name_idx=s3_name_idx,
            s3_address_idx=s3_addr_idx,
            s3_name_country_idx=s3_nc_idx,
            s3_address_country_idx=s3_ac_idx,
        )

        batch_result = batch_results[i]
        assert batch_result.s1_id == scalar_result.s1_id
        assert batch_result.s2_candidates == scalar_result.s2_candidates, \
            f"S2 candidates differ for {batch_result.s1_id}: batch={batch_result.s2_candidates}, scalar={scalar_result.s2_candidates}"
        assert batch_result.s3_candidates == scalar_result.s3_candidates, \
            f"S3 candidates differ for {batch_result.s1_id}: batch={batch_result.s3_candidates}, scalar={scalar_result.s3_candidates}"
        assert batch_counts[i] == scalar_result.total_candidates()


def test_batch_candidate_generation_union_strategies():
    """Verify that the batch function correctly unions candidates from all four strategies."""
    s2_name_idx = CompactInvertedIndex()
    s2_name_idx.add("orelee's barbershop", "S2-1")
    s2_addr_idx = CompactInvertedIndex()
    s2_addr_idx.add("1795 westchester drive high point nc", "S2-2")
    s2_nc_idx = CompactInvertedIndex()
    s2_nc_idx.add("orelee's barbershop|us", "S2-3")
    s2_ac_idx = CompactInvertedIndex()
    s2_ac_idx.add("1795 westchester drive high point nc|us", "S2-4")

    s3_name_idx = CompactInvertedIndex()
    s3_addr_idx = CompactInvertedIndex()
    s3_nc_idx = CompactInvertedIndex()
    s3_ac_idx = CompactInvertedIndex()

    s1_chunk = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Orelee's Barbershop"],
        "business_address": ["1795 Westchester Drive, High Point, NC"],
        "country": ["US"],
    })

    batch_results, batch_counts = generate_candidates_batch_vectorized(
        s1_chunk,
        s2_name_idx, s2_addr_idx, s2_nc_idx, s2_ac_idx,
        s3_name_idx, s3_addr_idx, s3_nc_idx, s3_ac_idx,
    )

    assert len(batch_results) == 1
    assert batch_results[0].s2_candidates == {"S2-1", "S2-2", "S2-3", "S2-4"}
    assert batch_results[0].s3_candidates == set()
    assert batch_counts[0] == 4