"""Unit tests for blocking_strategies module (Task 3B)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from business_entity_resolution.blocking import (
    CandidateResult,
    CompactInvertedIndex,
)
from business_entity_resolution.blocking_strategies import (
    ADDRESS_STOP_WORDS,
    ALL_STRATEGIES,
    BASELINE_STRATEGIES,
    NAME_STOP_WORDS,
    STRATEGY_NAME_NGRAM,
    STRATEGY_NAME_TOKEN,
    ExtendedKeySets,
    StrategyIndexes,
    extract_address_ngrams,
    extract_address_tokens,
    extract_char_ngrams,
    extract_name_ngrams,
    extract_name_tokens,
    generate_candidates_extended_batch,
    run_extended_blocking_benchmark,
)
from business_entity_resolution.normalization import (
    make_address_key_series,
    make_name_key_series,
)


# --------------------------------------------------------------------------
# Stop word tests
# --------------------------------------------------------------------------


def test_name_stop_words_contains_expected():
    assert "inc" in NAME_STOP_WORDS
    assert "ltd" in NAME_STOP_WORDS
    assert "llc" in NAME_STOP_WORDS
    assert "the" in NAME_STOP_WORDS
    assert "company" in NAME_STOP_WORDS


def test_address_stop_words_contains_expected():
    assert "st" in ADDRESS_STOP_WORDS
    assert "rd" in ADDRESS_STOP_WORDS
    assert "avenue" in ADDRESS_STOP_WORDS
    assert "road" in ADDRESS_STOP_WORDS
    assert "ltd" not in ADDRESS_STOP_WORDS  # business suffix, not address


# --------------------------------------------------------------------------
# Token extraction tests
# --------------------------------------------------------------------------


def test_extract_name_tokens_basic():
    series = pd.Series(["orelee's barbershop", "prime money"])
    tokens = extract_name_tokens(series)
    assert tokens.iloc[0] == ["orelee's", "barbershop"]
    assert tokens.iloc[1] == ["prime", "money"]


def test_extract_name_filters_stop_words():
    series = pd.Series(["abc inc llc", "the company group"])
    tokens = extract_name_tokens(series)
    assert tokens.iloc[0] == ["abc"]
    assert tokens.iloc[1] == []


def test_extract_name_filters_short_tokens():
    series = pd.Series(["a b ab abc"])
    tokens = extract_name_tokens(series)
    # "a", "b" are stop words, "ab" is 2 chars (kept), "abc" is 3 chars (kept)
    assert "ab" in tokens.iloc[0]
    assert "abc" in tokens.iloc[0]


def test_extract_address_tokens_basic():
    series = pd.Series(["1795 westchester drive high point nc"])
    tokens = extract_address_tokens(series)
    # "drive" is a stop word, others should remain
    assert "1795" in tokens.iloc[0]
    assert "westchester" in tokens.iloc[0]
    assert "drive" not in tokens.iloc[0]
    assert "high" in tokens.iloc[0]
    assert "point" in tokens.iloc[0]


def test_extract_address_filters_street_suffixes():
    series = pd.Series(["123 main street apt 4b"])
    tokens = extract_address_tokens(series)
    assert "123" in tokens.iloc[0]
    assert "main" in tokens.iloc[0]
    assert "street" not in tokens.iloc[0]
    assert "apt" not in tokens.iloc[0]
    assert "4b" in tokens.iloc[0]  # "4" alone would be filtered (< 2 chars)


def test_extract_tokens_handles_empty():
    series = pd.Series(["", "   "])
    tokens = extract_name_tokens(series)
    assert tokens.iloc[0] == []
    assert tokens.iloc[1] == []


# --------------------------------------------------------------------------
# N-gram extraction tests
# --------------------------------------------------------------------------


def test_extract_char_ngrams_basic():
    assert extract_char_ngrams("hello", 3) == ["hel", "ell", "llo"]
    assert extract_char_ngrams("ab", 3) == []
    assert extract_char_ngrams("", 3) == []


def test_extract_char_ngrams_size():
    assert extract_char_ngrams("abc", 3) == ["abc"]
    assert extract_char_ngrams("abcd", 3) == ["abc", "bcd"]
    assert extract_char_ngrams("abcde", 3) == ["abc", "bcd", "cde"]


def test_extract_name_ngrams_series():
    series = pd.Series(["abc", "hello"])
    ngrams = extract_name_ngrams(series, 3)
    assert ngrams.iloc[0] == ["abc"]
    assert ngrams.iloc[1] == ["hel", "ell", "llo"]


def test_extract_address_ngrams_series():
    series = pd.Series(["123 main"])
    ngrams = extract_address_ngrams(series, 3)
    assert "123" in ngrams.iloc[0]
    assert "23 " in ngrams.iloc[0]
    assert " ma" in ngrams.iloc[0]


# --------------------------------------------------------------------------
# Extended key sets tests
# --------------------------------------------------------------------------


def test_extended_key_sets_default():
    ks = ExtendedKeySets()
    assert ks.name_keys == set()
    assert ks.name_token_keys == set()
    assert ks.name_ngram_keys == set()


def test_extended_key_sets_from_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        s1_path = Path(tmpdir) / "s1.tsv"
        df = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["Orelee's Barbershop", "Prime Money"],
            "business_address": ["1795 Westchester Dr", "123 Main St"],
            "country": ["US", "US"],
        })
        df.to_csv(s1_path, sep="\t", index=False)

        ks = ExtendedKeySets.from_validation_s1(
            s1_path, {"S1-1", "S1-2"}, chunksize=100, strategies=ALL_STRATEGIES
        )

        # Baseline keys
        assert "orelee's barbershop" in ks.name_keys
        # Token keys
        assert "orelee's" in ks.name_token_keys
        assert "barbershop" in ks.name_token_keys
        # N-gram keys
        assert "ore" in ks.name_ngram_keys
        assert "bar" in ks.name_ngram_keys


def test_extended_key_sets_strategy_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        s1_path = Path(tmpdir) / "s1.tsv"
        df = pd.DataFrame({
            "entity_id": ["S1-1"],
            "business_name": ["ABC Corp"],
            "business_address": ["123 Main St"],
            "country": ["US"],
        })
        df.to_csv(s1_path, sep="\t", index=False)

        # Only token strategies
        ks = ExtendedKeySets.from_validation_s1(
            s1_path, {"S1-1"}, chunksize=100, strategies=[STRATEGY_NAME_TOKEN]
        )

        assert ks.name_token_keys  # token keys extracted
        assert ks.name_keys == set()  # baseline keys not extracted
        assert ks.name_ngram_keys == set()  # ngram keys not extracted


# --------------------------------------------------------------------------
# Candidate generation tests
# --------------------------------------------------------------------------


def test_candidate_generation_token_basic():
    """Verify token-based candidate generation finds matches."""
    # Build token index
    name_token_idx = CompactInvertedIndex()
    name_token_idx.add("orelee's", "S2-1")
    name_token_idx.add("barbershop", "S2-1")
    name_token_idx.add("orelee's", "S2-2")

    indexes = StrategyIndexes(name_token_idx=name_token_idx)

    s1_chunk = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Orelee's Barbershop"],
        "business_address": ["1795 Westchester Dr"],
        "country": ["US"],
    })

    results, counts = generate_candidates_extended_batch(
        s1_chunk, indexes, StrategyIndexes(), strategies=[STRATEGY_NAME_TOKEN]
    )

    assert len(results) == 1
    assert results[0].s1_id == "S1-1"
    assert results[0].s2_candidates == {"S2-1", "S2-2"}
    assert results[0].s3_candidates == set()
    assert counts[0] == 2


def test_candidate_generation_ngram_basic():
    """Verify n-gram-based candidate generation finds matches."""
    name_ngram_idx = CompactInvertedIndex()
    name_ngram_idx.add("abc", "S2-1")
    name_ngram_idx.add("bcr", "S2-1")  # overlap with "abcd"

    indexes = StrategyIndexes(name_ngram_idx=name_ngram_idx)

    s1_chunk = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["abc"],
        "business_address": ["123 Main St"],
        "country": ["US"],
    })

    results, counts = generate_candidates_extended_batch(
        s1_chunk, indexes, StrategyIndexes(), strategies=[STRATEGY_NAME_NGRAM]
    )

    assert len(results) == 1
    assert "S2-1" in results[0].s2_candidates


def test_candidate_generation_combined_strategies():
    """Verify combined strategies union candidates correctly."""
    name_idx = CompactInvertedIndex()
    name_idx.add("abc", "S2-1")

    name_token_idx = CompactInvertedIndex()
    name_token_idx.add("abc", "S2-2")

    indexes = StrategyIndexes(name_idx=name_idx, name_token_idx=name_token_idx)

    s1_chunk = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["abc"],
        "business_address": ["123 Main St"],
        "country": ["US"],
    })

    results, counts = generate_candidates_extended_batch(
        s1_chunk, indexes, StrategyIndexes(),
        strategies=["name", "name_token"]
    )

    assert results[0].s2_candidates == {"S2-1", "S2-2"}
    assert counts[0] == 2


# --------------------------------------------------------------------------
# Integration test
# --------------------------------------------------------------------------


def test_extended_benchmark_integration():
    """Integration test with small synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        s1_path = tmpdir / "source1.tsv"
        s2_path = tmpdir / "source2.tsv"
        s3_path = tmpdir / "source3.tsv"

        pd.DataFrame({
            "entity_id": ["S1-1", "S1-2", "S1-3"],
            "business_name": ["ABC Corp", "XYZ Ltd", "Orelee's Barbershop"],
            "business_address": ["123 Main St", "456 Oak Ave", "1795 Westchester Dr"],
            "country": ["US", "US", "US"],
        }).to_csv(s1_path, sep="\t", index=False)

        pd.DataFrame({
            "entity_id": ["S2-1", "S2-2"],
            "business_name": ["ABC Inc", "XYZ Corporation"],
            "business_address": ["123 Main Street", "456 Oak Avenue"],
            "country": ["US", "US"],
        }).to_csv(s2_path, sep="\t", index=False)

        pd.DataFrame({
            "entity_id": ["S3-1"],
            "business_name": ["Orelee's Barber Shop"],
            "business_address": ["1795 Westchester Drive"],
            "country": ["US"],
        }).to_csv(s3_path, sep="\t", index=False)

        val_s1_ids = {"S1-1", "S1-2", "S1-3"}
        gt_s1_matches = {
            "S1-1": (["S2-1"], []),
            "S1-2": (["S2-2"], []),
            "S1-3": ([], ["S3-1"]),
        }

        result = run_extended_blocking_benchmark(
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            val_s1_ids=val_s1_ids,
            gt_s1_matches=gt_s1_matches,
            chunksize=100,
            strategies=ALL_STRATEGIES,
        )

        assert result.total_s1_evaluated == 3
        assert result.total_candidate_pairs > 0
        assert result.true_match_recall > 0
