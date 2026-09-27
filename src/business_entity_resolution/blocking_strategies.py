"""Task 3B: Extended blocking strategies.

Adds token-based and character n-gram blocking for business names and addresses.

Strategies:
- Token-based name blocking: split normalized name into tokens, filter stop words,
  each remaining token is a blocking key.
- Token-based address blocking: split normalized address into tokens, filter stop words,
  each remaining token is a blocking key.
- Character n-gram name blocking: extract 3-character n-grams from normalized name,
  each n-gram is a blocking key.
- Character n-gram address blocking: extract 3-character n-grams from normalized address,
  each n-gram is a blocking key.

All strategies are combined with the 3A baseline (exact normalized name, address,
name+country, address+country) using UNION semantics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from business_entity_resolution.blocking import (
    BlockingBenchmarkResult,
    BlockingStats,
    CandidateResult,
    CompactInvertedIndex,
    DEFAULT_CHUNKSIZE,
)
from business_entity_resolution.normalization import (
    make_address_country_key_series,
    make_address_key_series,
    make_name_country_key_series,
    make_name_key_series,
)

# Column constants (matching blocking.py)
S1_COL = "entity_id"
S1_NAME_COL = "business_name"
S1_ADDRESS_COL = "business_address"
S1_COUNTRY_COL = "country"
S23_COLS = ["entity_id", "business_name", "business_address", "country"]

# Strategy identifiers
STRATEGY_NAME = "name"
STRATEGY_ADDRESS = "address"
STRATEGY_NAME_COUNTRY = "name_country"
STRATEGY_ADDRESS_COUNTRY = "address_country"
STRATEGY_NAME_TOKEN = "name_token"
STRATEGY_ADDRESS_TOKEN = "address_token"
STRATEGY_NAME_NGRAM = "name_ngram"
STRATEGY_ADDRESS_NGRAM = "address_ngram"

ALL_STRATEGIES = [
    STRATEGY_NAME,
    STRATEGY_ADDRESS,
    STRATEGY_NAME_COUNTRY,
    STRATEGY_ADDRESS_COUNTRY,
    STRATEGY_NAME_TOKEN,
    STRATEGY_ADDRESS_TOKEN,
    STRATEGY_NAME_NGRAM,
    STRATEGY_ADDRESS_NGRAM,
]

# Maximum number of source entities a single token/n-gram can appear in.
# Tokens appearing in more entities than this are too common to be useful
# for blocking and would create pathological candidate sets.
# This is a standard IR technique (posting list thresholding).
MAX_TOKEN_FREQUENCY = 50
MAX_NGRAM_FREQUENCY = 200

BASELINE_STRATEGIES = [
    STRATEGY_NAME,
    STRATEGY_ADDRESS,
    STRATEGY_NAME_COUNTRY,
    STRATEGY_ADDRESS_COUNTRY,
]

# --------------------------------------------------------------------------
# Stop words for token filtering
# --------------------------------------------------------------------------

NAME_STOP_WORDS = frozenset({
    # Articles and prepositions
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    # Conjunctions
    "and", "or",
    # Common business suffixes
    "inc", "ltd", "llc", "corp", "co", "company", "corporation", "private", "limited",
    "pvt", "pc", "llp", "pllc",
    # Common generic terms
    "group", "international", "services", "solutions", "systems",
    # India-specific common terms
    "india",
    # Common business descriptors (too frequent to be useful for blocking)
    "care", "associates", "partners", "center", "health", "clinic",
    "global", "trading", "technologies", "brothers", "industries",
    "enterprises", "ventures", "foundation", "medicine",
})

ADDRESS_STOP_WORDS = frozenset({
    # Articles and prepositions
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or",
    # Street suffix abbreviations
    "st", "rd", "ave", "ln", "dr", "blvd", "ct", "pl", "cir", "wy",
    # Address parts
    "apt", "ste", "suite", "floor", "fl", "building", "bldg", "unit", "no", "number",
    # Directions
    "n", "s", "e", "w", "ne", "nw", "se", "sw",
    "north", "south", "east", "west",
    # Street types
    "road", "street", "avenue", "lane", "drive", "boulevard", "court", "place", "circle", "highway",
    # Very common geographic terms (would create huge candidate sets)
    "city", "new", "null", "plot",
})

# --------------------------------------------------------------------------
# Token extraction
# --------------------------------------------------------------------------


def extract_name_tokens(series: pd.Series) -> pd.Series:
    """Extract filtered tokens from normalized business names."""
    tokens = series.str.split()

    def filter_tokens(token_list):
        if not isinstance(token_list, list) or not token_list:
            return []
        return [t for t in token_list if t not in NAME_STOP_WORDS and len(t) >= 2]

    return tokens.apply(filter_tokens)


def extract_address_tokens(series: pd.Series) -> pd.Series:
    """Extract filtered tokens from normalized business addresses."""
    tokens = series.str.split()

    def filter_tokens(token_list):
        if not isinstance(token_list, list) or not token_list:
            return []
        return [t for t in token_list if t not in ADDRESS_STOP_WORDS and len(t) >= 2]

    return tokens.apply(filter_tokens)


# --------------------------------------------------------------------------
# Character n-gram extraction
# --------------------------------------------------------------------------


def extract_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from a string."""
    if not text or len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def extract_name_ngrams(series: pd.Series, n: int = 3) -> pd.Series:
    """Extract character n-grams from normalized business names."""

    def get_ngrams(text):
        return extract_char_ngrams(text, n)

    return series.apply(get_ngrams)


def extract_address_ngrams(series: pd.Series, n: int = 3) -> pd.Series:
    """Extract character n-grams from normalized business addresses."""

    def get_ngrams(text):
        return extract_char_ngrams(text, n)

    return series.apply(get_ngrams)


# --------------------------------------------------------------------------
# Extended key sets
# --------------------------------------------------------------------------


@dataclass
class ExtendedKeySets:
    """Blocking key sets for all strategies (3A baseline + 3B extended)."""

    name_keys: Set[str] = field(default_factory=set)
    address_keys: Set[str] = field(default_factory=set)
    name_country_keys: Set[str] = field(default_factory=set)
    address_country_keys: Set[str] = field(default_factory=set)
    name_token_keys: Set[str] = field(default_factory=set)
    address_token_keys: Set[str] = field(default_factory=set)
    name_ngram_keys: Set[str] = field(default_factory=set)
    address_ngram_keys: Set[str] = field(default_factory=set)

    @classmethod
    def from_validation_s1(
        cls,
        s1_path: str | Path,
        val_s1_ids: Set[str],
        chunksize: int = DEFAULT_CHUNKSIZE,
        strategies: Optional[List[str]] = None,
    ) -> "ExtendedKeySets":
        """Extract all blocking keys from validation S1 entities."""
        if strategies is None:
            strategies = ALL_STRATEGIES

        key_sets = cls()

        for chunk in pd.read_csv(
            s1_path,
            sep="\t",
            usecols=[S1_COL, S1_NAME_COL, S1_ADDRESS_COL, S1_COUNTRY_COL],
            dtype={
                S1_COL: str,
                S1_NAME_COL: str,
                S1_ADDRESS_COL: str,
                S1_COUNTRY_COL: str,
            },
            chunksize=chunksize,
            keep_default_na=False,
            na_values=[""],
        ):
            chunk = chunk[chunk[S1_COL].isin(val_s1_ids)]
            if chunk.empty:
                continue

            norm_name = make_name_key_series(chunk[S1_NAME_COL])
            norm_addr = make_address_key_series(chunk[S1_ADDRESS_COL])

            if STRATEGY_NAME in strategies:
                key_sets.name_keys.update(norm_name.dropna().unique())
            if STRATEGY_ADDRESS in strategies:
                key_sets.address_keys.update(norm_addr.dropna().unique())
            if STRATEGY_NAME_COUNTRY in strategies:
                key_sets.name_country_keys.update(
                    make_name_country_key_series(norm_name, chunk[S1_COUNTRY_COL])
                    .dropna()
                    .unique()
                )
            if STRATEGY_ADDRESS_COUNTRY in strategies:
                key_sets.address_country_keys.update(
                    make_address_country_key_series(norm_addr, chunk[S1_COUNTRY_COL])
                    .dropna()
                    .unique()
                )
            if STRATEGY_NAME_TOKEN in strategies:
                for token_list in extract_name_tokens(norm_name):
                    key_sets.name_token_keys.update(token_list)
            if STRATEGY_ADDRESS_TOKEN in strategies:
                for token_list in extract_address_tokens(norm_addr):
                    key_sets.address_token_keys.update(token_list)
            if STRATEGY_NAME_NGRAM in strategies:
                for ngram_list in extract_name_ngrams(norm_name):
                    key_sets.name_ngram_keys.update(ngram_list)
            if STRATEGY_ADDRESS_NGRAM in strategies:
                for ngram_list in extract_address_ngrams(norm_addr):
                    key_sets.address_ngram_keys.update(ngram_list)

        return key_sets


# --------------------------------------------------------------------------
# Strategy indexes
# --------------------------------------------------------------------------


@dataclass
class StrategyIndexes:
    """Inverted indexes for all blocking strategies for one source."""

    name_idx: Optional[CompactInvertedIndex] = None
    address_idx: Optional[CompactInvertedIndex] = None
    name_country_idx: Optional[CompactInvertedIndex] = None
    address_country_idx: Optional[CompactInvertedIndex] = None
    name_token_idx: Optional[CompactInvertedIndex] = None
    address_token_idx: Optional[CompactInvertedIndex] = None
    name_ngram_idx: Optional[CompactInvertedIndex] = None
    address_ngram_idx: Optional[CompactInvertedIndex] = None


# --------------------------------------------------------------------------
# Extended index building
# --------------------------------------------------------------------------


def _build_token_index_for_chunk(
    entity_ids: pd.Series,
    token_series: pd.Series,
    token_keys: Set[str],
    index: CompactInvertedIndex,
    token_counts: dict,
    max_freq: int,
) -> pd.Series:
    """Build token index entries for a chunk with frequency filtering."""
    if not token_keys:
        return pd.Series(False, index=entity_ids.index)

    # Explode tokens: each entity -> multiple rows (one per token)
    df = pd.DataFrame({"entity_id": entity_ids.values, "token": token_series.values})
    df = df.explode("token")
    df = df.dropna(subset=["token"])

    # Filter to only tokens in validation key set
    mask = df["token"].isin(token_keys)
    filtered = df[mask]

    if filtered.empty:
        return pd.Series(False, index=entity_ids.index)

    # Count token frequencies and build index with threshold
    matched_entities = set()
    for token, eid in zip(filtered["token"], filtered["entity_id"]):
        count = token_counts.get(token, 0) + 1
        token_counts[token] = count
        if count <= max_freq:
            index.add(token, eid)
            matched_entities.add(eid)
        # If count > max_freq, we've already added entries; they stay in index
        # but future entities won't be added. This is acceptable for blocking.

    # Return mask: which original entities had at least one matching token
    return entity_ids.isin(list(matched_entities))


def _build_ngram_index_for_chunk(
    entity_ids: pd.Series,
    ngram_series: pd.Series,
    ngram_keys: Set[str],
    index: CompactInvertedIndex,
    ngram_counts: dict,
    max_freq: int,
) -> pd.Series:
    """Build n-gram index entries for a chunk with frequency filtering."""
    if not ngram_keys:
        return pd.Series(False, index=entity_ids.index)

    # Explode n-grams
    df = pd.DataFrame({"entity_id": entity_ids.values, "ngram": ngram_series.values})
    df = df.explode("ngram")
    df = df.dropna(subset=["ngram"])

    # Filter to only n-grams in validation key set
    mask = df["ngram"].isin(ngram_keys)
    filtered = df[mask]

    if filtered.empty:
        return pd.Series(False, index=entity_ids.index)

    # Count n-gram frequencies and build index with threshold
    matched_entities = set()
    for ngram, eid in zip(filtered["ngram"], filtered["entity_id"]):
        count = ngram_counts.get(ngram, 0) + 1
        ngram_counts[ngram] = count
        if count <= max_freq:
            index.add(ngram, eid)
            matched_entities.add(eid)

    # Return mask: which original entities had at least one matching n-gram
    return entity_ids.isin(list(matched_entities))


def _build_extended_indexes_for_source(
    source_path: str | Path,
    key_sets: ExtendedKeySets,
    chunksize: int = DEFAULT_CHUNKSIZE,
    strategies: Optional[List[str]] = None,
) -> Tuple[StrategyIndexes, BlockingStats]:
    """Build inverted indexes for a source, filtered by validation key sets."""
    if strategies is None:
        strategies = ALL_STRATEGIES

    indexes = StrategyIndexes()
    stats = BlockingStats(source=Path(source_path).stem.upper())

    has_name = STRATEGY_NAME in strategies
    has_addr = STRATEGY_ADDRESS in strategies
    has_nc = STRATEGY_NAME_COUNTRY in strategies
    has_ac = STRATEGY_ADDRESS_COUNTRY in strategies
    has_name_token = STRATEGY_NAME_TOKEN in strategies
    has_addr_token = STRATEGY_ADDRESS_TOKEN in strategies
    has_name_ngram = STRATEGY_NAME_NGRAM in strategies
    has_addr_ngram = STRATEGY_ADDRESS_NGRAM in strategies

    if has_name:
        indexes.name_idx = CompactInvertedIndex()
    if has_addr:
        indexes.address_idx = CompactInvertedIndex()
    if has_nc:
        indexes.name_country_idx = CompactInvertedIndex()
    if has_ac:
        indexes.address_country_idx = CompactInvertedIndex()
    if has_name_token:
        indexes.name_token_idx = CompactInvertedIndex()
    if has_addr_token:
        indexes.address_token_idx = CompactInvertedIndex()
    if has_name_ngram:
        indexes.name_ngram_idx = CompactInvertedIndex()
    if has_addr_ngram:
        indexes.address_ngram_idx = CompactInvertedIndex()

    print(f"  Building extended indexes for {source_path}...")
    chunk_num = 0

    # Track token/n-gram frequencies across chunks for thresholding
    token_counts: dict = {}
    ngram_counts: dict = {}

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        usecols=S23_COLS,
        dtype={col: str for col in S23_COLS},
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""],
    ):
        chunk_num += 1
        if chunk_num % 10 == 0:
            print(f"    Processed {chunk_num} chunks ({chunk_num * chunksize:,} rows)...")

        stats.total_entities += len(chunk)

        norm_name = make_name_key_series(chunk[S1_NAME_COL])
        norm_addr = make_address_key_series(chunk[S1_ADDRESS_COL])
        country = chunk[S1_COUNTRY_COL]
        entity_ids = chunk[S1_COL]

        # Build mask using vectorized operations for baseline strategies
        mask = pd.Series(False, index=entity_ids.index)
        if has_name:
            mask |= norm_name.isin(key_sets.name_keys)
        if has_addr:
            mask |= norm_addr.isin(key_sets.address_keys)
        if has_nc:
            nc_keys = norm_name + "|" + country
            mask |= nc_keys.isin(key_sets.name_country_keys)
        if has_ac:
            ac_keys = norm_addr + "|" + country
            mask |= ac_keys.isin(key_sets.address_country_keys)

        # Build token/n-gram indexes using explode-based approach with frequency filtering
        if has_name_token:
            name_tokens = extract_name_tokens(norm_name)
            token_mask = _build_token_index_for_chunk(
                entity_ids, name_tokens, key_sets.name_token_keys, indexes.name_token_idx,
                token_counts, MAX_TOKEN_FREQUENCY
            )
            mask |= token_mask

        if has_addr_token:
            addr_tokens = extract_address_tokens(norm_addr)
            token_mask = _build_token_index_for_chunk(
                entity_ids, addr_tokens, key_sets.address_token_keys, indexes.address_token_idx,
                token_counts, MAX_TOKEN_FREQUENCY
            )
            mask |= token_mask

        if has_name_ngram:
            name_ngrams = extract_name_ngrams(norm_name)
            ngram_mask = _build_ngram_index_for_chunk(
                entity_ids, name_ngrams, key_sets.name_ngram_keys, indexes.name_ngram_idx,
                ngram_counts, MAX_NGRAM_FREQUENCY
            )
            mask |= ngram_mask

        if has_addr_ngram:
            addr_ngrams = extract_address_ngrams(norm_addr)
            ngram_mask = _build_ngram_index_for_chunk(
                entity_ids, addr_ngrams, key_sets.address_ngram_keys, indexes.address_ngram_idx,
                ngram_counts, MAX_NGRAM_FREQUENCY
            )
            mask |= ngram_mask

        if not mask.any():
            continue

        # Build 3A baseline indexes for matched rows
        matched_ids = entity_ids[mask]
        matched_name = norm_name[mask]
        matched_addr = norm_addr[mask]
        matched_country = country[mask]

        for eid, nk, ak, c in zip(matched_ids, matched_name, matched_addr, matched_country):
            if has_name and nk in key_sets.name_keys:
                indexes.name_idx.add(nk, eid)
            if has_addr and ak in key_sets.address_keys:
                indexes.address_idx.add(ak, eid)
            if has_nc:
                nck = f"{nk}|{c}"
                if nck in key_sets.name_country_keys:
                    indexes.name_country_idx.add(nck, eid)
            if has_ac:
                ack = f"{ak}|{c}"
                if ack in key_sets.address_country_keys:
                    indexes.address_country_idx.add(ack, eid)

    print(f"  Done: {stats.total_entities:,} entities scanned")

    return indexes, stats


# --------------------------------------------------------------------------
# Extended candidate generation
# --------------------------------------------------------------------------


def generate_candidates_extended_batch(
    s1_chunk: pd.DataFrame,
    s2_indexes: StrategyIndexes,
    s3_indexes: StrategyIndexes,
    strategies: Optional[List[str]] = None,
) -> Tuple[List[CandidateResult], List[int]]:
    """Generate candidates for a batch of S1 entities using specified strategies."""
    if strategies is None:
        strategies = ALL_STRATEGIES

    norm_name = make_name_key_series(s1_chunk[S1_NAME_COL])
    norm_addr = make_address_key_series(s1_chunk[S1_ADDRESS_COL])

    # Pre-compute composite keys and tokens/ngrams
    nc_keys = None
    ac_keys = None
    name_tokens = None
    addr_tokens = None
    name_ngrams = None
    addr_ngrams = None

    if STRATEGY_NAME_COUNTRY in strategies:
        nc_keys = norm_name + "|" + s1_chunk[S1_COUNTRY_COL]
    if STRATEGY_ADDRESS_COUNTRY in strategies:
        ac_keys = norm_addr + "|" + s1_chunk[S1_COUNTRY_COL]
    if STRATEGY_NAME_TOKEN in strategies:
        name_tokens = extract_name_tokens(norm_name)
    if STRATEGY_ADDRESS_TOKEN in strategies:
        addr_tokens = extract_address_tokens(norm_addr)
    if STRATEGY_NAME_NGRAM in strategies:
        name_ngrams = extract_name_ngrams(norm_name)
    if STRATEGY_ADDRESS_NGRAM in strategies:
        addr_ngrams = extract_address_ngrams(norm_addr)

    s1_ids = s1_chunk[S1_COL].tolist()
    results: List[CandidateResult] = []
    counts: List[int] = []

    for i, s1_id in enumerate(s1_ids):
        result = CandidateResult(s1_id=s1_id)

        nk = norm_name.iloc[i]
        ak = norm_addr.iloc[i]

        # 3A strategies
        if STRATEGY_NAME in strategies and nk:
            if s2_indexes.name_idx:
                result.s2_candidates.update(s2_indexes.name_idx.get(nk))
            if s3_indexes.name_idx:
                result.s3_candidates.update(s3_indexes.name_idx.get(nk))

        if STRATEGY_ADDRESS in strategies and ak:
            if s2_indexes.address_idx:
                result.s2_candidates.update(s2_indexes.address_idx.get(ak))
            if s3_indexes.address_idx:
                result.s3_candidates.update(s3_indexes.address_idx.get(ak))

        if STRATEGY_NAME_COUNTRY in strategies and nc_keys is not None:
            nck = nc_keys.iloc[i]
            if nck:
                if s2_indexes.name_country_idx:
                    result.s2_candidates.update(s2_indexes.name_country_idx.get(nck))
                if s3_indexes.name_country_idx:
                    result.s3_candidates.update(s3_indexes.name_country_idx.get(nck))

        if STRATEGY_ADDRESS_COUNTRY in strategies and ac_keys is not None:
            ack = ac_keys.iloc[i]
            if ack:
                if s2_indexes.address_country_idx:
                    result.s2_candidates.update(s2_indexes.address_country_idx.get(ack))
                if s3_indexes.address_country_idx:
                    result.s3_candidates.update(s3_indexes.address_country_idx.get(ack))

        # Token strategies
        if STRATEGY_NAME_TOKEN in strategies and name_tokens is not None:
            for t in name_tokens.iloc[i]:
                if s2_indexes.name_token_idx:
                    result.s2_candidates.update(s2_indexes.name_token_idx.get(t))
                if s3_indexes.name_token_idx:
                    result.s3_candidates.update(s3_indexes.name_token_idx.get(t))

        if STRATEGY_ADDRESS_TOKEN in strategies and addr_tokens is not None:
            for t in addr_tokens.iloc[i]:
                if s2_indexes.address_token_idx:
                    result.s2_candidates.update(s2_indexes.address_token_idx.get(t))
                if s3_indexes.address_token_idx:
                    result.s3_candidates.update(s3_indexes.address_token_idx.get(t))

        # N-gram strategies
        if STRATEGY_NAME_NGRAM in strategies and name_ngrams is not None:
            for ng in name_ngrams.iloc[i]:
                if s2_indexes.name_ngram_idx:
                    result.s2_candidates.update(s2_indexes.name_ngram_idx.get(ng))
                if s3_indexes.name_ngram_idx:
                    result.s3_candidates.update(s3_indexes.name_ngram_idx.get(ng))

        if STRATEGY_ADDRESS_NGRAM in strategies and addr_ngrams is not None:
            for ng in addr_ngrams.iloc[i]:
                if s2_indexes.address_ngram_idx:
                    result.s2_candidates.update(s2_indexes.address_ngram_idx.get(ng))
                if s3_indexes.address_ngram_idx:
                    result.s3_candidates.update(s3_indexes.address_ngram_idx.get(ng))

        results.append(result)
        counts.append(result.total_candidates())

    return results, counts


# --------------------------------------------------------------------------
# Extended benchmark
# --------------------------------------------------------------------------


def run_extended_blocking_benchmark(
    s1_path: str | Path,
    s2_path: str | Path,
    s3_path: str | Path,
    val_s1_ids: Set[str],
    gt_s1_matches: Dict[str, Tuple[List[str], List[str]]],
    chunksize: int = DEFAULT_CHUNKSIZE,
    strategies: Optional[List[str]] = None,
) -> BlockingBenchmarkResult:
    """Run blocking benchmark with specified strategies."""
    if strategies is None:
        strategies = ALL_STRATEGIES

    result = BlockingBenchmarkResult()
    t_start = time.time()

    # Step 1: Extract validation S1 blocking keys
    print(f"  Extracting keys for strategies: {strategies}")
    t_keys = time.time()
    key_sets = ExtendedKeySets.from_validation_s1(s1_path, val_s1_ids, chunksize, strategies)
    key_time = time.time() - t_keys
    print(f"  Key extraction took {key_time:.2f}s")

    # Step 2: Build S2 indexes
    t_idx_start = time.time()
    s2_indexes, s2_stats = _build_extended_indexes_for_source(
        s2_path, key_sets, chunksize, strategies
    )
    result.build_index_time_sec += time.time() - t_idx_start
    result.s2_stats = s2_stats

    # Step 3: Build S3 indexes
    t_idx_start = time.time()
    s3_indexes, s3_stats = _build_extended_indexes_for_source(
        s3_path, key_sets, chunksize, strategies
    )
    result.build_index_time_sec += time.time() - t_idx_start
    result.s3_stats = s3_stats

    # Step 4: Generate candidates
    t_gen_start = time.time()

    all_candidate_counts: List[int] = []
    total_true_matches = 0
    total_recallable_matches = 0
    total_s2_true = 0
    total_s2_recallable = 0
    total_s3_true = 0
    total_s3_recallable = 0

    print(f"  Generating candidates for {len(val_s1_ids):,} validation S1 entities...")
    s1_processed = 0
    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        usecols=[S1_COL, S1_NAME_COL, S1_ADDRESS_COL, S1_COUNTRY_COL],
        dtype={
            S1_COL: str,
            S1_NAME_COL: str,
            S1_ADDRESS_COL: str,
            S1_COUNTRY_COL: str,
        },
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""],
    ):
        chunk = chunk[chunk[S1_COL].isin(val_s1_ids)]
        if chunk.empty:
            continue

        batch_results, counts = generate_candidates_extended_batch(
            chunk, s2_indexes, s3_indexes, strategies
        )

        for i, cr in enumerate(batch_results):
            result.total_s1_evaluated += 1

            n_total = counts[i]
            all_candidate_counts.append(n_total)
            result.total_candidate_pairs += n_total
            result.total_s2_candidates += cr.s2_count()
            result.total_s3_candidates += cr.s3_count()

            if n_total == 0:
                result.s1_zero_candidates += 1
            elif n_total == 1:
                result.s1_one_candidate += 1
            else:
                result.s1_multi_candidates += 1

            true_s2, true_s3 = gt_s1_matches.get(cr.s1_id, ([], []))
            total_true_matches += len(true_s2) + len(true_s3)
            total_s2_true += len(true_s2)
            total_s3_true += len(true_s3)

            found_s2 = set(true_s2) & cr.s2_candidates
            found_s3 = set(true_s3) & cr.s3_candidates

            total_recallable_matches += len(found_s2) + len(found_s3)
            total_s2_recallable += len(found_s2)
            total_s3_recallable += len(found_s3)

        s1_processed += len(chunk)
        if s1_processed % 50000 == 0:
            print(f"    Processed {s1_processed:,} / {len(val_s1_ids):,} validation S1 entities...")

    if s1_processed % 50000 != 0:
        print(f"    Processed {s1_processed:,} / {len(val_s1_ids):,} validation S1 entities...")
    print(f"  Done: evaluated {result.total_s1_evaluated:,} S1 entities")

    result.generate_candidates_time_sec = time.time() - t_gen_start
    result.total_time_sec = time.time() - t_start

    # Compute aggregate statistics
    if result.total_s1_evaluated > 0:
        result.avg_candidates_per_s1 = result.total_candidate_pairs / result.total_s1_evaluated
        sorted_counts = sorted(all_candidate_counts)
        n = len(sorted_counts)
        result.median_candidates_per_s1 = sorted_counts[n // 2] if n else 0
        result.p95_candidates_per_s1 = sorted_counts[int(n * 0.95)] if n else 0
        result.max_candidates_per_s1 = max(sorted_counts) if sorted_counts else 0

    if total_true_matches > 0:
        result.true_match_recall = total_recallable_matches / total_true_matches
    if total_s2_true > 0:
        result.s2_recall = total_s2_recallable / total_s2_true
    if total_s3_true > 0:
        result.s3_recall = total_s3_recallable / total_s3_true

    return result


def print_benchmark_summary(result: BlockingBenchmarkResult, label: str = "") -> None:
    """Print a concise benchmark summary."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    print(f"  Recall:       {result.true_match_recall:.4f} (S2: {result.s2_recall:.4f}, S3: {result.s3_recall:.4f})")
    print(f"  Candidates:   {result.total_candidate_pairs:,} pairs, avg {result.avg_candidates_per_s1:.2f}/S1")
    print(f"  Distribution: median={result.median_candidates_per_s1}, p95={result.p95_candidates_per_s1}, max={result.max_candidates_per_s1}")
    print(f"  Zero/1/>1:    {result.s1_zero_candidates:,} / {result.s1_one_candidate:,} / {result.s1_multi_candidates:,}")
    print(f"  Runtime:      {result.total_time_sec:.1f}s (index: {result.build_index_time_sec:.1f}s, gen: {result.generate_candidates_time_sec:.1f}s)")
