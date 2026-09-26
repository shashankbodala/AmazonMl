"""Optimized baseline blocking using vectorized operations and key filtering.

This module implements four exact-match blocking strategies using
memory-efficient approaches:

1. exact normalized business name
2. exact normalized business address
3. normalized name + country
4. normalized address + country

Key optimization: Only builds indexes for keys that appear in the
validation S1 entities, dramatically reducing memory usage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .normalization import (
    make_address_country_key,
    make_address_country_key_series,
    make_address_key,
    make_address_key_series,
    make_country_key_series,
    make_name_country_key,
    make_name_country_key_series,
    make_name_key,
    make_name_key_series,
)

DEFAULT_CHUNKSIZE = 200_000

S1_COL = "entity_id"
S1_NAME_COL = "business_name"
S1_ADDRESS_COL = "business_address"
S1_COUNTRY_COL = "country"

S23_COLS = ["entity_id", "business_name", "business_address", "country"]


@dataclass
class BlockingKeySets:
    """Sets of blocking keys extracted from validation S1 entities."""
    name_keys: Set[str]
    address_keys: Set[str]
    name_country_keys: Set[str]
    address_country_keys: Set[str]

    @classmethod
    def from_validation_s1(cls, s1_path: str | Path, val_s1_ids: Set[str], chunksize: int = DEFAULT_CHUNKSIZE) -> "BlockingKeySets":
        """Extract all blocking keys from validation S1 entities."""
        name_keys: Set[str] = set()
        address_keys: Set[str] = set()
        name_country_keys: Set[str] = set()
        address_country_keys: Set[str] = set()

        for chunk in pd.read_csv(
            s1_path,
            sep="\t",
            usecols=[S1_COL, S1_NAME_COL, S1_ADDRESS_COL, S1_COUNTRY_COL],
            dtype={S1_COL: str, S1_NAME_COL: str, S1_ADDRESS_COL: str, S1_COUNTRY_COL: str},
            chunksize=chunksize,
            keep_default_na=False,
            na_values=[""],
        ):
            chunk = chunk[chunk[S1_COL].isin(val_s1_ids)]
            if chunk.empty:
                continue

            # Vectorized normalization
            name_keys.update(make_name_key_series(chunk[S1_NAME_COL]).dropna().unique())
            address_keys.update(make_address_key_series(chunk[S1_ADDRESS_COL]).dropna().unique())
            name_country_keys.update(make_name_country_key_series(chunk[S1_NAME_COL], chunk[S1_COUNTRY_COL]).dropna().unique())
            address_country_keys.update(make_address_country_key_series(chunk[S1_ADDRESS_COL], chunk[S1_COUNTRY_COL]).dropna().unique())

        return cls(
            name_keys=name_keys,
            address_keys=address_keys,
            name_country_keys=name_country_keys,
            address_country_keys=address_country_keys,
        )


@dataclass
class CompactInvertedIndex:
    """Memory-efficient inverted index: key -> list of entity_ids (stored as array)."""
    index: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, key: str, entity_id: str) -> None:
        if key:
            self.index.setdefault(key, []).append(entity_id)

    def get(self, key: str) -> List[str]:
        return self.index.get(key, [])

    def __len__(self) -> int:
        return len(self.index)

    def total_entries(self) -> int:
        return sum(len(v) for v in self.index.values())


@dataclass
class BlockingStats:
    """Per-source blocking statistics for reporting."""
    source: str
    total_entities: int = 0
    name_index_size: int = 0
    name_index_entries: int = 0
    address_index_size: int = 0
    address_index_entries: int = 0
    name_country_index_size: int = 0
    name_country_index_entries: int = 0
    address_country_index_size: int = 0
    address_country_index_entries: int = 0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "total_entities": self.total_entities,
            "name_index": {"unique_keys": self.name_index_size, "total_entries": self.name_index_entries},
            "address_index": {"unique_keys": self.address_index_size, "total_entries": self.address_index_entries},
            "name_country_index": {"unique_keys": self.name_country_index_size, "total_entries": self.name_country_index_entries},
            "address_country_index": {"unique_keys": self.address_country_index_size, "total_entries": self.address_country_index_entries},
        }


@dataclass
class CandidateResult:
    """Result of candidate generation for one S1 entity."""
    s1_id: str
    s2_candidates: Set[str] = field(default_factory=set)
    s3_candidates: Set[str] = field(default_factory=set)

    def total_candidates(self) -> int:
        return len(self.s2_candidates) + len(self.s3_candidates)

    def s2_count(self) -> int:
        return len(self.s2_candidates)

    def s3_count(self) -> int:
        return len(self.s3_candidates)


def build_filtered_inverted_indexes(
    source_path: str | Path,
    key_sets: BlockingKeySets,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Tuple[CompactInvertedIndex, CompactInvertedIndex, CompactInvertedIndex, CompactInvertedIndex, BlockingStats]:
    """Build inverted indexes only for keys that appear in validation S1 entities.

    This is the key memory optimization: we only index S2/S3 entities
    whose blocking keys actually appear in the validation set.
    """
    name_idx = CompactInvertedIndex()
    address_idx = CompactInvertedIndex()
    name_country_idx = CompactInvertedIndex()
    address_country_idx = CompactInvertedIndex()

    stats = BlockingStats(source=Path(source_path).stem.upper())

    print(f"  Building filtered indexes for {source_path}...")
    chunk_num = 0
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

        # Vectorized normalization for the entire chunk
        name_keys = make_name_key_series(chunk[S1_NAME_COL])
        address_keys = make_address_key_series(chunk[S1_ADDRESS_COL])
        name_country_keys = make_name_country_key_series(chunk[S1_NAME_COL], chunk[S1_COUNTRY_COL])
        address_country_keys = make_address_country_key_series(chunk[S1_ADDRESS_COL], chunk[S1_COUNTRY_COL])

        # Only keep rows where at least one key matches our validation key sets
        mask = (
            name_keys.isin(key_sets.name_keys) |
            address_keys.isin(key_sets.address_keys) |
            name_country_keys.isin(key_sets.name_country_keys) |
            address_country_keys.isin(key_sets.address_country_keys)
        )

        if not mask.any():
            continue

        # Filter to matching rows
        matched_chunk = chunk[mask]
        matched_name_keys = name_keys[mask]
        matched_address_keys = address_keys[mask]
        matched_name_country_keys = name_country_keys[mask]
        matched_address_country_keys = address_country_keys[mask]

        # Build indexes (vectorized add would require iteration anyway)
        for eid, nk, ak, nck, ack in zip(
            matched_chunk[S1_COL],
            matched_name_keys,
            matched_address_keys,
            matched_name_country_keys,
            matched_address_country_keys,
        ):
            if nk in key_sets.name_keys:
                name_idx.add(nk, eid)
            if ak in key_sets.address_keys:
                address_idx.add(ak, eid)
            if nck in key_sets.name_country_keys:
                name_country_idx.add(nck, eid)
            if ack in key_sets.address_country_keys:
                address_country_idx.add(ack, eid)

    # Finalize stats
    stats.name_index_size = len(name_idx)
    stats.name_index_entries = name_idx.total_entries()
    stats.address_index_size = len(address_idx)
    stats.address_index_entries = address_idx.total_entries()
    stats.name_country_index_size = len(name_country_idx)
    stats.name_country_index_entries = name_country_idx.total_entries()
    stats.address_country_index_size = len(address_country_idx)
    stats.address_country_index_entries = address_country_idx.total_entries()

    print(f"  Done: {stats.total_entities:,} entities scanned, "
          f"{len(name_idx):,} name keys, {len(address_idx):,} address keys matched")
    return name_idx, address_idx, name_country_idx, address_country_idx, stats


def generate_candidates_for_s1(
    s1_id: str,
    s1_name: str,
    s1_address: str,
    s1_country: str,
    s2_name_idx: CompactInvertedIndex,
    s2_address_idx: CompactInvertedIndex,
    s2_name_country_idx: CompactInvertedIndex,
    s2_address_country_idx: CompactInvertedIndex,
    s3_name_idx: CompactInvertedIndex,
    s3_address_idx: CompactInvertedIndex,
    s3_name_country_idx: CompactInvertedIndex,
    s3_address_country_idx: CompactInvertedIndex,
) -> CandidateResult:
    """Generate candidate S2/S3 entity_ids for a single S1 entity."""
    result = CandidateResult(s1_id=s1_id)

    # S2 candidates
    s2_name_key = make_name_key(s1_name)
    result.s2_candidates.update(s2_name_idx.get(s2_name_key))

    s2_addr_key = make_address_key(s1_address)
    result.s2_candidates.update(s2_address_idx.get(s2_addr_key))

    s2_nc_key = make_name_country_key(s1_name, s1_country)
    result.s2_candidates.update(s2_name_country_idx.get(s2_nc_key))

    s2_ac_key = make_address_country_key(s1_address, s1_country)
    result.s2_candidates.update(s2_address_country_idx.get(s2_ac_key))

    # S3 candidates
    s3_name_key = make_name_key(s1_name)
    result.s3_candidates.update(s3_name_idx.get(s3_name_key))

    s3_addr_key = make_address_key(s1_address)
    result.s3_candidates.update(s3_address_idx.get(s3_addr_key))

    s3_nc_key = make_name_country_key(s1_name, s1_country)
    result.s3_candidates.update(s3_name_country_idx.get(s3_nc_key))

    s3_ac_key = make_address_country_key(s1_address, s1_country)
    result.s3_candidates.update(s3_address_country_idx.get(s3_ac_key))

    return result


def generate_candidates_batch_vectorized(
    s1_chunk: pd.DataFrame,
    s2_name_idx: CompactInvertedIndex,
    s2_address_idx: CompactInvertedIndex,
    s2_name_country_idx: CompactInvertedIndex,
    s2_address_country_idx: CompactInvertedIndex,
    s3_name_idx: CompactInvertedIndex,
    s3_address_idx: CompactInvertedIndex,
    s3_name_country_idx: CompactInvertedIndex,
    s3_address_country_idx: CompactInvertedIndex,
) -> Tuple[List[CandidateResult], List[int]]:
    """Generate candidates for a batch of S1 entities using vectorized operations.

    Returns (candidate_results, candidate_counts) for the chunk.
    """
    # Vectorized normalization for the entire chunk
    name_keys = make_name_key_series(s1_chunk[S1_NAME_COL])
    address_keys = make_address_key_series(s1_chunk[S1_ADDRESS_COL])
    name_country_keys = make_name_country_key_series(s1_chunk[S1_NAME_COL], s1_chunk[S1_COUNTRY_COL])
    address_country_keys = make_address_country_key_series(s1_chunk[S1_ADDRESS_COL], s1_chunk[S1_COUNTRY_COL])

    s1_ids = s1_chunk[S1_COL].tolist()
    results: List[CandidateResult] = []
    counts: List[int] = []

    # Process each row using the pre-computed vectorized keys
    for i, s1_id in enumerate(s1_ids):
        result = CandidateResult(s1_id=s1_id)

        # S2 candidates - lookup pre-computed keys
        nk = name_keys.iloc[i]
        if nk:
            result.s2_candidates.update(s2_name_idx.get(nk))

        ak = address_keys.iloc[i]
        if ak:
            result.s2_candidates.update(s2_address_idx.get(ak))

        nck = name_country_keys.iloc[i]
        if nck:
            result.s2_candidates.update(s2_name_country_idx.get(nck))

        ack = address_country_keys.iloc[i]
        if ack:
            result.s2_candidates.update(s2_address_country_idx.get(ack))

        # S3 candidates
        if nk:
            result.s3_candidates.update(s3_name_idx.get(nk))
        if ak:
            result.s3_candidates.update(s3_address_idx.get(ak))
        if nck:
            result.s3_candidates.update(s3_name_country_idx.get(nck))
        if ack:
            result.s3_candidates.update(s3_address_country_idx.get(ack))

        results.append(result)
        counts.append(result.total_candidates())

    return results, counts


def generate_candidates_batch(
    s1_chunk: pd.DataFrame,
    s2_indexes: Tuple[CompactInvertedIndex, CompactInvertedIndex, CompactInvertedIndex, CompactInvertedIndex],
    s3_indexes: Tuple[CompactInvertedIndex, CompactInvertedIndex, CompactInvertedIndex, CompactInvertedIndex],
) -> List[CandidateResult]:
    """Generate candidates for a batch of S1 entities (optimized vectorized version)."""
    (s2_name_idx, s2_address_idx, s2_name_country_idx, s2_address_country_idx) = s2_indexes
    (s3_name_idx, s3_address_idx, s3_name_country_idx, s3_address_country_idx) = s3_indexes

    results, _ = generate_candidates_batch_vectorized(
        s1_chunk,
        s2_name_idx, s2_address_idx, s2_name_country_idx, s2_address_country_idx,
        s3_name_idx, s3_address_idx, s3_name_country_idx, s3_address_country_idx,
    )
    return results


@dataclass
class BlockingBenchmarkResult:
    """Complete benchmark results for the validation split."""
    # Recall metrics
    true_match_recall: float = 0.0
    s2_recall: float = 0.0
    s3_recall: float = 0.0

    # Candidate count statistics
    total_s1_evaluated: int = 0
    total_candidate_pairs: int = 0
    avg_candidates_per_s1: float = 0.0
    median_candidates_per_s1: float = 0.0
    p95_candidates_per_s1: float = 0.0
    max_candidates_per_s1: int = 0

    # Per-source totals
    total_s2_candidates: int = 0
    total_s3_candidates: int = 0

    # Zero / one / >1 candidate distribution
    s1_zero_candidates: int = 0
    s1_one_candidate: int = 0
    s1_multi_candidates: int = 0

    # Runtime
    build_index_time_sec: float = 0.0
    generate_candidates_time_sec: float = 0.0
    total_time_sec: float = 0.0

    # Index stats
    s2_stats: Optional[BlockingStats] = None
    s3_stats: Optional[BlockingStats] = None

    def to_dict(self) -> dict:
        return {
            "true_match_recall": self.true_match_recall,
            "s2_recall": self.s2_recall,
            "s3_recall": self.s3_recall,
            "total_s1_evaluated": self.total_s1_evaluated,
            "total_candidate_pairs": self.total_candidate_pairs,
            "avg_candidates_per_s1": self.avg_candidates_per_s1,
            "median_candidates_per_s1": self.median_candidates_per_s1,
            "p95_candidates_per_s1": self.p95_candidates_per_s1,
            "max_candidates_per_s1": self.max_candidates_per_s1,
            "total_s2_candidates": self.total_s2_candidates,
            "total_s3_candidates": self.total_s3_candidates,
            "s1_zero_candidates": self.s1_zero_candidates,
            "s1_one_candidate": self.s1_one_candidate,
            "s1_multi_candidates": self.s1_multi_candidates,
            "build_index_time_sec": self.build_index_time_sec,
            "generate_candidates_time_sec": self.generate_candidates_time_sec,
            "total_time_sec": self.total_time_sec,
            "s2_index_stats": self.s2_stats.to_dict() if self.s2_stats else None,
            "s3_index_stats": self.s3_stats.to_dict() if self.s3_stats else None,
        }


def run_blocking_benchmark(
    s1_path: str | Path,
    s2_path: str | Path,
    s3_path: str | Path,
    val_s1_ids: Set[str],
    gt_s1_matches: Dict[str, Tuple[List[str], List[str]]],
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> BlockingBenchmarkResult:
    """Run the optimized baseline blocking benchmark on the validation split."""
    result = BlockingBenchmarkResult()
    t_start = time.time()

    # ---- Step 1: Extract validation S1 blocking keys ----
    print(f"  Extracting blocking keys from {len(val_s1_ids):,} validation S1 entities...")
    t_keys = time.time()
    key_sets = BlockingKeySets.from_validation_s1(s1_path, val_s1_ids, chunksize)
    print(f"  Key extraction took {time.time() - t_keys:.2f}s")
    print(f"  Unique keys: name={len(key_sets.name_keys):,}, address={len(key_sets.address_keys):,}, "
          f"name+country={len(key_sets.name_country_keys):,}, address+country={len(key_sets.address_country_keys):,}")

    # ---- Step 2: Build filtered S2 indexes ----
    t_idx_start = time.time()
    s2_name_idx, s2_addr_idx, s2_nc_idx, s2_ac_idx, s2_stats = build_filtered_inverted_indexes(s2_path, key_sets, chunksize)
    result.build_index_time_sec += time.time() - t_idx_start
    result.s2_stats = s2_stats

    # ---- Step 3: Build filtered S3 indexes ----
    t_idx_start = time.time()
    s3_name_idx, s3_addr_idx, s3_nc_idx, s3_ac_idx, s3_stats = build_filtered_inverted_indexes(s3_path, key_sets, chunksize)
    result.build_index_time_sec += time.time() - t_idx_start
    result.s3_stats = s3_stats

    # ---- Step 4: Generate candidates for validation S1 entities ----
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
        dtype={S1_COL: str, S1_NAME_COL: str, S1_ADDRESS_COL: str, S1_COUNTRY_COL: str},
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""],
    ):
        chunk = chunk[chunk[S1_COL].isin(val_s1_ids)]
        if chunk.empty:
            continue

        # Use vectorized batch function directly - returns (candidate_results, candidate_counts)
        batch_results, counts = generate_candidates_batch_vectorized(
            chunk,
            s2_name_idx, s2_addr_idx, s2_nc_idx, s2_ac_idx,
            s3_name_idx, s3_addr_idx, s3_nc_idx, s3_ac_idx,
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
        if s1_processed % 25000 == 0:
            print(f"    Processed {s1_processed:,} validation S1 entities...")

    print(f"  Done: evaluated {result.total_s1_evaluated:,} S1 entities")
    result.generate_candidates_time_sec = time.time() - t_gen_start
    result.total_time_sec = time.time() - t_start

    # ---- Compute aggregate statistics ----
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