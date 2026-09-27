#!/usr/bin/env python3
"""Run Task 3B refinement experiments: selective token blocking.

Tests frequency-based token filtering to reduce candidate volume while
maintaining recall improvement from token-based blocking strategies.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from business_entity_resolution.blocking import S1_COL, BlockingBenchmarkResult
from business_entity_resolution.blocking_strategies import (
    ALL_STRATEGIES,
    BASELINE_STRATEGIES,
    STRATEGY_ADDRESS_TOKEN,
    STRATEGY_NAME_TOKEN,
    ADDRESS_STOP_WORDS,
    NAME_STOP_WORDS,
    ExtendedKeySets,
    StrategyIndexes,
    _build_extended_indexes_for_source,
    generate_candidates_extended_batch,
    print_benchmark_summary,
    extract_name_tokens,
    extract_address_tokens,
    make_name_key_series,
    make_address_key_series,
)
from business_entity_resolution.validation import load_split, load_s1_matches

DEFAULT_CHUNKSIZE = 200_000


def compute_corpus_token_frequencies(
    s1_path, s2_path, s3_path, chunksize: int = DEFAULT_CHUNKSIZE
) -> Tuple[Counter, Counter]:
    """Compute document frequency of tokens across all source files."""
    name_token_freq = Counter()
    addr_token_freq = Counter()

    for source_path in [s1_path, s2_path, s3_path]:
        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            usecols=["business_name", "business_address"],
            dtype={"business_name": str, "business_address": str},
            chunksize=chunksize,
            keep_default_na=False,
            na_values=[""],
        ):
            norm_name = make_name_key_series(chunk["business_name"])
            norm_addr = make_address_key_series(chunk["business_address"])

            name_tokens = extract_name_tokens(norm_name)
            addr_tokens = extract_address_tokens(norm_addr)

            # Count unique tokens per entity (document frequency)
            for token_list in name_tokens:
                name_token_freq.update(set(token_list))
            for token_list in addr_tokens:
                addr_token_freq.update(set(token_list))

    return name_token_freq, addr_token_freq


def get_informative_tokens(
    token_freq: Counter, max_df: int, stop_words: frozenset, min_len: int = 2
) -> Set[str]:
    """Get informative tokens: not stop words, min length, below max document frequency."""
    return {
        token
        for token, freq in token_freq.items()
        if token not in stop_words and len(token) >= min_len and freq <= max_df
    }


def compute_metrics(
    batch_results,
    counts,
    gt_s1_matches: Dict[str, Tuple[List[str], List[str]]],
) -> BlockingBenchmarkResult:
    """Compute benchmark metrics from candidate results."""
    result = BlockingBenchmarkResult()

    all_counts = []
    total_true = 0
    total_recallable = 0
    total_s2_true = 0
    total_s2_recallable = 0
    total_s3_true = 0
    total_s3_recallable = 0

    for i, cr in enumerate(batch_results):
        result.total_s1_evaluated += 1
        n_total = counts[i]
        all_counts.append(n_total)
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
        total_true += len(true_s2) + len(true_s3)
        total_s2_true += len(true_s2)
        total_s3_true += len(true_s3)

        found_s2 = set(true_s2) & cr.s2_candidates
        found_s3 = set(true_s3) & cr.s3_candidates
        total_recallable += len(found_s2) + len(found_s3)
        total_s2_recallable += len(found_s2)
        total_s3_recallable += len(found_s3)

    if result.total_s1_evaluated > 0:
        result.avg_candidates_per_s1 = result.total_candidate_pairs / result.total_s1_evaluated
        sorted_counts = sorted(all_counts)
        n = len(sorted_counts)
        result.median_candidates_per_s1 = sorted_counts[n // 2] if n else 0
        result.p95_candidates_per_s1 = sorted_counts[int(n * 0.95)] if n else 0
        result.max_candidates_per_s1 = max(sorted_counts) if sorted_counts else 0

    if total_true > 0:
        result.true_match_recall = total_recallable / total_true
    if total_s2_true > 0:
        result.s2_recall = total_s2_recallable / total_s2_true
    if total_s3_true > 0:
        result.s3_recall = total_s3_recallable / total_s3_true

    return result


def run_experiment(
    s1_path,
    s2_indexes: StrategyIndexes,
    s3_indexes: StrategyIndexes,
    val_s1_ids,
    gt_s1_matches: Dict[str, Tuple[List[str], List[str]]],
    strategies: List[str],
    label: str,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> BlockingBenchmarkResult:
    """Run a single experiment with given strategies."""
    all_results = []
    all_counts = []

    print(f"\n  Experiment: {label}")
    t_gen = time.time()

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        usecols=[S1_COL, "business_name", "business_address", "country"],
        dtype={S1_COL: str, "business_name": str, "business_address": str, "country": str},
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
        all_results.extend(batch_results)
        all_counts.extend(counts)

    result = compute_metrics(all_results, all_counts, gt_s1_matches)
    result.generate_candidates_time_sec = time.time() - t_gen
    print_benchmark_summary(result, label)
    return result


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s1-path", type=Path, default=Path("dataset/train/train_source1.tsv"))
    p.add_argument("--s2-path", type=Path, default=Path("dataset/train/train_source2.tsv"))
    p.add_argument("--s3-path", type=Path, default=Path("dataset/train/train_source3.tsv"))
    p.add_argument("--ground-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    p.add_argument("--split", type=Path, default=Path("artifacts/validation/split_seed42.json"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/blocking"))
    p.add_argument("--chunksize", type=int, default=200_000)
    args = p.parse_args()

    for p in [args.s1_path, args.s2_path, args.s3_path, args.ground_truth, args.split]:
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 3B REFINEMENT: SELECTIVE TOKEN BLOCKING")
    print("=" * 70)

    # Load validation split
    print(f"\nLoading validation split...")
    all_s1_ids = set()
    for chunk in pd.read_csv(
        args.s1_path, sep="\t", usecols=[S1_COL], dtype={S1_COL: str},
        chunksize=args.chunksize, keep_default_na=False, na_values=[""],
    ):
        all_s1_ids.update(chunk[S1_COL].tolist())

    split = load_split(args.split, all_s1_ids=all_s1_ids)
    val_s1_ids = set(split.val_ids)
    print(f"  Validation S1: {len(val_s1_ids):,}")

    # Load ground truth
    print(f"Loading ground truth...")
    gt_matches = load_s1_matches(args.ground_truth, s1_ids=val_s1_ids, chunksize=args.chunksize)
    gt_s1_matches = {}
    for s1_id in val_s1_ids:
        gt_s1_matches[s1_id] = (gt_matches.s1_to_s2.get(s1_id, []), gt_matches.s1_to_s3.get(s1_id, []))

    # Compute corpus token frequencies
    print(f"\nComputing corpus token frequencies...")
    t0 = time.time()
    name_token_freq, addr_token_freq = compute_corpus_token_frequencies(
        args.s1_path, args.s2_path, args.s3_path, args.chunksize
    )
    t_freq = time.time() - t0
    print(f"  Name tokens: {len(name_token_freq):,}, Address tokens: {len(addr_token_freq):,} ({t_freq:.1f}s)")

    # Define frequency thresholds to test
    name_thresholds = [5, 10, 20, 50, 100]
    addr_thresholds = [5, 10, 20, 50, 100, 200]

    # Build informative token sets for each threshold
    name_token_sets = {}
    addr_token_sets = {}
    for thresh in name_thresholds:
        name_token_sets[thresh] = get_informative_tokens(name_token_freq, thresh, NAME_STOP_WORDS)
        print(f"  Name tokens (df<={thresh}): {len(name_token_sets[thresh]):,}")
    for thresh in addr_thresholds:
        addr_token_sets[thresh] = get_informative_tokens(addr_token_freq, thresh, ADDRESS_STOP_WORDS)
        print(f"  Addr tokens (df<={thresh}): {len(addr_token_sets[thresh]):,}")

    # Define refinement experiments
    experiments = []

    # Name token thresholds
    for thresh in name_thresholds:
        label = f"name_token_df{thresh}"
        experiments.append((label, [STRATEGY_NAME_TOKEN], {"name_token_keys": name_token_sets[thresh]}))

    # Address token thresholds
    for thresh in addr_thresholds:
        label = f"address_token_df{thresh}"
        experiments.append((label, [STRATEGY_ADDRESS_TOKEN], {"address_token_keys": addr_token_sets[thresh]}))

    # Combined with baseline: name token at different thresholds
    for thresh in name_thresholds:
        label = f"baseline + name_token_df{thresh}"
        experiments.append((label, BASELINE_STRATEGIES + [STRATEGY_NAME_TOKEN], {"name_token_keys": name_token_sets[thresh]}))

    # Combined with baseline: address token at different thresholds
    for thresh in addr_thresholds:
        label = f"baseline + address_token_df{thresh}"
        experiments.append((label, BASELINE_STRATEGIES + [STRATEGY_ADDRESS_TOKEN], {"address_token_keys": addr_token_sets[thresh]}))

    # Best combinations
    for name_thresh in [10, 20]:
        for addr_thresh in [20, 50]:
            label = f"baseline + name_token_df{name_thresh} + address_token_df{addr_thresh}"
            experiments.append((
                label,
                BASELINE_STRATEGIES + [STRATEGY_NAME_TOKEN, STRATEGY_ADDRESS_TOKEN],
                {
                    "name_token_keys": name_token_sets[name_thresh],
                    "address_token_keys": addr_token_sets[addr_thresh],
                }
            ))

    # Build indexes once with full token sets (we'll filter during candidate generation)
    print(f"\nBuilding indexes for ALL strategies...")
    t0 = time.time()
    key_sets = ExtendedKeySets.from_validation_s1(
        args.s1_path, val_s1_ids, args.chunksize, ALL_STRATEGIES
    )
    t_keys = time.time() - t0
    print(f"  Key extraction: {t_keys:.2f}s")

    t0 = time.time()
    s2_indexes, s2_stats = _build_extended_indexes_for_source(
        args.s2_path, key_sets, args.chunksize, ALL_STRATEGIES
    )
    s3_indexes, s3_stats = _build_extended_indexes_for_source(
        args.s3_path, key_sets, args.chunksize, ALL_STRATEGIES
    )
    t_idx = time.time() - t0
    print(f"  Index building: {t_idx:.2f}s")

    # Run experiments
    results = {}
    for label, strategies, token_filter in experiments:
        # Apply token filter by replacing index keys
        orig_name_token_idx = s2_indexes.name_token_idx
        orig_addr_token_idx = s2_indexes.address_token_idx
        orig_name_token_idx_s3 = s3_indexes.name_token_idx
        orig_addr_token_idx_s3 = s3_indexes.address_token_idx

        if "name_token_keys" in token_filter:
            allowed = token_filter["name_token_keys"]
            # Filter S2 index
            filtered_name_idx_s2 = {k: v for k, v in s2_indexes.name_token_idx.index.items() if k in allowed}
            s2_indexes.name_token_idx.index = filtered_name_idx_s2
            # Filter S3 index
            filtered_name_idx_s3 = {k: v for k, v in s3_indexes.name_token_idx.index.items() if k in allowed}
            s3_indexes.name_token_idx.index = filtered_name_idx_s3

        if "address_token_keys" in token_filter:
            allowed = token_filter["address_token_keys"]
            # Filter S2 index
            filtered_addr_idx_s2 = {k: v for k, v in s2_indexes.address_token_idx.index.items() if k in allowed}
            s2_indexes.address_token_idx.index = filtered_addr_idx_s2
            # Filter S3 index
            filtered_addr_idx_s3 = {k: v for k, v in s3_indexes.address_token_idx.index.items() if k in allowed}
            s3_indexes.address_token_idx.index = filtered_addr_idx_s3

        result = run_experiment(
            args.s1_path, s2_indexes, s3_indexes, val_s1_ids,
            gt_s1_matches, strategies, label, args.chunksize,
        )
        result.build_index_time_sec = t_keys + t_idx
        result.total_time_sec = result.build_index_time_sec + result.generate_candidates_time_sec
        results[label] = result.to_dict()

        # Restore original indexes for next experiment
        s2_indexes.name_token_idx.index = orig_name_token_idx.index
        s3_indexes.name_token_idx.index = orig_name_token_idx_s3.index
        s2_indexes.address_token_idx.index = orig_addr_token_idx.index
        s3_indexes.address_token_idx.index = orig_addr_token_idx_s3.index

    # Save results
    output_path = args.output_dir / "task3b_results.json"
    # Load existing results and merge
    if output_path.exists():
        existing = json.loads(output_path.read_text())
    else:
        existing = {}
    existing.update(results)
    output_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print(f"\nSaved results to: {output_path}")

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("REFINEMENT EXPERIMENT COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<55} {'Recall':>7} {'S2':>7} {'S3':>7} {'Pairs':>10} {'Avg/S1':>7} {'P95':>5}")
    print("-" * 100)
    for label, r in results.items():
        print(f"{label:<55} {r['true_match_recall']:>7.4f} {r['s2_recall']:>7.4f} {r['s3_recall']:>7.4f} {r['total_candidate_pairs']:>10,} {r['avg_candidates_per_s1']:>7.2f} {r['p95_candidates_per_s1']:>5.0f}")

    # Baseline comparison
    print(f"\n{'=' * 70}")
    print("BASELINE COMPARISON")
    print(f"{'=' * 70}")
    baseline = existing.get("baseline", {})
    if baseline:
        print(f"Baseline: {baseline['true_match_recall']:.4f} recall, {baseline['total_candidate_pairs']:,} pairs, {baseline['avg_candidates_per_s1']:.2f}/S1")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
