#!/usr/bin/env python3
"""Run baseline blocking benchmark on the validation split.

This script:
1. Loads the validation split (from Task 2)
2. Loads ground truth matches for validation S1 entities
3. Builds inverted indexes for S2 and S3 (four strategies each)
4. Generates candidates for each validation S1 entity
5. Measures recall and candidate statistics

Usage:
    python scripts/run_blocking_benchmark.py \\
        --s1-path dataset/train/train_source1.tsv \\
        --s2-path dataset/train/train_source2.tsv \\
        --s3-path dataset/train/train_source3.tsv \\
        --ground-truth dataset/train/train_ground_truth.tsv \\
        --split artifacts/validation/split_seed42.json \\
        --output-dir artifacts/blocking \\
        --chunksize 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from business_entity_resolution.blocking import run_blocking_benchmark, S1_COL
from business_entity_resolution.validation import load_split, load_s1_matches
from business_entity_resolution.validation.ground_truth import compute_s1_bucket_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s1-path", type=Path, default=Path("dataset/train/train_source1.tsv"))
    p.add_argument("--s2-path", type=Path, default=Path("dataset/train/train_source2.tsv"))
    p.add_argument("--s3-path", type=Path, default=Path("dataset/train/train_source3.tsv"))
    p.add_argument("--ground-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    p.add_argument("--split", type=Path, default=Path("artifacts/validation/split_seed42.json"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/blocking"))
    p.add_argument("--chunksize", type=int, default=200_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for p in [args.s1_path, args.s2_path, args.s3_path, args.ground_truth, args.split]:
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 3A: BASELINE BLOCKING BENCHMARK")
    print("=" * 70)

    # Load validation split - need all_s1_ids from source1 to reconstruct train_ids
    print(f"\nLoading validation split from {args.split}...")
    # First get all S1 IDs from source1
    all_s1_ids = set()
    for chunk in pd.read_csv(
        args.s1_path,
        sep="\t",
        usecols=[S1_COL],
        dtype={S1_COL: str},
        chunksize=args.chunksize,
        keep_default_na=False,
        na_values=[""],
    ):
        all_s1_ids.update(chunk[S1_COL].tolist())

    split = load_split(args.split, all_s1_ids=all_s1_ids)
    val_s1_ids = set(split.val_ids)
    print(f"Validation S1 entities: {len(val_s1_ids):,}")

    # Load ground truth for validation S1 entities
    print(f"\nLoading ground truth for validation split from {args.ground_truth}...")
    gt_matches = load_s1_matches(args.ground_truth, s1_ids=val_s1_ids, chunksize=args.chunksize)

    # Convert to dict format expected by blocking benchmark
    gt_s1_matches: dict = {}
    for s1_id in val_s1_ids:
        s2_ids = gt_matches.s1_to_s2.get(s1_id, [])
        s3_ids = gt_matches.s1_to_s3.get(s1_id, [])
        gt_s1_matches[s1_id] = (s2_ids, s3_ids)

    total_gt_s2 = sum(len(v[0]) for v in gt_s1_matches.values())
    total_gt_s3 = sum(len(v[1]) for v in gt_s1_matches.values())
    print(f"Ground truth: {total_gt_s2:,} S2 matches, {total_gt_s3:,} S3 matches")

    # Run benchmark
    print(f"\nRunning blocking benchmark (chunksize={args.chunksize:,})...")
    t0 = time.time()
    result = run_blocking_benchmark(
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        val_s1_ids=val_s1_ids,
        gt_s1_matches=gt_s1_matches,
        chunksize=args.chunksize,
    )
    elapsed = time.time() - t0
    print(f"Benchmark completed in {elapsed:.1f}s")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nRECALL METRICS:")
    print(f"  True-match recall (S2+S3): {result.true_match_recall:.4f} ({result.true_match_recall*100:.2f}%)")
    print(f"  S2 recall:                 {result.s2_recall:.4f} ({result.s2_recall*100:.2f}%)")
    print(f"  S3 recall:                 {result.s3_recall:.4f} ({result.s3_recall*100:.2f}%)")

    print(f"\nCANDIDATE STATISTICS:")
    print(f"  Total S1 evaluated:        {result.total_s1_evaluated:,}")
    print(f"  Total candidate pairs:     {result.total_candidate_pairs:,}")
    print(f"  Avg candidates per S1:     {result.avg_candidates_per_s1:.2f}")
    print(f"  Median candidates per S1:  {result.median_candidates_per_s1:.2f}")
    print(f"  P95 candidates per S1:     {result.p95_candidates_per_s1:.2f}")
    print(f"  Max candidates per S1:     {result.max_candidates_per_s1:,}")
    print(f"  Total S2 candidates:       {result.total_s2_candidates:,}")
    print(f"  Total S3 candidates:       {result.total_s3_candidates:,}")

    print(f"\nCANDIDATE DISTRIBUTION (per S1):")
    print(f"  Zero candidates:           {result.s1_zero_candidates:,} ({result.s1_zero_candidates/result.total_s1_evaluated*100:.2f}%)")
    print(f"  One candidate:             {result.s1_one_candidate:,} ({result.s1_one_candidate/result.total_s1_evaluated*100:.2f}%)")
    print(f"  >1 candidates:             {result.s1_multi_candidates:,} ({result.s1_multi_candidates/result.total_s1_evaluated*100:.2f}%)")

    print(f"\nRUNTIME:")
    print(f"  Build indexes:    {result.build_index_time_sec:.2f}s")
    print(f"  Generate cands:   {result.generate_candidates_time_sec:.2f}s")
    print(f"  Total:            {result.total_time_sec:.2f}s")

    if result.s2_stats:
        print(f"\nS2 INDEX STATS:")
        print(f"  Entities indexed:  {result.s2_stats.total_entities:,}")
        print(f"  Name index:        {result.s2_stats.name_index_size:,} keys, {result.s2_stats.name_index_entries:,} entries")
        print(f"  Address index:     {result.s2_stats.address_index_size:,} keys, {result.s2_stats.address_index_entries:,} entries")
        print(f"  Name+Country:      {result.s2_stats.name_country_index_size:,} keys, {result.s2_stats.name_country_index_entries:,} entries")
        print(f"  Address+Country:   {result.s2_stats.address_country_index_size:,} keys, {result.s2_stats.address_country_index_entries:,} entries")

    if result.s3_stats:
        print(f"\nS3 INDEX STATS:")
        print(f"  Entities indexed:  {result.s3_stats.total_entities:,}")
        print(f"  Name index:        {result.s3_stats.name_index_size:,} keys, {result.s3_stats.name_index_entries:,} entries")
        print(f"  Address index:     {result.s3_stats.address_index_size:,} keys, {result.s3_stats.address_index_entries:,} entries")
        print(f"  Name+Country:      {result.s3_stats.name_country_index_size:,} keys, {result.s3_stats.name_country_index_entries:,} entries")
        print(f"  Address+Country:   {result.s3_stats.address_country_index_size:,} keys, {result.s3_stats.address_country_index_entries:,} entries")

    # Save results
    output_path = args.output_dir / "blocking_benchmark.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(f"\nSaved results to: {output_path}")

    # Also save per-S1 candidate counts for analysis
    # (not implemented here to keep memory bounded - only aggregates saved)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()