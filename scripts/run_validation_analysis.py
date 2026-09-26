#!/usr/bin/env python3
"""Run the complete Task 2 validation analysis against the real local
dataset and write compact artifacts under ``artifacts/validation/``.

This script does NOT implement blocking, candidate generation, or
matching -- it only parses ground truth, detects cross-links, builds
the leakage-safe split, and sanity-checks the F0.5 evaluator against a
handful of trivial in-memory cases (not against real predictions, since
no matcher exists yet).

Usage
-----
    python scripts/run_validation_analysis.py \\
        --ground-truth dataset/train/train_ground_truth.tsv \\
        --output-dir artifacts/validation \\
        --chunksize 200000 \\
        --val-fraction 0.2 \\
        --seed 42

Run from the repository root (so the relative default paths resolve),
or pass absolute paths explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Allow running this script directly from `scripts/` without installing
# the package first (`pip install -e .` also works and is preferred).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from business_entity_resolution.validation import (  # noqa: E402
    compute_cross_link_statistics,
    compute_ground_truth_statistics,
    compute_s1_bucket_map,
    create_validation_split,
    evaluate_predictions,
    save_split,
)
from business_entity_resolution.validation.ground_truth import (  # noqa: E402
    verify_internal_consistency,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("dataset/train/train_ground_truth.tsv"),
        help="Path to train_ground_truth.tsv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/validation"),
        help="Directory to write JSON artifacts into",
    )
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--example-limit",
        type=int,
        default=5,
        help="How many concrete cross-link examples to keep for docs/reporting",
    )
    return p.parse_args()


def _print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    args = parse_args()
    if not args.ground_truth.exists():
        raise SystemExit(
            f"Ground truth file not found: {args.ground_truth}\n"
            f"Run this script from the repository root, or pass "
            f"--ground-truth explicitly."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Section 1: ground-truth match statistics -----------------------
    _print_header("SECTION 1: Ground-truth match statistics")
    s2_link_counts: Counter = Counter()
    s3_link_counts: Counter = Counter()
    gt_stats = compute_ground_truth_statistics(
        args.ground_truth,
        chunksize=args.chunksize,
        s2_link_counts=s2_link_counts,
        s3_link_counts=s3_link_counts,
    )
    problems = verify_internal_consistency(gt_stats)
    gt_stats_path = args.output_dir / "ground_truth_stats.json"
    gt_stats.save_json(gt_stats_path)

    print(f"total_s1_entities        = {gt_stats.total_s1_entities:,}")
    print(f"total_s2_links           = {gt_stats.total_s2_links:,}")
    print(f"total_s3_links           = {gt_stats.total_s3_links:,}")
    print(f"total_links              = {gt_stats.total_links:,}")
    print(f"mean_total_matches       = {gt_stats.mean_total_matches:.4f}")
    print(f"max_s2_matches           = {gt_stats.max_s2_matches}")
    print(f"max_s3_matches           = {gt_stats.max_s3_matches}")
    print(f"max_total_matches        = {gt_stats.max_total_matches}")
    print(f"only_s2_count            = {gt_stats.only_s2_count:,}")
    print(f"only_s3_count            = {gt_stats.only_s3_count:,}")
    print(f"both_s2_and_s3_count     = {gt_stats.both_s2_and_s3_count:,}")
    print(f"neither_count            = {gt_stats.neither_count:,}")
    print(f"total_bucket_counts      = {gt_stats.total_bucket_counts}")
    if problems:
        print("\nINTERNAL CONSISTENCY CHECK: FAILED")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nINTERNAL CONSISTENCY CHECK: passed")
    print(f"\nWrote: {gt_stats_path}")

    # ---- Section 2: cross-link (leakage) statistics ----------------------
    _print_header("SECTION 2: Cross-link (leakage) statistics")
    cross_stats, full_mapping = compute_cross_link_statistics(
        args.ground_truth,
        chunksize=args.chunksize,
        example_limit=args.example_limit,
        s2_link_counts=s2_link_counts,
        s3_link_counts=s3_link_counts,
    )
    cross_stats_path = args.output_dir / "cross_link_stats.json"
    cross_stats.save_json(cross_stats_path)

    print(f"unique_s2_ids                    = {cross_stats.unique_s2_ids:,}")
    print(
        f"s2_ids_linked_to_multiple_s1      = "
        f"{cross_stats.s2_ids_linked_to_multiple_s1:,}"
    )
    print(f"max_s1_per_s2                     = {cross_stats.max_s1_per_s2}")
    print(f"unique_s3_ids                     = {cross_stats.unique_s3_ids:,}")
    print(
        f"s3_ids_linked_to_multiple_s1      = "
        f"{cross_stats.s3_ids_linked_to_multiple_s1:,}"
    )
    print(f"max_s1_per_s3                     = {cross_stats.max_s1_per_s3}")
    print(f"\nExample S2 cross-links (up to {args.example_limit}):")
    print(json.dumps(cross_stats.example_s2_cross_links, indent=2))
    print(f"\nExample S3 cross-links (up to {args.example_limit}):")
    print(json.dumps(cross_stats.example_s3_cross_links, indent=2))
    print(f"\nWrote: {cross_stats_path}")

    # ---- Section 3: leakage-safe validation split -------------------------
    _print_header("SECTION 3: Leakage-safe validation split")
    s1_bucket = compute_s1_bucket_map(args.ground_truth, chunksize=args.chunksize)
    split = create_validation_split(
        s1_bucket,
        cross_link_mapping=full_mapping,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    split_path = save_split(split, args.output_dir)

    print(f"seed                       = {split.seed}")
    print(f"val_fraction (target)      = {split.val_fraction}")
    print(f"n_train                    = {len(split.train_ids):,}")
    print(f"n_val                      = {len(split.val_ids):,}")
    realized_frac = len(split.val_ids) / (len(split.train_ids) + len(split.val_ids))
    print(f"val_fraction (realized)    = {realized_frac:.4f}")
    print(f"n_components               = {split.n_components:,}")
    print(f"n_multi_member_components  = {split.n_multi_member_components:,}")
    print(f"largest_component_size     = {split.largest_component_size}")
    print(f"train_bucket_counts        = {split.train_bucket_counts}")
    print(f"val_bucket_counts          = {split.val_bucket_counts}")
    print(f"\nWrote: {split_path}")

    # ---- Section 4: F0.5 evaluator smoke test -----------------------------
    _print_header("SECTION 4: F0.5 evaluator smoke test (trivial in-memory case)")
    toy_gt = {"S1-1": {"S2-1", "S2-2"}, "S1-2": set(), "S1-3": {"S3-9"}}
    toy_pred = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-9", "S3-8"}}
    result = evaluate_predictions(toy_gt, toy_pred)
    print("(This is a hard-coded 3-entity toy example, NOT real predictions --")
    print(" no matcher exists yet. It only demonstrates the evaluator runs.)")
    print(json.dumps(result.to_dict(include_per_entity=True), indent=2))

    elapsed = time.time() - t0
    _print_header("DONE")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
