"""Leakage-safe validation infrastructure (Task 2).

This subpackage is intentionally self-contained and has no dependency
on any future blocking/candidate-generation/matching code. It provides:

- ``ground_truth``: streaming parsing of ``train_ground_truth.tsv`` and
  cross-link (leakage) detection between S1 entities that share a
  matched S2/S3 record.
- ``split``: a reproducible, leakage-aware train/validation split at
  the S1-entity level (grouped by connected component when
  cross-links exist).
- ``metrics``: a reusable macro-F0.5-per-S1-entity evaluator with
  explicit, documented edge-case handling.

Nothing in this subpackage reads S2/S3 business_name/business_address
content -- it only ever touches entity IDs, so it is cheap to run
repeatedly and safe to import from later pipeline stages.
"""

from .ground_truth import (
    CrossLinkStats,
    GroundTruthStats,
    S1Matches,
    compute_cross_link_statistics,
    compute_ground_truth_statistics,
    compute_s1_bucket_map,
    iter_ground_truth_chunks,
    load_s1_matches,
    parse_matched_ids,
    verify_internal_consistency,
)
from .metrics import (
    EntityScore,
    MacroF05Result,
    evaluate_entity,
    evaluate_predictions,
    f_beta,
)
from .split import (
    SplitResult,
    build_s1_components,
    create_validation_split,
    load_split,
    save_split,
)

__all__ = [
    # ground_truth
    "CrossLinkStats",
    "GroundTruthStats",
    "S1Matches",
    "compute_cross_link_statistics",
    "compute_ground_truth_statistics",
    "compute_s1_bucket_map",
    "iter_ground_truth_chunks",
    "load_s1_matches",
    "parse_matched_ids",
    "verify_internal_consistency",
    # metrics
    "EntityScore",
    "MacroF05Result",
    "evaluate_entity",
    "evaluate_predictions",
    "f_beta",
    # split
    "SplitResult",
    "build_s1_components",
    "create_validation_split",
    "load_split",
    "save_split",
]
