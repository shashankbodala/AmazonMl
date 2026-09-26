"""Streaming parser for ``train_ground_truth.tsv`` and cross-link (leakage)
detection between S1 entities.

Design notes
------------
The ground-truth file has two columns::

    source1_entity_id    matched_entity_ids

where ``matched_entity_ids`` is a single comma-separated string mixing
``S2-`` and ``S3-`` prefixed IDs, e.g.::

    S2-681193310,S2-743505751,S3-775321672

This module NEVER loads the whole file into a single DataFrame. It reads
it with ``pandas.read_csv(..., chunksize=...)`` and updates running
aggregates chunk by chunk. Only small, bounded objects are kept in
memory across the whole file:

- integer counters (total counts, histograms keyed by small integers)
- one ``collections.Counter`` per source (S2, S3) mapping
  ``entity_id -> number of distinct S1 entities that matched it``.
  This is unavoidable to answer "does this S2/S3 id appear for more
  than one S1" exactly -- there is no way to know without counting
  every occurrence. The counters hold at most one entry per distinct
  S2/S3 id that appears in the ground truth (bounded by the number of
  matched links, not the full 5M-row source tables), and store a
  single int per key, so memory stays roughly linear in the number of
  *matched* ids, not in the size of the raw source tables.

Building the *full* list of S1 ids for every cross-linked S2/S3 id
(needed by ``split.py`` to keep leakage-connected S1 entities on the
same side of the split) is done as an optional, targeted **second
pass** (``build_full_cross_link_mapping``) that only tracks ids already
known (from pass 1) to be linked to more than one S1 entity. This keeps
the second pass's memory bounded by the number of *cross-linked* ids,
which is expected to be a small fraction of the ~7.6M total links.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

DEFAULT_CHUNKSIZE = 200_000

S1_COL = "source1_entity_id"
MATCH_COL = "matched_entity_ids"


# --------------------------------------------------------------------------
# Low-level parsing
# --------------------------------------------------------------------------


def parse_matched_ids(raw: object) -> Tuple[List[str], List[str]]:
    """Split one ``matched_entity_ids`` cell into (s2_ids, s3_ids).

    Handles:
    - NaN / None / empty string -> ([], [])
    - a single id with no commas
    - ids with unexpected/unknown prefixes -> raises ValueError, since a
      silent drop would corrupt match counts (fail loudly rather than
      under-count).

    Whitespace around ids is stripped; empty tokens produced by stray
    commas (e.g. a trailing comma) are ignored.
    """
    if raw is None:
        return [], []
    if isinstance(raw, float) and pd.isna(raw):
        return [], []
    text = str(raw).strip()
    if text == "" or text.lower() == "nan":
        return [], []

    s2_ids: List[str] = []
    s3_ids: List[str] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("S2-"):
            s2_ids.append(token)
        elif token.startswith("S3-"):
            s3_ids.append(token)
        else:
            raise ValueError(
                f"Unrecognized matched-id prefix in ground truth: {token!r} "
                f"(expected an 'S2-' or 'S3-' prefix)"
            )
    return s2_ids, s3_ids


def iter_ground_truth_chunks(
    path: str | Path, chunksize: int = DEFAULT_CHUNKSIZE
) -> Iterator[pd.DataFrame]:
    """Yield ``train_ground_truth.tsv`` in chunks, never loading it whole.

    Only the two expected columns are read (``usecols``), and both are
    forced to ``str`` dtype so that ID-like strings are never coerced to
    numbers/NaN by pandas' type inference.
    """
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=[S1_COL, MATCH_COL],
        dtype={S1_COL: str, MATCH_COL: str},
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""],
    )
    for chunk in reader:
        yield chunk


# --------------------------------------------------------------------------
# Pass 1: per-S1 statistics + cross-link *counts* (not full mapping)
# --------------------------------------------------------------------------


@dataclass
class GroundTruthStats:
    """Aggregate, per-S1 ground-truth statistics computed in one streaming
    pass over ``train_ground_truth.tsv``."""

    total_s1_entities: int = 0

    total_s2_links: int = 0
    total_s3_links: int = 0

    # count_of_matches -> number_of_s1_entities_with_that_count
    s2_match_count_distribution: Dict[int, int] = field(default_factory=dict)
    s3_match_count_distribution: Dict[int, int] = field(default_factory=dict)

    # total (s2+s3) 0 / 1 / >1 bucket counts, kept for direct comparability
    # with docs/analysis.md (Task 1 EDA), which reported only these buckets.
    total_bucket_counts: Dict[str, int] = field(
        default_factory=lambda: {"0": 0, "1": 0, ">1": 0}
    )

    only_s2_count: int = 0
    only_s3_count: int = 0
    both_s2_and_s3_count: int = 0
    neither_count: int = 0

    max_s2_matches: int = 0
    max_s3_matches: int = 0
    max_total_matches: int = 0

    @property
    def total_links(self) -> int:
        return self.total_s2_links + self.total_s3_links

    @property
    def mean_total_matches(self) -> float:
        if self.total_s1_entities == 0:
            return 0.0
        return self.total_links / self.total_s1_entities

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_links"] = self.total_links
        d["mean_total_matches"] = self.mean_total_matches
        # JSON object keys must be strings.
        d["s2_match_count_distribution"] = {
            str(k): v for k, v in self.s2_match_count_distribution.items()
        }
        d["s3_match_count_distribution"] = {
            str(k): v for k, v in self.s3_match_count_distribution.items()
        }
        return d

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def _bump_histogram(hist: Dict[int, int], key: int) -> None:
    hist[key] = hist.get(key, 0) + 1


def compute_ground_truth_statistics(
    path: str | Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    s2_link_counts: Optional[Counter] = None,
    s3_link_counts: Optional[Counter] = None,
) -> GroundTruthStats:
    """Stream ``train_ground_truth.tsv`` once and compute all per-S1
    statistics required by Task 2 section 1.

    If ``s2_link_counts`` / ``s3_link_counts`` (empty ``Counter`` objects)
    are passed in, they are updated in place with
    ``entity_id -> number of S1 entities matching it``. This lets a
    single pass over the (potentially large) ground-truth file serve
    both section 1 (match statistics) and the first half of section 2
    (cross-link counts), instead of reading the file twice.
    """
    stats = GroundTruthStats()

    for chunk in iter_ground_truth_chunks(path, chunksize=chunksize):
        for s1_id, raw_matches in zip(chunk[S1_COL], chunk[MATCH_COL]):
            s2_ids, s3_ids = parse_matched_ids(raw_matches)

            n_s2 = len(s2_ids)
            n_s3 = len(s3_ids)
            n_total = n_s2 + n_s3

            stats.total_s1_entities += 1
            stats.total_s2_links += n_s2
            stats.total_s3_links += n_s3

            _bump_histogram(stats.s2_match_count_distribution, n_s2)
            _bump_histogram(stats.s3_match_count_distribution, n_s3)

            if n_total == 0:
                stats.total_bucket_counts["0"] += 1
            elif n_total == 1:
                stats.total_bucket_counts["1"] += 1
            else:
                stats.total_bucket_counts[">1"] += 1

            if n_s2 > 0 and n_s3 > 0:
                stats.both_s2_and_s3_count += 1
            elif n_s2 > 0:
                stats.only_s2_count += 1
            elif n_s3 > 0:
                stats.only_s3_count += 1
            else:
                stats.neither_count += 1

            stats.max_s2_matches = max(stats.max_s2_matches, n_s2)
            stats.max_s3_matches = max(stats.max_s3_matches, n_s3)
            stats.max_total_matches = max(stats.max_total_matches, n_total)

            if s2_link_counts is not None:
                for sid in s2_ids:
                    s2_link_counts[sid] += 1
            if s3_link_counts is not None:
                for sid in s3_ids:
                    s3_link_counts[sid] += 1

    return stats


def verify_internal_consistency(stats: GroundTruthStats) -> List[str]:
    """Cross-check derived totals against each other.

    Returns a list of human-readable problem descriptions; an empty list
    means all checks passed. This does not re-read the file -- it only
    checks that the numbers computed in a single pass are mutually
    consistent (e.g. the 0/1/>1 buckets sum to total_s1_entities, and
    only/both/neither sums to total_s1_entities).
    """
    problems: List[str] = []

    bucket_sum = sum(stats.total_bucket_counts.values())
    if bucket_sum != stats.total_s1_entities:
        problems.append(
            f"0/1/>1 bucket counts sum to {bucket_sum}, expected "
            f"total_s1_entities={stats.total_s1_entities}"
        )

    only_both_sum = (
        stats.only_s2_count
        + stats.only_s3_count
        + stats.both_s2_and_s3_count
        + stats.neither_count
    )
    if only_both_sum != stats.total_s1_entities:
        problems.append(
            f"only_s2 + only_s3 + both + neither = {only_both_sum}, expected "
            f"total_s1_entities={stats.total_s1_entities}"
        )

    hist_s2_sum = sum(
        count * freq for count, freq in stats.s2_match_count_distribution.items()
    )
    if hist_s2_sum != stats.total_s2_links:
        problems.append(
            f"sum(count * freq) over s2_match_count_distribution = "
            f"{hist_s2_sum}, expected total_s2_links={stats.total_s2_links}"
        )

    hist_s3_sum = sum(
        count * freq for count, freq in stats.s3_match_count_distribution.items()
    )
    if hist_s3_sum != stats.total_s3_links:
        problems.append(
            f"sum(count * freq) over s3_match_count_distribution = "
            f"{hist_s3_sum}, expected total_s3_links={stats.total_s3_links}"
        )

    freq_s1_sum = sum(stats.s2_match_count_distribution.values())
    if freq_s1_sum != stats.total_s1_entities:
        problems.append(
            f"s2_match_count_distribution frequencies sum to {freq_s1_sum}, "
            f"expected total_s1_entities={stats.total_s1_entities}"
        )

    freq_s1_sum_s3 = sum(stats.s3_match_count_distribution.values())
    if freq_s1_sum_s3 != stats.total_s1_entities:
        problems.append(
            f"s3_match_count_distribution frequencies sum to "
            f"{freq_s1_sum_s3}, expected "
            f"total_s1_entities={stats.total_s1_entities}"
        )

    return problems


# --------------------------------------------------------------------------
# Section 2: cross-link (leakage) statistics
# --------------------------------------------------------------------------


@dataclass
class CrossLinkStats:
    unique_s2_ids: int = 0
    s2_ids_linked_to_multiple_s1: int = 0
    max_s1_per_s2: int = 0

    unique_s3_ids: int = 0
    s3_ids_linked_to_multiple_s1: int = 0
    max_s1_per_s3: int = 0

    # A handful of concrete (id -> [s1_ids]) examples for documentation /
    # sanity-checking. Bounded in size by `example_limit` in
    # `build_full_cross_link_mapping`.
    example_s2_cross_links: Dict[str, List[str]] = field(default_factory=dict)
    example_s3_cross_links: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def summarize_cross_links(
    s2_link_counts: Counter, s3_link_counts: Counter
) -> CrossLinkStats:
    """Compute the aggregate cross-link statistics from the per-id
    occurrence counters built during ``compute_ground_truth_statistics``.

    This is pure in-memory aggregation over the already-built counters --
    it does not touch the ground-truth file again.
    """
    stats = CrossLinkStats()

    stats.unique_s2_ids = len(s2_link_counts)
    stats.unique_s3_ids = len(s3_link_counts)

    if s2_link_counts:
        stats.max_s1_per_s2 = max(s2_link_counts.values())
        stats.s2_ids_linked_to_multiple_s1 = sum(
            1 for c in s2_link_counts.values() if c > 1
        )
    if s3_link_counts:
        stats.max_s1_per_s3 = max(s3_link_counts.values())
        stats.s3_ids_linked_to_multiple_s1 = sum(
            1 for c in s3_link_counts.values() if c > 1
        )

    return stats


def build_full_cross_link_mapping(
    path: str | Path,
    multi_linked_ids: set,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Dict[str, List[str]]:
    """Second, targeted pass over the ground-truth file.

    Given the set of S2/S3 ids already known (from pass 1's counters) to
    be linked to more than one S1 entity, re-scan the file and build the
    *complete* ``id -> [s1_ids that matched it]`` mapping, but only for
    ids in ``multi_linked_ids``. This keeps memory proportional to the
    number of cross-linked ids (expected to be small), not to the total
    number of rows in the ground-truth file.

    This mapping is what ``split.build_s1_components`` uses to group
    leakage-connected S1 entities into the same split.
    """
    mapping: Dict[str, List[str]] = {sid: [] for sid in multi_linked_ids}
    if not multi_linked_ids:
        return mapping

    for chunk in iter_ground_truth_chunks(path, chunksize=chunksize):
        for s1_id, raw_matches in zip(chunk[S1_COL], chunk[MATCH_COL]):
            s2_ids, s3_ids = parse_matched_ids(raw_matches)
            for sid in s2_ids:
                if sid in mapping:
                    mapping[sid].append(s1_id)
            for sid in s3_ids:
                if sid in mapping:
                    mapping[sid].append(s1_id)
    return mapping


def compute_cross_link_statistics(
    path: str | Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    example_limit: int = 5,
    s2_link_counts: Optional[Counter] = None,
    s3_link_counts: Optional[Counter] = None,
) -> Tuple[CrossLinkStats, Dict[str, List[str]]]:
    """Full section-2 workflow.

    If ``s2_link_counts``/``s3_link_counts`` are already populated (e.g.
    from a prior call to ``compute_ground_truth_statistics`` that was
    given empty ``Counter`` objects to fill), they are reused and the
    file is only re-scanned once, for the targeted second pass. If they
    are not supplied, this function builds them itself via a first pass.

    Returns ``(CrossLinkStats, full_multi_linked_mapping)`` where the
    mapping is ``id -> [s1_ids]`` for every id linked to more than one
    S1 entity (S2 and S3 ids share one dict; prefixes disambiguate them).
    This mapping is the leakage-detection input required by
    ``split.build_s1_components``.
    """
    if s2_link_counts is None or s3_link_counts is None:
        s2_link_counts = Counter()
        s3_link_counts = Counter()
        compute_ground_truth_statistics(
            path,
            chunksize=chunksize,
            s2_link_counts=s2_link_counts,
            s3_link_counts=s3_link_counts,
        )

    stats = summarize_cross_links(s2_link_counts, s3_link_counts)

    multi_linked_ids = {sid for sid, c in s2_link_counts.items() if c > 1} | {
        sid for sid, c in s3_link_counts.items() if c > 1
    }

    full_mapping = build_full_cross_link_mapping(
        path, multi_linked_ids, chunksize=chunksize
    )

    # Populate a small, bounded set of examples for docs/reporting.
    s2_examples = {
        sid: ids
        for sid, ids in full_mapping.items()
        if sid.startswith("S2-") and len(ids) > 1
    }
    s3_examples = {
        sid: ids
        for sid, ids in full_mapping.items()
        if sid.startswith("S3-") and len(ids) > 1
    }
    stats.example_s2_cross_links = dict(list(s2_examples.items())[:example_limit])
    stats.example_s3_cross_links = dict(list(s3_examples.items())[:example_limit])

    return stats, full_mapping


# --------------------------------------------------------------------------
# Convenience: per-S1 matched-id sets (used by split.py and the evaluator)
# --------------------------------------------------------------------------


@dataclass
class S1Matches:
    """The full parsed ground truth as ``s1_id -> (s2_ids, s3_ids)``.

    Only used for building the validation split and for constructing the
    ``ground_truth`` argument to the F0.5 evaluator on the *validation*
    subset. This is NOT meant to be materialized for the full 2.2M-row
    training set inside a long-lived process without care -- see the
    memory note in ``load_s1_matches``.
    """

    s1_to_s2: Dict[str, List[str]] = field(default_factory=dict)
    s1_to_s3: Dict[str, List[str]] = field(default_factory=dict)

    def total_matches(self, s1_id: str) -> Tuple[str, ...]:
        return tuple(self.s1_to_s2.get(s1_id, [])) + tuple(
            self.s1_to_s3.get(s1_id, [])
        )

    def bucket(self, s1_id: str) -> str:
        n = len(self.s1_to_s2.get(s1_id, [])) + len(self.s3_to_s2_placeholder(s1_id))
        return "0" if n == 0 else ("1" if n == 1 else ">1")

    def s3_to_s2_placeholder(self, s1_id: str) -> List[str]:
        # Small helper kept separate only for readability of `bucket`.
        return self.s1_to_s3.get(s1_id, [])


def compute_s1_bucket_map(
    path: str | Path, chunksize: int = DEFAULT_CHUNKSIZE
) -> Dict[str, str]:
    """Stream the ground truth and return ``s1_id -> "0"|"1"|">1"``.

    This is the cheap input ``split.create_validation_split`` needs: it
    stores one short string per S1 entity (not the match-id lists
    themselves), so materializing it for all 2.2M training S1 entities
    is inexpensive (tens of MB), unlike ``load_s1_matches``.
    """
    bucket_map: Dict[str, str] = {}
    for chunk in iter_ground_truth_chunks(path, chunksize=chunksize):
        for s1_id, raw_matches in zip(chunk[S1_COL], chunk[MATCH_COL]):
            s2_ids, s3_ids = parse_matched_ids(raw_matches)
            n = len(s2_ids) + len(s3_ids)
            bucket_map[s1_id] = "0" if n == 0 else ("1" if n == 1 else ">1")
    return bucket_map


def load_s1_matches(
    path: str | Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    s1_ids: Optional[set] = None,
) -> S1Matches:
    """Load the full per-S1 matched-id lists.

    Memory footprint: roughly proportional to ``total_links``
    (~7.6M short strings per the Task 1 EDA estimate), which is a few
    hundred MB to low GB of Python object overhead -- noticeably more
    than the bounded counters above. Use ``s1_ids`` to restrict this to
    only the S1 entities you actually need (e.g. only the validation
    split's ids), which is the intended usage pattern: this function is
    for building small validation-only ground-truth dicts to hand to the
    evaluator, not for holding the entire training ground truth in
    memory at once. If ``s1_ids`` is None, every S1 entity is loaded --
    only do this if you have verified you have enough memory headroom.
    """
    result = S1Matches()
    for chunk in iter_ground_truth_chunks(path, chunksize=chunksize):
        for s1_id, raw_matches in zip(chunk[S1_COL], chunk[MATCH_COL]):
            if s1_ids is not None and s1_id not in s1_ids:
                continue
            s2_ids, s3_ids = parse_matched_ids(raw_matches)
            if s2_ids:
                result.s1_to_s2[s1_id] = s2_ids
            if s3_ids:
                result.s1_to_s3[s1_id] = s3_ids
    return result
