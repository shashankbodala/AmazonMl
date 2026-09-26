# Validation Infrastructure (Task 2)

Scope: this document covers **ground-truth parsing, cross-link (leakage)
detection, the train/validation split, and the F0.5 evaluator only**.
No blocking, candidate generation, matching, or model code exists yet
— see `docs/analysis.md` (Task 1) for the dataset analysis that this
work builds on, and the repository's Task 2 request for the full list
of exclusions.

> **Note on numbers in this document:** this code has not yet been run
> against the real `dataset/train/train_ground_truth.tsv` (it isn't
> available in the environment this was written in). Every statistic
> below that looks like a placeholder (e.g. "*(fill in after running)*")
> is intentionally left blank. Run
> `scripts/run_validation_analysis.py` locally and paste its output
> here once you have it — see "How to run" below.

---

## 1. How ground truth is represented

`dataset/train/train_ground_truth.tsv` has two columns:

```
source1_entity_id    matched_entity_ids
```

`matched_entity_ids` is a single comma-separated string mixing `S2-`
and `S3-` prefixed ids, e.g.:

```
S2-681193310,S2-743505751,S3-775321672
```

`business_entity_resolution.validation.ground_truth` never loads this
file as one DataFrame. `iter_ground_truth_chunks()` reads it with
`pandas.read_csv(..., chunksize=...)`, forcing both columns to `str`
dtype (so numeric-looking IDs are never silently coerced/NaN'd), and
`parse_matched_ids()` splits each cell into `(s2_ids, s3_ids)` lists.

Unknown prefixes (anything not `S2-`/`S3-`) raise a `ValueError` rather
than being silently dropped — a parsing bug that under-counts matches
is worse than a loud crash.

## 2. S2/S3 parsing and per-S1 match statistics

`compute_ground_truth_statistics(path, chunksize=...)` streams the file
once and returns a `GroundTruthStats` object with:

- `total_s1_entities`, `total_s2_links`, `total_s3_links`, `total_links`
- `s2_match_count_distribution` / `s3_match_count_distribution`:
  `{match_count: number_of_s1_entities_with_that_count}`
- `total_bucket_counts`: `{"0": ..., "1": ..., ">1": ...}` — directly
  comparable to the Task 1 EDA's `match_distribution.csv`, which only
  had these three aggregate buckets (Task 1 explicitly could not tell
  S2 from S3 apart; this module fixes that)
- `only_s2_count` / `only_s3_count` / `both_s2_and_s3_count` /
  `neither_count`
- `max_s2_matches`, `max_s3_matches`, `max_total_matches`
- `mean_total_matches` (property)

`verify_internal_consistency(stats)` cross-checks that these derived
numbers agree with each other (bucket counts sum to
`total_s1_entities`, histogram-weighted sums equal the link totals,
etc.) without re-reading the file. `scripts/run_validation_analysis.py`
runs this automatically and prints `PASSED`/`FAILED`.

### Results against the real training ground truth

*(fill in after running `scripts/run_validation_analysis.py` locally)*

| Metric | Value |
|---|---|
| total_s1_entities | |
| total_s2_links | |
| total_s3_links | |
| total_links | |
| mean_total_matches | |
| max_s2_matches | |
| max_s3_matches | |
| max_total_matches | |
| only_s2_count (%) | |
| only_s3_count (%) | |
| both_s2_and_s3_count (%) | |
| neither_count (%) | |
| bucket "0" / "1" / ">1" (counts, %) | |
| Internal consistency check | |

## 3. Cross-link findings

Two S1 entities are **cross-linked** if their true-match sets share at
least one S2 or S3 id. `compute_cross_link_statistics()`:

1. Reuses the `Counter`s built during step 2 (`s2_link_counts`,
   `s3_link_counts`: `id -> number of S1 entities matching it`) to
   compute, in memory, `unique_s2_ids`, `s2_ids_linked_to_multiple_s1`,
   `max_s1_per_s2`, and the S3 equivalents — no second file read needed
   for these aggregate numbers.
2. Runs one **targeted second pass** over the file
   (`build_full_cross_link_mapping`) to build the *complete*
   `id -> [s1_ids]` mapping, but only for ids already known (from step
   1) to be linked to more than one S1 entity. This keeps the second
   pass's memory bounded by the number of cross-linked ids, not by the
   size of the whole file.
3. Surfaces a handful of concrete examples (`example_s2_cross_links`,
   `example_s3_cross_links`) for this document / manual inspection.

This targeted mapping is exactly what `split.build_s1_components()`
needs to guarantee cross-linked S1 entities land on the same side of
the train/validation split (see section 4).

### Results against the real training ground truth

*(fill in after running)*

| Metric | Value |
|---|---|
| unique_s2_ids | |
| s2_ids_linked_to_multiple_s1 (count, %) | |
| max_s1_per_s2 | |
| unique_s3_ids | |
| s3_ids_linked_to_multiple_s1 (count, %) | |
| max_s1_per_s3 | |

Example cross-links: see `artifacts/validation/cross_link_stats.json`
→ `example_s2_cross_links` / `example_s3_cross_links` after running.

**Why this matters for splitting:** if `S1-A` and `S1-B` both have
`S2-X` as a true match, and a naive random split puts `S1-A` in train
and `S1-B` in validation, a future matcher could see `S2-X` "confirmed
as a match-type record" during training and get credited again for
recognizing it during validation — a soft, indirect form of leakage
that a plain per-S1 random split does not prevent. Section 4 describes
how this implementation avoids it.

## 4. Validation split methodology

### 4.1 Why S1-level splitting alone is not sufficient

Splitting rows by S1 id keeps each S1 entity's *own* row intact, but
does nothing to prevent the cross-link leakage above, since the
S2/S3 side of a match is a separate, un-partitioned table. The split
must therefore operate on **groups of S1 entities that share a
cross-linked match**, not on individual S1 ids.

### 4.2 Algorithm

1. **Union-Find over S1 ids.** For every S2/S3 id linked to more than
   one S1 entity (from section 3's full mapping), union all of that
   id's S1 entities into one component. This is transitive: if `S2-X`
   links `S1-A`/`S1-B` and `S2-Y` links `S1-B`/`S1-C`, then `S1-A`,
   `S1-B`, and `S1-C` all end up in the same component even though no
   single id directly connects `S1-A` and `S1-C`
   (`tests/test_split.py::test_build_s1_components_transitively_merges_chained_links`).
   S1 entities with no cross-linked match are singleton components —
   expected to be the large majority.
2. **Stratify by bucket at the component level.** Each component is
   assigned the 0/1/>1 bucket of its lexicographically-smallest member
   S1 id. For singleton components this is exact; for the rare
   multi-member components it's an approximation, consistent with the
   "preserve *approximately*" requirement.
3. **Shuffle deterministically and split per stratum.** Within each of
   the three strata, components are sorted (for determinism) and then
   shuffled with `random.Random(seed)`, and the first
   `round(len(stratum) * val_fraction)` components go to validation.
4. **Emit `train_ids` / `val_ids`** as the flattened, sorted member
   lists of each side's components.

### 4.3 Reproducibility

Same `(bucket_map, cross_link_mapping, val_fraction, seed)` input
always produces an identical split — verified by
`test_split_is_reproducible_with_same_seed`. Default seed: `42`.

### 4.4 A known limitation to check for on the real data

If cross-linking turns out to be pervasive (i.e. a large fraction of
S1 entities transitively connect into one or a few giant components),
component-level splitting can (a) make it impossible to hit the exact
target `val_fraction`, since whole giant components can't be divided,
and (b) distort the preserved 0/1/>1 balance for whichever bucket the
giant component's representative happens to fall into. This is exactly
why `SplitResult` reports `n_components`, `n_multi_member_components`,
and `largest_component_size` — **inspect these after running against
the real ground truth** (they are printed by
`scripts/run_validation_analysis.py` and saved in
`split_seed{seed}.json`). If `largest_component_size` turns out to be
a large fraction of all training S1 entities, treat that as a finding
to bring back for review before trusting the split, rather than a
silent edge case — this document does not assume in advance which
outcome the real data will show.

### 4.5 Results against the real training ground truth

*(fill in after running)*

| Metric | Value |
|---|---|
| seed | 42 |
| val_fraction (target) | 0.2 |
| n_train / n_val | |
| val_fraction (realized) | |
| n_components | |
| n_multi_member_components | |
| largest_component_size | |
| train_bucket_counts | |
| val_bucket_counts | |

### 4.6 Persisted artifact

`save_split()` writes `artifacts/validation/split_seed{seed}.json`
containing `val_ids` (the smaller side) plus all metadata above —
**not** the full `train_ids` list by default (pass
`store_train_ids=True` if you'd rather persist it explicitly).
`load_split()` reconstructs `train_ids` as the complement of `val_ids`
against `all_s1_ids` (e.g. every id from `train_source1.tsv`) when it
wasn't stored directly. This keeps the artifact small — proportional
to `val_fraction * total_s1_entities`, not to the full training set —
per the "do not duplicate millions of rows" requirement.

## 5. F0.5 definition and edge-case behavior

**Official metric: macro F0.5, averaged per S1 entity.** This is
computed once per S1 entity from that entity's own true/predicted ID
sets, and the per-entity F0.5 values are averaged **unweighted** across
entities. This is *not* the same as pooling every entity's TP/FP/FN
into one global count and computing a single F0.5 from the pooled
totals (micro-averaging) — `evaluate_predictions()` also returns a
`micro_f0_5` field purely as a secondary diagnostic, explicitly labeled
as such, and callers must not report it as the official score.

```
F0.5 = (1 + 0.5^2) * P * R / ((0.5^2 * P) + R)
```

### Edge-case convention

| Case | precision | recall | F0.5 |
|---|---|---|---|
| `TP + FP == 0` (nothing predicted) | `1.0` (vacuous) | as usual | as usual |
| `TP + FN == 0` (ground truth empty) | as usual | `1.0` (vacuous) | as usual |
| both above | `1.0` | `1.0` | `1.0` |
| `TP == 0` and `FP > 0` (ground truth empty, but something predicted) | `0.0` | `1.0` (vacuous) | `0.0` |

The two vacuous-`1.0` rules only ever fire when there is genuinely
nothing to be penalized for in that direction (nothing predicted → no
possible false positives; nothing true → no possible false negatives).
Whenever precision is genuinely `0`, F0.5 is `0` regardless of recall —
this is a direct consequence of the F-beta formula, not a special case,
but it has an important practical implication worth stating plainly:
**an S1 entity with an empty true-match set scores 0.0 the instant any
candidate is predicted for it, and an S1 entity whose entire prediction
misses a non-empty ground truth also scores 0.0.**

`predictions` missing an S1 entity that exists in `ground_truth` is
treated as if an empty set had been predicted for it (not skipped —
skipping would silently shrink the scored universe). Duplicate IDs in
either `true_ids` or `predicted_ids` are coalesced by converting to a
`set` before counting, so they can never inflate TP/FP.

### The 8 required test scenarios

All implemented as individual functions in `tests/test_metrics.py` so
a failure names the exact scenario:

1. Perfect prediction → F0.5 = 1.0
2. Ground truth empty + prediction empty → F0.5 = 1.0 (vacuous)
3. Ground truth empty + false prediction → F0.5 = 0.0
4. Partial recall (2 of 3 found) → F0.5 = 10/11 ≈ 0.909
5. Correct matches + one extra false positive → F0.5 = 5/7 ≈ 0.714
6. Multiple correct matches (size-4 exact match) → F0.5 = 1.0
7. Prediction missing an S1 entity entirely → F0.5 = 0.0 (recall 0)
8. Prediction containing duplicate IDs → duplicates collapse via `set`,
   scored identically to the de-duplicated prediction

## 6. Files created

```
src/business_entity_resolution/
  __init__.py
  validation/
    __init__.py           # public API re-exports
    ground_truth.py        # streaming parsing, match stats, cross-link detection
    metrics.py              # macro F0.5 evaluator
    split.py                 # union-find + stratified, reproducible split

tests/
  test_ground_truth.py     # synthetic-data unit tests for parsing/stats/cross-links
  test_metrics.py           # the 8 required F0.5 edge-case scenarios + aggregation tests
  test_split.py              # union-find, stratification, reproducibility, persistence

scripts/
  run_validation_analysis.py  # CLI: runs sections 1-4 against the real dataset

docs/
  validation.md              # this file

artifacts/validation/
  .gitkeep                    # directory placeholder; generated *.json is git-ignored
```

`docs/analysis.md` (Task 1) was **not modified** — nothing computed in
Task 2 contradicts its conclusions; if the real cross-link run reveals
something that does (see 4.4), update `docs/analysis.md` at that point
rather than here.

## 7. How to run

```bash
# from the repository root
pip install -e ".[dev]"          # installs the package + pytest
# or, at minimum: pip install pandas pytest

# unit tests (synthetic data only, seconds to run)
pytest tests/ -v

# full analysis against the real training ground truth
python scripts/run_validation_analysis.py \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --output-dir artifacts/validation \
    --chunksize 200000 \
    --val-fraction 0.2 \
    --seed 42
```

Expected console output from the script: four sections (ground-truth
stats, cross-link stats, split stats, an F0.5 smoke-test on a 3-entity
toy example — not real predictions, since no matcher exists yet),
finishing with a `DONE` banner and a runtime. Expected files written:

- `artifacts/validation/ground_truth_stats.json`
- `artifacts/validation/cross_link_stats.json`
- `artifacts/validation/split_seed42.json` (filename varies with `--seed`)

None of these are raw dataset rows — all three are small, bounded
summaries/id-lists as required.

## 8. Verification checklist (run before trusting this for Task 3 design)

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `scripts/run_validation_analysis.py` run against the real
      `train_ground_truth.tsv` — no exceptions, "INTERNAL CONSISTENCY
      CHECK: passed"
- [ ] Section 1 numbers pasted into this doc's tables above
- [ ] Section 2 (cross-link) numbers pasted into this doc's tables above,
      and `largest_component_size` inspected per §4.4
- [ ] Section 3 (split) numbers pasted into this doc's tables above
- [ ] No files under `dataset/` were modified (this code only reads)
- [ ] No network access was used (there is none in this codebase)
- [ ] No blocking/candidate-generation/matching code exists in this PR
