# Task 2 Validation Analysis Report

**Generated from**: `dataset/train/train_ground_truth.tsv`  
**Analysis script**: `scripts/run_validation_analysis.py`  
**Runtime**: ~20.5 seconds  
**Date**: 2026-09-26

> **All results below are measured directly from the training dataset.**  
> No synthetic data, no manual overrides, no extrapolation.

---

## 1. Ground Truth Match Statistics (Measured)

| Metric | Value |
|--------|-------|
| **Total S1 entities** | 2,206,821 |
| **Total S2 links** | 3,693,619 |
| **Total S3 links** | 3,944,746 |
| **Total links (S2 + S3)** | 7,638,365 |
| **Mean total matches per S1** | 3.461 |
| **Max S2 matches per S1** | 5 |
| **Max S3 matches per S1** | 6 |
| **Max total matches per S1** | 11 |

### S1 Entity Classification by Match Type

| Category | Count | Percentage |
|----------|-------|------------|
| Only S2 matches | 143,029 | 6.48% |
| Only S3 matches | 164,498 | 7.45% |
| Both S2 and S3 matches | 1,776,047 | 80.48% |
| Neither (0 matches) | 123,247 | 5.58% |

### 0 / 1 / >1 Match Distribution (Total Matches per S1)

| Bucket | Count | Percentage |
|--------|-------|------------|
| 0 matches | 123,247 | 5.58% |
| 1 match | 119,157 | 5.40% |
| >1 matches | 1,964,417 | 89.02% |

### S2 Matches per S1 Distribution

| Matches | S1 Entities |
|---------|-------------|
| 0 | 287,745 |
| 1 | 789,108 |
| 2 | 652,779 |
| 3 | 333,957 |
| 4 | 119,078 |
| 5 | 24,154 |

### S3 Matches per S1 Distribution

| Matches | S1 Entities |
|---------|-------------|
| 0 | 266,276 |
| 1 | 716,417 |
| 2 | 668,375 |
| 3 | 372,443 |
| 4 | 145,116 |
| 5 | 35,378 |
| 6 | 2,816 |

### Internal Consistency Check

✅ **PASSED** — All derived totals cross-validate (bucket counts sum to total S1 entities, histogram-weighted sums equal link totals).

---

## 2. Cross-Link / Leakage Statistics (Measured)

| Metric | Value |
|--------|-------|
| Unique S2 IDs | 3,693,619 |
| S2 IDs linked to multiple S1 | **0** (0.00%) |
| Max S1 entities per S2 ID | 1 |
| Unique S3 IDs | 3,944,746 |
| S3 IDs linked to multiple S1 | **0** (0.00%) |
| Max S1 entities per S3 ID | 1 |

**Finding**: No cross-links detected. Every S2 and S3 entity maps to exactly one S1 entity. This means:
- No S2 ID is shared across multiple S1 entities
- No S3 ID is shared across multiple S1 entities
- Leakage via shared true matches is **not present** in this dataset

---

## 3. Leakage-Safe Validation Split (Measured)

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Target validation fraction | 0.20 |
| **Train S1 count** | 1,765,458 |
| **Validation S1 count** | 441,363 |
| **Realized validation fraction** | 0.2000 |
| Number of components | 2,206,821 |
| Multi-member components | 0 |
| Largest component size | 1 |

### Bucket Distribution Comparison (Train vs Validation)

| Bucket | Train Count | Train % | Val Count | Val % | Delta (pp) |
|--------|-------------|---------|-----------|-------|------------|
| 0 matches | 98,598 | 5.58% | 24,649 | 5.58% | 0.00 |
| 1 match | 95,326 | 5.40% | 23,831 | 5.40% | 0.00 |
| >1 matches | 1,571,534 | 89.02% | 392,883 | 89.02% | 0.00 |

**Finding**: The split preserves the 0/1/>1 distribution **exactly** (deltas ≤ 0.00 percentage points). This is expected because:
- No cross-links exist → all components are singletons
- Stratified split operates on individual S1 entities directly
- Each bucket is independently subsampled at 20%

---

## 4. F0.5 Evaluator Smoke Test

A hard-coded 3-entity toy example (not real predictions) was run to verify the evaluator executes correctly:

| S1 Entity | True IDs | Predicted IDs | Precision | Recall | F0.5 |
|-----------|----------|---------------|-----------|--------|------|
| S1-1 | {S2-1, S2-2} | {S2-1} | 1.000 | 0.500 | 0.833 |
| S1-2 | ∅ | ∅ | 1.000 | 1.000 | 1.000 |
| S1-3 | {S3-9} | {S3-9, S3-8} | 0.500 | 1.000 | 0.556 |

**Macro F0.5 (official)**: 0.796  
**Micro F0.5 (diagnostic)**: 0.667

---

## 5. Summary of Warnings / Anomalies

| Check | Status | Notes |
|-------|--------|-------|
| Cross-links (S2) | ✅ None found | 0 S2 IDs map to multiple S1 |
| Cross-links (S3) | ✅ None found | 0 S3 IDs map to multiple S1 |
| Component size | ✅ All singletons | Largest component = 1 |
| Split distribution preservation | ✅ Exact | 0/1/>1 buckets match perfectly |
| Internal consistency | ✅ Passed | All derived stats cross-validate |

**No anomalies detected.** The ground truth is clean (no cross-links), the split is exact, and all consistency checks pass.

---

## 6. Artifacts Generated

| File | Description |
|------|-------------|
| `artifacts/validation/ground_truth_stats.json` | Full match statistics + distributions |
| `artifacts/validation/cross_link_stats.json` | Cross-link analysis (empty — no cross-links) |
| `artifacts/validation/split_seed42.json` | Validation split (val_ids + metadata) |

---

## 7. Assumptions & Implementation Notes

1. **Chunked processing**: All file reads used `chunksize=200,000` — no full-file loads into memory.
2. **No dataset modification**: The raw `train_ground_truth.tsv` was only read, never written.
3. **No external dependencies**: Analysis used only local code and standard libraries (pandas, stdlib).
4. **Split methodology**: Union-find on cross-links → component-level stratified split. Since no cross-links exist, this degenerates to per-S1 stratified sampling.
5. **Reproducibility**: Seed=42 guarantees identical split on re-run.
6. **Artifact size**: `split_seed42.json` stores only `val_ids` (441,363 entries) + metadata; `train_ids` is reconstructed as complement at load time.

---

## 8. Commands Run

```bash
# From repository root
uv run python scripts/run_validation_analysis.py \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --output-dir artifacts/validation \
    --chunksize 200000 \
    --val-fraction 0.2 \
    --seed 42
```

**Total runtime**: ~20.5 seconds