# Task 3A: Baseline Blocking Benchmark Report

**Generated from**: 20% validation split (seed=42)  
**Benchmark script**: `scripts/run_blocking_benchmark.py`  
**Date**: 2026-09-27

---

## 1. Methodology

This benchmark evaluates a baseline blocking strategy that uses **exact normalized matches** on four composite keys. No fuzzy matching, no ML, no character n-grams -- just deterministic string canonicalization followed by exact key lookups.

### Blocking Strategies

Four strategies are applied independently; the candidate set is the **union** of all four:

| # | Strategy | Key Construction |
|---|----------|-----------------|
| 1 | Exact normalized business name | `normalize(business_name)` |
| 2 | Exact normalized business address | `normalize(business_address)` |
| 3 | Normalized name + country | `normalize(business_name) \| normalize(country)` |
| 4 | Normalized address + country | `normalize(business_address) \| normalize(country)` |

### Normalization Pipeline

For **business names**:
1. Strip whitespace, handle NaN/None
2. Strip diacritics from Latin characters (NFD decomposition, remove combining marks)
3. Unicode casefold
4. Remove punctuation except `&`, `-`, `'`, ` `, `+`; replace `/` with space
5. Collapse whitespace

For **business addresses**:
1. Strip whitespace, handle NaN/None
2. Strip diacritics from Latin characters
3. Unicode casefold
4. Replace `/` with space, remove all punctuation
5. Collapse whitespace

For **country**:
1. Strip whitespace, handle NaN/None
2. Strip diacritics from Latin characters
3. Unicode casefold
4. Remove all punctuation
5. Collapse whitespace

### S2 and S3 Handling

S2 and S3 candidates are tracked **separately**. Each S1 entity produces two candidate sets: one from S2 and one from S3. The inverted indexes for S2 and S3 are built independently.

### Optimization Approach

- Only build source indexes for blocking keys that appear in the validation S1 set
- Use vectorized pandas operations for normalization (with pre-built `str.translate` tables for punctuation removal)
- Chunk-based processing (200,000 rows per chunk) to bound memory
- Filter source chunks with `isin()` masks before adding to inverted indexes

---

## 2. Validation Split

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Validation fraction | 20% |
| **Validation S1 entities** | **441,363** |
| Total source S1 entities | 2,206,821 |
| Total source S2 entities | 5,034,616 |
| Total source S3 entities | 5,285,603 |
| Ground truth S2 links (validation) | 738,933 |
| Ground truth S3 links (validation) | 788,417 |

---

## 3. Measured Results

### Candidate Recall

| Metric | Value |
|--------|-------|
| **Overall candidate recall (S2+S3)** | **0.3080 (30.80%)** |
| S2 candidate recall | 0.3326 (33.26%) |
| S3 candidate recall | 0.2849 (28.49%) |

Candidate recall measures: for each true match (S2 or S3), was that entity returned as a candidate by at least one blocking strategy?

### Candidate Pair Counts

| Metric | Value |
|--------|-------|
| **Total candidate pairs** | **4,759,549** |
| Total S2 candidates | 2,306,603 |
| Total S3 candidates | 2,452,946 |

### Per-S1 Candidate Statistics

| Metric | Value |
|--------|-------|
| **Avg candidates per S1** | **10.78** |
| **Median candidates per S1** | **2.00** |
| **P95 candidates per S1** | **64.00** |
| **Max candidates per S1** | **476** |

### Zero / One / >1 Candidate Distribution

| Bucket | Count | Percentage |
|--------|-------|------------|
| **Zero candidates** | **93,348** | **21.15%** |
| One candidate | 103,729 | 23.50% |
| >1 candidates | 244,286 | 55.35% |

### Runtime

| Phase | Time |
|-------|------|
| Key extraction (validation S1) | 15.56s |
| Build S2 indexes | ~157s |
| Build S3 indexes | ~156s |
| Generate candidates | 22.85s |
| **Total** | **351.55s** |

### Source Index Statistics

**S2 Indexes:**
| Index | Unique Keys | Total Entries |
|-------|-------------|---------------|
| Name | 163,920 | 440,345 |
| Address | 75,235 | 106,443 |
| Name + Country | 163,902 | 438,874 |
| Address + Country | 75,235 | 106,443 |

**S3 Indexes:**
| Index | Unique Keys | Total Entries |
|-------|-------------|---------------|
| Name | 172,163 | 491,078 |
| Address | 37,571 | 38,710 |
| Name + Country | 172,161 | 488,480 |
| Address + Country | 37,571 | 38,710 |

---

## 4. Interpretation

### What the Numbers Tell Us

1. **Modest recall (30.80%)**: Exact-match blocking on normalized fields misses ~69% of true matches. This is expected -- real-world business records have many variations that survive normalization but still represent the same entity (abbreviations, typos, reordering, etc.).

2. **21% of S1 entities get zero candidates**: These entities have no exact normalized match on name, address, name+country, or address+country with any S2/S3 entity. This could be due to:
   - Significant name/address differences between sources
   - Missing or inconsistent country fields
   - Data quality issues

3. **Median of 2 candidates per S1**: Most S1 entities that do match have very few candidates, which is good for downstream matching efficiency.

4. **Long tail (P95=64, max=476)**: A small number of S1 entities generate very large candidate sets, likely due to common business names or addresses.

5. **S2 recall (33.26%) slightly higher than S3 (28.49%)**: S2 may have more consistent formatting with S1, or the S2 data may be cleaner.

### Limitations

- **No fuzzy matching**: Any variation that survives normalization but differs in spelling, abbreviation, or word order will be missed.
- **No ML scoring**: All candidates are treated equally; there's no way to rank them by likelihood of being a true match.
- **No character n-grams**: Typos and minor variations are not captured.
- **Country dependency**: Strategies 3 and 4 require correct country fields; missing or wrong country values prevent matches.

### Next Steps

This baseline establishes the floor for blocking recall. Future improvements (fuzzy matching, ML scoring, character n-grams) will be layered on top to increase recall while managing candidate set size.

---

## 5. Artifacts

| File | Description |
|------|-------------|
| `artifacts/blocking/blocking_benchmark.json` | Machine-readable benchmark results |
| `docs/blocking_baseline.md` | This report |

---

## 6. Commands Run

```bash
# Run benchmark
PYTHONUNBUFFERED=1 uv run python scripts/run_blocking_benchmark.py

# Verify tests
uv run pytest -v
```

**Total benchmark runtime**: ~352 seconds
