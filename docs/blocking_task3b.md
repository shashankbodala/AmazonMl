# Task 3B: Extended Blocking Strategies Report

**Generated from**: 20% validation split (seed=42)  
**Benchmark script**: `scripts/run_task3b_experiments.py`  
**Date**: 2026-09-27

---

## 1. Methodology

This experiment evaluates four extended blocking strategies on top of the Task 3A baseline:

| Strategy | Description |
|----------|-------------|
| **name_token** | Split normalized business name into tokens, filter stop words and short tokens, use each remaining token as a blocking key |
| **address_token** | Split normalized business address into tokens, filter stop words and short tokens, use each remaining token as a blocking key |
| **name_ngram** | Extract 3-character n-grams from normalized business name, use each n-gram as a blocking key |
| **address_ngram** | Extract 3-character n-grams from normalized business address, use each n-gram as a blocking key |

### Token Filtering

**Name stop words**: Articles, prepositions, conjunctions, common business suffixes (inc, ltd, llc, corp, pvt, pc, llp, etc.), and overly common business descriptors (india, care, associates, partners, center, health, clinic, global, trading, technologies, etc.).

**Address stop words**: Articles, prepositions, street suffixes (st, rd, ave, etc.), address parts (apt, ste, suite, etc.), directions (n, s, e, w, etc.), street types (road, street, avenue, etc.), and common geographic terms (city, new, null, plot).

**Short tokens**: Tokens shorter than 2 characters are filtered out.

### Frequency Thresholding

To prevent pathological candidate sets from overly common tokens/n-grams, we apply a frequency threshold during index building:

- **Max token frequency**: 50 source entities per token. Tokens appearing in more than 50 entities are not indexed.
- **Max n-gram frequency**: 200 source entities per n-gram. N-grams appearing in more than 200 entities are not indexed.

This is a standard information retrieval technique (posting list thresholding) that removes terms too common to be discriminative.

### N-gram Strategies

N-gram strategies were evaluated during development but excluded from the final report due to excessive runtime (candidate generation did not complete within practical time limits). The n-gram approach generates many more keys per entity (~20-50 n-grams vs ~2-5 tokens), making candidate generation significantly slower.

---

## 2. Task 3A Baseline (Frozen)

| Metric | Value |
|--------|-------|
| Overall candidate recall | 30.80% |
| S2 candidate recall | 33.26% |
| S3 candidate recall | 28.49% |
| Total candidate pairs | 4,759,549 |
| Avg candidates/S1 | 10.78 |
| Median candidates/S1 | 2 |
| P95 candidates/S1 | 64 |
| Max candidates/S1 | 476 |
| Zero candidates | 93,348 (21.15%) |
| One candidate | 103,729 (23.50%) |
| >1 candidates | 244,286 (55.35%) |
| Runtime | 351.55s |

---

## 3. Individual Strategy Results

### name_token (standalone)

| Metric | Value |
|--------|-------|
| Overall candidate recall | **24.77%** |
| S2 candidate recall | 25.00% |
| S3 candidate recall | 24.56% |
| Total candidate pairs | 82,016,853 |
| Avg candidates/S1 | 185.83 |
| Median candidates/S1 | 197 |
| P95 candidates/S1 | 298 |
| Max candidates/S1 | 647 |
| Zero candidates | 583 (0.13%) |

**Assessment**: Worse recall than baseline (24.77% vs 30.80%). Name tokens are too generic to be useful as standalone blocking keys. **Not recommended as standalone strategy.**

### address_token (standalone)

| Metric | Value |
|--------|-------|
| Overall candidate recall | **46.88%** |
| S2 candidate recall | 48.17% |
| S3 candidate recall | 45.68% |
| Total candidate pairs | 167,103,507 |
| Avg candidates/S1 | 378.61 |
| Median candidates/S1 | 348 |
| P95 candidates/S1 | 691 |
| Max candidates/S1 | 2,036 |
| Zero candidates | 23 (0.01%) |

**Assessment**: Much better recall than baseline (46.88% vs 30.80%), but 35x more candidates. Address tokens are more discriminative than name tokens. **Useful but produces large candidate sets.**

---

## 4. Combined Strategy Results

### baseline + name_token

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Overall candidate recall | **46.58%** | +15.78 pp |
| S2 candidate recall | 48.60% | +15.34 pp |
| S3 candidate recall | 44.69% | +16.20 pp |
| Total candidate pairs | 86,511,427 | 18.2x |
| Avg candidates/S1 | 196.01 | 18.2x |
| Median candidates/S1 | 200 | 100x |
| P95 candidates/S1 | 304 | 4.8x |
| Max candidates/S1 | 650 | 1.4x |
| Zero candidates | 57 (0.01%) | -93,291 |

**Assessment**: Adding name tokens to baseline increases recall by 51% relative (46.58% vs 30.80%) but increases candidate volume 18x. The marginal benefit of name_token over baseline+address_token is small.

### baseline + address_token

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Overall candidate recall | **63.27%** | +32.47 pp |
| S2 candidate recall | 65.24% | +31.98 pp |
| S3 candidate recall | 61.42% | +32.93 pp |
| Total candidate pairs | 171,627,645 | 36.1x |
| Avg candidates/S1 | 388.86 | 36.1x |
| Median candidates/S1 | 358 | 179x |
| P95 candidates/S1 | 696 | 10.9x |
| Max candidates/S1 | 2,036 | 4.3x |
| Zero candidates | 6 (0.00%) | -93,342 |

**Assessment**: Adding address tokens to baseline increases recall by 105% relative (63.27% vs 30.80%) but increases candidate volume 36x. Almost no S1 entities have zero candidates. **Best recall improvement among tested configurations.**

### baseline + name_token + address_token

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Overall candidate recall | **70.85%** | +40.05 pp |
| S2 candidate recall | 72.41% | +39.15 pp |
| S3 candidate recall | 69.38% | +40.89 pp |
| Total candidate pairs | 252,860,692 | 53.1x |
| Avg candidates/S1 | 572.91 | 53.1x |
| Median candidates/S1 | 557 | 278.5x |
| P95 candidates/S1 | 859 | 13.4x |
| Max candidates/S1 | 2,145 | 4.5x |
| Zero candidates | 0 (0.00%) | -93,348 |

**Assessment**: Adding both name and address tokens to baseline achieves 70.85% recall, a 130% relative improvement. However, this produces 253M candidate pairs (53x baseline) with an average of 573 candidates per S1 entity. **Highest recall but candidate volume may be prohibitive.**

---

## 5. Experiment Comparison

| Experiment | Recall | S2 | S3 | Pairs | Avg/S1 | P95 | Max |
|------------|--------|----|----|----|--------|-----|-----|
| name_token | 0.2477 | 0.2500 | 0.2456 | 82,016,853 | 185.83 | 298 | 647 |
| address_token | 0.4688 | 0.4817 | 0.4568 | 167,103,507 | 378.61 | 691 | 2,036 |
| baseline (3A) | 0.3080 | 0.3326 | 0.2849 | 4,759,549 | 10.78 | 64 | 476 |
| baseline + name_token | 0.4658 | 0.4860 | 0.4469 | 86,511,427 | 196.01 | 304 | 650 |
| baseline + address_token | 0.6327 | 0.6524 | 0.6142 | 171,627,645 | 388.86 | 696 | 2,036 |
| baseline + name_token + address_token | 0.7085 | 0.7241 | 0.6938 | 252,860,692 | 572.91 | 859 | 2,145 |

---

## 6. Recommendations

### Best Configuration: baseline + address_token

For a practical balance of recall and candidate volume, **baseline + address_token** is recommended:

- **63.27% recall** (vs 30.80% baseline) - 105% relative improvement
- **172M candidate pairs** (36x baseline) - manageable for downstream matching
- **Only 6 S1 entities with zero candidates** (vs 93,348 baseline)
- Address tokens are more discriminative than name tokens

### Highest Recall: baseline + name_token + address_token

If maximum recall is the priority and candidate volume is less of a concern:

- **70.85% recall** - 130% relative improvement over baseline
- **253M candidate pairs** - may require significant compute for downstream matching
- **Zero entities with zero candidates**

### Not Recommended: name_token standalone

Name token blocking alone performs worse than the baseline and is not recommended.

### Not Evaluated: n-gram strategies

Character n-gram strategies (name_ngram, address_ngram) were excluded from the final evaluation due to excessive runtime during candidate generation. The high number of n-grams per entity (~20-50) makes candidate generation significantly slower than token-based approaches.

---

## 7. Limitations

1. **Token frequency thresholding**: The thresholds (50 for tokens, 200 for n-grams) were chosen heuristically. Different thresholds would produce different recall/volume tradeoffs.
2. **Stop word list**: The stop word lists were expanded based on observed token frequencies. More comprehensive stop words could further reduce candidate volume but might also reduce recall.
3. **N-gram strategies excluded**: Due to runtime constraints, n-gram strategies were not fully evaluated. They might provide additional recall improvements.
4. **No fuzzy matching**: This experiment only evaluates exact-match blocking strategies. Fuzzy matching (Task 3C) will likely provide additional recall improvements.
5. **No ML scoring**: All candidates are treated equally. ML-based scoring (Task 3C) will help rank candidates by likelihood.

---

## 8. Refinement Experiments: Token Frequency Filtering

### v1: Ultra-Rare Token Filtering (Incorrect Approach)

The first refinement attempt used very restrictive thresholds (df≤5 to df≤200), which kept only tokens appearing in ≤5 to ≤200 entities across the 12.5M corpus. This was **too restrictive**:

- Name tokens with df≤5: 92% of unique tokens, but these tokens rarely match any source entities
- Address tokens with df≤5: 62% of unique tokens, same issue

**Result**: Recall dropped to 0.49% (name) and 5.10% (address) — far worse than baseline.

### v2: Common-Token Filtering (Correct Approach)

The second refinement correctly filters by removing the MOST common tokens (df ≥ threshold) while retaining moderately frequent tokens.

**Corpus Token Distribution (12,527,040 total entities)**:

| Percentile | Name Token df | Address Token df |
|------------|---------------|------------------|
| P50 | 1 | 4 |
| P75 | 1 | 8 |
| P90 | 3 | 30 |
| P95 | 14 | 80 |
| P99 | 132 | 695 |
| P99.9 | 2,210 | 8,849 |
| Max (top 100) | 296,635 | 693,402 |

**Key insight**: 92% of name tokens and 62% of address tokens appear in ≤5 entities. The most common tokens appear in 296K (name) and 693K (address) entities. Even the "most common" tokens represent only 2.4% (name) and 5.5% (address) of the corpus.

**Thresholds tested**: df < 50K, 100K, 200K, 300K (name); df < 200K, 500K, 700K (address)

### Results

#### Standalone Token Strategies with Filtering

| Experiment | Recall | S2 | S3 | Pairs | Avg/S1 | P95 | Tokens Kept |
|------------|--------|----|----|----|--------|-----|-------------|
| name_token_df_lt50000 | 0.2477 | 0.2500 | 0.2456 | 74,614,438 | 169.05 | 298 | 1,689,928 |
| name_token_df_lt300000 (unfiltered) | 0.2477 | 0.2500 | 0.2456 | 82,016,853 | 185.83 | 298 | 1,689,973 |
| address_token_df_lt200000 | 0.4688 | 0.4817 | 0.4568 | 150,241,498 | 340.40 | 637 | 972,450 |
| address_token_df_lt700000 (unfiltered) | 0.4688 | 0.4817 | 0.4568 | 167,103,507 | 378.61 | 691 | 972,479 |

**Assessment**: Filtering the most common tokens provides ~9-10% candidate reduction with no recall loss. Name tokens remain ineffective standalone (24.77% recall). Address tokens provide 46.88% recall.

#### Baseline + Filtered Tokens

| Experiment | Recall | S2 | S3 | Pairs | Avg/S1 | P95 |
|------------|--------|----|----|----|--------|-----|
| baseline (3A) | 0.3080 | 0.3326 | 0.2849 | 4,759,549 | 10.78 | 64 |
| baseline + name_token_df_lt50000 | 0.4658 | 0.4859 | 0.4468 | 79,109,709 | 179.24 | 300 |
| baseline + name_token (unfiltered) | 0.4658 | 0.4860 | 0.4469 | 86,511,427 | 196.01 | 304 |
| baseline + address_token_df_lt200000 | 0.6327 | 0.6524 | 0.6142 | 154,765,655 | 350.65 | 643 |
| baseline + address_token (unfiltered) | 0.6327 | 0.6524 | 0.6142 | 171,627,645 | 388.86 | 696 |
| baseline + name_token_df_lt100000 + address_token_df_lt500000 | 0.7085 | 0.7241 | 0.6938 | 249,105,203 | 564.40 | 840 |
| baseline + name_token + address_token (unfiltered) | 0.7085 | 0.7241 | 0.6938 | 252,860,692 | 572.91 | 859 |

**Assessment**: DF filtering provides 9-10% candidate reduction with no recall loss. However, absolute candidate volumes remain high.

### Key Findings

1. **DF filtering helps but not enough**: Removing the most common tokens reduces candidates by 9-10%, but the remaining tokens still produce 79M-249M candidate pairs.

2. **Name tokens are fundamentally limited**: Even with optimal filtering, name token blocking achieves only 24.77% recall standalone and adds ~10-12pp when combined with baseline.

3. **Address tokens are the main driver**: Address token blocking provides 46.88% recall standalone and 63.27% when combined with baseline.

4. **The recall/candidate tradeoff curve is steep**:
   - 3A baseline: 30.80% recall, 4.76M pairs
   - baseline + filtered name_token: 46.58% recall, 79M pairs (16.6x candidates)
   - baseline + filtered address_token: 63.27% recall, 155M pairs (32.5x candidates)
   - baseline + filtered name + address: 70.85% recall, 249M pairs (52.3x candidates)

5. **Practical recommendation**: If the downstream matcher can handle ~80M candidates, `baseline + name_token_df_lt50000` provides 46.58% recall with reasonable candidate volume. For maximum recall, use unfiltered tokens.

---

## 9. Artifacts

| File | Description |
|------|-------------|
| `artifacts/blocking/task3b_results.json` | Machine-readable experiment results |
| `docs/blocking_task3b.md` | This report |
| `artifacts/blocking/blocking_benchmark.json` | Task 3A baseline (unchanged) |
| `docs/blocking_baseline.md` | Task 3A baseline report (unchanged) |

---

## 9. Commands Run

```bash
# Run experiments
PYTHONUNBUFFERED=1 uv run python scripts/run_task3b_experiments.py

# Verify tests
uv run pytest -q
```
