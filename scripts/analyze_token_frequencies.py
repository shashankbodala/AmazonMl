#!/usr/bin/env python3
"""Analyze token frequencies in source corpus for informed blocking design."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from business_entity_resolution.normalization import make_name_key_series, make_address_key_series
from business_entity_resolution.blocking_strategies import (
    extract_name_tokens, extract_address_tokens,
    NAME_STOP_WORDS, ADDRESS_STOP_WORDS,
)

S1_COL = "entity_id"
S1_NAME_COL = "business_name"
S1_ADDRESS_COL = "business_address"
S1_COUNTRY_COL = "country"
S23_COLS = ["entity_id", "business_name", "business_address", "country"]
CHUNKSIZE = 200_000


def count_token_frequencies(source_path: Path, source_label: str) -> tuple:
    """Count token document frequencies across a source file."""
    name_token_counts = Counter()
    addr_token_counts = Counter()
    total_entities = 0

    print(f"  Counting tokens in {source_label}...")
    chunk_num = 0
    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        usecols=S23_COLS,
        dtype={col: str for col in S23_COLS},
        chunksize=CHUNKSIZE,
        keep_default_na=False,
        na_values=[""],
    ):
        chunk_num += 1
        if chunk_num % 10 == 0:
            print(f"    Processed {chunk_num} chunks ({chunk_num * CHUNKSIZE:,} rows)...")

        total_entities += len(chunk)

        norm_name = make_name_key_series(chunk[S1_NAME_COL])
        norm_addr = make_address_key_series(chunk[S1_ADDRESS_COL])

        name_tokens = extract_name_tokens(norm_name)
        addr_tokens = extract_address_tokens(norm_addr)

        # Count unique tokens per entity (document frequency)
        for token_list in name_tokens:
            name_token_counts.update(set(token_list))
        for token_list in addr_tokens:
            addr_token_counts.update(set(token_list))

    print(f"  Done: {total_entities:,} entities")
    return name_token_counts, addr_token_counts, total_entities


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s1-path", type=Path, default=Path("dataset/train/train_source1.tsv"))
    p.add_argument("--s2-path", type=Path, default=Path("dataset/train/train_source2.tsv"))
    p.add_argument("--s3-path", type=Path, default=Path("dataset/train/train_source3.tsv"))
    args = p.parse_args()

    all_name_tokens = Counter()
    all_addr_tokens = Counter()
    total_entities = 0

    for source_path, label in [
        (args.s1_path, "S1"),
        (args.s2_path, "S2"),
        (args.s3_path, "S3"),
    ]:
        name_counts, addr_counts, n_entities = count_token_frequencies(source_path, label)
        all_name_tokens.update(name_counts)
        all_addr_tokens.update(addr_counts)
        total_entities += n_entities

    print(f"\nTotal entities across all sources: {total_entities:,}")
    print(f"Unique name tokens: {len(all_name_tokens):,}")
    print(f"Unique address tokens: {len(all_addr_tokens):,}")

    # Analyze name token frequency distribution
    print(f"\n=== NAME TOKEN FREQUENCY DISTRIBUTION ===")
    name_freq_bins = {
        "1-5": 0, "6-10": 0, "11-20": 0, "21-50": 0,
        "51-100": 0, "101-200": 0, "201-500": 0, "501-1000": 0,
        "1001-5000": 0, "5001_plus": 0,
    }
    for token, freq in all_name_tokens.items():
        if freq <= 5:
            name_freq_bins["1-5"] += 1
        elif freq <= 10:
            name_freq_bins["6-10"] += 1
        elif freq <= 20:
            name_freq_bins["11-20"] += 1
        elif freq <= 50:
            name_freq_bins["21-50"] += 1
        elif freq <= 100:
            name_freq_bins["51-100"] += 1
        elif freq <= 200:
            name_freq_bins["101-200"] += 1
        elif freq <= 500:
            name_freq_bins["201-500"] += 1
        elif freq <= 1000:
            name_freq_bins["501-1000"] += 1
        elif freq <= 5000:
            name_freq_bins["1001-5000"] += 1
        else:
            name_freq_bins["5001_plus"] += 1

    print(f"{'Bin':<15} {'Tokens':>10} {'% of total':>12}")
    for bin_name, count in name_freq_bins.items():
        print(f"{bin_name:<15} {count:>10,} {count/len(all_name_tokens)*100:>11.2f}%")

    # Analyze address token frequency distribution
    print(f"\n=== ADDRESS TOKEN FREQUENCY DISTRIBUTION ===")
    addr_freq_bins = {
        "1-5": 0, "6-10": 0, "11-20": 0, "21-50": 0,
        "51-100": 0, "101-200": 0, "201-500": 0, "501-1000": 0,
        "1001-5000": 0, "5001_plus": 0,
    }
    for token, freq in all_addr_tokens.items():
        if freq <= 5:
            addr_freq_bins["1-5"] += 1
        elif freq <= 10:
            addr_freq_bins["6-10"] += 1
        elif freq <= 20:
            addr_freq_bins["11-20"] += 1
        elif freq <= 50:
            addr_freq_bins["21-50"] += 1
        elif freq <= 100:
            addr_freq_bins["51-100"] += 1
        elif freq <= 200:
            addr_freq_bins["101-200"] += 1
        elif freq <= 500:
            addr_freq_bins["201-500"] += 1
        elif freq <= 1000:
            addr_freq_bins["501-1000"] += 1
        elif freq <= 5000:
            addr_freq_bins["1001-5000"] += 1
        else:
            addr_freq_bins["5001_plus"] += 1

    print(f"{'Bin':<15} {'Tokens':>10} {'% of total':>12}")
    for bin_name, count in addr_freq_bins.items():
        print(f"{bin_name:<15} {count:>10,} {count/len(all_addr_tokens)*100:>11.2f}%")

    # Show most common name tokens
    print(f"\n=== TOP 30 MOST COMMON NAME TOKENS ===")
    for token, freq in all_name_tokens.most_common(30):
        print(f"  {token:<25} {freq:>8,} entities ({freq/total_entities*100:.2f}%)")

    # Show most common address tokens
    print(f"\n=== TOP 30 MOST COMMON ADDRESS TOKENS ===")
    for token, freq in all_addr_tokens.most_common(30):
        print(f"  {token:<25} {freq:>8,} entities ({freq/total_entities*100:.2f}%)")

    # Estimate candidate volume at different thresholds
    print(f"\n=== ESTIMATED CANDIDATE VOLUME AT DIFFERENT THRESHOLDS ===")
    for threshold in [5, 10, 20, 50, 100, 200, 500, 1000]:
        # Count entities covered by tokens at or below threshold
        name_tokens_ok = sum(1 for _, f in all_name_tokens.items() if f <= threshold)
        addr_tokens_ok = sum(1 for _, f in all_addr_tokens.items() if f <= threshold)
        # Estimate candidate volume: each token appears in ~freq entities
        # but we capped at threshold, so estimate = sum(min(freq, threshold)) for each token in validation
        # This is a rough estimate
        print(f"  Threshold {threshold:>5}: {name_tokens_ok:>8,} name tokens, {addr_tokens_ok:>8,} addr tokens")

    # Save token frequencies for use in experiments
    output_dir = Path("artifacts/blocking")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save top tokens and frequencies
    token_freq_data = {
        "total_entities": total_entities,
        "name_token_count": len(all_name_tokens),
        "addr_token_count": len(all_addr_tokens),
        "name_freq_bins": name_freq_bins,
        "addr_freq_bins": addr_freq_bins,
        "top_name_tokens": dict(all_name_tokens.most_common(100)),
        "top_addr_tokens": dict(all_addr_tokens.most_common(100)),
    }
    output_path = output_dir / "token_frequencies.json"
    output_path.write_text(json.dumps(token_freq_data, indent=2, sort_keys=True))
    print(f"\nSaved token frequencies to: {output_path}")


if __name__ == "__main__":
    main()
