import argparse
import ast
import difflib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_OVERRIDES = {
    "train_source1.tsv": {"id": None, "name": None, "address": None, "country": None},
    "train_source2.tsv": {"id": None, "name": None, "address": None, "country": None},
    "train_source3.tsv": {"id": None, "name": None, "address": None, "country": None},
    "test_source1.tsv": {"id": None, "name": None, "address": None, "country": None},
    "test_source2.tsv": {"id": None, "name": None, "address": None, "country": None},
    "test_source3.tsv": {"id": None, "name": None, "address": None, "country": None},
    "train_ground_truth.tsv": {"s1_id": None, "s2_id": None, "s3_id": None},
}

ID_CANDIDATES = [
    "id",
    "record_id",
    "entity_id",
    "row_id",
    "uid",
    "unique_id",
    "business_id",
    "s1_id",
    "source1_id",
    "poi_id",
]

NAME_CANDIDATES = [
    "business_name",
    "name",
    "company_name",
    "entity_name",
    "org_name",
    "organization_name",
    "poi_name",
]

ADDRESS_CANDIDATES = [
    "business_address",
    "address",
    "full_address",
    "company_address",
    "registered_address",
    "addr",
    "location_address",
]

COUNTRY_CANDIDATES = [
    "country",
    "country_code",
    "country_name",
    "nation",
    "ctry",
    "cc",
]

S1_ID_CANDIDATES = [
    "s1_id",
    "source1_id",
    "src1_id",
    "id_1",
    "left_id",
    "s1",
]

S2_ID_CANDIDATES = [
    "s2_id",
    "source2_id",
    "src2_id",
    "id_2",
    "right_id",
    "s2",
    "matched_id_2",
]

S3_ID_CANDIDATES = [
    "s3_id",
    "source3_id",
    "src3_id",
    "id_3",
    "s3",
    "matched_id_3",
]

LEGAL_SUFFIXES = [
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "gmbh",
    "ag",
    "kg",
    "ohg",
    "bv",
    "nv",
    "pty",
    "pte",
    "plc",
    "llp",
    "lp",
    "srl",
    "spa",
    "sl",
    "sa",
    "sas",
    "sarl",
    "eurl",
    "snc",
    "oy",
    "oyj",
    "ab",
    "aps",
    "pvt",
    "bhd",
    "kk",
]

_SUFFIX_ALT = "|".join(LEGAL_SUFFIXES)

LEGAL_SUFFIX_CONTAINS_RE = re.compile(
    r"\b(?:" + _SUFFIX_ALT + r")\b\.?",
    re.IGNORECASE,
)

LEGAL_SUFFIX_TRAILING_RE = re.compile(
    r"[,.]?\s*\b(?:" + _SUFFIX_ALT + r")\b\.?\s*$",
    re.IGNORECASE,
)

PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def log(msg):
    stamp = time.strftime("%H:%M:%S")
    print("[" + stamp + "] " + msg, flush=True)


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, set):
        return list(obj)

    return str(obj)


def tokenize_colname(col):
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(col))

    return [
        t
        for t in re.split(r"[^a-zA-Z0-9]+", s.lower())
        if t
    ]


def find_column(columns, exact_candidates, override=None):
    if override:
        if override in columns:
            return override

        log(
            "  WARNING: override column "
            + repr(override)
            + " not found in "
            + str(list(columns))
        )

    cols_lower = {}

    for c in columns:
        cols_lower[c.lower()] = c

    for cand in exact_candidates:
        if cand in cols_lower:
            return cols_lower[cand]

    for c in columns:
        toks = tokenize_colname(c)

        for cand in exact_candidates:
            cand_toks = cand.split("_")

            if all(t in toks for t in cand_toks):
                return c

    return None


def resolve_source_columns(columns, overrides):
    return {
        "id": find_column(
            columns,
            ID_CANDIDATES,
            overrides.get("id"),
        ),
        "name": find_column(
            columns,
            NAME_CANDIDATES,
            overrides.get("name"),
        ),
        "address": find_column(
            columns,
            ADDRESS_CANDIDATES,
            overrides.get("address"),
        ),
        "country": find_column(
            columns,
            COUNTRY_CANDIDATES,
            overrides.get("country"),
        ),
    }


def detect_id_list_format(values):
    vals = [
        str(v).strip()
        for v in values
        if (
            v is not None
            and str(v).strip() != ""
            and str(v).strip().lower() != "nan"
        )
    ]

    if not vals:
        return None

    sample = vals[:2000]

    bracket_like = sum(
        1
        for v in sample
        if v.startswith("[") and v.endswith("]")
    )

    if bracket_like / len(sample) > 0.3:
        return "list_literal"

    for delim in [",", ";", "|", " "]:
        hits = sum(
            1
            for v in sample
            if delim in v
        )

        if hits / len(sample) > 0.05:
            return delim

    return None


def parse_id_list(raw, fmt):
    if raw is None:
        return []

    s = str(raw).strip()

    if not s or s.lower() == "nan":
        return []

    if fmt == "list_literal":
        try:
            parsed = ast.literal_eval(s)

            if isinstance(parsed, (list, tuple, set)):
                out = [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip() != ""
                ]

                return out

        except Exception:
            pass

        inner = s.strip("[]() ")
        parts = re.split(r"[,\s]+", inner)

        return [
            p.strip(" '\"")
            for p in parts
            if p.strip(" '\"") != ""
        ]

    if fmt is None:
        return [s]

    parts = s.split(fmt)

    return [
        p.strip()
        for p in parts
        if p.strip() != ""
    ]


def compute_id_list_fmt(cols, sample_df):
    fmt = {}

    for c in cols:
        if not c:
            continue

        if sample_df is not None and c in sample_df.columns:
            fmt[c] = detect_id_list_format(
                sample_df[c].tolist()
            )
        else:
            fmt[c] = None

    return fmt


def resolve_gt_columns(columns, overrides, sample_df=None):
    s1 = find_column(
        columns,
        S1_ID_CANDIDATES,
        overrides.get("s1_id"),
    )

    s2 = find_column(
        columns,
        S2_ID_CANDIDATES,
        overrides.get("s2_id"),
    )

    s3 = find_column(
        columns,
        S3_ID_CANDIDATES,
        overrides.get("s3_id"),
    )

    warnings = []

    other_cols = [
        c
        for c in columns
        if c != s1
    ]

    if s2 and s3 and s2 != s3:
        id_list_fmt = compute_id_list_fmt(
            [s2, s3],
            sample_df,
        )

        warnings.append(
            "match-id list formats detected: "
            + str(id_list_fmt)
        )

        return {
            "mode": "wide",
            "s1_id": s1,
            "s2_id": s2,
            "s3_id": s3,
            "match_cols": [s2, s3],
            "match_id_col": None,
            "label_col": None,
            "label_map": None,
            "id_list_fmt": id_list_fmt,
            "warnings": warnings,
        }

    id_like = []
    label_like = []

    n = len(sample_df) if sample_df is not None else 0

    for c in other_cols:
        if c in (s2, s3):
            id_like.append(c)
            continue

        if (
            sample_df is None
            or c not in sample_df.columns
            or n == 0
        ):
            id_like.append(c)
            continue

        nunique = sample_df[c].nunique(dropna=True)
        frac = nunique / n

        if nunique <= 10 and frac <= 0.5:
            label_like.append(c)
        else:
            id_like.append(c)

    if len(id_like) >= 2:
        chosen_s2 = s2 or id_like[0]

        remaining = [
            c
            for c in id_like
            if c != chosen_s2
        ]

        chosen_s3 = (
            s3
            or (remaining[0] if remaining else None)
        )

        warnings.append(
            "no exact S2/S3 column names matched; guessed "
            "match columns "
            + repr(chosen_s2)
            + " (s2) and "
            + repr(chosen_s3)
            + " (s3) by elimination. Verify with "
            "COLUMN_OVERRIDES if this looks wrong."
        )

        id_list_fmt = compute_id_list_fmt(
            [chosen_s2, chosen_s3],
            sample_df,
        )

        warnings.append(
            "match-id list formats detected: "
            + str(id_list_fmt)
        )

        return {
            "mode": "wide",
            "s1_id": s1,
            "s2_id": chosen_s2,
            "s3_id": chosen_s3,
            "match_cols": [
                c
                for c in (chosen_s2, chosen_s3)
                if c
            ],
            "match_id_col": None,
            "label_col": None,
            "label_map": None,
            "id_list_fmt": id_list_fmt,
            "warnings": warnings,
        }

    if len(id_like) == 1 and label_like:
        match_col = id_like[0]
        label_col = label_like[0]

        if sample_df is not None:
            vals = (
                sample_df[label_col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
        else:
            vals = []

        label_map = {}

        for v in vals:
            vl = v.lower()

            if "3" in vl:
                label_map[v] = "s3"
            elif "2" in vl:
                label_map[v] = "s2"
            else:
                label_map[v] = "unknown"

        id_list_fmt = compute_id_list_fmt(
            [match_col],
            sample_df,
        )

        warnings.append(
            "detected long-format ground truth: "
            "match-id column "
            + repr(match_col)
            + ", source-label column "
            + repr(label_col)
            + ", label values mapped as "
            + str(label_map)
            + ", match-id list format "
            + str(id_list_fmt.get(match_col))
            + ". Verify by eye if this looks wrong."
        )

        return {
            "mode": "long_labeled",
            "s1_id": s1,
            "match_id_col": match_col,
            "label_col": label_col,
            "label_map": label_map,
            "match_cols": [
                match_col,
                label_col,
            ],
            "id_list_fmt": id_list_fmt,
            "warnings": warnings,
        }

    if len(id_like) == 1:
        id_list_fmt = compute_id_list_fmt(
            [id_like[0]],
            sample_df,
        )

        warnings.append(
            "only one match-id column detected "
            + repr(id_like[0])
            + " and no source-label column; "
            "total match counts will be correct but "
            "S2 vs S3 cannot be told apart. "
            "Match-id list format detected: "
            + str(id_list_fmt.get(id_like[0]))
            + "."
        )

        return {
            "mode": "long_unlabeled",
            "s1_id": s1,
            "match_id_col": id_like[0],
            "match_cols": id_like,
            "label_col": None,
            "label_map": None,
            "id_list_fmt": id_list_fmt,
            "warnings": warnings,
        }

    warnings.append(
        "could not confidently detect any S2/S3 "
        "match column in ground truth; set "
        "COLUMN_OVERRIDES by hand."
    )

    return {
        "mode": "none",
        "s1_id": s1,
        "match_cols": [],
        "match_id_col": None,
        "label_col": None,
        "label_map": None,
        "id_list_fmt": {},
        "warnings": warnings,
    }


def normalize_text(s):
    if s is None:
        return ""

    s = str(s)

    if s.lower() == "nan":
        return ""

    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def strip_legal_suffix(name):
    if not name:
        return ""

    s = str(name)
    prev = None

    while prev != s:
        prev = s
        s = LEGAL_SUFFIX_TRAILING_RE.sub(
            "",
            s,
        ).strip(" ,.")

    return s


def char_similarity(a, b):
    return difflib.SequenceMatcher(
        None,
        a or "",
        b or "",
    ).ratio()


def classify_name_pair(name_a, name_b):
    na_raw = str(name_a or "")
    nb_raw = str(name_b or "")

    na = normalize_text(na_raw)
    nb = normalize_text(nb_raw)

    if not na or not nb:
        return "missing_field"

    if na == nb:
        return "exact_match"

    sa = normalize_text(
        strip_legal_suffix(na_raw)
    )

    sb = normalize_text(
        strip_legal_suffix(nb_raw)
    )

    if sa == sb:
        return "legal_suffix_diff"

    toks_a = set(na.split())
    toks_b = set(nb.split())

    if toks_a == toks_b:
        return "token_reorder"

    has_non_ascii = bool(
        NON_ASCII_RE.search(na_raw)
        or NON_ASCII_RE.search(nb_raw)
    )

    ratio = char_similarity(na, nb)

    if has_non_ascii and ratio < 0.6:
        return "possible_transliteration"

    if ratio >= 0.90:
        return "minor_typo"

    overlap = (
        len(toks_a & toks_b)
        / max(1, len(toks_a | toks_b))
    )

    if overlap >= 0.5:
        return "abbreviation_or_partial_overlap"

    return "low_similarity_other"


class Reservoir:
    def __init__(self, k):
        self.k = k
        self.items = []
        self.n = 0

    def add(self, item):
        self.n += 1

        if len(self.items) < self.k:
            self.items.append(item)

        else:
            j = random.randint(
                0,
                self.n - 1,
            )

            if j < self.k:
                self.items[j] = item


class RunningStats:
    def __init__(self, sample_cap):
        self.count = 0
        self.total = 0
        self.total_sq = 0
        self.vmin = None
        self.vmax = None
        self.sample_cap = sample_cap
        self.sample_chunks = []
        self.sampled = 0

    def update(self, arr):
        if arr.size == 0:
            return

        arr = arr.astype(np.int64)

        n = arr.size

        self.count += n
        self.total += int(arr.sum())
        self.total_sq += int(
            (arr ** 2).sum()
        )

        cmin = int(arr.min())
        cmax = int(arr.max())

        self.vmin = (
            cmin
            if self.vmin is None
            else min(self.vmin, cmin)
        )

        self.vmax = (
            cmax
            if self.vmax is None
            else max(self.vmax, cmax)
        )

        if self.sampled < self.sample_cap:
            self.sample_chunks.append(arr)
            self.sampled += n

    def summary(self):
        if self.count == 0:
            return {"count": 0}

        mean = self.total / self.count

        var = max(
            0.0,
            self.total_sq / self.count - mean ** 2,
        )

        std = var ** 0.5

        out = {
            "count": self.count,
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": self.vmin,
            "max": self.vmax,
            "percentiles_approx": self.sampled < self.count,
            "percentile_sample_size": self.sampled,
        }

        if self.sample_chunks:
            all_vals = np.concatenate(
                self.sample_chunks
            )

            p = np.percentile(
                all_vals,
                [25, 50, 75, 90, 95, 99],
            )

            out["p25"] = float(p[0])
            out["p50"] = float(p[1])
            out["p75"] = float(p[2])
            out["p90"] = float(p[3])
            out["p95"] = float(p[4])
            out["p99"] = float(p[5])

        return out


class NoiseAccum:
    def __init__(self, is_name_col):
        self.n = 0
        self.non_ascii = 0
        self.has_digit = 0
        self.has_legal_suffix = (
            0
            if is_name_col
            else None
        )
        self.all_caps = 0
        self.multi_space = 0
        self.lead_trail_space = 0
        self.token_count_sum = 0
        self.is_name_col = is_name_col

    def update(self, s):
        s = s.fillna("").astype(str)

        n = len(s)

        if n == 0:
            return

        self.n += n

        self.non_ascii += int(
            s.str.contains(
                NON_ASCII_RE,
                regex=True,
            ).sum()
        )

        self.has_digit += int(
            s.str.contains(
                r"\d",
                regex=True,
            ).sum()
        )

        if self.is_name_col:
            self.has_legal_suffix += int(
                s.str.contains(
                    LEGAL_SUFFIX_CONTAINS_RE,
                    regex=True,
                ).sum()
            )

        self.all_caps += int(
            s.str.isupper()
            .fillna(False)
            .sum()
        )

        self.multi_space += int(
            s.str.contains(
                r"\s{2,}",
                regex=True,
            ).sum()
        )

        self.lead_trail_space += int(
            s.str.contains(
                r"^\s|\s$",
                regex=True,
            ).sum()
        )

        self.token_count_sum += int(
            s.str.split()
            .str.len()
            .fillna(0)
            .sum()
        )

    def summary(self):
        if self.n == 0:
            return {"count": 0}

        out = {
            "count": self.n,
            "pct_non_ascii": round(
                100 * self.non_ascii / self.n,
                2,
            ),
            "pct_has_digit": round(
                100 * self.has_digit / self.n,
                2,
            ),
            "pct_all_caps": round(
                100 * self.all_caps / self.n,
                2,
            ),
            "pct_multiple_spaces": round(
                100 * self.multi_space / self.n,
                2,
            ),
            "pct_leading_or_trailing_space": round(
                100 * self.lead_trail_space / self.n,
                2,
            ),
            "avg_token_count": round(
                self.token_count_sum / self.n,
                2,
            ),
        }

        if self.is_name_col:
            out["pct_has_legal_suffix"] = round(
                100 * self.has_legal_suffix / self.n,
                2,
            )

        return out


def peek_schema(path):
    try:
        sample = pd.read_csv(
            path,
            sep="\t",
            nrows=5000,
            on_bad_lines="skip",
            engine="python",
        )

    except Exception as e:
        log(
            "  WARNING: could not read sample of "
            + str(path)
            + ": "
            + str(e)
        )

        return [], {}, None

    dtypes = {}

    for c, t in sample.dtypes.items():
        dtypes[c] = str(t)

    return (
        list(sample.columns),
        dtypes,
        sample,
    )


def analyze_source_file(
    path,
    overrides,
    args,
    want_ids=None,
    collect_all_ids=False,
    hard_neg_pool_size=0,
):
    fname = path.name

    columns, dtypes_sample, _schema_sample = peek_schema(
        path
    )

    if not columns:
        return (
            {"error": "could not read " + str(path)},
            None,
            {},
            None,
        )

    resolved = resolve_source_columns(
        columns,
        overrides,
    )

    log(
        "["
        + fname
        + "] columns: "
        + str(columns)
    )

    log(
        "["
        + fname
        + "] resolved -> id="
        + repr(resolved["id"])
        + " name="
        + repr(resolved["name"])
        + " address="
        + repr(resolved["address"])
        + " country="
        + repr(resolved["country"])
    )

    warnings = []

    for f in (
        "id",
        "name",
        "address",
        "country",
    ):
        if resolved[f] is None:
            warnings.append(
                "could not detect a "
                + f
                + " column; related stats skipped"
            )

    if args.dry_run:
        result = {
            "columns": columns,
            "inferred_dtypes_sample": dtypes_sample,
            "resolved_columns": resolved,
            "warnings": warnings,
        }

        return (
            result,
            None,
            {},
            None,
        )

    id_col = resolved["id"]
    name_col = resolved["name"]
    addr_col = resolved["address"]
    country_col = resolved["country"]

    row_count = 0

    missing_counts = Counter()
    country_counter = Counter()

    seen_ids = set()
    dup_ids = set()

    name_stats = RunningStats(
        args.length_sample_cap
    )

    addr_stats = RunningStats(
        args.length_sample_cap
    )

    name_noise = NoiseAccum(
        is_name_col=True
    )

    addr_noise = NoiseAccum(
        is_name_col=False
    )

    all_ids = (
        set()
        if collect_all_ids
        else None
    )

    want_rows_out = {}

    want_ids = want_ids or set()

    hn_reservoir = (
        Reservoir(hard_neg_pool_size)
        if hard_neg_pool_size > 0
        else None
    )

    bad_chunks = 0

    need_row_capture = (
        bool(want_ids)
        or hn_reservoir is not None
    )

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=args.chunksize,
        keep_default_na=True,
        na_values=[
            "",
            "NA",
            "N/A",
            "null",
            "NULL",
            "None",
            "none",
        ],
        on_bad_lines="skip",
        engine="c",
    )

    for chunk in reader:
        try:
            n = len(chunk)

            row_count += n

            for col in columns:
                if col not in chunk.columns:
                    continue

                s = chunk[col]

                miss_mask = (
                    s.isna()
                    | s.fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                )

                missing_counts[col] += int(
                    miss_mask.sum()
                )

            if id_col and id_col in chunk.columns:
                ids = (
                    chunk[id_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                id_list = ids.tolist()

                for v in id_list:
                    if not v:
                        continue

                    if v in seen_ids:
                        dup_ids.add(v)
                    else:
                        seen_ids.add(v)

                if collect_all_ids:
                    for v in id_list:
                        if v:
                            all_ids.add(v)

            else:
                id_list = [None] * n

            name_series = (
                chunk[name_col]
                if name_col
                and name_col in chunk.columns
                else None
            )

            addr_series = (
                chunk[addr_col]
                if addr_col
                and addr_col in chunk.columns
                else None
            )

            country_series = (
                chunk[country_col]
                if country_col
                and country_col in chunk.columns
                else None
            )

            if country_series is not None:
                vals = (
                    country_series
                    .fillna("<missing>")
                    .astype(str)
                    .str.strip()
                )

                vals = vals.replace(
                    "",
                    "<missing>",
                )

                country_counter.update(
                    vals.tolist()
                )

            if name_series is not None:
                name_clean = (
                    name_series
                    .fillna("")
                    .astype(str)
                )

                name_lengths = (
                    name_clean
                    .str.len()
                    .to_numpy()
                )

                name_stats.update(
                    name_lengths
                )

                name_noise.update(
                    name_series
                )

            if addr_series is not None:
                addr_clean = (
                    addr_series
                    .fillna("")
                    .astype(str)
                )

                addr_lengths = (
                    addr_clean
                    .str.len()
                    .to_numpy()
                )

                addr_stats.update(
                    addr_lengths
                )

                addr_noise.update(
                    addr_series
                )

            if need_row_capture and id_col:
                name_list = (
                    name_series.tolist()
                    if name_series is not None
                    else [None] * n
                )

                addr_list = (
                    addr_series.tolist()
                    if addr_series is not None
                    else [None] * n
                )

                country_list = (
                    country_series.tolist()
                    if country_series is not None
                    else [None] * n
                )

                for (
                    rid,
                    nm,
                    ad,
                    ctry,
                ) in zip(
                    id_list,
                    name_list,
                    addr_list,
                    country_list,
                ):
                    if not rid:
                        continue

                    rec = {
                        "id": rid,
                        "name": nm,
                        "address": ad,
                        "country": ctry,
                    }

                    if rid in want_ids:
                        want_rows_out[rid] = rec

                    if hn_reservoir is not None:
                        hn_reservoir.add(rec)

        except Exception as e:
            bad_chunks += 1

            log(
                "  WARNING: error processing a chunk "
                "of "
                + fname
                + ": "
                + str(e)
            )

            continue

    missing_pct = {}

    for c in columns:
        if row_count:
            missing_pct[c] = round(
                100
                * missing_counts.get(c, 0)
                / row_count,
                2,
            )
        else:
            missing_pct[c] = None

    duplicate_ids = None

    if id_col:
        duplicate_ids = {
            "distinct_ids_seen": len(seen_ids),
            "distinct_ids_with_duplicates": len(
                dup_ids
            ),
            "duplicate_row_overhang": max(
                0,
                row_count - len(seen_ids),
            ),
        }

    result = {
        "path": str(path),
        "columns": columns,
        "inferred_dtypes_sample": dtypes_sample,
        "resolved_columns": resolved,
        "row_count": row_count,
        "missing_pct": missing_pct,
        "duplicate_ids": duplicate_ids,
        "country_distribution_top200": (
            dict(
                country_counter.most_common(200)
            )
            if country_col
            else None
        ),
        "country_cardinality": (
            len(country_counter)
            if country_col
            else None
        ),
        "name_length": (
            name_stats.summary()
            if name_col
            else None
        ),
        "address_length": (
            addr_stats.summary()
            if addr_col
            else None
        ),
        "name_noise": (
            name_noise.summary()
            if name_col
            else None
        ),
        "address_noise": (
            addr_noise.summary()
            if addr_col
            else None
        ),
        "bad_chunks_skipped": bad_chunks,
        "warnings": warnings,
    }

    hn_items = (
        hn_reservoir.items
        if hn_reservoir
        else None
    )

    return (
        result,
        all_ids,
        want_rows_out,
        hn_items,
    )


def process_ground_truth(
    path,
    overrides,
    args,
    sample_target=400,
):
    fname = path.name

    columns, dtypes_sample, schema_sample = peek_schema(
        path
    )

    if not columns:
        return (
            {"error": "could not read " + str(path)},
            {},
            [],
        )

    resolved = resolve_gt_columns(
        columns,
        overrides,
        schema_sample,
    )

    log(
        "["
        + fname
        + "] columns: "
        + str(columns)
    )

    log(
        "["
        + fname
        + "] resolved -> mode="
        + str(resolved["mode"])
        + " s1_id="
        + repr(resolved["s1_id"])
        + " match_cols="
        + str(resolved["match_cols"])
    )

    for w in resolved["warnings"]:
        log("  NOTE: " + w)

    warnings = list(
        resolved["warnings"]
    )

    if resolved["s1_id"] is None:
        warnings.append(
            "could not detect the S1-id column "
            "in ground truth; match distribution "
            "will be empty"
        )

    if args.dry_run:
        result = {
            "columns": columns,
            "inferred_dtypes_sample": dtypes_sample,
            "resolved_columns": resolved,
            "warnings": warnings,
        }

        return (
            result,
            {},
            [],
        )

    s1_col = resolved["s1_id"]
    mode = resolved["mode"]

    row_count = 0

    match_counts = defaultdict(int)
    reservoir = Reservoir(sample_target)

    bad_chunks = 0

    if not s1_col or mode == "none":
        result = {
            "path": str(path),
            "columns": columns,
            "row_count": None,
            "resolved_columns": resolved,
            "warnings": warnings,
        }

        return (
            result,
            {},
            [],
        )

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=args.chunksize,
        keep_default_na=True,
        na_values=[
            "",
            "NA",
            "N/A",
            "null",
            "NULL",
            "None",
            "none",
        ],
        on_bad_lines="skip",
        engine="c",
    )

    for chunk in reader:
        try:
            row_count += len(chunk)

            if s1_col not in chunk.columns:
                continue

            s1_vals = (
                chunk[s1_col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            id_list_fmt = resolved.get(
                "id_list_fmt",
                {},
            )

            if mode == "wide":
                match_cols = [
                    c
                    for c in resolved["match_cols"]
                    if c in chunk.columns
                ]

                if not match_cols:
                    continue

                s2_col = resolved["s2_id"]
                s3_col = resolved["s3_id"]

                s1_list = s1_vals.tolist()

                for c in match_cols:
                    if c == s2_col:
                        which = "s2"
                    elif c == s3_col:
                        which = "s3"
                    else:
                        which = "unknown"

                    fmt = id_list_fmt.get(c)
                    raw_list = chunk[c].tolist()

                    for (
                        s1v,
                        raw,
                    ) in zip(
                        s1_list,
                        raw_list,
                    ):
                        if not s1v:
                            continue

                        ids = parse_id_list(
                            raw,
                            fmt,
                        )

                        if not ids:
                            continue

                        match_counts[s1v] += len(ids)

                        for mid in ids:
                            mid_str = str(mid).strip()

                            if mid_str.startswith("S2-"):
                                which_source = "s2"
                            elif mid_str.startswith("S3-"):
                                which_source = "s3"
                            else:
                                which_source = "unknown"

                            reservoir.add(
                                {
                                    "s1_id": s1v,
                                    "which_source": which_source,
                                    "match_id": mid_str,
                                }
                            )

            elif mode == "long_labeled":
                match_col = resolved[
                    "match_id_col"
                ]

                label_col = resolved[
                    "label_col"
                ]

                if (
                    match_col not in chunk.columns
                    or label_col not in chunk.columns
                ):
                    continue

                fmt = id_list_fmt.get(
                    match_col
                )

                raw_list = chunk[
                    match_col
                ].tolist()

                label_vals = (
                    chunk[label_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                label_map = resolved[
                    "label_map"
                ]

                for (
                    s1v,
                    raw,
                    lv,
                ) in zip(
                    s1_vals.tolist(),
                    raw_list,
                    label_vals.tolist(),
                ):
                    if not s1v:
                        continue

                    ids = parse_id_list(
                        raw,
                        fmt,
                    )

                    if not ids:
                        continue

                    which = label_map.get(
                        lv,
                        "unknown",
                    )

                    match_counts[s1v] += len(ids)

                    for mid in ids:
                        reservoir.add(
                            {
                                "s1_id": s1v,
                                "which_source": which,
                                "match_id": mid,
                            }
                        )

            elif mode == "long_unlabeled":
                match_col = resolved[
                    "match_id_col"
                ]

                if match_col not in chunk.columns:
                    continue

                fmt = id_list_fmt.get(
                    match_col
                )

                raw_list = chunk[
                    match_col
                ].tolist()

                for (
                    s1v,
                    raw,
                ) in zip(
                    s1_vals.tolist(),
                    raw_list,
                ):
                    if not s1v:
                        continue

                    ids = parse_id_list(
                        raw,
                        fmt,
                    )

                    if not ids:
                        continue

                    match_counts[s1v] += len(ids)

                    for mid in ids:
                        reservoir.add(
                            {
                                "s1_id": s1v,
                                "which_source": "unknown",
                                "match_id": mid,
                            }
                        )

        except Exception as e:
            bad_chunks += 1

            log(
                "  WARNING: error processing a chunk "
                "of "
                + fname
                + ": "
                + str(e)
            )

            continue

    result = {
        "path": str(path),
        "columns": columns,
        "inferred_dtypes_sample": dtypes_sample,
        "resolved_columns": resolved,
        "row_count": row_count,
        "bad_chunks_skipped": bad_chunks,
        "warnings": warnings,
    }

    return (
        result,
        dict(match_counts),
        reservoir.items,
    )


def mine_hard_negatives(
    pool,
    source_label,
    sim_threshold=0.82,
    max_block_size=250,
    max_results=150,
    max_comparisons=3000000,
):
    blocks = defaultdict(list)

    for rec in pool:
        norm = normalize_text(
            strip_legal_suffix(
                str(rec.get("name") or "")
            )
        )

        if len(norm) < 3:
            continue

        key = (
            str(rec.get("country") or ""),
            norm[:4],
        )

        blocks[key].append(rec)

    results = []
    comparisons = 0

    block_items = list(
        blocks.items()
    )

    random.shuffle(block_items)

    for key, items in block_items:
        if len(items) < 2:
            continue

        if len(items) > max_block_size:
            items = random.sample(
                items,
                max_block_size,
            )

        for i in range(len(items)):
            for j in range(
                i + 1,
                len(items),
            ):
                comparisons += 1

                if comparisons > max_comparisons:
                    return results

                a = items[i]
                b = items[j]

                if a["id"] == b["id"]:
                    continue

                na = normalize_text(
                    a.get("name")
                )

                nb = normalize_text(
                    b.get("name")
                )

                if not na or not nb:
                    continue

                sim = char_similarity(
                    na,
                    nb,
                )

                if sim >= sim_threshold:
                    addr_sim = char_similarity(
                        normalize_text(
                            a.get("address")
                        ),
                        normalize_text(
                            b.get("address")
                        ),
                    )

                    results.append(
                        {
                            "source": source_label,
                            "id_a": a["id"],
                            "name_a": a.get("name"),
                            "address_a": a.get("address"),
                            "country_a": a.get("country"),
                            "id_b": b["id"],
                            "name_b": b.get("name"),
                            "address_b": b.get("address"),
                            "country_b": b.get("country"),
                            "name_similarity": round(
                                sim,
                                3,
                            ),
                            "address_similarity": round(
                                addr_sim,
                                3,
                            ),
                        }
                    )

                    if len(results) >= max_results:
                        return results

    return results


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset-dir",
        default="dataset",
    )

    ap.add_argument(
        "--output-dir",
        default="eda_report",
    )

    ap.add_argument(
        "--chunksize",
        type=int,
        default=200000,
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "only detect and print schema for "
            "all 7 files, write nothing, then exit"
        ),
    )

    ap.add_argument(
        "--inspect-gt",
        action="store_true",
        help=(
            "read 20 rows of "
            "train_ground_truth.tsv, print how "
            "matched ids are parsed, then exit"
        ),
    )

    ap.add_argument(
        "--positive-sample-size",
        type=int,
        default=250,
    )

    ap.add_argument(
        "--hard-neg-pool-size",
        type=int,
        default=15000,
        help=(
            "rows reservoir-sampled per source "
            "file for hard-negative mining"
        ),
    )

    ap.add_argument(
        "--hard-neg-max-results",
        type=int,
        default=150,
        help="per source file",
    )

    ap.add_argument(
        "--length-sample-cap",
        type=int,
        default=20000000,
        help=(
            "cap on rows kept for exact "
            "percentile calc per column"
        ),
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()

    ds = Path(args.dataset_dir)
    out = Path(args.output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "train_source1.tsv":
            ds / "train" / "train_source1.tsv",

        "train_source2.tsv":
            ds / "train" / "train_source2.tsv",

        "train_source3.tsv":
            ds / "train" / "train_source3.tsv",

        "train_ground_truth.tsv":
            ds / "train" / "train_ground_truth.tsv",

        "test_source1.tsv":
            ds / "test" / "test_source1.tsv",

        "test_source2.tsv":
            ds / "test" / "test_source2.tsv",

        "test_source3.tsv":
            ds / "test" / "test_source3.tsv",
    }

    if args.inspect_gt:
        gt_path = paths[
            "train_ground_truth.tsv"
        ]

        if not gt_path.exists():
            log(
                "ERROR: "
                + str(gt_path)
                + " not found."
            )
            return

        sample = pd.read_csv(
            gt_path,
            sep="\t",
            dtype=str,
            nrows=20,
        )

        columns = list(
            sample.columns
        )

        resolved = resolve_gt_columns(
            columns,
            COLUMN_OVERRIDES[
                "train_ground_truth.tsv"
            ],
            sample,
        )

        log(
            "columns: "
            + str(columns)
        )

        log(
            "resolved -> mode="
            + str(resolved["mode"])
            + " s1_id="
            + repr(resolved["s1_id"])
            + " match_cols="
            + str(resolved["match_cols"])
            + " id_list_fmt="
            + str(
                resolved.get(
                    "id_list_fmt"
                )
            )
        )

        for w in resolved["warnings"]:
            log(
                "  NOTE: "
                + w
            )

        s1_col = resolved[
            "s1_id"
        ]

        id_list_fmt = resolved.get(
            "id_list_fmt",
            {},
        )

        shown = 0

        for _, row in sample.iterrows():
            if shown >= 10:
                break

            s1v = (
                row.get(s1_col)
                if s1_col
                else None
            )

            if resolved["mode"] == "wide":
                parts = []
                total = 0

                for c in resolved[
                    "match_cols"
                ]:
                    raw = row.get(c)

                    ids = parse_id_list(
                        raw,
                        id_list_fmt.get(c),
                    )

                    total += len(ids)

                    parts.append(
                        c
                        + "="
                        + repr(raw)
                        + " -> "
                        + str(ids)
                    )

                log(
                    "s1_id="
                    + str(s1v)
                    + " | "
                    + " | ".join(parts)
                    + " | total_matches="
                    + str(total)
                )

            elif resolved["mode"] in (
                "long_labeled",
                "long_unlabeled",
            ):
                match_col = resolved.get(
                    "match_id_col"
                )

                raw = (
                    row.get(match_col)
                    if match_col
                    else None
                )

                ids = parse_id_list(
                    raw,
                    id_list_fmt.get(
                        match_col
                    ),
                )

                log(
                    "s1_id="
                    + str(s1v)
                    + " raw="
                    + repr(raw)
                    + " parsed_ids="
                    + str(ids)
                    + " count="
                    + str(len(ids))
                )

            else:
                log(
                    "s1_id="
                    + str(s1v)
                    + " mode=none, "
                    "no match column resolved"
                )

            shown += 1

        log(
            "Inspection complete. "
            "No output files written. "
            "Re-run without --inspect-gt "
            "for the full EDA."
        )

        return

    missing_files = [
        str(p)
        for p in paths.values()
        if not p.exists()
    ]

    if missing_files:
        log(
            "ERROR: the following expected "
            "files were not found:"
        )

        for m in missing_files:
            log(
                "  - "
                + m
            )

        log(
            "Fix --dataset-dir or the "
            "file layout and rerun."
        )

        return

    summary = {
        "generated_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        "args":
            vars(args),
        "files": {},
    }

    log(
        "=== train_ground_truth.tsv ==="
    )

    gt_result, match_counts, sampled_matches = (
        process_ground_truth(
            paths[
                "train_ground_truth.tsv"
            ],
            COLUMN_OVERRIDES[
                "train_ground_truth.tsv"
            ],
            args,
            sample_target=args.positive_sample_size,
        )
    )

    summary["files"][
        "train_ground_truth.tsv"
    ] = gt_result

    if args.dry_run:
        for fname, path in paths.items():
            if fname == "train_ground_truth.tsv":
                continue

            log(
                "=== "
                + fname
                + " ==="
            )

            res, _, _, _ = analyze_source_file(
                path,
                COLUMN_OVERRIDES[fname],
                args,
            )

            summary["files"][
                fname
            ] = res

        (
            out / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                default=json_default,
            )
        )

        log(
            "Dry run complete. "
            "Schema written to "
            + str(out / "summary.json")
            + "."
        )

        log(
            "Review the resolved mapping "
            "printed above for every file."
        )

        log(
            "If anything looks wrong, fix "
            "COLUMN_OVERRIDES, then rerun "
            "without --dry-run."
        )

        return

    want_s1_ids = set()
    want_s2_ids = set()
    want_s3_ids = set()

    for m in sampled_matches:
        want_s1_ids.add(
            m["s1_id"]
        )

        if m["which_source"] in (
            "s2",
            "unknown",
        ):
            want_s2_ids.add(
                m["match_id"]
            )

        if m["which_source"] in (
            "s3",
            "unknown",
        ):
            want_s3_ids.add(
                m["match_id"]
            )

    log(
        "=== train_source1.tsv ==="
    )

    (
        s1_result,
        all_s1_ids,
        s1_lookup,
        s1_hn_pool,
    ) = analyze_source_file(
        paths[
            "train_source1.tsv"
        ],
        COLUMN_OVERRIDES[
            "train_source1.tsv"
        ],
        args,
        want_ids=want_s1_ids,
        collect_all_ids=True,
        hard_neg_pool_size=args.hard_neg_pool_size,
    )

    summary["files"][
        "train_source1.tsv"
    ] = s1_result

    log(
        "=== train_source2.tsv ==="
    )

    (
        s2_result,
        _,
        s2_lookup,
        s2_hn_pool,
    ) = analyze_source_file(
        paths[
            "train_source2.tsv"
        ],
        COLUMN_OVERRIDES[
            "train_source2.tsv"
        ],
        args,
        want_ids=want_s2_ids,
        hard_neg_pool_size=args.hard_neg_pool_size,
    )

    summary["files"][
        "train_source2.tsv"
    ] = s2_result

    log(
        "=== train_source3.tsv ==="
    )

    (
        s3_result,
        _,
        s3_lookup,
        s3_hn_pool,
    ) = analyze_source_file(
        paths[
            "train_source3.tsv"
        ],
        COLUMN_OVERRIDES[
            "train_source3.tsv"
        ],
        args,
        want_ids=want_s3_ids,
        hard_neg_pool_size=args.hard_neg_pool_size,
    )

    summary["files"][
        "train_source3.tsv"
    ] = s3_result

    for fname in (
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ):
        log(
            "=== "
            + fname
            + " ==="
        )

        res, _, _, _ = analyze_source_file(
            paths[fname],
            COLUMN_OVERRIDES[fname],
            args,
        )

        summary["files"][
            fname
        ] = res

    log(
        "Computing ground-truth "
        "match distribution ..."
    )

    buckets = {
        "0": 0,
        "1": 0,
        ">1": 0,
    }

    all_counts = []

    if all_s1_ids:
        for sid in all_s1_ids:
            c = match_counts.get(
                sid,
                0,
            )

            all_counts.append(c)

            if c == 0:
                buckets["0"] += 1

            elif c == 1:
                buckets["1"] += 1

            else:
                buckets[">1"] += 1

    total_s1 = (
        len(all_s1_ids)
        if all_s1_ids
        else 0
    )

    bucket_pct = {}

    if total_s1:
        for k, v in buckets.items():
            bucket_pct[k] = round(
                100 * v / total_s1,
                2,
            )

    extra_gt_ids = None

    if all_s1_ids:
        extra_gt_ids = len(
            set(match_counts)
            - all_s1_ids
        )

    match_dist_summary = {
        "total_s1_entities": total_s1,
        "bucket_counts": buckets,
        "bucket_pct": bucket_pct,
        "max_matches_observed": (
            int(max(all_counts))
            if all_counts
            else 0
        ),
        "mean_matches_per_s1": (
            round(
                sum(all_counts)
                / total_s1,
                4,
            )
            if total_s1
            else 0
        ),
        "s1_ids_in_ground_truth_but_not_in_source1":
            extra_gt_ids,
    }

    summary[
        "ground_truth_match_distribution"
    ] = match_dist_summary

    log(
        "Match distribution: "
        + str(buckets)
        + " (out of "
        + str(total_s1)
        + " S1 entities)"
    )

    match_dist_rows = []

    for k, v in buckets.items():
        match_dist_rows.append(
            {
                "bucket": k,
                "count": v,
                "pct": bucket_pct.get(k),
            }
        )

    pd.DataFrame(
        match_dist_rows
    ).to_csv(
        out / "match_distribution.csv",
        index=False,
    )

    log(
        "Writing country_stats.csv ..."
    )

    country_rows = []

    source_file_names = (
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    )

    for fname in source_file_names:
        finfo = summary[
            "files"
        ].get(fname, {})

        dist = finfo.get(
            "country_distribution_top200"
        )

        rc = finfo.get(
            "row_count"
        )

        if not dist:
            continue

        for country, cnt in dist.items():
            pct = (
                round(
                    100 * cnt / rc,
                    3,
                )
                if rc
                else None
            )

            country_rows.append(
                {
                    "file": fname,
                    "country": country,
                    "count": cnt,
                    "pct": pct,
                }
            )

    pd.DataFrame(
        country_rows
    ).to_csv(
        out / "country_stats.csv",
        index=False,
    )

    log(
        "Writing noise_statistics.csv ..."
    )

    noise_rows = []

    for fname in source_file_names:
        finfo = summary[
            "files"
        ].get(fname, {})

        for field in (
            "name",
            "address",
        ):
            length_summary = (
                finfo.get(
                    field
                    + "_length"
                )
                or {}
            )

            noise_summary = (
                finfo.get(
                    field
                    + "_noise"
                )
                or {}
            )

            if (
                not length_summary
                and not noise_summary
            ):
                continue

            row = {
                "file": fname,
                "field": field,
            }

            for k, v in length_summary.items():
                row[
                    "length_" + k
                ] = v

            row.update(
                noise_summary
            )

            noise_rows.append(row)

    pd.DataFrame(
        noise_rows
    ).to_csv(
        out / "noise_statistics.csv",
        index=False,
    )

    log(
        "Building positive_samples.csv ..."
    )

    pos_rows = []

    for m in sampled_matches:
        s1_rec = s1_lookup.get(
            m["s1_id"],
            {},
        )

        matched_source = m[
            "which_source"
        ]

        if matched_source == "s2":
            other_rec = s2_lookup.get(
                m["match_id"],
                {},
            )

        elif matched_source == "s3":
            other_rec = s3_lookup.get(
                m["match_id"],
                {},
            )

        else:
            other_rec = (
                s2_lookup.get(
                    m["match_id"],
                    {},
                )
                or s3_lookup.get(
                    m["match_id"],
                    {},
                )
            )

        row = {
            "s1_id": m["s1_id"],
            "match_id": m["match_id"],
            "source": matched_source,
            "s1_name": s1_rec.get("name"),
            "s1_address": s1_rec.get("address"),
            "s1_country": s1_rec.get("country"),
            "match_name": other_rec.get("name"),
            "match_address": other_rec.get("address"),
            "match_country": other_rec.get("country"),
        }

        row["name_noise_type"] = classify_name_pair(
            row["s1_name"],
            row["match_name"],
        )

        pos_rows.append(row)

    pd.DataFrame(
        pos_rows
    ).to_csv(
        out / "positive_samples.csv",
        index=False,
    )

    log(
        "Building hard_negative_samples.csv ..."
    )

    hard_neg_rows = []

    for source_label, pool in (
        ("source1", s1_hn_pool),
        ("source2", s2_hn_pool),
        ("source3", s3_hn_pool),
    ):
        if not pool:
            continue

        hard_neg_rows.extend(
            mine_hard_negatives(
                pool,
                source_label,
                max_results=args.hard_neg_max_results,
            )
        )

    pd.DataFrame(
        hard_neg_rows
    ).to_csv(
        out / "hard_negative_samples.csv",
        index=False,
    )

    noise_breakdown = Counter(
        r.get(
            "name_noise_type"
        )
        for r in pos_rows
    )

    summary[
        "positive_sample_noise_breakdown"
    ] = dict(
        noise_breakdown
    )

    summary[
        "hard_negative_sample_count"
    ] = len(hard_neg_rows)

    summary[
        "runtime_seconds"
    ] = round(
        time.time() - t0,
        2,
    )

    (
        out / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=json_default,
        )
    )

    log(
        "EDA complete in "
        + str(
            round(
                time.time() - t0,
                1,
            )
        )
        + " seconds."
    )


if __name__ == "__main__":
    main()