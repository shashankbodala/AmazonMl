"""Reusable text normalization for business entity fields.

This module provides deterministic, dependency-free normalization for:
- business_name
- business_address
- country

The goal is exact-match blocking: two records that should match must
normalize to exactly the same key. No fuzzy matching, no ML, no external
resources -- just aggressive but consistent string canonicalization.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd


# --------------------------------------------------------------------------
# Pre-built translation tables for punctuation removal (built once at import)
# --------------------------------------------------------------------------

def _build_punct_trans_table(keep: str = "") -> dict:
    """Build a str.translate table that removes punctuation/symbols except keep."""
    table = {}
    keep_set = set(keep)
    for cp in range(0x110000):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        if cat.startswith('L') or cat.startswith('M') or cat.startswith('N') or cat.startswith('Z') or cat == 'Cc' or ch in keep_set:
            pass  # keep as-is
        elif ch == '/':
            table[cp] = ord(' ')
        else:
            table[cp] = None
    return str.maketrans(table)


_NAME_PUNCT_TABLE = _build_punct_trans_table(keep="&-' +")
_ADDRESS_PUNCT_TABLE = _build_punct_trans_table(keep="")


# --------------------------------------------------------------------------
# Vectorized normalization functions (operate on pandas Series)
# --------------------------------------------------------------------------


def _strip_accents_series(series: pd.Series) -> pd.Series:
    """Remove diacritics from Latin characters in a Series."""
    def strip_accents(text: str) -> str:
        result = []
        for ch in text:
            cp = ord(ch)
            if (0x0041 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
                for c in unicodedata.normalize('NFD', ch):
                    if unicodedata.category(c) != 'Mn':
                        result.append(c)
            else:
                result.append(ch)
        return ''.join(result)
    return series.apply(strip_accents)


def _collapse_whitespace_series(series: pd.Series) -> pd.Series:
    """Collapse all runs of whitespace to a single space."""
    return series.str.replace(r"\s+", " ", regex=True).str.strip()


def _remove_punctuation_series(series: pd.Series, keep: str = "") -> pd.Series:
    """Remove punctuation except characters in ``keep`` using str.translate."""
    if keep == "&-' +":
        table = _NAME_PUNCT_TABLE
    else:
        table = _ADDRESS_PUNCT_TABLE
    return series.str.translate(table)


def _normalize_case_series(series: pd.Series) -> pd.Series:
    """Lowercase with Unicode-aware case folding."""
    return series.str.casefold()


def _clean_series(series: pd.Series) -> pd.Series:
    """Basic cleaning: strip, handle None/NaN/empty."""
    return series.fillna("").astype(str).str.strip().replace({"nan": "", "NaN": ""})


# --------------------------------------------------------------------------
# Vectorized field normalizers
# --------------------------------------------------------------------------


def normalize_business_name_series(series: pd.Series) -> pd.Series:
    """Vectorized business name normalization."""
    s = _clean_series(series)
    s = _strip_accents_series(s)
    s = _normalize_case_series(s)
    s = _remove_punctuation_series(s, keep="&-' +")
    s = _collapse_whitespace_series(s)
    return s


def normalize_business_address_series(series: pd.Series) -> pd.Series:
    """Vectorized business address normalization."""
    s = _clean_series(series)
    s = _strip_accents_series(s)
    s = _normalize_case_series(s)
    s = s.str.replace('/', ' ', regex=False)
    s = _remove_punctuation_series(s, keep="")
    s = _collapse_whitespace_series(s)
    return s


def normalize_country_series(series: pd.Series) -> pd.Series:
    """Vectorized country normalization."""
    s = _clean_series(series)
    s = _strip_accents_series(s)
    s = _normalize_case_series(s)
    s = _remove_punctuation_series(s, keep="")
    s = _collapse_whitespace_series(s)
    return s


# --------------------------------------------------------------------------
# Scalar normalizers (for single values / small sets)
# --------------------------------------------------------------------------


def _strip_accents(text: str) -> str:
    result = []
    for ch in text:
        cp = ord(ch)
        if (0x0041 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            for c in unicodedata.normalize('NFD', ch):
                if unicodedata.category(c) != 'Mn':
                    result.append(c)
        else:
            result.append(ch)
    return ''.join(result)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_punctuation(text: str, keep: str = "") -> str:
    if keep == "&-' +":
        return text.translate(_NAME_PUNCT_TABLE)
    else:
        return text.translate(_ADDRESS_PUNCT_TABLE)


def _normalize_case(text: str) -> str:
    return text.casefold()


def _clean_scalar(raw: Optional[object]) -> str:
    if raw is None:
        return ""
    if isinstance(raw, float):
        import math
        if math.isnan(raw):
            return ""
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def normalize_business_name(raw: Optional[object]) -> str:
    text = _clean_scalar(raw)
    text = _strip_accents(text)
    text = _normalize_case(text)
    text = _remove_punctuation(text, keep="&-' +")
    text = _collapse_whitespace(text)
    return text


def normalize_business_address(raw: Optional[object]) -> str:
    text = _clean_scalar(raw)
    text = _strip_accents(text)
    text = _normalize_case(text)
    text = text.replace('/', ' ')
    text = _remove_punctuation(text, keep="")
    text = _collapse_whitespace(text)
    return text


def normalize_country(raw: Optional[object]) -> str:
    text = _clean_scalar(raw)
    text = _strip_accents(text)
    text = _normalize_case(text)
    text = _remove_punctuation(text, keep="")
    text = _collapse_whitespace(text)
    return text


# --------------------------------------------------------------------------
# Composite keys for blocking
# --------------------------------------------------------------------------


def make_name_key(name: str) -> str:
    return normalize_business_name(name)


def make_address_key(address: str) -> str:
    return normalize_business_address(address)


def make_country_key(country: str) -> str:
    return normalize_country(country)


def make_name_country_key(name: str, country: str) -> str:
    return f"{make_name_key(name)}|{make_country_key(country)}"


def make_address_country_key(address: str, country: str) -> str:
    return f"{make_address_key(address)}|{make_country_key(country)}"


# --------------------------------------------------------------------------
# Vectorized composite keys
# --------------------------------------------------------------------------


def make_name_key_series(name_series: pd.Series) -> pd.Series:
    return normalize_business_name_series(name_series)


def make_address_key_series(address_series: pd.Series) -> pd.Series:
    return normalize_business_address_series(address_series)


def make_country_key_series(country_series: pd.Series) -> pd.Series:
    return normalize_country_series(country_series)


def make_name_country_key_series(name_series: pd.Series, country_series: pd.Series) -> pd.Series:
    return make_name_key_series(name_series) + "|" + make_country_key_series(country_series)


def make_address_country_key_series(address_series: pd.Series, country_series: pd.Series) -> pd.Series:
    return make_address_key_series(address_series) + "|" + make_country_key_series(country_series)