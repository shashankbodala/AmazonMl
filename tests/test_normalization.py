"""Unit tests for normalization module."""

from __future__ import annotations

import pytest

from business_entity_resolution.normalization import (
    make_address_country_key,
    make_address_key,
    make_country_key,
    make_name_country_key,
    make_name_key,
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)


# --------------------------------------------------------------------------
# normalize_business_name
# --------------------------------------------------------------------------


def test_normalize_business_name_basic():
    assert normalize_business_name("Orelee's Barbershop") == "orelee's barbershop"
    assert normalize_business_name("B+ Retail Inc") == "b+ retail inc"
    assert normalize_business_name("Prime Money") == "prime money"


def test_normalize_business_name_accents():
    assert normalize_business_name("Café") == "cafe"
    assert normalize_business_name("Montréal") == "montreal"
    assert normalize_business_name("São Paulo") == "sao paulo"


def test_normalize_business_name_casefold():
    assert normalize_business_name("Orelee's Barbershop") == normalize_business_name("ORELEE'S BARBERSHOP")
    assert normalize_business_name("B+ Retail Inc") == normalize_business_name("b+ retail inc")


def test_normalize_business_name_punctuation():
    # Kept: & - '
    assert normalize_business_name("A & B") == "a & b"
    assert normalize_business_name("Smith-Jones") == "smith-jones"
    assert normalize_business_name("Orelee's") == "orelee's"
    # Removed: . , # etc.
    assert normalize_business_name("A.B.C. Inc") == "abc inc"
    assert normalize_business_name("Hello, World!") == "hello world"


def test_normalize_business_name_whitespace():
    assert normalize_business_name("  Orelee's   Barbershop  ") == "orelee's barbershop"
    assert normalize_business_name("Orelee's\tBarbershop") == "orelee's barbershop"
    assert normalize_business_name("Orelee's\nBarbershop") == "orelee's barbershop"


def test_normalize_business_name_none_nan():
    assert normalize_business_name(None) == ""
    assert normalize_business_name(float("nan")) == ""
    assert normalize_business_name("") == ""
    assert normalize_business_name("nan") == ""
    assert normalize_business_name("NaN") == ""


def test_normalize_business_name_unicode():
    # Hindi - should preserve the text (no accent stripping)
    assert normalize_business_name("राम मार्केटिंग") == "राम मार्केटिंग"
    # Arabic
    assert normalize_business_name("شركة") == "شركة"


# --------------------------------------------------------------------------
# normalize_business_address
# --------------------------------------------------------------------------


def test_normalize_business_address_basic():
    addr = "1795 Westchester Drive, High Point, NC"
    expected = "1795 westchester drive high point nc"
    assert normalize_business_address(addr) == expected


def test_normalize_business_address_punctuation_removed():
    # All punctuation removed for addresses
    assert normalize_business_address("123 Main St.") == "123 main st"
    assert normalize_business_address("123 Main St, Apt 4") == "123 main st apt 4"
    assert normalize_business_address("123 Main St #4") == "123 main st 4"
    assert normalize_business_address("123 Main St/Ste 4") == "123 main st ste 4"
    assert normalize_business_address("123 Main St - Suite 4") == "123 main st suite 4"


def test_normalize_business_address_casefold():
    assert normalize_business_address("123 MAIN ST") == "123 main st"


def test_normalize_business_address_accents():
    assert normalize_business_address("Montréal, QC") == "montreal qc"


def test_normalize_business_address_none_nan():
    assert normalize_business_address(None) == ""
    assert normalize_business_address(float("nan")) == ""
    assert normalize_business_address("") == ""


# --------------------------------------------------------------------------
# normalize_country
# --------------------------------------------------------------------------


def test_normalize_country_basic():
    assert normalize_country("US") == "us"
    assert normalize_country("India") == "india"
    assert normalize_country("United States") == "united states"


def test_normalize_country_casefold():
    assert normalize_country("US") == normalize_country("us")
    assert normalize_country("India") == normalize_country("INDIA")


def test_normalize_country_none_nan():
    assert normalize_country(None) == ""
    assert normalize_country(float("nan")) == ""
    assert normalize_country("") == ""


# --------------------------------------------------------------------------
# Composite keys
# --------------------------------------------------------------------------


def test_make_name_key():
    assert make_name_key("Orelee's Barbershop") == "orelee's barbershop"


def test_make_address_key():
    assert make_address_key("123 Main St") == "123 main st"


def test_make_country_key():
    assert make_country_key("US") == "us"


def test_make_name_country_key():
    assert make_name_country_key("Orelee's Barbershop", "US") == "orelee's barbershop|us"
    assert make_name_country_key("Prime Money", "India") == "prime money|india"


def test_make_address_country_key():
    assert make_address_country_key("123 Main St", "US") == "123 main st|us"
    assert make_address_country_key("123 Main St", "India") == "123 main st|india"


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_normalize_business_name_special_chars():
    # Various special characters
    assert normalize_business_name("A&B") == "a&b"
    assert normalize_business_name("A@B") == "ab"  # @ removed
    assert normalize_business_name("A#B") == "ab"  # # removed
    assert normalize_business_name("A$B") == "ab"  # $ removed


def test_normalize_business_address_indian_format():
    addr = "H.No.16-11-23/37/A, 2Nd Floor, Flat No.207, Sagar Hotel Building, Opp.Rta Office, Mo, Osarambagh, Hyderabad, Telangana"
    result = normalize_business_address(addr)
    # Should not contain punctuation
    assert "." not in result
    assert "," not in result
    assert "/" not in result
    assert "#" not in result
    assert "-" not in result
    # Should contain the words
    assert "hno" in result
    assert "16" in result
    assert "11" in result
    assert "23" in result
    assert "37" in result
    assert "a" in result