"""Deterministic field-opportunity ranking for `catalog fields --opportunity`."""

import math

from research_app.catalog import field_opportunity


def test_unmined_high_coverage_field_outranks_crowded_one():
    fresh = field_opportunity({"coverage": 0.95, "alphaCount": 0})
    crowded = field_opportunity({"coverage": 0.95, "alphaCount": 5000})
    assert fresh > crowded
    assert fresh == 0.95 / math.log(2)


def test_missing_coverage_is_unscorable():
    assert field_opportunity({"alphaCount": 3}) is None
    assert field_opportunity({"coverage": 0, "alphaCount": 3}) is None
    assert field_opportunity({"coverage": 1.5, "alphaCount": 3}) is None


def test_absent_or_invalid_alpha_count_counts_as_unmined():
    assert field_opportunity({"coverage": 0.5}) == 0.5 / math.log(2)
    assert field_opportunity({"coverage": 0.5, "alphaCount": None}) == 0.5 / math.log(2)
    assert field_opportunity({"coverage": 0.5, "alphaCount": -4}) == 0.5 / math.log(2)


def test_reported_theme_multiplier_scales_the_score():
    base = field_opportunity({"coverage": 0.8, "alphaCount": 10})
    themed = field_opportunity({"coverage": 0.8, "alphaCount": 10, "pyramidMultiplier": 1.5})
    assert themed > base
    assert math.isclose(themed, base * 1.5)
    # A non-finite or non-numeric multiplier falls back to 1.0 instead of poisoning the sort.
    assert field_opportunity({"coverage": 0.8, "alphaCount": 10, "pyramidMultiplier": "x"}) == base
