"""Unit tests for the macro-F0.5 evaluator.

Includes the 8 scenarios explicitly enumerated in the Task 2 brief, each
as its own test function so a failure clearly names which scenario
broke.
"""

from __future__ import annotations

import pytest

from business_entity_resolution.validation.metrics import (
    evaluate_entity,
    evaluate_predictions,
    f_beta,
)


def approx(x: float) -> pytest.approx:
    return pytest.approx(x, abs=1e-9)


# --------------------------------------------------------------------------
# f_beta building block
# --------------------------------------------------------------------------


def test_f_beta_matches_known_formula():
    # P=0.8, R=0.6, beta=0.5 -> hand-computed reference value.
    p, r = 0.8, 0.6
    expected = (1.25 * p * r) / (0.25 * p + r)
    assert f_beta(p, r) == approx(expected)


def test_f_beta_zero_zero_is_zero():
    assert f_beta(0.0, 0.0) == 0.0


# --------------------------------------------------------------------------
# The 8 required scenarios (Task 2 brief, section 4)
# --------------------------------------------------------------------------


def test_scenario_1_perfect_prediction():
    score = evaluate_entity(true_ids={"S2-1", "S2-2"}, predicted_ids={"S2-1", "S2-2"})
    assert (score.tp, score.fp, score.fn) == (2, 0, 0)
    assert score.precision == approx(1.0)
    assert score.recall == approx(1.0)
    assert score.f0_5 == approx(1.0)


def test_scenario_2_ground_truth_empty_and_prediction_empty():
    score = evaluate_entity(true_ids=set(), predicted_ids=set())
    assert (score.tp, score.fp, score.fn) == (0, 0, 0)
    # Vacuous-truth convention: nothing predicted, nothing to miss.
    assert score.precision == approx(1.0)
    assert score.recall == approx(1.0)
    assert score.f0_5 == approx(1.0)


def test_scenario_3_ground_truth_empty_but_false_prediction():
    score = evaluate_entity(true_ids=set(), predicted_ids={"S2-99"})
    assert (score.tp, score.fp, score.fn) == (0, 1, 0)
    assert score.precision == approx(0.0)
    # recall vacuously 1.0 (TP+FN == 0), but F0.5 is still 0 since
    # precision is 0 -- documented behaviour, asserted explicitly here.
    assert score.recall == approx(1.0)
    assert score.f0_5 == approx(0.0)


def test_scenario_4_partial_recall():
    score = evaluate_entity(true_ids={"a", "b", "c"}, predicted_ids={"a", "b"})
    assert (score.tp, score.fp, score.fn) == (2, 0, 1)
    assert score.precision == approx(1.0)
    assert score.recall == approx(2 / 3)
    assert score.f0_5 == approx(10 / 11)


def test_scenario_5_correct_matches_plus_extra_false_positives():
    score = evaluate_entity(true_ids={"a", "b"}, predicted_ids={"a", "b", "x"})
    assert (score.tp, score.fp, score.fn) == (2, 1, 0)
    assert score.precision == approx(2 / 3)
    assert score.recall == approx(1.0)
    assert score.f0_5 == approx(5 / 7)


def test_scenario_6_multiple_correct_matches():
    score = evaluate_entity(
        true_ids={"a", "b", "c", "d"}, predicted_ids={"a", "b", "c", "d"}
    )
    assert (score.tp, score.fp, score.fn) == (4, 0, 0)
    assert score.f0_5 == approx(1.0)


def test_scenario_7_prediction_missing_an_s1_entity():
    ground_truth = {"S1-1": {"S2-1"}}
    predictions: dict = {}  # S1-1 has no entry at all
    result = evaluate_predictions(ground_truth, predictions)
    entity_score = result.per_entity["S1-1"]
    assert (entity_score.tp, entity_score.fp, entity_score.fn) == (0, 0, 1)
    assert entity_score.precision == approx(1.0)  # vacuous, nothing predicted
    assert entity_score.recall == approx(0.0)
    assert entity_score.f0_5 == approx(0.0)


def test_scenario_8_prediction_containing_duplicate_ids():
    score = evaluate_entity(
        true_ids={"S2-1", "S2-2"}, predicted_ids=["S2-1", "S2-1", "S2-2"]
    )
    assert (score.tp, score.fp, score.fn) == (2, 0, 0)
    assert score.f0_5 == approx(1.0)


# --------------------------------------------------------------------------
# evaluate_predictions: macro aggregation, universe handling, diagnostics
# --------------------------------------------------------------------------


def test_evaluate_predictions_macro_is_unweighted_mean_of_per_entity_f0_5():
    ground_truth = {
        "S1-1": {"a", "b"},  # perfect -> F0.5 = 1.0
        "S1-2": set(),  # false positive -> F0.5 = 0.0
    }
    predictions = {
        "S1-1": {"a", "b"},
        "S1-2": {"x"},
    }
    result = evaluate_predictions(ground_truth, predictions)
    assert result.macro_f0_5 == approx((1.0 + 0.0) / 2)
    assert result.n_entities == 2


def test_evaluate_predictions_macro_differs_from_micro_when_entities_differ_in_size():
    # One large easy entity, one small hard entity -- macro treats them
    # equally; micro would be dominated by the large entity's counts.
    ground_truth = {
        "S1-big": {f"id-{i}" for i in range(100)},
        "S1-small": {"only-id"},
    }
    predictions = {
        "S1-big": {f"id-{i}" for i in range(100)},  # perfect, F0.5=1.0
        "S1-small": set(),  # miss -> F0.5=0.0
    }
    result = evaluate_predictions(ground_truth, predictions)
    assert result.macro_f0_5 == approx(0.5)
    # Micro is dominated by the 100 correct big-entity matches.
    assert result.micro_f0_5 > 0.9
    assert result.macro_f0_5 != result.micro_f0_5


def test_evaluate_predictions_flags_unexpected_prediction_ids():
    ground_truth = {"S1-1": {"a"}}
    predictions = {"S1-1": {"a"}, "S1-unexpected": {"b"}}
    result = evaluate_predictions(ground_truth, predictions)
    assert result.unexpected_prediction_ids == ["S1-unexpected"]
    # The unexpected entity must not be scored / must not affect n_entities.
    assert result.n_entities == 1


def test_evaluate_predictions_deterministic():
    ground_truth = {"S1-1": {"a", "b"}, "S1-2": {"c"}}
    predictions = {"S1-1": {"a"}, "S1-2": {"c", "d"}}
    r1 = evaluate_predictions(ground_truth, predictions)
    r2 = evaluate_predictions(ground_truth, predictions)
    assert r1.to_dict() == r2.to_dict()
