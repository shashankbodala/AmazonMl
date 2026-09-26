"""Reusable, deterministic macro-F0.5-per-S1-entity evaluator.

The official challenge metric is **macro F0.5, averaged per S1 entity**
(explicitly NOT micro-averaged pooled TP/FP/FN). This module computes a
per-entity precision/recall/F0.5, then averages the per-entity F0.5
values unweighted across entities. A micro-averaged number is also made
available on the result object purely as a secondary diagnostic, and is
clearly labeled as such -- callers should never report it as "the"
score.

Edge-case convention (documented explicitly, since F-beta involves 0/0
divisions whenever a set is empty)
--------------------------------------------------------------------
For one S1 entity, let TP/FP/FN be defined the usual way over the
predicted vs. true ID sets. Then:

- ``precision = TP / (TP + FP)`` if ``TP + FP > 0``, else ``precision = 1.0``.
  Rationale: if nothing was predicted, there are no false positives to
  be penalized for -- vacuously "everything predicted was correct".
- ``recall = TP / (TP + FN)`` if ``TP + FN > 0``, else ``recall = 1.0``.
  Rationale: if the true match set is empty, there is nothing to have
  missed -- vacuously "everything true was recovered".
- If both precision and recall are defined as 1.0 via the vacuous
  cases above, F0.5 = 1.0.
- If precision = 0 (i.e. TP=0 and FP>0), F0.5 = 0.0 regardless of
  recall, which is the mathematically correct behaviour of the F-beta
  formula (the numerator is 0 whenever precision is 0), so it does not
  need special-casing -- it is included here only to spell out the
  practical implication: **an entity with no true matches that receives
  any predicted candidate always scores 0.0**, and an entity whose
  prediction is entirely wrong (no overlap with a non-empty ground
  truth) also always scores 0.0.

This exact convention is what is exercised by
``tests/test_metrics.py``; see the 8 scenarios enumerated in the Task 2
brief, reproduced there as individual test functions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

BETA = 0.5
BETA_SQUARED = BETA * BETA


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    """Standard F-beta combination of an already-computed precision/recall.

    Returns 0.0 if both precision and recall are 0.0 (avoids a 0/0 in the
    combination formula itself; this is distinct from -- and applied
    after -- the vacuous-1.0 conventions for precision/recall themselves,
    which are handled in ``evaluate_entity``).
    """
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    numerator = (1 + b2) * precision * recall
    denominator = (b2 * precision) + recall
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


@dataclass
class EntityScore:
    s1_id: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f0_5: float

    def to_dict(self) -> dict:
        return asdict(self)


def _to_set(ids: Optional[Iterable[str]]) -> Set[str]:
    """Coerce a list/tuple/set/None of ids into a set, de-duplicating.

    This is where "prediction containing duplicate IDs" (Task 2 edge
    case #8) is handled: duplicates collapse naturally once coerced to
    a set, so they can never inflate TP or FP counts.
    """
    if ids is None:
        return set()
    return set(ids)


def evaluate_entity(
    true_ids: Optional[Iterable[str]],
    predicted_ids: Optional[Iterable[str]],
    s1_id: str = "",
) -> EntityScore:
    """Compute TP/FP/FN/precision/recall/F0.5 for a single S1 entity.

    ``true_ids`` / ``predicted_ids`` may be any iterable of strings
    (list, set, tuple) or ``None``; both are treated as possibly
    containing duplicates and are de-duplicated internally.
    """
    true_set = _to_set(true_ids)
    pred_set = _to_set(predicted_ids)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    score = f_beta(precision, recall)

    return EntityScore(
        s1_id=s1_id, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f0_5=score
    )


@dataclass
class MacroF05Result:
    macro_f0_5: float
    macro_precision: float
    macro_recall: float

    # Secondary diagnostic ONLY -- never the official metric. Computed
    # from pooled TP/FP/FN across all entities.
    micro_f0_5: float
    micro_precision: float
    micro_recall: float

    n_entities: int
    total_tp: int
    total_fp: int
    total_fn: int

    # S1 ids present in `predictions` but absent from `ground_truth`'s
    # universe -- these are never scored (the S1 universe is defined by
    # ground truth), but are surfaced here since they usually indicate a
    # bug upstream (e.g. predicting for a test-set id against a
    # train-ground-truth dict).
    unexpected_prediction_ids: List[str] = field(default_factory=list)

    per_entity: Dict[str, EntityScore] = field(default_factory=dict, repr=False)

    def to_dict(self, include_per_entity: bool = False) -> dict:
        d = {
            "macro_f0_5": self.macro_f0_5,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "micro_f0_5": self.micro_f0_5,
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "n_entities": self.n_entities,
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_fn": self.total_fn,
            "unexpected_prediction_ids": self.unexpected_prediction_ids,
        }
        if include_per_entity:
            d["per_entity"] = {k: v.to_dict() for k, v in self.per_entity.items()}
        return d

    def save_json(self, path: str | Path, include_per_entity: bool = False) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(include_per_entity=include_per_entity), indent=2)
        )


def evaluate_predictions(
    ground_truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
    s1_ids: Optional[Iterable[str]] = None,
) -> MacroF05Result:
    """Score a full set of predictions against ground truth.

    Parameters
    ----------
    ground_truth:
        ``s1_id -> iterable of true S2/S3 ids``. Defines the scored
        universe of S1 entities unless ``s1_ids`` is given explicitly.
        An S1 entity with an empty/absent true-match iterable is scored
        as a legitimate zero-match entity (see module docstring).
    predictions:
        ``s1_id -> iterable of predicted S2/S3 ids``. An S1 entity
        present in ``ground_truth`` (or ``s1_ids``) but absent from
        ``predictions`` is scored as if it had been predicted with an
        empty set (Task 2 edge case #7), NOT skipped.
    s1_ids:
        Optional explicit universe of S1 ids to score. Defaults to
        ``ground_truth.keys()``.
    """
    universe = list(s1_ids) if s1_ids is not None else list(ground_truth.keys())

    per_entity: Dict[str, EntityScore] = {}
    total_tp = total_fp = total_fn = 0
    sum_f0_5 = 0.0
    sum_precision = 0.0
    sum_recall = 0.0

    for s1_id in universe:
        true_ids = ground_truth.get(s1_id, [])
        pred_ids = predictions.get(s1_id, [])
        score = evaluate_entity(true_ids, pred_ids, s1_id=s1_id)
        per_entity[s1_id] = score

        total_tp += score.tp
        total_fp += score.fp
        total_fn += score.fn
        sum_f0_5 += score.f0_5
        sum_precision += score.precision
        sum_recall += score.recall

    n = len(universe)
    macro_f0_5 = sum_f0_5 / n if n else 0.0
    macro_precision = sum_precision / n if n else 0.0
    macro_recall = sum_recall / n if n else 0.0

    micro_precision = (
        1.0 if (total_tp + total_fp) == 0 else total_tp / (total_tp + total_fp)
    )
    micro_recall = 1.0 if (total_tp + total_fn) == 0 else total_tp / (total_tp + total_fn)
    micro_f0_5 = f_beta(micro_precision, micro_recall)

    universe_set = set(universe)
    unexpected_prediction_ids = sorted(set(predictions.keys()) - universe_set)

    return MacroF05Result(
        macro_f0_5=macro_f0_5,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        micro_f0_5=micro_f0_5,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        n_entities=n,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        unexpected_prediction_ids=unexpected_prediction_ids,
        per_entity=per_entity,
    )
