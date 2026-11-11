"""Tests for the metric definitions used by every experiment.

`paper/lib/metrics.py` is the single source of truth for RMSE / accuracy /
ROC-AUC in this project, so its behaviour is pinned down here. The module
lives inside the `paper/` project, whose import machinery requires the
current working directory to be `paper/`.
"""

import functools
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

PAPER_DIR = Path(__file__).resolve().parents[1] / 'paper'


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """Temporarily change the working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@functools.cache
def calculate_metrics():
    """Import `lib.metrics.calculate_metrics` from the `paper/` project."""
    with working_directory(PAPER_DIR):
        from lib.metrics import calculate_metrics as function

    return function


def test_regression_metrics_match_their_definitions() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    result = calculate_metrics()(y_true, y_pred, 'regression', 'labels')
    assert result['rmse'] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert result['mae'] == pytest.approx(1.0 / 3.0)


def test_perfect_regression_prediction_has_zero_error() -> None:
    y = np.array([0.5, -1.5, 4.0, 2.25])
    result = calculate_metrics()(y, y, 'regression', 'labels')
    assert result['rmse'] == pytest.approx(0.0)
    assert result['r2'] == pytest.approx(1.0)


def test_binary_logits_are_thresholded_at_half() -> None:
    y_true = np.array([1, 0, 1])
    logits = np.array([[2.0], [-2.0], [2.0]])
    result = calculate_metrics()(y_true, logits, 'binclass', 'logits')
    assert result['accuracy'] == pytest.approx(1.0)
    assert result['roc-auc'] == pytest.approx(1.0)
    assert result['cross-entropy'] < 0.2


def test_binary_accuracy_reflects_a_wrong_prediction() -> None:
    y_true = np.array([1, 0, 1, 0])
    logits = np.array([[2.0], [2.0], [2.0], [-2.0]])
    result = calculate_metrics()(y_true, logits, 'binclass', 'logits')
    assert result['accuracy'] == pytest.approx(0.75)


def test_multiclass_probabilities_are_argmaxed() -> None:
    y_true = np.array([0, 2, 1])
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.2, 0.5, 0.3]])
    result = calculate_metrics()(y_true, probs, 'multiclass', 'probs')
    assert result['accuracy'] == pytest.approx(1.0)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_metrics()(
            np.array([0.0, 1.0]), np.array([0.0]), 'regression', 'labels'
        )


def test_unknown_task_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_metrics()(
            np.array([0.0, 1.0]), np.array([0.0, 1.0]), 'ranking', 'labels'
        )
