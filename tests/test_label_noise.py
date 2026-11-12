"""Tests for training-only label noise.

The helper must leave the input array untouched, ignore the global NumPy
random state, and match the protocol used by E8: Gaussian noise on
already-standardized regression targets, and class flips for classification.
"""

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


def apply_train_label_noise():
    """Import the noise helper from the paper package."""
    with working_directory(PAPER_DIR):
        from lib.data import apply_train_label_noise as function

    return function


def test_zero_noise_returns_the_same_array() -> None:
    apply = apply_train_label_noise()
    labels = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    assert apply(labels, level=0.0, seed=0, regression=True) is labels


def test_regression_noise_is_deterministic_and_does_not_mutate_input() -> None:
    apply = apply_train_label_noise()
    labels = np.zeros(20_000, dtype=np.float32)
    original = labels.copy()
    first = apply(labels, level=0.1, seed=3, regression=True)
    second = apply(labels, level=0.1, seed=3, regression=True)
    other = apply(labels, level=0.1, seed=4, regression=True)

    assert np.array_equal(labels, original)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.dtype == np.float32
    assert abs(float(first.std()) - 0.1) < 0.005


def test_noise_does_not_consume_the_global_generator() -> None:
    apply = apply_train_label_noise()
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    apply(np.zeros(100, dtype=np.float32), level=0.2, seed=1, regression=True)
    after = np.random.random()
    assert before == after


def test_binary_flip_rate_matches_the_level() -> None:
    apply = apply_train_label_noise()
    labels = np.zeros(50_000, dtype=np.int64)
    labels[25_000:] = 1
    original = labels.copy()
    noisy = apply(labels, level=0.2, seed=0, regression=False)
    assert np.array_equal(labels, original)
    rate = float(np.mean(noisy != labels))
    assert abs(rate - 0.2) < 0.01
    assert set(np.unique(noisy)).issubset({0, 1})


def test_multiclass_flip_never_keeps_the_true_label() -> None:
    apply = apply_train_label_noise()
    labels = np.array([0, 1, 2, 3] * 1000, dtype=np.int64)
    noisy = apply(labels, level=1.0, seed=2, regression=False)
    assert np.all(noisy != labels)
    assert set(np.unique(noisy)).issubset({0, 1, 2, 3})


def test_level_out_of_range_is_rejected() -> None:
    apply = apply_train_label_noise()
    labels = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError):
        apply(labels, level=1.5, seed=0, regression=True)
    with pytest.raises(ValueError):
        apply(labels, level=-0.1, seed=0, regression=True)
