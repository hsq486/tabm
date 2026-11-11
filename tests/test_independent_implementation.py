"""Regression tests for the independent TabM implementation used in E7."""

import importlib.util
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / 'paper' / 'scripts' / 'verify_implementation.py'


def load_verification() -> types.ModuleType:
    """Load the E7 verification script without turning it into a package."""
    spec = importlib.util.spec_from_file_location('verify_implementation', SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def verification() -> types.ModuleType:
    torch.set_num_threads(1)
    return load_verification()


@pytest.fixture(scope='module')
def reference(verification: types.ModuleType) -> types.ModuleType:
    return verification.load_reference(REPO_ROOT)


@pytest.mark.parametrize('k', [1, 8])
def test_forward_matches_official_implementation(
    verification: types.ModuleType, reference: types.ModuleType, k: int
) -> None:
    row = verification.check_forward_equivalence(reference, k)
    assert row['passed'], f'forward mismatch for k={k}: {row}'


def test_parameter_count_formula_is_exact(
    verification: types.ModuleType, reference: types.ModuleType
) -> None:
    row = verification.check_parameter_formula(reference, 8)
    assert row['passed'], f'parameter formula mismatch: {row}'


def test_k1_reduces_to_a_plain_mlp(
    verification: types.ModuleType, reference: types.ModuleType
) -> None:
    row = verification.check_k1_reduces_to_mlp(reference)
    assert row['passed'], f'k=1 equivalence broken: {row}'
