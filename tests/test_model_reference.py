"""Tests for the reference implementation of TabM (`tabm_reference.py`).

These tests pin down the two properties that the experiment design relies on:

* the output shape of every architecture (so logits/probabilities are read correctly),
* the parameter-efficiency ordering used as evidence for the weight-sharing claim.
"""

from __future__ import annotations

import pytest
import torch

import tabm_reference

BACKBONE = {
    'type': 'MLP',
    'n_blocks': 2,
    'd_block': 16,
    'dropout': 0.0,
}
N_NUM_FEATURES = 5
BATCH_SIZE = 4


def make_model(
    arch_type: str,
    k: int | None = None,
    n_classes: int | None = None,
) -> torch.nn.Module:
    """Build a small model on synthetic feature counts."""
    return tabm_reference.Model(
        n_num_features=N_NUM_FEATURES,
        cat_cardinalities=[],
        n_classes=n_classes,
        backbone=BACKBONE,
        bins=None,
        arch_type=arch_type,
        k=k,
    )


def count_parameters(module: torch.nn.Module) -> int:
    """Return the total number of scalar parameters."""
    return sum(parameter.numel() for parameter in module.parameters())


def forward_shape(model: torch.nn.Module) -> tuple[int, ...]:
    """Run one forward pass on random features and return the output shape."""
    model.eval()
    with torch.no_grad():
        output = model(x_num=torch.randn(BATCH_SIZE, N_NUM_FEATURES))
    return tuple(output.shape)


def test_plain_returns_single_prediction() -> None:
    model = make_model('plain')
    assert forward_shape(model) == (BATCH_SIZE, 1, 1)


def test_tabm_returns_one_prediction_per_submodel() -> None:
    model = make_model('tabm', k=8)
    assert forward_shape(model) == (BATCH_SIZE, 8, 1)


def test_tabm_keeps_class_dimension_for_classification() -> None:
    model = make_model('tabm', k=4, n_classes=3)
    assert forward_shape(model) == (BATCH_SIZE, 4, 3)


def test_packed_ensemble_matches_k_independent_models() -> None:
    """TabM-packed trains k independent MLPs, so its size is exactly k times."""
    k = 8
    plain = count_parameters(make_model('plain'))
    packed = count_parameters(make_model('tabm-packed', k=k))
    assert packed == k * plain


def test_shared_weights_use_fewer_parameters_than_packed() -> None:
    """The central efficiency claim: weight sharing is cheaper than packing."""
    tabm = count_parameters(make_model('tabm', k=8))
    packed = count_parameters(make_model('tabm-packed', k=8))
    assert tabm < packed


def test_mini_variant_is_cheaper_than_tabm() -> None:
    mini = count_parameters(make_model('tabm-mini', k=8))
    tabm = count_parameters(make_model('tabm', k=8))
    assert mini < tabm


def test_parameter_growth_in_k_stays_sublinear_for_tabm() -> None:
    """Doubling k must not double the parameter count of the shared version."""
    small = count_parameters(make_model('tabm', k=8))
    large = count_parameters(make_model('tabm', k=16))
    assert large < 2 * small


def test_k_is_required_for_ensembled_architectures() -> None:
    with pytest.raises(AssertionError):
        make_model('tabm', k=None)


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match='Unknown arch_type'):
        make_model('not-an-architecture', k=2)


def test_one_optimizer_step_reduces_loss_on_synthetic_data() -> None:
    """Minimal end-to-end smoke test: forward, backward and update must work."""
    torch.manual_seed(0)
    model = make_model('tabm', k=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(32, N_NUM_FEATURES)
    y = x.sum(dim=1, keepdim=True)

    def loss_value() -> float:
        prediction = model(x_num=x).mean(dim=1)
        return float(torch.nn.functional.mse_loss(prediction, y))

    before = loss_value()
    optimizer.zero_grad()
    prediction = model(x_num=x).mean(dim=1)
    loss = torch.nn.functional.mse_loss(prediction, y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert loss_value() < before
