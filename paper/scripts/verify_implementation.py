"""E7: verify the TabM adapter math with an independent implementation.

The official ensemble layer is replaced here by a hand-written equivalent, and
three claims are checked numerically.

1. **Forward equivalence**: for identical weights, the hand-written layer must
   reproduce `tabm_reference.Model` for k in {1, 2, 4, 8}.
2. **Closed-form parameter count**: `params = sum_layers (D_in*D_out + k*(D_in
   + 2*D_out)) + k*(D + d_out)` must equal the real parameter count.
3. **k=1 reduces to a plain MLP**: with r=s=1 and the shared bias, a k=1 TabM
   must produce exactly the same output as an MLP with the same weights. This is
   why the paper's "ensembling without changing the architecture" holds.

Example:
    python paper/scripts/verify_implementation.py
"""

import argparse
import csv
import importlib.util
import types
import warnings
from pathlib import Path

import torch

TOLERANCE = 1e-5


def load_reference(repo_root: Path) -> types.ModuleType:
    """Load `tabm_reference.py` from the repository root by explicit path."""
    path = repo_root / 'tabm_reference.py'
    if not path.exists():
        raise FileNotFoundError(f'Reference implementation not found: {path}')
    spec = importlib.util.spec_from_file_location('tabm_reference', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        spec.loader.exec_module(module)
    return module


class IndependentTabMLP(torch.nn.Module):
    """Hand-written TabM: one shared weight matrix plus k adapters per layer.

    Layer equation (BatchEnsemble, Eq. 5):

        y[b, i, :] = ((x[b, :] * r[i]) @ W.T) * s[i] + bias[i]

    This class does not import any ensembling code from the reference
    implementation; weights are copied in with `load_from_reference`.
    """

    def __init__(
        self,
        n_features: int,
        n_blocks: int,
        d_block: int,
        k: int,
        d_out: int = 1,
    ) -> None:
        super().__init__()
        self.k = k
        shapes = [
            (n_features if block == 0 else d_block, d_block)
            for block in range(n_blocks)
        ]
        self.layers = torch.nn.ParameterList()
        self.shapes = shapes
        self.d_block = d_block
        self.d_out = d_out

        # Flat parameter vector: [W | r | s | bias] per layer, then the output.
        for d_in, d_out_layer in shapes:
            self.layers.append(torch.nn.Parameter(torch.zeros(d_in * d_out_layer)))
            self.layers.append(torch.nn.Parameter(torch.zeros(k * d_in)))
            self.layers.append(torch.nn.Parameter(torch.zeros(k * d_out_layer)))
            self.layers.append(torch.nn.Parameter(torch.zeros(k * d_out_layer)))
        self.output_weight = torch.nn.Parameter(torch.zeros(k * d_block * d_out))
        self.output_bias = torch.nn.Parameter(torch.zeros(k * d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape (B, K, D_OUT)."""
        k = self.k
        x = x[:, None].expand(-1, k, -1)
        cursor = 0
        for d_in, d_out_layer in self.shapes:
            weight, r, s, bias = (
                self.layers[cursor],
                self.layers[cursor + 1],
                self.layers[cursor + 2],
                self.layers[cursor + 3],
            )
            cursor += 4
            x = x * r.view(k, d_in)
            x = x @ weight.view(d_out_layer, d_in).T
            x = x * s.view(k, d_out_layer)
            x = x + bias.view(k, d_out_layer)
            x = torch.relu(x)
        output_weight = self.output_weight.view(k, self.d_block, self.d_out)
        output_bias = self.output_bias.view(k, self.d_out)
        return torch.einsum('bkd,kdo->bko', x, output_weight) + output_bias

    def parameter_count(self) -> int:
        """Count parameters with the closed-form formula, not by enumeration."""
        if self.d_out != 1:
            raise ValueError('The closed form below assumes a single output')
        total = 0
        for d_in, d_out_layer in self.shapes:
            total += d_in * d_out_layer + self.k * (d_in + 2 * d_out_layer)
        return total + self.k * (self.d_block + 1)


def load_from_reference(
    model: IndependentTabMLP, reference: torch.nn.Module
) -> IndependentTabMLP:
    """Copy the weights of `tabm_reference.Model` into the independent model."""
    cursor = 0
    for block in range(len(model.shapes)):
        layer = reference.backbone.blocks[block][0]
        values = [layer.weight, layer.r, layer.s, layer.bias]
        for parameter, value in zip(
            [model.layers[cursor + index] for index in range(4)], values
        ):
            if value is None:
                raise ValueError('This variant always uses r, s and bias')
            with torch.no_grad():
                parameter.copy_(value.reshape(-1))
        cursor += 4
    with torch.no_grad():
        model.output_weight.copy_(reference.output.weight.reshape(-1))
        model.output_bias.copy_(reference.output.bias.reshape(-1))
    return model


def build_reference(reference: types.ModuleType, k: int, d_block: int = 16) -> object:
    return reference.Model(
        n_num_features=5,
        cat_cardinalities=[],
        n_classes=None,
        backbone={'type': 'MLP', 'n_blocks': 2, 'd_block': d_block, 'dropout': 0.0},
        bins=None,
        arch_type='tabm',
        k=k,
    )


def check_forward_equivalence(reference: types.ModuleType, k: int) -> dict:
    torch.manual_seed(0)
    official = build_reference(reference, k).eval()
    mine = load_from_reference(IndependentTabMLP(5, 2, 16, k), official).eval()
    x = torch.randn(32, 5)
    with torch.no_grad():
        expected = official(x_num=x)
        actual = mine(x)
    difference = float((expected - actual).abs().max())
    return {
        'check': f'forward_equivalence_k{k}',
        'reference': float(expected.abs().mean()),
        'independent': float(actual.abs().mean()),
        'difference': difference,
        'tolerance': TOLERANCE,
        'passed': difference <= TOLERANCE,
    }


def check_parameter_formula(reference: types.ModuleType, k: int) -> dict:
    official = build_reference(reference, k)
    mine = IndependentTabMLP(5, 2, 16, k)
    actual = sum(parameter.numel() for parameter in official.parameters())
    formula = mine.parameter_count()
    enumerated = sum(parameter.numel() for parameter in mine.parameters())
    return {
        'check': f'parameter_formula_k{k}',
        'reference': actual,
        'independent': formula,
        'difference': abs(actual - formula) + abs(enumerated - formula),
        'tolerance': 0,
        'passed': actual == formula == enumerated,
    }


def check_k1_reduces_to_mlp(reference: types.ModuleType) -> dict:
    """With r = s = 1, a k=1 TabM must equal a plain MLP of the same weights."""
    torch.manual_seed(0)
    plain = reference.Model(
        n_num_features=5,
        cat_cardinalities=[],
        n_classes=None,
        backbone={'type': 'MLP', 'n_blocks': 2, 'd_block': 16, 'dropout': 0.0},
        bins=None,
        arch_type='plain',
    ).eval()
    tabm = build_reference(reference, 1).eval()
    with torch.no_grad():
        for block in range(2):
            source = plain.backbone.blocks[block][0]
            target = tabm.backbone.blocks[block][0]
            target.weight.copy_(source.weight)
            target.r.fill_(1.0)
            target.s.fill_(1.0)
            target.bias.copy_(source.bias)
        # nn.Linear stores (out, in); NLinear stores (k, in, out).
        tabm.output.weight.copy_(plain.output.weight.T.unsqueeze(0))
        tabm.output.bias.copy_(plain.output.bias.unsqueeze(0))

    x = torch.randn(32, 5)
    with torch.no_grad():
        expected = plain(x_num=x)
        actual = tabm(x_num=x)
    difference = float((expected - actual).abs().max())
    return {
        'check': 'k1_reduces_to_mlp',
        'reference': float(expected.abs().mean()),
        'independent': float(actual.abs().mean()),
        'difference': difference,
        'tolerance': TOLERANCE,
        'passed': difference <= TOLERANCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help='Repository root containing tabm_reference.py',
    )
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    reference = load_reference(args.repo_root)

    rows = [check_forward_equivalence(reference, k) for k in (1, 2, 4, 8)]
    rows += [check_parameter_formula(reference, k) for k in (1, 8)]
    rows.append(check_k1_reduces_to_mlp(reference))

    output = (
        args.output or args.repo_root / 'paper' / 'results' / 'E7' / 'verification.csv'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        status = 'PASS' if row['passed'] else 'FAIL'
        print(
            f'{status}  {row["check"]:26} difference={row["difference"]:.3e}'
            f'  (tolerance {row["tolerance"]})'
        )
    if not all(row['passed'] for row in rows):
        raise SystemExit('At least one verification check failed')
    print(f'\nWrote {output}')


if __name__ == '__main__':
    main()
