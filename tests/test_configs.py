"""Tests for the frozen experiment configurations under `paper/configs/`.

The configs are committed artifacts, so they are validated here as data:
they must be BOM-free TOML (a UTF-8 BOM made training crash once), and they
must stay inside the parameter budget declared in the README.
"""

from pathlib import Path

import pytest
import tomli

CONFIGS_DIR = Path(__file__).resolve().parents[1] / 'paper' / 'configs'
CONFIG_PATHS = sorted(CONFIGS_DIR.glob('*.toml'))

ALLOWED_ARCH_TYPES = {'plain', 'tabm', 'tabm-packed', 'tabm-piecewiselinear'}
ALLOWED_K = {1, 2, 4, 8, 16, 32}
REQUIRED_SECTIONS = {'data', 'optimizer', 'model', 'model.backbone'}


def test_frozen_configs_exist() -> None:
    assert CONFIG_PATHS, f'no frozen configs found in {CONFIGS_DIR}'


@pytest.mark.parametrize('path', CONFIG_PATHS, ids=lambda path: path.name)
def test_config_is_valid_utf8_without_bom(path: Path) -> None:
    raw = path.read_bytes()
    assert not raw.startswith(b'\xef\xbb\xbf'), 'a UTF-8 BOM breaks tomli'
    assert raw.decode('utf-8').strip()


@pytest.mark.parametrize('path', CONFIG_PATHS, ids=lambda path: path.name)
def test_config_parses_and_declares_required_sections(path: Path) -> None:
    config = tomli.loads(path.read_text(encoding='utf-8'))
    nested = {
        'data': 'data' in config,
        'optimizer': 'optimizer' in config,
        'model': 'model' in config,
        'model.backbone': 'backbone' in config.get('model', {}),
    }
    assert {name for name, present in nested.items() if present} == REQUIRED_SECTIONS


@pytest.mark.parametrize('path', CONFIG_PATHS, ids=lambda path: path.name)
def test_config_uses_the_training_protocol(path: Path) -> None:
    config = tomli.loads(path.read_text(encoding='utf-8'))
    assert config['batch_size'] == 256
    assert config['patience'] == 16
    assert config['optimizer']['type'] == 'AdamW'
    assert config['data']['num_policy'] == 'noisy-quantile'


@pytest.mark.parametrize('path', CONFIG_PATHS, ids=lambda path: path.name)
def test_config_respects_the_cpu_parameter_budget(path: Path) -> None:
    """Ensembled configs must use a k from the pre-registered grid."""
    config = tomli.loads(path.read_text(encoding='utf-8'))
    model = config['model']
    if model['arch_type'] != 'plain':
        assert model['k'] in ALLOWED_K
        assert config['model']['backbone']['n_blocks'] <= 2
        assert config['model']['backbone']['d_block'] <= 256


@pytest.mark.parametrize(
    'path',
    [path for path in CONFIG_PATHS if path.name.startswith(('E3-', 'E3b-'))],
    ids=lambda path: path.name,
)
def test_main_method_configs_use_the_frozen_k(path: Path) -> None:
    """E3/E3b are the headline experiments, so they all share k=8."""
    config = tomli.loads(path.read_text(encoding='utf-8'))
    assert config['model']['k'] == 8
    assert config['model']['share_training_batches'] is True


@pytest.mark.parametrize('path', CONFIG_PATHS, ids=lambda path: path.name)
def test_config_arch_type_is_known(path: Path) -> None:
    config = tomli.loads(path.read_text(encoding='utf-8'))
    assert config['model']['arch_type'] in ALLOWED_ARCH_TYPES
