from copy import deepcopy

import pytest

from utils.config import DEFAULT_CONFIG, validate_config


def test_validate_config_accepts_default():
    cfg = deepcopy(DEFAULT_CONFIG)
    validate_config(cfg)


def test_validate_config_rejects_bad_lr_schedule_factor():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["training"]["lr_schedule"]["enabled"] = True
    cfg["training"]["lr_schedule"]["factor"] = 1.2
    with pytest.raises(ValueError, match="factor"):
        validate_config(cfg)


def test_validate_config_rejects_bad_lr_schedule_alpha():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["training"]["lr_schedule"]["enabled"] = True
    cfg["training"]["lr_schedule"]["monitor_ema_alpha"] = 0.0
    with pytest.raises(ValueError, match="monitor_ema_alpha"):
        validate_config(cfg)


def test_validate_config_rejects_bad_diagnostics_every():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["training"]["diagnostics_every"] = 0
    with pytest.raises(ValueError, match="diagnostics_every"):
        validate_config(cfg)


def test_validate_config_rejects_bad_early_stopping_validation_nq():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["training"]["early_stopping"]["enabled"] = True
    cfg["training"]["early_stopping"]["validation_nq"] = 1
    with pytest.raises(ValueError, match="validation_nq"):
        validate_config(cfg)


def test_validate_config_rejects_bad_pair_orbital_count():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["pair"]["num_orbitals"] = 1
    with pytest.raises(ValueError, match="pair.num_orbitals"):
        validate_config(cfg)


def test_validate_config_rejects_bad_pair_sector():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["pair"]["sectors"] = ["singlet", "quartet"]
    with pytest.raises(ValueError, match="pair.sectors"):
        validate_config(cfg)


def test_validate_config_rejects_bad_pair_coulomb_params():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["pair"]["coulomb"]["epsilon_r"] = 0.0
    with pytest.raises(ValueError, match="epsilon_r"):
        validate_config(cfg)

    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["pair"]["coulomb"]["softening"] = -0.1
    with pytest.raises(ValueError, match="softening"):
        validate_config(cfg)
