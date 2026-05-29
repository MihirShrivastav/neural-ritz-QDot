import torch

from models.siren import BlockSIREN, SIRENConfig
from training.trainer import train_block
from utils.config import DEFAULT_CONFIG


def _zero_potential(coords: torch.Tensor) -> torch.Tensor:
    return torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)


def test_train_block_stops_early_when_validation_stalls():
    cfg = {
        **DEFAULT_CONFIG,
        "domain": {**DEFAULT_CONFIG["domain"], "nq": 8},
        "solver": {**DEFAULT_CONFIG["solver"], "K": 1, "M": 2},
        "model": {
            **DEFAULT_CONFIG["model"],
            "hidden_features": 8,
            "hidden_layers": 1,
        },
        "training": {
            **DEFAULT_CONFIG["training"],
            "steps": 20,
            "lr": 0.0,
            "sampling_mode": "fixed_grid",
            "diagnostics_every": 5,
            "early_stopping": {
                "enabled": True,
                "eval_every": 2,
                "min_steps": 4,
                "patience_evals": 1,
                "min_delta_rel": 0.0,
                "validation_nq": 8,
            },
        },
    }
    model = BlockSIREN(
        SIRENConfig(
            in_features=2,
            hidden_features=8,
            hidden_layers=1,
            out_features=2,
            first_omega_0=10.0,
            hidden_omega_0=10.0,
        )
    ).to(dtype=torch.float64)

    seen_steps = []
    outputs = train_block(
        model=model,
        potential_fn=_zero_potential,
        cfg=cfg,
        metrics_cb=lambda step, metrics: seen_steps.append(step),
        device=torch.device("cpu"),
    )

    assert outputs.stopped_early is True
    assert outputs.stop_reason is not None
    assert len(outputs.history) < cfg["training"]["steps"]
    assert seen_steps[-1] == len(outputs.history)
    assert outputs.best_validation_eigsum is not None
