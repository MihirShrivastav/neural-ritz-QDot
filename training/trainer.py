"""Training loop for the block variational solver."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from numerics.autodiff import basis_gradients
from numerics.ritz import project_states, solve_ritz
from numerics.sampling import jittered_grid, monte_carlo_batch, uniform_grid
from training.losses import eigsum_loss, s_condition_penalty


@dataclass
class TrainOutputs:
    eigvals: torch.Tensor
    psi: torch.Tensor
    phi: torch.Tensor
    S: torch.Tensor
    H: torch.Tensor
    history: list[dict]
    stopped_early: bool
    stop_reason: str | None
    best_validation_eigsum: float | None


def _compute_eval_eigsum(
    model: torch.nn.Module,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    X: float,
    Y: float,
    nq: int,
    K: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    batch = uniform_grid(X, Y, nq, device=device, dtype=dtype)
    coords = batch.coords.clone().detach().requires_grad_(True)
    potential = potential_fn(coords).detach()
    phi = model(coords)
    grad_phi = basis_gradients(coords, phi, create_graph=False)
    ritz = solve_ritz(phi, grad_phi, potential, batch.weights)
    return float(eigsum_loss(ritz.eigvals, K).detach().cpu())


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float((g * g).sum().item())
    return total**0.5


def train_block(
    model: torch.nn.Module,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    cfg: dict,
    metrics_cb: Callable[[int, dict], None],
    device: torch.device,
) -> TrainOutputs:
    domain = cfg["domain"]
    solver = cfg["solver"]
    training = cfg["training"]
    loss_w = training["loss_weights"]

    nq = int(domain["nq"])
    K = int(solver["K"])

    dtype = next(model.parameters()).dtype
    sampling_mode = str(training.get("sampling_mode", "jittered_grid")).lower()
    jitter_frac = float(training.get("grid_jitter_frac", 0.35))
    mc_points = int(training.get("mc_points", nq * nq))
    diagnostics_every = int(training.get("diagnostics_every", 25))

    fixed_batch = None
    if sampling_mode == "fixed_grid":
        fixed_batch = uniform_grid(domain["X"], domain["Y"], nq, device=device, dtype=dtype)
    elif sampling_mode not in {"jittered_grid", "monte_carlo"}:
        raise ValueError("training.sampling_mode must be one of: fixed_grid, jittered_grid, monte_carlo")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]))
    sched_cfg = dict(training.get("lr_schedule", {}))
    use_scheduler = bool(sched_cfg.get("enabled", False))
    scheduler = None
    monitor_ema = None
    monitor_alpha = float(sched_cfg.get("monitor_ema_alpha", 0.1))
    if use_scheduler:
        sched_type = str(sched_cfg.get("type", "plateau")).lower()
        if sched_type != "plateau":
            raise ValueError("training.lr_schedule.type must be 'plateau'")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 200)),
            threshold=float(sched_cfg.get("threshold", 1e-4)),
            threshold_mode=str(sched_cfg.get("threshold_mode", "rel")),
            cooldown=int(sched_cfg.get("cooldown", 100)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
            eps=float(sched_cfg.get("eps", 1e-12)),
        )

    grad_clip = float(training.get("grad_clip", 0.0))
    steps = int(training["steps"])
    early_stop_cfg = dict(training.get("early_stopping", {}))
    use_early_stop = bool(early_stop_cfg.get("enabled", False))
    early_eval_every = int(early_stop_cfg.get("eval_every", 50))
    early_min_steps = int(early_stop_cfg.get("min_steps", 0))
    early_patience = int(early_stop_cfg.get("patience_evals", 4))
    early_min_delta_rel = float(early_stop_cfg.get("min_delta_rel", 5e-4))
    validation_nq_raw = early_stop_cfg.get("validation_nq", None)
    validation_nq = int(validation_nq_raw) if validation_nq_raw is not None else nq

    history = []
    last = {}
    last_diag = {
        "s_min_eig": float("nan"),
        "s_max_eig": float("nan"),
        "s_cond_est": float("nan"),
        "grad_norm": float("nan"),
    }
    best_validation = None
    stale_validations = 0
    stopped_early = False
    stop_reason = None

    for step in range(1, steps + 1):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        if fixed_batch is not None:
            batch = fixed_batch
        elif sampling_mode == "jittered_grid":
            batch = jittered_grid(
                domain["X"],
                domain["Y"],
                nq,
                device=device,
                dtype=dtype,
                jitter_frac=jitter_frac,
            )
        else:
            batch = monte_carlo_batch(
                domain["X"],
                domain["Y"],
                mc_points,
                device=device,
                dtype=dtype,
            )

        coords = batch.coords.clone().detach().requires_grad_(True)
        potential = potential_fn(coords).detach()
        weights = batch.weights

        phi = model(coords)
        grad_phi = basis_gradients(coords, phi)
        ritz = solve_ritz(phi, grad_phi, potential, weights)

        loss_e = eigsum_loss(ritz.eigvals, K)
        needs_diag = step == 1 or step % diagnostics_every == 0
        s_eigs = None
        if needs_diag:
            loss_s, s_eigs = s_condition_penalty(ritz.S, return_eigvals=True)
        else:
            loss_s = s_condition_penalty(ritz.S)
        loss = loss_w["eigsum"] * loss_e + loss_w["S_condition"] * loss_s

        loss.backward()
        if needs_diag:
            last_diag["grad_norm"] = _grad_norm(model)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        lr_before_sched = float(optimizer.param_groups[0]["lr"])

        lr_monitor = float(loss_e.detach().cpu())
        if use_scheduler and scheduler is not None:
            if monitor_ema is None:
                monitor_ema = lr_monitor
            else:
                monitor_ema = monitor_alpha * lr_monitor + (1.0 - monitor_alpha) * monitor_ema
            scheduler.step(monitor_ema)
            lr_monitor = float(monitor_ema)

        lr_current = float(optimizer.param_groups[0]["lr"])
        lr_reduced = lr_current < lr_before_sched

        if needs_diag and s_eigs is not None:
            s_eigs_detached = s_eigs.detach()
            s_min = float(s_eigs_detached[0].cpu())
            s_max = float(s_eigs_detached[-1].cpu())
            s_cond = s_max / max(s_min, 1e-12)
            last_diag["s_min_eig"] = s_min
            last_diag["s_max_eig"] = s_max
            last_diag["s_cond_est"] = float(s_cond)

        validation_eigsum = None
        if use_early_stop and step >= early_min_steps and step % early_eval_every == 0:
            validation_eigsum = _compute_eval_eigsum(
                model,
                potential_fn,
                X=float(domain["X"]),
                Y=float(domain["Y"]),
                nq=validation_nq,
                K=K,
                device=device,
                dtype=dtype,
            )
            if best_validation is None:
                best_validation = validation_eigsum
                stale_validations = 0
            else:
                improve_threshold = best_validation * (1.0 - early_min_delta_rel)
                if validation_eigsum < improve_threshold:
                    best_validation = validation_eigsum
                    stale_validations = 0
                else:
                    stale_validations += 1

        metrics = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "loss_eigsum": float(loss_e.detach().cpu()),
            "loss_S_condition": float(loss_s.detach().cpu()),
            "eigvals": [float(v) for v in ritz.eigvals[:K].detach().cpu()],
            "s_min_eig": last_diag["s_min_eig"],
            "s_max_eig": last_diag["s_max_eig"],
            "s_cond_est": last_diag["s_cond_est"],
            "grad_norm": last_diag["grad_norm"],
            "sampling_mode": sampling_mode,
            "num_points": int(coords.shape[0]),
            "lr": lr_current,
            "lr_reduced": bool(lr_reduced),
            "lr_monitor": lr_monitor,
            "validation_eigsum": validation_eigsum,
            "dt_sec": time.time() - t0,
        }
        last = {
            "phi": phi.detach(),
            "S": ritz.S.detach(),
            "H": ritz.H.detach(),
            "eigvals": ritz.eigvals.detach(),
            "eigvecs": ritz.eigvecs.detach(),
        }

        history.append(metrics)
        metrics_cb(step, metrics)

        if use_early_stop and stale_validations >= early_patience:
            stopped_early = True
            stop_reason = (
                "validation eigsum did not improve enough on the deterministic grid "
                f"for {stale_validations} evaluations"
            )
            break

    psi = project_states(last["phi"], last["eigvecs"], K)
    return TrainOutputs(
        eigvals=last["eigvals"][:K],
        psi=psi,
        phi=last["phi"],
        S=last["S"],
        H=last["H"],
        history=history,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        best_validation_eigsum=best_validation,
    )
