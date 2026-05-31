# Artifact Contract

Each run uses this layout:

```text
results/<experiment>/<timestamp>_<name>_k<K>_m<M>_s<seed>/
  config.yaml
  run_meta.json
  logs/run.log
  arrays/*.npy
  reports/*.json
  plots/*.png
  checkpoints/*.pt
```

Required one-electron reports:

- `one_electron_energies.json`
- `orthonormality.json`
- `one_electron_quality.json`
- `localized_orbitals.json`
- `training_metrics.json`
- `final_summary.json`

Finite-difference baseline reports:

- `finite_difference_energies.json`
- `orthonormality.json`
- `final_summary.json`

Required pair reports:

- `pair_energies.json`
- `pair_exchange.json`
- `density_checks.json`
- `ci_weights.json`
- `hubbard_report.json`
- `correlation_report.json`
- `pair_correlation_report.json`
- `final_summary.json`

`correlation_report.json` includes normalized natural-occupation entropy, linear entropy, effective orbital count, dominant occupation fraction, charge-sector probabilities, left/right density integrals, and heuristic interpretation labels. The labels are run-triage aids, not calibrated physical phase boundaries.

`hubbard_report.json` includes approximate two-site parameters from the lowest two localized orbitals plus a simple singlet/triplet Hubbard estimate. It is an interpretation and validation diagnostic, not the primary solver output.

`training_metrics.json` includes logged eigsum/E0 history and `training_status` with completed steps, configured steps, early-stop flag, stop reason, and best eigsum. `final_summary.json` repeats the training status for one-electron and pair-CI runs.

Important one-electron arrays:

- `orbitals.npy`
- `orbital_energies.npy`
- `ritz_coefficients.npy`
- `localized_orbital_left.npy`
- `localized_orbital_right.npy`

Important finite-difference arrays:

- `fd_orbitals.npy`
- `fd_energies.npy`

Important pair arrays:

- `one_body_density_singlet.npy`
- `one_body_density_triplet.npy`
- `conditional_density_singlet.npy`
- `conditional_density_triplet.npy`
- `pair_correlation_singlet.npy`
- `pair_correlation_triplet.npy`

Exchange-map studies write:

```text
results/<study>/<study-id>/
  study_config.json
  base_config.yaml
  manifest.json
  points.csv
  maps/J_meV.npy
  maps/J_GHz.npy
  maps/dJ_d_detuning.npy
  maps/dJ_d_barrier.npy
  maps/sensitivity_norm.npy
  plots/exchange_map_meV.png
  plots/log_exchange_map.png
  plots/sensitivity_map.png
  summary.json
  points/<point-id>/<run-id>/
```

`manifest.json` records study status, total points, completed points, failed points, resume mode, and links to summary files.

One-electron convergence studies write:

```text
results/<study>/<study-id>/
  study_config.json
  base_config.yaml
  manifest.json
  points.csv
  plots/one_electron_convergence.png
  summary.json
  points/<point-id>/<run-id>/
```

`points.csv` records grid size, neural ground energy, neural energy sum, optional finite-difference ground energy, neural-minus-finite-difference ground-energy error, projected residual, basis-overlap condition number, final normalization deviation, and the point run directory.
