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
- `final_summary.json`

Required pair reports:

- `pair_energies.json`
- `pair_exchange.json`
- `density_checks.json`
- `ci_weights.json`
- `correlation_report.json`
- `pair_correlation_report.json`
- `final_summary.json`

`correlation_report.json` includes normalized natural-occupation entropy, linear entropy, effective orbital count, dominant occupation fraction, charge-sector probabilities, and left/right density integrals.

Important one-electron arrays:

- `orbitals.npy`
- `orbital_energies.npy`
- `ritz_coefficients.npy`
- `localized_orbital_left.npy`
- `localized_orbital_right.npy`

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
