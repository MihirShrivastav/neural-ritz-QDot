# Neural-Orbital Maps

Standalone research scaffold for neural-orbital configuration-interaction maps in semiconductor quantum-dot qubits.

The v1 scope is intentionally narrow:

- two electrons only,
- 2D effective-mass Hamiltonian,
- no magnetic field, spin-orbit, valley physics, or time dynamics,
- neural Block-Ritz one-electron orbitals,
- singlet/triplet two-electron CI,
- exchange, densities, CI weights, and basic plots.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Quickstart

```bash
nom-pair-ci --config configs/smoke_pair.yaml
nom-analyze-run --run-dir results/smoke_pair/<latest-run>
```

For a direct one-electron run:

```bash
nom-one-electron --config configs/smoke_pair.yaml
```

For a tiny detuning/barrier exchange map:

```bash
nom-exchange-map --config configs/smoke_exchange_map.yaml
```

To resume a known study directory:

```bash
nom-exchange-map --config configs/smoke_exchange_map.yaml --study-dir results/smoke_exchange_map/<study-id>
```

## Outputs

Each run writes:

```text
results/<experiment>/<timestamp>_<name>_k<K>_m<M>_s<seed>/
  config.yaml
  run_meta.json
  logs/
  arrays/
  reports/
  plots/
  checkpoints/
```

Important pair reports include `pair_energies.json`, `pair_exchange.json`, `density_checks.json`, `ci_weights.json`, and `final_summary.json`.

`correlation_report.json` adds natural occupations, orbital entropy, CI participation ratio, left/right density integrals, and charge-sector probabilities `P_20`, `P_11`, and `P_02`.

`one_electron_quality.json` records projected Ritz residuals, basis-overlap conditioning, coefficient orthonormality, and energy gaps. `localized_orbitals.json` records the left/right orbitals constructed from the two lowest states.
