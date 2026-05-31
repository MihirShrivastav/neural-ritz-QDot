# Neural-Orbital Maps Tracker

Last updated: 2026-05-31

Status legend:

- `done`: implemented and smoke/test verified.
- `in_progress`: partially implemented; usable but not paper-grade.
- `planned`: accepted next work, not implemented yet.
- `backlog`: important later extension.

## Done

| Status | Item | Notes |
|---|---|---|
| `done` | Standalone project scaffold | Created `neural-orbital-maps/` with `pyproject.toml`, `src/`, `tests/`, `configs/`, `docs/`, `examples/`, `notebooks/`, `scripts/`, and `results/.gitkeep`. |
| `done` | Package entrypoints | Added `nom-one-electron`, `nom-pair-ci`, `nom-analyze-run`, `nom-exchange-map`, `nom-convergence-study`, `nom-fd-baseline`, `nom-fd-pair-ci`, `nom-compare-pair-baselines`, and `nom-pair-baseline-benchmark`. |
| `done` | Config system | Added Pydantic/YAML config models for material, domain, potential, solver, training, pair CI, and exchange-map sweeps. |
| `done` | Run/artifact layout | Runs write `config.yaml`, `run_meta.json`, `logs/`, `arrays/`, `reports/`, `plots/`, and `checkpoints/`. |
| `done` | Logging | CLIs write structured run logs via standard Python logging. |
| `done` | Units/material support | Added effective-mass energy scale and softened Coulomb material scale. |
| `done` | Analytic double-dot potential | Added dimensionless biquadratic double-dot potential with separation, barrier, y confinement, and detuning. |
| `done` | Uniform grid/quadrature | Added deterministic rectangular grid and cell-area weights. |
| `done` | Neural basis model | Added SIREN/MLP basis network with Gaussian envelope. |
| `done` | One-electron Block-Ritz trainer | Added weak-form first-derivative Ritz assembly, generalized eigensolve, Adam training, checkpoint, energies, orbitals, plots, and reports. |
| `done` | Two-electron CI | Added singlet/triplet sector basis construction, Coulomb tensor, product Hamiltonian, sector diagonalization, and one-body densities. |
| `done` | Pair reports | Added `pair_energies.json`, `pair_exchange.json`, `density_checks.json`, `ci_weights.json`, and `final_summary.json`. |
| `done` | Pair plots | Added potential, per-state density, singlet/triplet one-body density, conditional density, exchange summary, and singlet-triplet density-difference plots. |
| `done` | Correlation observables | Added natural occupations, orbital entropy, CI participation ratio, and left/right density integrals. |
| `done` | Localized orbital diagnostics | Added left/right localized orbital construction from the two lowest orbitals, reports, arrays, and plots. |
| `done` | One-electron quality report | Added projected Ritz residuals, basis-overlap conditioning, coefficient orthonormality diagnostics, and energy gaps. |
| `done` | Training termination diagnostics | Added relative early-stop tolerance, non-finite loss failure guard, and persisted training-stop metadata in metrics/final summaries. |
| `done` | Pair-correlation diagnostics | Added conditional pair-correlation ratio maps, reports, arrays, and plots. |
| `done` | Pair summary dashboard | Added compact multi-panel dashboard with potential, singlet density, triplet density, and pair correlation. |
| `done` | Natural-occupation entanglement summaries | Added normalized entropy, linear entropy, effective orbital count, and dominant occupation fraction. |
| `done` | Correlation interpretation labels | Added heuristic correlation, CI-mixing, and charge-regime labels to `correlation_report.json` for run triage. |
| `done` | Final-grid orbital normalization | Added configurable final orbital renormalization and before/after norm reporting. |
| `done` | Post-run analyzer | `nom-analyze-run` summarizes final, energy, exchange, density, CI, and correlation reports. |
| `done` | Exchange-map runner | `nom-exchange-map` sweeps detuning/barrier, runs pair CI at each point, writes `points.csv`, map arrays, map plots, and summary. |
| `done` | Exchange sensitivity maps | Added finite-difference `dJ/detuning`, `dJ/dbarrier`, and sensitivity norm maps. |
| `done` | Sweep resume and manifest | Added explicit `--study-dir`, `--no-resume`, reusable `points.csv`, failed-point accounting, and `manifest.json`. |
| `done` | One-electron convergence study runner | Added `nom-convergence-study`, grid/basis/hidden-dim/seed axes, convergence CSV/JSON summaries, and per-axis convergence plots. |
| `done` | Finite-difference one-electron baseline | Added a five-point Dirichlet finite-difference solver, baseline CLI, reports, arrays, plots, and tests. |
| `done` | Finite-difference pair-CI baseline | Added a workflow/CLI that feeds finite-difference orbitals through the same singlet/triplet pair-CI solver and writes standard pair artifacts. |
| `done` | Pair baseline comparison report | Added neural-vs-finite-difference pair-CI comparison JSON and bar plot for exchange `J`. |
| `done` | Controlled pair-baseline benchmark | Added a runner that executes neural pair CI, finite-difference pair CI, and baseline comparison in one reproducible study folder. |
| `done` | Baseline-aware convergence summaries | One-electron convergence studies can include finite-difference reference energies and neural-minus-baseline errors. |
| `done` | Charge-sector probabilities | Added spatial-projector estimates of `P_20`, `P_11`, `P_02`, double occupancy, and charge imbalance. |
| `done` | Two-site Hubbard diagnostic | Added approximate `t`, `U`, `V`, exchange `K`, and Hubbard singlet/triplet estimate from the lowest localized orbital pair. |
| `done` | CI-vs-Hubbard comparison plot | Pair runs now plot full CI exchange against the two-site Hubbard estimate. |
| `done` | Smoke configs | Added `configs/smoke_pair.yaml`, `configs/default_pair.yaml`, `configs/smoke_exchange_map.yaml`, `configs/smoke_convergence.yaml`, and `configs/smoke_convergence_axes.yaml`. |
| `done` | Future schema stub | Added `configs/exchange_map_stub.yaml` documenting intended future map schema. |
| `done` | Documentation | Added README plus docs for physics model, architecture, artifact contract, and research roadmap. |
| `done` | Test coverage | Added tests for config validation, units, grid, Ritz assembly, pair CI, observables, CLI workflow, convergence-study workflow, and exchange-map workflow. |
| `done` | Verification | Verified `pytest` passes with 32 tests; verified `nom-pair-ci`, `nom-one-electron`, `nom-analyze-run`, `nom-exchange-map`, `nom-convergence-study`, `nom-fd-baseline`, `nom-fd-pair-ci`, `nom-compare-pair-baselines`, and `nom-pair-baseline-benchmark` smoke paths. |

## In Progress

| Status | Item | Current State | Next Step |
|---|---|---|---|
| `in_progress` | Exchange-map framework | Rectangular detuning/barrier sweeps and resume work, but execution is cold-started and serial. | Add parallel execution and warm starts. |
| `in_progress` | Correlation/entanglement metrics | Natural-occupation entropy, linear entropy, effective orbital count, CI PR, charge sectors, localized orbitals, pair-correlation maps, heuristic interpretation labels, one-electron and pair-CI finite-difference validation support, Hubbard diagnostics, baseline comparison reports, and controlled benchmark runner exist. | Add disorder-aware validation examples after disorder generators are implemented. |
| `in_progress` | Plotting | Basic scientific plots, single-run dashboard, exchange maps, sensitivity maps, one-electron convergence plots, CI-vs-Hubbard comparison plots, neural-vs-FD pair baseline plots, and controlled benchmark comparison plots exist. | Add publication-style comparison plots for disorder ensembles after disorder workflows exist. |
| `in_progress` | One-electron solver quality | Thin runnable neural Block-Ritz core, quality diagnostics, final normalization, absolute/relative early stopping, non-finite loss guards, grid/basis/hidden-dim/seed convergence automation, finite-difference reference comparisons, and controlled pair-baseline benchmark exist. | Add runtime benchmarks. |

## Planned Next

| Status | Item | Implementation Target |
|---|---|---|
| `planned` | Parallel sweep execution | Add configurable worker count for independent sweep points. |
| `planned` | Warm-start continuation | Allow each sweep point to initialize from a nearby completed checkpoint; add state-overlap alignment. |
| `planned` | Orbital alignment | Track orbital signs/order across parameter sweeps using overlap matrices. |
| `planned` | Disorder generators | Add Gaussian random field, impurity, barrier disorder, and detuning-bias disorder. |
| `planned` | Disorder ensemble runner | Run many disorder realizations and report mean/std/quantiles of `J`, sensitivity, and sweet-spot shifts. |
| `planned` | Noise-aware sweet-spot score | Combine `J`, gradients, and gate-noise covariance into a quality metric. |
| `planned` | Imported potential maps | Support `.npy`/`.npz` potential grids with unit metadata. |
| `planned` | Runtime benchmarks | Compare neural one-electron/pair workflows against finite-difference baselines on common smoke and realistic configs. |
| `planned` | CI-vs-Hubbard validation plots | Compare CI exchange against Hubbard estimates across controlled parameter sets. |
| `planned` | Pair-CI convergence studies | Extend convergence workflow to pair-CI controls such as `num_orbitals` and Coulomb softening. |
| `planned` | Paper figure recipes | Add reproducible commands/configs for each expected manuscript figure. |

## Backlog

| Status | Item | Notes |
|---|---|---|
| `backlog` | Magnetic field | Needed for Zeeman splitting and Fock-Darwin-style comparisons. |
| `backlog` | Spin-orbit coupling | Needed for hole qubits and spin-orbit leakage studies. |
| `backlog` | Valley physics | Needed for realistic Si/SiGe qubit modelling. |
| `backlog` | Finite-thickness Coulomb kernel | Replace simple softening with a more physical quasi-2D kernel. |
| `backlog` | Electrostatic TCAD bridge | Import potentials from external device simulators. |
| `backlog` | Multi-dot arrays | Extend beyond a single two-electron double dot toward residual exchange/crosstalk maps. |
| `backlog` | More than two electrons | Requires larger CI spaces or different many-body strategy. |
| `backlog` | Neural operators/surrogates | Amortize exchange-map prediction once high-fidelity data exists. |
| `backlog` | VMC/Jastrow correction | Add correlated neural correction after CI baseline is validated. |
| `backlog` | Notebook dashboard | Add Colab/local notebook for running maps and inspecting outputs. |

## Current Verification Commands

```powershell
cd neural-orbital-maps
python -m pip install -e ".[dev]"
pytest
nom-pair-ci --config configs\smoke_pair.yaml
nom-exchange-map --config configs\smoke_exchange_map.yaml
nom-convergence-study --config configs\smoke_convergence.yaml
nom-convergence-study --config configs\smoke_convergence_axes.yaml
nom-fd-baseline --config configs\smoke_pair.yaml
nom-fd-pair-ci --config configs\smoke_pair.yaml
nom-compare-pair-baselines --neural-run <neural-run> --fd-run <fd-run> --output-dir <output-dir>
nom-pair-baseline-benchmark --config configs\smoke_pair.yaml
```

Expected current test result:

```text
32 passed
```
