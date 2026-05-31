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
| `done` | Package entrypoints | Added `nom-one-electron`, `nom-pair-ci`, `nom-analyze-run`, `nom-exchange-map`, `nom-convergence-study`, and `nom-fd-baseline`. |
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
| `done` | One-electron convergence study runner | Added `nom-convergence-study`, `configs/smoke_convergence.yaml`, convergence CSV/JSON summaries, and convergence plot. |
| `done` | Finite-difference one-electron baseline | Added a five-point Dirichlet finite-difference solver, baseline CLI, reports, arrays, plots, and tests. |
| `done` | Baseline-aware convergence summaries | One-electron convergence studies can include finite-difference reference energies and neural-minus-baseline errors. |
| `done` | Charge-sector probabilities | Added spatial-projector estimates of `P_20`, `P_11`, `P_02`, double occupancy, and charge imbalance. |
| `done` | Smoke configs | Added `configs/smoke_pair.yaml`, `configs/default_pair.yaml`, `configs/smoke_exchange_map.yaml`, and `configs/smoke_convergence.yaml`. |
| `done` | Future schema stub | Added `configs/exchange_map_stub.yaml` documenting intended future map schema. |
| `done` | Documentation | Added README plus docs for physics model, architecture, artifact contract, and research roadmap. |
| `done` | Test coverage | Added tests for config validation, units, grid, Ritz assembly, pair CI, observables, CLI workflow, convergence-study workflow, and exchange-map workflow. |
| `done` | Verification | Verified `pytest` passes with 25 tests; verified `nom-pair-ci`, `nom-one-electron`, `nom-analyze-run`, `nom-exchange-map`, `nom-convergence-study`, and `nom-fd-baseline` smoke paths. |

## In Progress

| Status | Item | Current State | Next Step |
|---|---|---|---|
| `in_progress` | Exchange-map framework | Rectangular detuning/barrier sweeps and resume work, but execution is cold-started and serial. | Add parallel execution and warm starts. |
| `in_progress` | Correlation/entanglement metrics | Natural-occupation entropy, linear entropy, effective orbital count, CI PR, charge sectors, localized orbitals, pair-correlation maps, heuristic interpretation labels, and one-electron finite-difference validation support exist. | Add Hubbard-style two-site reduction and CI-vs-Hubbard validation examples. |
| `in_progress` | Plotting | Basic scientific plots, single-run dashboard, exchange maps, sensitivity maps, and one-electron convergence plots exist. | Add publication-style comparison plots for baselines and disorder ensembles. |
| `in_progress` | One-electron solver quality | Thin runnable neural Block-Ritz core, quality diagnostics, final normalization, absolute/relative early stopping, non-finite loss guards, grid convergence automation, and finite-difference reference comparisons exist. | Add expanded convergence axes and pair-CI baseline comparisons. |

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
| `planned` | Pair-CI finite-difference baseline | Feed finite-difference orbitals through the same pair CI path for direct exchange benchmarking. |
| `planned` | Hubbard baseline | Extract approximate `t`, `U`, `V`, and compare CI exchange against Hubbard estimates. |
| `planned` | Expanded convergence studies | Extend current grid convergence runner to `num_orbitals`, Coulomb softening, training seed, and network capacity convergence. |
| `planned` | Runtime benchmarks | Compare cold neural runs, warm-started neural runs, and finite-difference baseline runtimes. |
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
nom-fd-baseline --config configs\smoke_pair.yaml
```

Expected current test result:

```text
25 passed
```
