# Neural-Orbital Maps Research Plan

Research target: a fast, interpretable neural-orbital framework for exchange, entanglement, disorder sensitivity, and operating-point discovery in semiconductor quantum-dot qubits.

This document is the working research and build specification for the `neural-orbital-maps` branch. It is intentionally broader than an implementation ticket: it defines the scientific problem, current landscape, gap, proposed contribution, required physics, software architecture, validation plan, and paper path.

## 1. Working Thesis

Semiconductor quantum-dot qubits need predictive, fast, and interpretable models of few-electron spectra as device geometry, gate voltages, disorder, and material parameters vary. The most important low-energy quantity for many spin-qubit operations is the exchange splitting

```text
J = E_T - E_S
```

where `E_S` is the two-electron singlet energy and `E_T` is the lowest triplet energy. `J` controls singlet-triplet qubits, exchange gates, residual coupling, leakage risk, and sensitivity to charge noise.

The central paper thesis should be:

> A symmetry-preserving neural-orbital configuration-interaction framework can learn compact device-adapted orbitals and produce fast, interpretable exchange, entanglement, and noise-sensitivity maps for semiconductor quantum-dot qubits across gate and disorder parameter spaces.

The contribution is not merely "a neural network solves the Schrodinger equation." The stronger contribution is a device-analysis framework that combines:

- continuous-space neural Block-Ritz orbitals,
- exact spin-symmetry-resolved two-electron CI,
- efficient Coulomb tensor assembly,
- parameter sweeps over physically meaningful device controls,
- uncertainty/disorder analysis,
- exchange and entanglement observables,
- interpretability through orbitals, CI weights, densities, Coulomb matrix elements, and sensitivity maps.

## 2. Problem Statement

Scaling semiconductor spin qubits requires reliable control of Hamiltonian parameters in dense gate-defined quantum-dot arrays. The hard part is not only producing quantum dots. It is predicting and tuning the effective low-energy Hamiltonian:

```text
H_eff ~= sum_i epsilon_i n_i + sum_<ij> t_ij c_i^dag c_j + sum_i U_i n_i_up n_i_down
       + sum_<ij> V_ij n_i n_j + sum_<ij> J_ij S_i . S_j + ...
```

For two-electron double dots, the key experimentally relevant quantity is exchange:

```text
J(epsilon, barrier, geometry, disorder, material) = E_triplet_0 - E_singlet_0
```

where `epsilon` is detuning and the barrier controls tunnel coupling. The practical questions are:

- Where is `J` large enough for fast gates?
- Where is `J` small enough for idle isolation?
- Where is `J` insensitive to detuning or barrier noise?
- How much does disorder shift the exchange landscape?
- Can a design tolerate fabrication variability?
- Which changes in the wavefunction explain a change in `J`?
- When does a simple Hubbard model fail?
- How many orbitals are needed for reliable exchange?

These questions are directly connected to quantum computing because exchange-coupled spins are a leading route to compact, scalable, electrically controlled qubits. Experimental singlet-triplet qubit work describes the qubit Hamiltonian in terms of exchange `J(epsilon)` and a magnetic-field gradient, with `J` manipulated by detuning between dots. Closed-loop GaAs singlet-triplet control reached high fidelity, but the same literature makes clear that exchange, leakage, and noise sensitivity are central operating concerns.

## 3. Current Landscape

### 3.1 Device-scale exact diagonalization

Commercial/device simulators such as QTCAD use the standard route:

1. solve electrostatics and the single-particle Schrodinger problem,
2. take a truncated set of single-particle orbitals,
3. evaluate Coulomb integrals,
4. build a many-body Hamiltonian,
5. diagonalize it in fixed-particle-number subspaces.

QTCAD's many-body theory states that mean-field Schrodinger-Poisson does not exactly treat exchange and correlation, while exact diagonalization in a truncated basis does include exchange and correlation effects within that basis. It writes the many-body Hamiltonian in terms of single-particle energies and Coulomb integrals `V_ijkl`, exactly matching the skeleton of our electron-pair solver.

Relevant source:

- QTCAD many-body theory: https://docs.nanoacademic.com/qtcad/theory_spin_fem/manybody/

QTCAD's exchange tutorial computes exchange in a double quantum dot using exact diagonalization. It explicitly says the expensive part is the Coulomb integral calculation, not diagonalization, and gives the usual exchange definition from singlet-triplet energy difference.

Relevant source:

- QTCAD exchange exact diagonalization tutorial: https://docs.nanoacademic.com/qtcad/tutorials/device/exchange_2/

Implication for us:

- We are aligned with accepted physics.
- The opportunity is speed, sweepability, continuous parameterization, and interpretability.
- Reviewers will expect convergence studies over basis size and grid resolution.

### 3.2 Perturbative Hubbard-style modelling

Simplified models estimate exchange from tunnel coupling and Coulomb parameters, for example via Fermi-Hubbard reductions. These models are useful but can fail outside their validity regime. QTCAD's tutorial shows exact diagonalization and perturbation theory can differ significantly for a coarse basis/mesh and only converge with better resolution and larger basis.

Implication for us:

- We should report both exact neural-orbital CI values and extracted Hubbard parameters.
- A paper can show when Hubbard reductions are reliable and when they miss correlation, disorder, or orbital deformation effects.

### 3.3 ML for quantum-dot control and tuning

Recent ML work in quantum dots often focuses on automated control, charge-stability-diagram interpretation, crosstalk calibration, and disorder-parameter inference. A 2025 PRX/NIST work introduces an autonomous virtualization system for 2D quantum-dot arrays, using ML to extract features from charge stability diagrams and determine cross-capacitance matrices. A 2024 neural-network method predicts disorder parameters in extended Hubbard models from charge stability diagrams with high reported accuracy.

Relevant sources:

- MAViS autonomous virtual gates: https://www.nist.gov/publications/modular-autonomous-virtualization-system-two-dimensional-semiconductor-quantum-dot
- CNN disorder inference for quantum-dot qubits: https://arxiv.org/abs/2405.04524

Implication for us:

- Existing ML is strong at experimental image/control inference.
- Our distinct angle is physics-resolved simulation of wavefunctions, exchange, pair correlations, and sensitivity landscapes from continuous potentials.
- Our outputs could eventually feed auto-tuning systems: not just "where are the charge transitions?", but "where is exchange fast, quiet, and robust?"

### 3.4 Ab initio and real-space spin-qubit modelling

Recent real-space modelling work emphasizes that foundry-compatible quantum processors require predictive modelling of interacting electrons in realistic geometries and non-ideal environments. Such work computes exchange from electrode geometry and simulates gate dynamics/charge-noise robustness.

Relevant source:

- Ab initio modelling of quantum dot qubits: https://arxiv.org/abs/2403.00191

Implication for us:

- A realistic paper must connect from potentials and gate knobs to device observables, not stop at abstract wells.
- We do not need full foundry electrostatics in the first paper, but our potential families must be physically interpretable and extensible to imported electrostatic maps.

### 3.5 Residual exchange and scalable arrays

Residual exchange cannot always be turned fully off. A 2024 Physical Review Research paper studies residual exchange in linear spin-qubit arrays and emphasizes its importance for scaling, gate errors, and charge-noise tradeoffs.

Relevant source:

- Residual exchange in spin-qubit arrays: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.013153

Implication for us:

- A strong second paper or extension is "residual exchange maps under fabrication disorder."
- Even v1 double-dot maps should include low-`J` tails and idle-point robustness, not only high-`J` gate points.

### 3.6 Disorder and material variability

Disorder is not a nuisance detail. Si/SiGe interface-disorder modelling reconstructed from microscopy found large qubit-relevant variability, including roughly 50% valley-splitting variability from alloy disorder and roughness-induced double-dot detuning bias variability of order `1-10 meV`, depending on well thickness.

Relevant source:

- Si/SiGe interface disorder and qubit variability: https://www.osti.gov/pages/biblio/2329339

Germanium hole-qubit work emphasizes anisotropic noise sensitivity, site-dependent qubit properties, and sweet-spot complications. It reports that optimal orientations can improve charge-noise-limited coherence metrics by about an order of magnitude in that setting.

Relevant sources:

- Ge hole-qubit sweet spots and anisotropic noise: https://www.nature.com/articles/s41563-024-01857-5
- Planar germanium hole-qubit modelling: https://www.nature.com/articles/s41534-024-00897-8

Implication for us:

- Disorder and sensitivity maps are not optional if we want practical relevance.
- First framework should include scalar disorder in the 2D confinement potential; later versions can include valley, spin-orbit, g-tensor, and multi-band effects.

### 3.7 Neural Schrodinger solvers and neural wavefunctions

There is prior work on neural networks for Schrodinger eigenproblems, arbitrary quantum wells, PINN eigenfunctions, and multi-electron neural wavefunctions.

Representative sources:

- Neural networks for arbitrary quantum wells: https://www.nature.com/articles/s41598-022-06442-x
- PINNs for quantum eigenvalue problems: https://arxiv.org/abs/2203.00451
- FermiNet: https://arxiv.org/abs/1909.02487
- PauliNet: https://arxiv.org/abs/1909.08423
- Transferable fermionic neural wavefunctions: https://www.nature.com/articles/s41467-023-44216-9

Implication for us:

- A generic "PINN solves Schrodinger" paper is not enough.
- Direct many-electron neural wavefunctions are powerful but expensive and impose antisymmetry by architecture. They are a long-term method direction, not the fastest path to a device paper.
- Our novelty should be neural orbitals plus exact CI plus device observables, not replacing exact CI with a black-box many-body network.

## 4. Gap We Can Fill

Existing methods form three clusters:

| Method class | Strength | Weakness / gap |
|---|---|---|
| FEM/finite-difference + exact diagonalization | Accurate and accepted | Expensive sweeps; Coulomb integrals/convergence bottlenecks; less ML-native; limited amortization |
| Hubbard/perturbative models | Fast and interpretable | Requires parameter extraction; may fail with strong orbital deformation, disorder, asymmetry, or correlation |
| ML tuning/control models | Useful for experimental automation | Usually infer controls from charge diagrams; do not directly resolve continuous-space wavefunctions, Coulomb integrals, pair densities, or exchange landscapes |
| Neural many-electron VMC | General and high-accuracy | Heavy optimization; not tailored to device sweeps; often overkill for two-electron qubit maps |

Our target gap:

> A fast, continuous-space, symmetry-preserving, interpretable exchange-map engine that is accurate enough to benchmark against exact diagonalization and fast enough to sweep design/noise/disorder parameters.

## 5. Proposed Framework

Working name: Neural-Orbital Maps.

Pipeline:

```text
device/gate/disorder parameters
        |
        v
continuous 2D confinement potential V(x, y; p)
        |
        v
shared neural Block-Ritz orbital solver
        |
        v
device-adapted one-electron orbitals and energies
        |
        v
FFT/direct Coulomb tensor assembly
        |
        v
singlet/triplet CI diagonalization
        |
        v
exchange, densities, pair correlations, CI weights, entanglement metrics
        |
        v
sweeps, sensitivity maps, sweet spots, robustness reports
```

### 5.1 Current repo foundation

Already implemented:

- Shared neural Block-Ritz one-electron solver.
- Deterministic fixed-grid post-training artifacts.
- Two-electron neural-orbital CI solver.
- Singlet/triplet sector construction.
- Softened material-scaled Coulomb interaction.
- FFT-convolution Coulomb assembly path.
- Pair outputs: energies, exchange, densities, CI coefficients, conditional densities, plots.

### 5.2 Required extension

To become a paper-grade framework, we need:

- Parameterized physical potential families.
- Batch/sweep orchestration.
- Warm-started training across nearby parameters.
- Imported potential-map support.
- Disorder generators.
- Sensitivity derivatives.
- Entanglement and localization metrics.
- Benchmark baselines.
- Convergence and uncertainty reporting.
- Publication-quality analysis scripts.

## 6. Physics We Must Model

### 6.1 Two-electron exchange

Core model:

```text
H_2 = h(1) + h(2) + e^2 / (4 pi eps0 eps_r sqrt(|r1-r2|^2 + a_soft^2))
```

where `h` is the one-electron Hamiltonian:

```text
h = -nabla^2 + V(x, y)
```

Outputs:

- `E_S0`
- `E_T0`
- `J = E_T0 - E_S0`
- `J_meV`
- `J_GHz`
- excited singlet/triplet levels
- singlet-triplet gap to leakage states

### 6.2 Gate controls

Minimum controls:

- detuning `epsilon`: left-right potential bias,
- barrier height/control `B`: central barrier and tunnel coupling,
- dot separation `d`,
- confinement frequency/curvature `hbar_omega_x`, `hbar_omega_y`,
- dot asymmetry,
- tilt/displacement,
- material `m_eff`, `epsilon_r`, `L0_nm`.

Derived controls:

- effective tunnel coupling `t`,
- on-site Coulomb `U_L`, `U_R`,
- interdot Coulomb `V_LR`,
- Hubbard exchange estimate for comparison.

### 6.3 Noise and sensitivity

For charge noise and gate-noise analysis:

```text
dJ/depsilon
dJ/dB
d2J/depsilon2
d2J/dB2
||grad_p log J||
```

Sweet-spot definitions:

- detuning sweet spot: `|dJ/depsilon|` small,
- barrier sweet spot: `|dJ/dB|` small,
- multi-parameter sweet region: low gradient norm under a gate-noise covariance matrix,
- practical gate region: sufficiently high `J` and sufficiently low sensitivity.

Noise-aware score:

```text
sigma_J^2 ~= grad_p J^T Sigma_p grad_p J
quality ~= J / sigma_J
```

### 6.4 Disorder

First disorder models should be simple, controllable, and explainable:

- Gaussian random field disorder with amplitude and correlation length.
- Interface-roughness-like smooth potential perturbations.
- Dot-position disorder.
- Barrier disorder.
- Detuning bias disorder.
- Random local charge impurities.

Later models:

- imported atomistic/disorder maps,
- valley splitting disorder for Si/SiGe,
- g-tensor disorder for Ge hole qubits,
- dielectric-boundary/image-charge corrections.

### 6.5 Entanglement and correlation

We need to define which entanglement we mean. There are multiple useful notions:

1. Spin entanglement between the two electron spins.
2. Spatial/orbital entanglement between left and right dots.
3. Mode entanglement in a localized orbital basis.
4. Correlation beyond a single Slater determinant.

For our two-electron CI state, implement:

- one-body reduced density matrix `gamma_ij`,
- natural orbital occupations,
- von Neumann entropy of normalized one-body density matrix,
- participation ratio of CI coefficients,
- double-occupancy probability,
- left/right occupation probabilities,
- charge-sector probabilities: `(2,0)`, `(1,1)`, `(0,2)`,
- singlet-triplet spin sector label,
- pair-correlation function `g(r1, r2)`,
- conditional density maps.

Candidate metrics:

```text
S_1 = -Tr(gamma_tilde log gamma_tilde)
PR_CI = 1 / sum_k |c_k|^4
P_11, P_20, P_02
D = P_20 + P_02
localization imbalance = <n_L - n_R>
```

For actual qubit entanglement between two spin qubits, we need a larger Hilbert space than one two-electron double dot. That is a later extension to coupled double dots or multi-dot arrays.

### 6.6 Leakage

For singlet-triplet qubits, leakage states matter:

- triplet `T+`, `T-` in magnetic field,
- orbital excited singlets/triplets,
- charge leakage `(0,2)`/`(2,0)` depending on operating point,
- valley leakage in silicon,
- spin-orbit leakage for holes.

In v1, spin-independent zero-field model can report:

- orbital excitation gap above qubit subspace,
- singlet-triplet separation,
- charge-sector probabilities.

Magnetic field, valley, and spin-orbit leakage require later physics.

## 7. Why This Should Work

### 7.1 Device-adapted orbitals are compact

For few-electron quantum dots, low-energy physics is dominated by a small number of bound orbitals localized in the confinement region. A neural Block-Ritz solver learns a compact subspace tailored to the continuous potential, avoiding a large generic finite-difference basis.

### 7.2 Exact CI handles the crucial two-electron physics

Exchange and correlation arise from Coulomb interaction plus fermionic symmetry. By solving singlet and triplet CI sectors explicitly, we preserve the physics that matters for two-electron qubits. We avoid a black-box supervised regression model for `J`.

### 7.3 Parameter sweeps are structured

Neighboring gate settings produce nearby potentials and nearby orbitals. This creates opportunities for:

- warm starts,
- continuation in parameter space,
- caching orbital products and Coulomb kernels,
- adaptive sampling near transitions,
- surrogate fitting on top of high-fidelity solver outputs.

### 7.4 Interpretability is built in

Unlike a direct `p -> J` regressor, the framework exposes:

- orbitals,
- energies,
- Coulomb integrals,
- CI coefficients,
- densities,
- charge-sector weights,
- pair correlations,
- sensitivity gradients.

That makes it scientifically defensible and useful for device design.

## 8. Concrete Software Workstreams

### Workstream A: Parametric potentials

Add physically meaningful potential families:

- `biquadratic_dqd`: already present, extend controls for detuning/barrier/separation.
- `two_gaussian_dots`: two attractive wells plus central barrier.
- `quartic_with_barrier_gate`: analytic model with direct barrier knob.
- `imported_grid`: load potential from `.npy`, `.npz`, `.csv`, or HDF5.
- `disordered_base`: wraps any base potential with disorder.

Config target:

```json
{
  "study": {
    "type": "exchange_map",
    "parameters": {
      "detuning": {"min": -2.0, "max": 2.0, "num": 41},
      "barrier_meV": {"min": 2.0, "max": 12.0, "num": 31}
    }
  }
}
```

### Workstream B: Sweep runner

Add:

- `experiments/run_exchange_map.py`
- sweep manifest JSON/CSV,
- resume support,
- failed-run isolation,
- deterministic seeds per point,
- optional parallelism,
- warm-start from nearest completed point,
- adaptive refinement where `|grad J|` or uncertainty is high.

Outputs:

```text
results/<study>/<study-id>/
  manifest.json
  points.csv
  maps/exchange_grid.npy
  maps/sensitivity_grid.npy
  maps/quality_grid.npy
  plots/J_map.png
  plots/logJ_map.png
  plots/sweet_spot_map.png
  plots/disorder_variability.png
```

### Workstream C: Warm-started neural orbital solving

Needed for speed:

- save/load previous checkpoint,
- initialize from nearest parameter point,
- optionally freeze lower layers for early continuation,
- early-stop based on deterministic validation,
- detect orbital swaps/sign flips,
- align states across sweep points by overlap matrix.

State tracking:

```text
O_ij(p, p+dp) = <psi_i(p) | psi_j(p+dp)>
```

Use `O` to maintain consistent labels and detect avoided crossings.

### Workstream D: Exchange and sensitivity analysis

Add analysis module:

- compute finite-difference derivatives,
- compute central differences on rectangular sweeps,
- fit local polynomial response surfaces,
- propagate gate-noise covariance to `sigma_J`,
- identify sweet regions under constraints.

Report:

```json
{
  "J_meV": ...,
  "J_GHz": ...,
  "dJ_d_detuning": ...,
  "dJ_d_barrier": ...,
  "noise_sigma_J": ...,
  "quality_J_over_sigma": ...,
  "sweet_spot_score": ...
}
```

### Workstream E: Disorder ensemble engine

Add:

- disorder config schema,
- Gaussian random fields,
- random impurities,
- asymmetric gate perturbations,
- correlated left/right detuning offsets,
- ensemble statistics.

Outputs:

- mean/std/quantiles of `J`,
- probability of `J` outside tolerance,
- disorder-induced sweet-spot shift,
- maps of exchange variability.

### Workstream F: Entanglement and correlation metrics

Add:

- one-body reduced density matrix in orbital basis,
- natural occupations,
- orbital entropy,
- CI participation ratio,
- left/right charge-sector classifier,
- double-occupancy probability,
- pair-correlation maps.

Need localized basis:

- construct left/right localized orbitals from bonding/antibonding orbitals,
- or use spatial projectors onto left/right regions.

Practical left/right projectors:

```text
P_L = integral_{x < x_cut} rho(x,y) dxdy
P_R = integral_{x >= x_cut} rho(x,y) dxdy
```

Better localized basis:

```text
phi_L = (psi_0 + psi_1) / sqrt(2)
phi_R = (psi_0 - psi_1) / sqrt(2)
```

with sign alignment and overlap checks.

### Workstream G: Benchmarks

Minimum baselines:

- finite-difference one-electron solver + same pair CI,
- analytic harmonic double-dot or Fock-Darwin checks where possible,
- Hubbard estimate `J_Hubbard` from extracted `t`, `U`, `V`,
- convergence against `num_orbitals`,
- grid convergence,
- softening convergence,
- seed convergence,
- runtime comparison.

Ideal external comparison:

- QTCAD-style exchange tutorial reproduction or parameter-matched simplified benchmark.

### Workstream H: Publication-grade visualization

Required plots:

- exchange map `J(detuning, barrier)`,
- `log10 J` map,
- sensitivity map `|grad J|`,
- sweet-spot score map,
- disorder ensemble mean/std maps,
- representative singlet/triplet densities,
- conditional pair-density slices,
- CI weight spectrum across regimes,
- natural occupation / entropy maps,
- convergence curves,
- runtime/accuracy Pareto plot versus baseline.

## 9. Architecture Sketch

Proposed modules:

```text
studies/
  exchange_map.py
  disorder_ensemble.py
  convergence.py

analysis/
  exchange.py
  sensitivity.py
  entanglement.py
  localization.py
  sweep_tables.py

physics/
  disorder.py
  gates.py
  materials.py
  potentials.py

numerics/
  pair_ci.py
  finite_difference.py
  orbital_alignment.py
  coulomb.py

utils/
  sweep_manager.py
  artifact_index.py
```

Entrypoints:

```text
python -m experiments.run_electron_pair
python -m experiments.run_exchange_map
python -m experiments.run_disorder_ensemble
python -m experiments.run_convergence_study
```

## 10. Validation Strategy

### 10.1 Unit tests

- potential parameter changes produce expected qualitative changes,
- disorder generation is seed-deterministic,
- orbital alignment handles sign flips,
- finite-difference derivatives match analytic toy functions,
- density and CI metrics normalize correctly,
- charge-sector probabilities sum to one,
- sweep resume avoids rerunning completed points.

### 10.2 Physics tests

- `J > 0` for normal symmetric zero-field double-dot cases after convergence,
- `J` increases as barrier decreases,
- `J` generally decreases with dot separation,
- detuning changes charge-sector weights,
- Coulomb repulsion raises two-electron energy versus noninteracting case,
- triplet forbids same-orbital spatial occupation,
- singlet can mix double-occupancy configurations.

### 10.3 Convergence tests

- `J(P)` convergence with number of orbitals,
- `J(nq)` grid convergence,
- `J(a_soft)` softening convergence or extrapolation,
- training seed variance,
- sweep warm-start versus cold-start consistency.

### 10.4 Benchmark tests

- finite-difference orbital baseline,
- Hubbard estimate comparison,
- exact diagonalization in same finite basis,
- QTCAD-style reference if available.

## 11. Paper Plan

### Candidate title

Neural-Orbital Configuration Interaction Maps for Exchange and Noise Sensitivity in Semiconductor Quantum-Dot Qubits

### Abstract-level claim

We introduce a neural-orbital CI framework that learns compact continuous-space single-electron orbitals with a Block-Ritz solver and performs symmetry-resolved exact diagonalization of interacting two-electron states. The method produces exchange, density, pair-correlation, entanglement, and charge-noise-sensitivity maps across quantum-dot design parameters. It enables faster and more interpretable parameter sweeps than direct grid-based repeated diagonalization while retaining the exchange/correlation physics required for spin-qubit modelling.

### Main figures

1. Method diagram: neural orbital solver + Coulomb tensor + singlet/triplet CI + exchange maps.
2. Validation against finite-difference or QTCAD-style exact diagonalization.
3. `J(detuning, barrier)` and `log J` maps.
4. Sweet-spot and noise-sensitivity maps.
5. Disorder ensemble: exchange variability and sweet-spot displacement.
6. Interpretability: densities, conditional pair density, CI weights, charge-sector probabilities.
7. Runtime/accuracy/convergence Pareto.

### Claims that are realistic

- Faster sweeps after warm-starting and caching.
- Interpretable exchange landscapes.
- Accurate within controlled basis/grid convergence.
- Useful disorder and noise-sensitivity diagnostics.

### Claims to avoid unless proven

- Universal replacement for QTCAD.
- Full ab initio device accuracy.
- General many-electron scalability.
- Direct prediction for real fabricated devices without electrostatic calibration.
- Spin-orbit/valley effects before implemented.

## 12. Immediate Build Roadmap

### Phase 1: Analysis metrics on existing pair runs

Implement:

- pair density difference plots,
- one-body reduced density matrix,
- natural occupations,
- CI participation ratio,
- left/right density integrals,
- conditional density validation report,
- pair run summary table.

Goal:

- Extract more physics from the existing Colab `P=8` pair run.

### Phase 2: Single-parameter exchange sweeps

Implement:

- detuning sweep,
- barrier sweep,
- warm-start optional,
- CSV output,
- `J` and density plots across sweep.

Goal:

- Verify expected monotonic trends and detect numerical issues.

### Phase 3: 2D exchange maps

Implement:

- rectangular sweep over detuning and barrier,
- central-difference sensitivity,
- sweet-spot score,
- adaptive plotting.

Goal:

- First paper-style figure set.

### Phase 4: Disorder ensembles

Implement:

- Gaussian random disorder,
- impurity disorder,
- ensemble stats,
- variability plots.

Goal:

- Show practical robustness and fabrication variability impact.

### Phase 5: Benchmark and convergence package

Implement:

- finite-difference baseline,
- convergence scripts,
- runtime comparison,
- reproducible manuscript configs.

Goal:

- Make the paper defensible to reviewers.

## 13. Open Technical Risks

- Neural training may be slower than finite-difference for single points; the advantage must come from warm starts, compact basis, reusable maps, and interpretability.
- Exchange can be very small, making relative errors large; convergence and uncertainty reporting are mandatory.
- Orbital labels can swap across sweeps; state alignment is mandatory.
- The 2D softened Coulomb model is an approximation; paper claims must frame it as an effective 2D model unless we add finite-thickness Coulomb kernels.
- Real Si/SiGe needs valley physics; real Ge holes need spin-orbit/g-tensor physics. These are future extensions unless explicitly implemented.
- Imported electrostatic potentials require careful units and boundary conventions.

## 14. Minimal Publishable Milestone

The minimum credible manuscript package is:

- one validated double-dot potential family,
- two-electron singlet/triplet CI,
- `J(detuning, barrier)` map,
- sensitivity/sweet-spot analysis,
- disorder ensemble,
- entanglement/correlation metrics,
- finite-difference baseline,
- convergence in `num_orbitals`, grid size, and seed,
- runtime comparison,
- all plots reproducible from configs.

If this package shows that neural-orbital warm-started sweeps produce comparable `J` values faster than repeated baseline solves, while exposing richer physical diagnostics, it is a credible applied ML/physics paper.

## 15. Reference Map

Core device/exchange modelling:

- QTCAD many-body exact diagonalization theory: https://docs.nanoacademic.com/qtcad/theory_spin_fem/manybody/
- QTCAD double-dot exchange exact diagonalization tutorial: https://docs.nanoacademic.com/qtcad/tutorials/device/exchange_2/
- Ab initio modelling of quantum-dot qubits: https://arxiv.org/abs/2403.00191
- Exchange interaction review: https://www.jos.ac.cn/en/article/doi/10.1088/1674-4926/24050043

Scaling, residual exchange, and control:

- Residual exchange in spin-qubit arrays: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.013153
- MAViS autonomous quantum-dot virtualization: https://www.nist.gov/publications/modular-autonomous-virtualization-system-two-dimensional-semiconductor-quantum-dot
- Closed-loop singlet-triplet qubit control: https://www.nature.com/articles/s41467-020-17865-3

Disorder and materials:

- Si/SiGe interface disorder variability: https://www.osti.gov/pages/biblio/2329339
- Ge hole-qubit sweet spots and anisotropic noise: https://www.nature.com/articles/s41563-024-01857-5
- Planar germanium hole-qubit modelling: https://www.nature.com/articles/s41534-024-00897-8

ML and neural wavefunction context:

- Neural networks for arbitrary quantum wells: https://www.nature.com/articles/s41598-022-06442-x
- PINNs for quantum eigenvalue problems: https://arxiv.org/abs/2203.00451
- CNN disorder inference for quantum-dot qubits: https://arxiv.org/abs/2405.04524
- FermiNet: https://arxiv.org/abs/1909.02487
- PauliNet: https://arxiv.org/abs/1909.08423
- Transferable fermionic neural wavefunctions: https://www.nature.com/articles/s41467-023-44216-9

