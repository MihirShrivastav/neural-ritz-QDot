Superseded by the full branch roadmap:

- `neural-orbital-maps-research-plan.md`
- `docs/neural-orbital-maps-research-plan.md`

Original scoping note:

The strongest paper-worthy direction is:

Neural-orbital CI maps for exchange, entanglement, and disorder sensitivity in semiconductor quantum-dot qubits.

In plain terms: use our solver to rapidly compute how a real-ish double quantum dot’s two-electron spectrum changes with gate geometry, detuning, barrier height, disorder, and material parameters, then extract the quantities qubit people actually care about: J, charge-noise sensitivity, singlet-triplet mixing tendencies, electron localization, pair correlation, and entanglement.

This is a real problem. Exchange coupling is central to quantum-dot spin qubits and two-qubit gates. Recent work emphasizes that exchange cannot be perfectly turned off, residual exchange causes coherent errors, and charge noise/disorder strongly affects scalable arrays. QTCAD also uses exactly the same physics skeleton we now implemented: single-particle states, Coulomb integrals, and exact diagonalization for few-electron states, but the Coulomb integral step is expensive. Their docs explicitly note that exact diagonalization captures exchange/correlation beyond mean field, while Coulomb integral evaluation is the bottleneck. Sources: QTCAD many-body theory and exchange tutorial, residual exchange in spin-qubit arrays, and recent quantum-dot exchange reviews. (docs.nanoacademic.com) (docs.nanoacademic.com) (journals.aps.org) (jos.ac.cn)

Best Paper Thesis
Our contribution would not be “we solved Schrödinger with a neural net.” That is too generic.

The sharper thesis is:

A symmetry-preserving neural-orbital CI framework can learn compact device-adapted orbitals and produce interpretable exchange/entanglement/noise-sensitivity landscapes for quantum-dot qubits across gate and disorder parameter spaces.

That is much more defensible.

What Makes It Novel Enough
The novelty would be the combination of:

Neural Block-Ritz orbitals learned directly from continuous 2D confinement potentials.
Exact singlet/triplet CI on top, preserving fermionic exchange physics.
Fast parameter sweeps over device knobs.
Automatic extraction of J, dJ/dV, d²J/dV², charge-noise sweet spots, pair densities, and CI composition.
Disorder studies: how interface roughness, detuning bias, dot asymmetry, and barrier fluctuations distort exchange.
Interpretability: we can show why J changes using orbital localization, Coulomb integrals, CI weights, and conditional densities.
This aligns with current pain points. For example, Si/SiGe interface disorder has been shown to create large spectral variability, including valley splitting variability and double-dot detuning bias variability of order 1-10 meV. (osti.gov) Ge/SiGe hole qubits also have anisotropic noise sensitivity and sweet-spot complications. (nature.com)

Most Promising Study
I would focus the first serious paper on:

“Exchange Landscape Learning And Sweet-Spot Discovery In Disordered Double Quantum Dots”

Core experiment:

Inputs:
  dot separation
  barrier height
  detuning
  confinement anisotropy
  disorder amplitude/correlation length
  dielectric/effective mass

Outputs:
  singlet/triplet spectrum
  J = E_T - E_S
  dJ/d(detuning)
  dJ/d(barrier)
  charge-noise sensitivity
  one-body densities
  conditional pair densities
  CI weight decomposition
  entanglement/correlation metrics
The key plots would be:

J(detuning, barrier)
log J landscape
sweet-spot map where |dJ/dV| is small
exchange variability under disorder
pair-density evolution across regimes
CI weights showing Hubbard-like vs correlated regimes
benchmark against finite-difference / analytic / QTCAD-style ED
What We Need To Add
To make this publishable, our current repo needs these capabilities:

Parametric sweep runner.
Disorder potential generator.
Exchange derivative/sensitivity analysis.
Entanglement/correlation metrics.
Better pair visualizations: difference density, exchange hole, pair-correlation function.
Benchmark baselines: finite-difference single-particle solver + CI, analytic harmonic/Hubbard limits, and ideally comparison to QTCAD-style examples.
Convergence studies over num_orbitals, grid size, softening, and training seeds.
What Not To Do Yet
I would not jump straight to FermiNet/PauliNet-style many-electron neural wavefunctions. That becomes a different project, and top reviewers will ask why we need that complexity for two-electron devices. Our current path is cleaner: neural orbitals plus exact CI is physically interpretable and directly relevant to quantum-dot qubit engineering.

Ranked Research Directions

Exchange/sweet-spot/disorder maps for double quantum-dot spin qubits. Best fit, highest relevance, closest to current solver.
Residual exchange and crosstalk in small quantum-dot arrays. Strong applied-physics angle, but requires extending beyond one pair.
Entanglement and Wigner-molecule formation in tunable artificial molecules. More physics-rich, less directly device-engineering.
Neural-orbital CI as a fast surrogate for QTCAD-like many-body workflows. Strong ML angle, but needs rigorous benchmarking.
Correlated neural correction on top of CI. Good method paper later, but premature before the CI baseline is validated.
My recommendation: build toward direction 1. It is concrete, relevant, measurable, and paper-shaped.
