# Architecture

Package layout:

```text
src/neural_orbital_maps/
  analysis/     exchange, density, CI summaries
  cli/          command-line entrypoints
  io/           config, artifacts, logging, run directories
  models/       neural basis networks
  numerics/     grids, Ritz assembly, Coulomb, pair CI
  physics/      materials, units, potentials
  plotting/     publication-oriented diagnostics
  studies/      convergence and parameter-study runners
  training/     Block-Ritz trainer
```

The intended data flow is:

```text
config -> run directory -> grid/potential -> neural Block-Ritz training
       -> one-electron orbitals -> Coulomb tensor -> pair CI
       -> reports/arrays/plots -> post-run analysis
```

Study runners wrap this single-run flow, vary one controlled parameter set, and collect machine-readable CSV/JSON summaries plus comparison plots.

No code is imported from the parent repository. This project can be moved into a clean repo later.
