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
  training/     Block-Ritz trainer
```

The intended data flow is:

```text
config -> run directory -> grid/potential -> neural Block-Ritz training
       -> one-electron orbitals -> Coulomb tensor -> pair CI
       -> reports/arrays/plots -> post-run analysis
```

No code is imported from the parent repository. This project can be moved into a clean repo later.
