# Physics Model

The v1 model solves a 2D effective-mass single-electron problem,

```text
h psi = E psi
h = -nabla^2 + V(x, y)
```

in dimensionless units. The physical scale is

```text
E0 = hbar^2 / (2 m* L0^2).
```

Two-electron states are built from the learned one-electron orbitals. The interacting Hamiltonian is

```text
H2 = h(1) + h(2) + lambda / sqrt(|r1-r2|^2 + softening^2).
```

The pair solver diagonalizes symmetric spatial states for singlets and antisymmetric spatial states for triplets. Exchange is reported as

```text
J = E_triplet_0 - E_singlet_0.
```

Limitations: no magnetic field, no spin-orbit coupling, no valley physics, no finite-temperature occupations, no time dynamics, and no more than two electrons.
