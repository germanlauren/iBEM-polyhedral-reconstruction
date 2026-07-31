# iBEM polyhedral eigenstrain potentials — reconstruction and validation

Reconstruction of the **uniform-eigenstrain polyhedral potential class** that is
absent from the publicly released inclusion-based boundary element method (iBEM)
library, together with the validation suite and a mean-field homogenization harness.
Restoring this class re-enables the three-dimensional polyhedral execution path of
the publicly released implementation.

This repository accompanies the manuscript *"Reconstruction of the uniform polyhedral
eigenstrain potentials and restoration of the three-dimensional inclusion-based
boundary element (iBEM) method"* (Romero-Suárez, Villegas-Bermúdez, Martínez).

## What is reconstructed
The linear and quadratic polyhedral integral classes of iBEM are built on a
`Uniform_Polyhedral` class exposing seven functions — the harmonic potential Φ and
Φ,i, Φ,ij, and the biharmonic Ψ,i, Ψ,ij, Ψ,ijk, Ψ,ijkl — which is not distributed.
`src/Uniform_Polyhedral_integral.{h,cpp}` provides it by a **semi-analytical
face-and-edge reduction** — the three-dimensional volume potentials are reduced
analytically to face contributions and one-dimensional edge integrals; the harmonic
edge contribution is analytical, and the biharmonic edge integral is evaluated by a
converged one-dimensional Gauss–Legendre quadrature (it is therefore semi-analytical,
not fully closed form):

- harmonic edge kernel  w = b·ln[(r⁺+l⁺)/(r⁻+l⁻)] − |a|[atan2(bl⁺, d²+|a|r⁺) − atan2(bl⁻, d²+|a|r⁻)]
- biharmonic edge kernel v = ∫_{l⁻}^{l⁺} b[(a²+b²+l²)^{3/2} − |a|³] / (3(b²+l²)) dl
- all derivatives assembled by contracting the mixed parameter partials of the
  kernels with the constant parameter gradients (96-point Gauss–Legendre for v,
  retained after the convergence study in `convergence/`).

`src/Uniform_polygon_integral.h` restores the missing 2-D face-potential header.

## Validation (near-machine precision)
`validation/` reproduces the pointwise checks reported in the paper:
- `potentials_reference.py` — direct tetra-quadrature ground truth for φ, ψ.
- `harmonic_closed.py`, `harmonic_hessian.py` — analytical φ, φ,i, φ,ij vs reference
  (φ,ij vs finite difference: 3×10⁻¹¹; symmetric to 2×10⁻¹⁶; traceless outside body).
- `biharmonic_full.py` — ψ,i…ψ,ijkl; identities ∇²ψ = 2φ and ∇²(ψ,ij) = 2φ,ij to ≤8×10⁻¹⁵.
The C++ port reproduces the Python reference to 4×10⁻¹⁵ and is free of non-finite
values for the near-degenerate limits tested (a→0, b→0, l±→0; near-face/edge/vertex).

## Convergence and verification studies
`convergence/` reproduces every convergence study and error figure in the paper:
- `convergence_edge.py` — edge-quadrature convergence of ψ,ij and ψ,ijkl vs Nₘ ∈ {12,24,48,96,192}.
- `verification_set.py` — φ vs reference volume quadrature by normalized boundary
  distance (45 tetrahedra, 1799 exterior points) and the differential-identity /
  symmetry residuals (near-machine precision).
- `bem_convergence.py` — BEM discretization convergence on the unit-contrast cube
  (normal-component bias 14.2 % → 7.1 % → 4.2 %; symmetry residual ~10⁻¹⁶).
- `porosity_sweep.py` — building-block RVE effective modulus vs core pore fraction.
- `eshelby_run.py` — dilute Eshelby/Mori–Tanaka benchmark (stiff sphere, contrast 10):
  the effective modulus agrees with the analytical sphere to 1.9 % (φ=0.05) and 0.2 % (φ=0.10),
  with reciprocity C=Cᵀ to ≤5×10⁻¹⁵ and positive-definite recovered tensors.
- `make_conv_figs.py` — regenerates Figures 4a and 4b.

The BEM-solver scripts (`bem_convergence.py`, `porosity_sweep.py`) require a compiled
iBEM binary; set its path in `homogenization/ibem_homog.py` (`BIN`). The potential-only
studies (`convergence_edge.py`, `verification_set.py`) run from Python + NumPy + SymPy alone.

## Build
1. Obtain the iBEM library (github.com/iBemResearch/iBEM).
2. Copy `src/Uniform_Polyhedral_integral.{h,cpp}` and `src/Uniform_polygon_integral.h`
   into the library directory.
3. Build with a C++17 compiler and Eigen (`src/build.sh` documents the flags and the
   two upstream fixes required — the missing headers and a `setUp()` ordering issue).

## Homogenization harness
`homogenization/ibem_homog.py` — displacement-controlled (KUBC) RVE homogenization
with an exact boundary-traction stress average; `f_u_validation.py` — structural
force–displacement conversion and leave-one-curve-out validation.

## Regenerate the C++ from symbolic source
`src/gen_uniform_polyhedral.py` regenerates `Uniform_Polyhedral_integral.cpp` from the
symbolic integrands (SymPy), including the analytic integrand partials.

## License
GNU General Public License v3 (see LICENSE), consistent with the iBEM library on which
this work builds.

## Citation
See CITATION.cff. Please cite both this repository and the accompanying manuscript.
