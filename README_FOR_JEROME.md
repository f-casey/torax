# README for Jerome — dipole TORAX fork

Branch: `coupled-neutral-evolution`. This repo is a fork of stock TORAX with the
changes below layered on top. **The dipole-specific transport model and every driver/
analysis script live in a separate directory, `torax_dipole_lib/`, on the machine this
was developed on — not in this repo, and not currently under version control.** That's
the biggest gap in "reproducible for Jerome" right now; see the last section.

## What's modified in this repo, file by file

All changes are additive and gated behind new config flags/fields that default to
values reproducing stock TORAX behaviour exactly (most are regression-tested
bit-identical against the pre-change baseline). Nothing here changes stock TORAX's
behaviour for a config that doesn't touch the dipole-specific flags.

- **`_src/fvm/calc_coeffs.py`**
  - Coordinate Jacobian fix for restricted-domain (`rho_min>0`) geometries: the FVM
    assembles fluxes in TORAX's internal `[0,1]` mesh coordinate, but this project's
    coefficients are defined in the physical `rho_hat = rho_min + (1-rho_min)*rho_mesh`.
    Missing the `1/(1-rho_min)` chain-rule factor silently scaled diffusion/advection
    wrong on any restricted domain. Exact no-op at `rho_min=0`.
  - `n_n`'s convection coefficient: was using `geo.g0_face`, should be `geo.vpr_face`
    (two different metric quantities) — fixed to match both the validated external
    reference solver and Eq. 174 of the writeup.
  - Scaling-factor correction on the `('n_e','n_n')` bilinear ionisation cross-coupling
    term (the solver's state vector is scaled per-variable; a cross-block matrix entry
    needs the ratio of the two variables' own scaling factors, not 1).
  - A Voronov (1997) atomic-hydrogen ionisation rate coefficient
    (`_voronov_R_iz_H`), wired into every `R_iz` use site, gated behind
    `use_local_R_iz` (default off — uses the existing fixed placeholder).
  - The new `v_heat_face_el` heat-convection slot (see below) is added into the
    existing convection accumulator, mirroring how Pereverzev/Phi_b_dot already do
    the same thing there.

- **`_src/core_profiles/profile_conditions.py`, `getters.py`, `updaters.py`,
  `initialization.py`** — new, independent boundary-condition options, all off by
  default:
  - `use_sheath_outer` / `n_e_sheath_outer_c_s` — outer boundary as a sheath sink,
    `Gamma_out = n(1)*c_s(T_e)`, replacing the Dirichlet `n_e_right_bc` reservoir for
    levitated-configuration work.
  - `use_graze_recycling` / `n_e_graze_recycling_c_s` — inner grazing-incidence
    recycling as a net particle influx.
  - `use_zero_heat_flux_inner`, `use_sheath_heat_inner` / `gamma_sheath`,
    `T_e_floor_keV` — three alternative inner `T_e` boundary conditions alongside the
    existing Dirichlet.
  - `use_local_R_iz`, `use_local_robin_c_s` — use the actual local `core_profiles.T_e`
    for the ionisation rate / sheath sound speed instead of a fixed placeholder.

- **`_src/state.py`, `_src/transport_model/transport_model.py`** — a new
  `v_heat_face_el: jax.Array | None` field on `TurbulentTransport` and `CoreTransport`
  (default `None` everywhere). A physical electron heat-CONVECTION channel, distinct
  from the existing `chi_face_el` (diffusive). Populated only by the dipole project's
  own transport model (`QLDipole_nU`, in `torax_dipole_lib/`, not this repo) when its
  entropy closure is active; every other transport model leaves it `None` and is
  unaffected.

## Known bug, not yet fixed, visible in the code comments

`getters.py`'s sheath-heat BC block (`use_sheath_heat_inner`) had two unit-consistency
bugs (a missing `n_e*keV_to_J` factor, a constant/variable mixup) that made the BC's
boundary term dominate its diffusive term by 10+ orders of magnitude — **fixed** and
regression-verified. Separately, and **still open**: the convective heat term
(`v_heat_face_el`, computed in `torax_dipole_lib/ql_dipole_model.py`, not this repo) is
clip-saturated at every tested value of its own diffusivity parameter — its true
unclipped value is many orders of magnitude too large. Not fixed here. See
`torax_dipole_lib`'s own notes (goal-1-2-6 Task 25/28/30) for the full diagnosis.

## Config flags at a glance

| Flag | Default | Effect when on |
|---|---|---|
| `use_sheath_outer` | off | outer BC = sheath sink instead of Dirichlet |
| `use_graze_recycling` | off | inner BC gains a recycling influx term |
| `use_zero_heat_flux_inner` | off | inner `T_e` BC = pure Neumann |
| `use_sheath_heat_inner` | off | inner `T_e` BC = sheath heat loss (`gamma_sheath`) |
| `T_e_floor_keV` | 0 | floors the sheath-heat BC's face value |
| `use_local_R_iz` | off | ionisation rate from local `T_e` (Voronov) vs. fixed `R_iz` |
| `use_local_robin_c_s` | off | Robin BC sound speed from local `T_e` vs. fixed `n_e_robin_bc_c_s` |

## Reproducing the current baseline

1. This repo installed editable (or copied) into your TORAX environment.
2. The dipole-specific model and every driver script are in `torax_dipole_lib/` — ask
   Finian for a current copy; it is not yet a git repo. Key files there:
   `ql_dipole_model.py` (the `QLDipole_nU` transport model: the QL closure, the K_n
   interchange-stability switch, and the entropy-closure `chi_e`/`v_heat_face_el`
   derivation), `rhomin_geometry.py` (restricted-domain geometry builder), and a large
   number of `step*.py`/`task*.py` driver scripts, one per investigation.
3. Every driver script builds its own `ToraxConfig` directly in Python (not a config
   file) — this is the pattern to follow for a new run, not the CLI/YAML path stock
   TORAX examples use.

## The actual gap

`torax_dipole_lib/` is not under version control. Everything in *this* repo is
reproducible from git history; the dipole-specific model, every figure-generating
script, and the full investigation log (a markdown vault, also not in this repo) are
not. If you need to reproduce or extend this work, that directory needs to move under
version control first — flagged here rather than silently left as a gap.
