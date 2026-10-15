# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implicit recycling boundary condition (2026-10-16, behind a static flag).

WHAT THIS FIXES. Boundary conditions are built once per step, from the OLD time
level (step_function.py -> updaters.provide_core_profiles_t_plus_dt), and are
frozen for the whole nonlinear solve. For a recycling wall the neutral inflow
is R x the ion flux into the coil, so freezing it lags recycling by one step.
The lag test (dipole step 1b) measured dN/dt scaling LINEARLY with dt, which is
the signature of a drive that is cancelled by its own lag rather than of slow
physics.

WHAT IT DOES. With profile_conditions.implicit_recycling_bc=True, the neutral
Dirichlet constraint at the inner face is re-solved from the TRIAL state on
every residual evaluation, so that

    Gamma_n(coil)  =  R_RECYCLE * Gamma_w,ion(coil)

holds at the DISCRETE level, with both fluxes taken from TORAX's own operator
at that iteration -- the same coefficients, the same face values, the same
Dirichlet treatment.

HOW THE FLUXES ARE OBTAINED. By masking, exactly as in
torax_dipole_lib/boundary_flux.py: zero every face but face 0 in d_face/v_face
and drop the sources, so the telescoping sum of F = Cx + c over a channel's
cells collapses to that channel's inner-face flux. Nothing about the flux
expression is restated here, so there is no second implementation to keep in
step with the solver's.

HOW THE CONSTRAINT IS SOLVED. calc_c builds C from the shapes and the boundary
conditions and c from the boundary conditions alone, and the Dirichlet
constraint enters c LINEARLY. The neutral inner flux is therefore an exact
affine function of the constraint,
    Gamma_n(n_bc) = a0 + n_bc (a1 - a0),
evaluated at n_bc = 0 and 1, and inverted in closed form. That is convention-
free: it holds for 'ghost' or any other Dirichlet mode without this module
knowing which is in force, which is what the earlier harness-side ghost
inversion had to hard-code.

NO CLIPPING. max(Gamma_w, 0) inside the residual puts a kink in the Newton
Jacobian that stalls the line search; the caller asserts Gamma_w >= 0 at each
converged step instead.

JACOBIAN. Everything here is traced jnp, called from inside
residual_and_loss.theta_method_block_residual, so jax.jacfwd in
jax_root_finding.py picks up the dGamma_w/dn_e path automatically. No manual
term.

COEFFICIENTS. The PDE coefficients do not depend on the neutral BC -- n_n
enters calc_coeffs only through cell values (the bilinear ionisation term), and
source_n_n is identically zero -- so calc_coeffs is evaluated ONCE per residual
and the constraint is updated afterwards. Gate G1 (per-step discrete closure at
round-off, measured with the FINAL coefficients) is what would catch it if that
ever stopped being true.
"""
import dataclasses

import jax.numpy as jnp

from torax._src.core_profiles import convertors
from torax._src.fvm import cell_variable
from torax._src.fvm import discrete_system

_SCALE = convertors.SCALING_FACTORS


def _inner_face_fluxes(x, coeffs, dirichlet_mode, neumann_mode, drho):
  """Per-channel flux through the inner face, from TORAX's own operator.

  Returns an array of length len(x), in each channel's SOLVER-SCALED units.
  Sign follows F = Cx + c: positive adds particles to the domain.
  """
  num_faces = coeffs.d_face[0].shape[0]
  mask = jnp.zeros(num_faces).at[0].set(1.0)
  d_new = tuple(df * mask for df in coeffs.d_face)
  v_new = (
      None
      if coeffs.v_face is None
      else tuple(vf * mask for vf in coeffs.v_face)
  )
  masked = dataclasses.replace(
      coeffs, d_face=d_new, v_face=v_new, source_mat_cell=None, source_cell=None
  )
  c_mat, c_vec = discrete_system.calc_c(
      x, masked, dirichlet_mode, neumann_mode
  )
  x_flat = jnp.concatenate([var.value for var in x])
  flux_per_cell = c_mat @ x_flat + c_vec
  num_cells = x[0].value.shape[0]
  return jnp.sum(
      flux_per_cell.reshape(len(x), num_cells) * jnp.atleast_1d(drho), axis=1
  )


def _with_neutral_bc(x, index, value):
  """Copy of x with channel `index`'s left Dirichlet constraint replaced."""
  return tuple(
      dataclasses.replace(var, left_face_constraint=value) if i == index else var
      for i, var in enumerate(x)
  )


def solve_neutral_bc(
    x: tuple[cell_variable.CellVariable, ...],
    coeffs,
    evolving_names: tuple[str, ...],
    recycling_coefficient,
    dirichlet_mode: str,
    neumann_mode: str,
    drho,
):
  """Neutral inner-face constraint that makes recycling exact, and Gamma_w.

  Args:
    x: trial CellVariables, in solver-scaled units.
    coeffs: Block1DCoeffs at the trial state.
    evolving_names: names of the evolving channels, in x's order. Must contain
      both 'n_e' and 'n_n'.
    recycling_coefficient: R, the fraction of the ion wall flux returned as
      neutrals.
    dirichlet_mode: solver's convection_dirichlet_mode.
    neumann_mode: solver's convection_neumann_mode.
    drho: cell width in the normalised coordinate (geo.drho_norm).

  Returns:
    (x_with_updated_bc, gamma_wall) with gamma_wall in physical units, positive
    for ions flowing INTO the coil.
  """
  if 'n_e' not in evolving_names or 'n_n' not in evolving_names:
    raise ValueError(
        'implicit_recycling_bc requires both n_e and n_n to be evolving; got'
        f' evolving_names={evolving_names}. Set numerics.evolve_n_n=True and'
        ' evolve the density, or turn the flag off.'
    )
  i_e = evolving_names.index('n_e')
  i_n = evolving_names.index('n_n')

  fluxes = _inner_face_fluxes(x, coeffs, dirichlet_mode, neumann_mode, drho)
  # Ions flowing into the coil leave the domain, so the wall flux is -F.
  gamma_wall = -fluxes[i_e] * _SCALE['n_e']

  # The neutral inner flux is exactly affine in its Dirichlet constraint.
  a0 = (
      _inner_face_fluxes(
          _with_neutral_bc(x, i_n, jnp.zeros(())),
          coeffs,
          dirichlet_mode,
          neumann_mode,
          drho,
      )[i_n]
      * _SCALE['n_n']
  )
  a1 = (
      _inner_face_fluxes(
          _with_neutral_bc(x, i_n, jnp.ones(())),
          coeffs,
          dirichlet_mode,
          neumann_mode,
          drho,
      )[i_n]
      * _SCALE['n_n']
  )
  n_bc = (recycling_coefficient * gamma_wall - a0) / (a1 - a0)
  return _with_neutral_bc(x, i_n, n_bc), gamma_wall


def check_supported(torax_config) -> None:
  """Raise a clear error if the flag is on with a solver that cannot honour it.

  Call this once, from the harness, before building the step function. The
  checks cannot live inside the jitted solve: theta_implicit and
  use_predictor_corrector are traced values there, not static ones.
  """
  pc = torax_config.profile_conditions
  if not getattr(pc, 'implicit_recycling_bc', False):
    return
  solver = torax_config.solver
  numerics = torax_config.numerics
  problems = []
  if not numerics.evolve_n_n:
    problems.append(
        'numerics.evolve_n_n is False -- there is no neutral equation whose'
        ' boundary condition could be made implicit.'
    )
  if not numerics.evolve_density:
    problems.append(
        'numerics.evolve_density is False -- there is no ion wall flux to'
        ' recycle.'
    )
  theta = float(solver.theta_implicit)
  if theta != 1.0:
    problems.append(
        f'solver.theta_implicit is {theta}, not 1.0. The implicit BC is'
        ' applied to the new-time coefficients only; with theta < 1 the'
        ' old-time flux terms would still carry the frozen constraint stored'
        ' on x_old, which this change does not refresh.'
    )
  name = type(solver).__name__
  if 'Linear' in name and not solver.use_predictor_corrector:
    problems.append(
        f'{name} with use_predictor_corrector=False solves once with the'
        ' trial state equal to the old state, so the recycling BC would still'
        ' be lagged by a full step. Use the Newton-Raphson solver, or enable'
        ' the predictor-corrector loop.'
    )
  if problems:
    raise ValueError(
        'implicit_recycling_bc=True is not supported with this'
        ' configuration:\n  - ' + '\n  - '.join(problems)
    )
