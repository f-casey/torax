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

"""Calculates Block1DCoeffs for a time step."""
import functools

import jax
import jax.numpy as jnp
from torax._src import constants
from torax._src import math_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.pedestal_model import adaptive_source
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
import typing_extensions


# Voronov (1997) electron-impact ionisation rate coefficient, atomic
# hydrogen (Table 2: dE=13.6eV, P=0, A=2.91e-8 cm^3/s, X=0.232, K=0.39).
# R_iz(T_e) = A*(1+P*sqrt(U))/(X+U)*U^K*exp(-U), U=dE/T_e[eV]. Pure
# elementwise jnp -- no data-dependent control flow, traces cleanly.
# Reproduces this project's constant default (3.17e-14 m^3/s,
# profile_conditions.py's R_iz) at T_e=100eV to the given precision,
# confirmed by direct calculation before wiring in (2026-08-07, Stage 0).
def _voronov_R_iz_H(T_e_eV: jax.Array) -> jax.Array:
  dE, A, X, K = 13.6, 2.91e-8, 0.232, 0.39  # A in cm^3/s
  U = dE / jnp.maximum(T_e_eV, 1e-3)
  R_iz_cm3 = A / (X + U) * U**K * jnp.exp(-U)
  return R_iz_cm3 * 1e-6  # cm^3/s -> m^3/s


# pylint: disable=invalid-name
class CoeffsCallback:
  """Calculates Block1DCoeffs for a state."""

  def __init__(
      self,
      physics_models: physics_models_lib.PhysicsModels,
      evolving_names: tuple[str, ...],
  ):
    self.physics_models = physics_models
    self.evolving_names = evolving_names

  def __hash__(self) -> int:
    return hash((
        self.physics_models,
        self.evolving_names,
    ))

  def __eq__(self, other: typing_extensions.Self) -> bool:
    return (
        self.physics_models == other.physics_models
        and self.evolving_names == other.evolving_names
    )

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      x: tuple[cell_variable.CellVariable, ...],
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      allow_pereverzev: bool = False,
      # Checks if reduced calc_coeffs for explicit terms when theta_implicit=1
      # should be called
      explicit_call: bool = False,
  ) -> block_1d_coeffs.Block1DCoeffs:
    """Returns coefficients given a state x.

    Used to calculate the coefficients for the implicit or explicit components
    of the PDE system.

    Args:
      runtime_params: Runtime configuration parameters. These values are
        potentially time-dependent and should correspond to the time step of the
        state x.
      geo: The geometry of the system at this time step.
      core_profiles: The core profiles of the system at this time step.
      x: The state with cell-grid values of the evolving variables.
      explicit_source_profiles: Precomputed explicit source profiles. These
        profiles were configured to always depend on state and parameters at
        time t during the solver step. They can thus be inputs, since they are
        not recalculated at time t+plus_dt with updated state during the solver
        iterations. For sources that are implicit, their explicit profiles are
        set to all zeros.
      allow_pereverzev: If True, then the coeffs are being called within a
        linear solver. Thus could be either the use_predictor_corrector solver
        or as part of calculating the initial guess for the nonlinear solver. In
        either case, we allow the inclusion of Pereverzev-Corrigan terms which
        aim to stabilize the linear solver when being used with highly nonlinear
        (stiff) transport coefficients. The nonlinear solver solves the system
        more rigorously and Pereverzev-Corrigan terms are not needed.
      explicit_call: If True, then if theta_implicit=1, only a reduced
        Block1DCoeffs is calculated since most explicit coefficients will not be
        used.

    Returns:
      coeffs: The diffusion, convection, etc. coefficients for this state.
    """

    # Update core_profiles with the subset of new values of evolving variables
    core_profiles = updaters.update_core_profiles_during_step(
        x,
        runtime_params,
        geo,
        core_profiles,
        self.evolving_names,
    )
    if allow_pereverzev:
      use_pereverzev = runtime_params.solver.use_pereverzev
    else:
      use_pereverzev = False

    return calc_coeffs(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit_source_profiles=explicit_source_profiles,
        physics_models=self.physics_models,
        evolving_names=self.evolving_names,
        use_pereverzev=use_pereverzev,
        explicit_call=explicit_call,
    )


def _calculate_pereverzev_flux(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Adds Pereverzev-Corrigan flux to diffusion terms."""

  consts = constants.CONSTANTS

  geo_factor = jnp.concatenate(
      [jnp.ones(1), geo.g1_over_vpr_face[1:] / geo.g0_face[1:]]
  )

  chi_face_per_ion = (
      geo.g1_over_vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
      * runtime_params.solver.chi_pereverzev
  )

  chi_face_per_el = (
      geo.g1_over_vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
      * runtime_params.solver.chi_pereverzev
  )

  d_face_per_el = runtime_params.solver.D_pereverzev
  # Epsilon floor (2026-07-16) -- matches the safe_D pattern used for the
  # analogous division in getters.py's Robin BC calc (safe_D = max(D_eff,
  # 1e-30)). Without this floor, n_e.face_value() can legitimately be exactly
  # 0.0 at the left face: with a Robin BC and zero initial transport (D_eff=
  # V_eff=0 everywhere, e.g. a uniform IC with the K_n switch fully off), the
  # Robin formula's numerator is 0 regardless of denominator, so it degenerates
  # to n_face=0 exactly -- dividing by that here produces inf/nan that
  # corrupts the whole solve. Every other division in this codebase already
  # guards its denominator this way; this one didn't.
  safe_n_e_face_value = jnp.maximum(core_profiles.n_e.face_value(), 1e-30)
  v_face_per_el = (
      core_profiles.n_e.face_grad()
      / safe_n_e_face_value
      * d_face_per_el
      * geo_factor
  )

  # remove Pereverzev flux from boundary region if pedestal model is on
  # (for PDE stability)
  chi_face_per_ion = jnp.where(
      geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
      0.0,
      chi_face_per_ion,
  )
  chi_face_per_el = jnp.where(
      geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
      0.0,
      chi_face_per_el,
  )

  # set heat convection terms to zero out Pereverzev-Corrigan heat diffusion
  v_heat_face_ion = (
      core_profiles.T_i.face_grad()
      / core_profiles.T_i.face_value()
      * chi_face_per_ion
  )
  v_heat_face_el = (
      core_profiles.T_e.face_grad()
      / core_profiles.T_e.face_value()
      * chi_face_per_el
  )

  d_face_per_el = jnp.where(
      geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
      0.0,
      d_face_per_el * geo.g1_over_vpr_face,
  )

  v_face_per_el = jnp.where(
      geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
      0.0,
      v_face_per_el * geo.g0_face,
  )

  chi_face_per_ion = chi_face_per_ion.at[0].set(chi_face_per_ion[1])
  chi_face_per_el = chi_face_per_el.at[0].set(chi_face_per_el[1])

  return (
      chi_face_per_ion,
      chi_face_per_el,
      v_heat_face_ion,
      v_heat_face_el,
      d_face_per_el,
      v_face_per_el,
  )


def calc_coeffs(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
    explicit_call: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates Block1DCoeffs for the time step described by `core_profiles`.

  Args:
    runtime_params: General input parameters that can change from time step to
      time step or simulation run to run, and do so without triggering a
      recompile.
    geo: Geometry describing the torus.
    core_profiles: Core plasma profiles for this time step during this iteration
      of the solver. Depending on the type of solver being used, this may or may
      not be equal to the original plasma profiles at the beginning of the time
      step.
    explicit_source_profiles: Precomputed explicit source profiles. These
      profiles either do not depend on the core profiles or depend on the
      original core profiles at the start of the time step, not the "live"
      updating core profiles. For sources that are implicit, their explicit
      profiles are set to all zeros.
    physics_models: The physics models to use for the simulation.
    evolving_names: The names of the evolving variables in the order that their
      coefficients should be written to `coeffs`.
    use_pereverzev: Toggle whether to calculate Pereverzev terms
    explicit_call: If True, indicates that calc_coeffs is being called for the
      explicit component of the PDE. Then calculates a reduced Block1DCoeffs if
      theta_implicit=1. This saves computation for the default fully implicit
      implementation.

  Returns:
    coeffs: Block1DCoeffs containing the coefficients at this time step.
  """

  # If we are fully implicit and we are making a call for calc_coeffs for the
  # explicit components of the PDE, only return a cheaper reduced Block1DCoeffs
  if explicit_call and runtime_params.solver.theta_implicit == 1.0:
    return _calc_coeffs_reduced(
        geo,
        core_profiles,
        evolving_names,
    )
  else:
    return _calc_coeffs_full(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit_source_profiles=explicit_source_profiles,
        physics_models=physics_models,
        evolving_names=evolving_names,
        use_pereverzev=use_pereverzev,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        'physics_models',
        'evolving_names',
    ],
)
def _calc_coeffs_full(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """See `calc_coeffs` for details."""

  consts = constants.CONSTANTS

  pedestal_model_output = physics_models.pedestal_model(
      runtime_params, geo, core_profiles
  )

  conductivity = (
      physics_models.neoclassical_models.conductivity.calculate_conductivity(
          geo, core_profiles
      )
  )

  # Calculate the implicit source profiles and combines with the explicit
  merged_source_profiles = source_profile_builders.build_source_profiles(
      source_models=physics_models.source_models,
      neoclassical_models=physics_models.neoclassical_models,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      explicit=False,
      explicit_source_profiles=explicit_source_profiles,
      conductivity=conductivity,
  )

  # psi source terms. Source matrix is zero for all psi sources
  source_mat_psi = jnp.zeros_like(geo.rho)

  # fill source vector based on both original and updated core profiles
  source_psi = merged_source_profiles.total_psi_sources(geo)

  # Transient term coefficient vector (has radial dependence through r, n)
  toc_T_i = 1.5 * geo.vpr ** (-2.0 / 3.0) * consts.keV_to_J
  tic_T_i = core_profiles.n_i.value * geo.vpr ** (5.0 / 3.0)
  toc_T_e = 1.5 * geo.vpr ** (-2.0 / 3.0) * consts.keV_to_J
  tic_T_e = core_profiles.n_e.value * geo.vpr ** (5.0 / 3.0)
  toc_psi = (
      1.0
      / runtime_params.numerics.resistivity_multiplier
      * geo.rho_norm
      * conductivity.sigma
      * consts.mu_0
      * 16
      * jnp.pi**2
      * geo.Phi_b**2
      / geo.F**2
  )
  tic_psi = jnp.ones_like(toc_psi)
  toc_dens_el = jnp.ones_like(geo.vpr)
  tic_dens_el = geo.vpr
  toc_n_n = jnp.ones_like(geo.vpr)
  tic_n_n = geo.vpr

  # Diffusion term coefficients
  turbulent_transport = physics_models.transport_model(
      runtime_params, geo, core_profiles, pedestal_model_output
  )
  neoclassical_transport = physics_models.neoclassical_models.transport(
      runtime_params, geo, core_profiles
  )

  chi_face_ion_total = (
      turbulent_transport.chi_face_ion + neoclassical_transport.chi_neo_i
  )
  chi_face_el_total = (
      turbulent_transport.chi_face_el + neoclassical_transport.chi_neo_e
  )
  d_face_el_total = (
      turbulent_transport.d_face_el + neoclassical_transport.D_neo_e
  )
  v_face_el_total = (
      turbulent_transport.v_face_el
      + neoclassical_transport.V_neo_e
      + neoclassical_transport.V_neo_ware_e
  )
  d_face_psi = geo.g2g3_over_rhon_face
  # No poloidal flux convection term
  v_face_psi = jnp.zeros_like(d_face_psi)

  # Neutral density (n_n) transport -- Phase 1 scaffolding (2026-07-09).
  # D=0 (Jerome confirmed, no artificial diffusion floor needed), pure
  # outward advection V=v_n/h_rho. Analytically fixed, not a turbulent-
  # transport closure, so computed directly here rather than via
  # physics_models.transport_model -- mirrors psi's pattern above, not n_e's.
  #
  # rho_min (2026-07-14): the computational domain starts at the first
  # closed flux surface (fcfs), rho_hat=rho_min, not at the magnetic axis
  # rho_hat=0 -- everything inside rho_min is inside the magnet coil and
  # non-physical for the Junior equilibrium. geo.rho_face_norm/geo.rho_norm
  # remain TORAX's own internal [0,1] mesh coordinate (required structurally
  # by Grid1D); rho_hat_face/rho_hat_cell below are the TRUE physical
  # flux-surface label, affinely mapped from that internal coordinate. This
  # means rho_hat never actually reaches 0 in this domain -- the h_rho
  # coordinate singularity at rho_hat=0 that drove the entire clip/face[0]
  # regularization investigation (see concepts/coordinate-velocity-clipping
  # in the project vault) simply does not exist in the physical domain, so
  # neither that fix nor a velocity clip is needed here.
  rho_min = runtime_params.profile_conditions.rho_min
  rho_hat_face = rho_min + (1.0 - rho_min) * geo.rho_face_norm
  rho_hat_cell = rho_min + (1.0 - rho_min) * geo.rho_norm
  psi_b = geo.psi[-1]
  B_face_n_n = jnp.sqrt(jnp.maximum(geo.gm5_face, 1e-10))
  # h_rho (2026-10-01). The Eq. 178 form below is DIMENSIONALLY WRONG: Eq. 178
  # misquotes Eq. 109 as |grad psi| ~ B where Eq. 109 is |grad psi| = R B, and
  # Eq. 113 gives h_rho = (1/(R B)) dpsi/drho in METRES -- the dropped R is a
  # missing power of length (2 psi_b rho_hat/B has units Wb/T = m^2). It also
  # assumes psi = psi_b rho_hat^2 (Eq. 106), which is false where psi falls
  # outward, and it vanishes at rho_hat = 0 (the July h_rho_face singularity).
  # use_metric_h_rho=True selects the geometric metric instead,
  # h_rho = sqrt(vpr/(g1/vpr)) = 1/sqrt(<|grad rho_hat|^2>), which is in metres
  # and, in SLAB, equals dx/drho_hat EXACTLY. Default False is an exact no-op.
  _h_rho_metric_face = jnp.sqrt(
      jnp.maximum(geo.vpr_face / jnp.maximum(geo.g1_over_vpr_face, 1e-300), 0.0)
  )
  h_rho_face = jnp.where(
      runtime_params.profile_conditions.use_metric_h_rho,
      _h_rho_metric_face,
      2.0 * psi_b * rho_hat_face / jnp.maximum(B_face_n_n, 1e-10),
  )
  # CANDIDATE (c), the ADVECTIVE average (2026-10-12). _h_rho_metric_face above
  # is candidate (b), 1/sqrt(<|grad rho|^2>) -- the second moment, which is what
  # DIFFUSION needs. Neutral advection needs the first moment: g0 = vpr<|grad
  # rho|>, so h_rho = vpr/|g0| = 1/<|grad rho|>. Adopted by Jerome 2026-10-08.
  # abs() because g0 is stored NEGATIVE in this dipole equilibrium unless
  # standard_geometry.set_positive_g0(True) is in force; with it on the abs() is
  # redundant and harmless. Default False is an exact no-op.
  _h_rho_adv_face = geo.vpr_face / jnp.maximum(jnp.abs(geo.g0_face), 1e-300)
  h_rho_face = jnp.where(
      runtime_params.profile_conditions.use_h_rho_advective,
      _h_rho_adv_face,
      h_rho_face,
  )

  d_face_n_n = jnp.zeros_like(geo.rho_face)
  # Geometric factor fix (2026-07-14): use geo.vpr_face (V_rho = dV/drho_hat),
  # NOT geo.g0_face (<grad V>), matching the validated external
  # evolve_neutral_pde reference's own flux formula exactly:
  #   F = Vrho_arr * u * n_n   (u = v_th_n/h_rho)
  # This is n_n's own PDE (writeup Eq 174): d(V_rho*n_n)/dt =
  # -d/drho_hat(V_rho*u*n_n) - V_rho*n_e*R_iz*n_n. The original
  # implementation (2026-07-09) copied n_e's QLDipole_nU advection pattern
  # (full_v_face_el = geo.g0_face * v_face_el_total) without checking
  # whether n_n's own equation uses the same geometric factor -- it doesn't.
  # g0 (<grad V>, units m^2) and vpr (dV/drho_hat, units m^3) are physically
  # different quantities that diverge across this geometry (vpr/g0 grows by
  # ~4x from rho_min to rho_hat~1), producing a smoothly accumulating
  # multiplicative error that was mistaken for a discretization/CFL problem
  # through several rounds of investigation (see
  # concepts/coordinate-velocity-clipping in the project vault for the full
  # history). vpr_face is always positive (volume strictly increases with
  # rho_hat), unlike g0_face which is negative in this dipole equilibrium --
  # no jnp.abs() needed here.
  # Coordinate Jacobian (2026-07-22, Stage B0 fix): the FVM assembles this
  # advection divergence in the internal [0,1] MESH coordinate, but the n_n
  # equation (Eq 174) is in the physical rho_hat = rho_min+(1-rho_min)*rho_mesh.
  # d/drho_hat = (1/(1-rho_min)) d/drho_mesh, so the mesh-coordinate advection
  # coefficient carries one factor of 1/(1-rho_min). n_n diffusion is exactly 0
  # (D=0, pure advection) so no J^2 term is needed. EXACT no-op at rho_min=0.
  # Applied at coefficient construction; n_n is not Pereverzev-augmented. See
  # Stage B0 in goal-1-2-6 and concepts/restricted-domain-migration-checklist.
  _jac_n_n = 1.0 / (1.0 - rho_min)
  full_v_face_n_n = (
      geo.vpr_face
      * runtime_params.profile_conditions.v_th_n
      / h_rho_face
      * _jac_n_n
  )
  # No velocity clip: with the domain starting at rho_min>0, there is no
  # coordinate singularity for a clip to regularize, and TORAX's solver
  # (theta=1, fully implicit backward Euler) is unconditionally stable
  # regardless (TORAX paper, section III.C.1). A velocity clip was tried
  # (2026-07-10 through 2026-07-13) when the domain still incorrectly
  # started at rho_hat=0; investigation traced the observed discrepancies
  # to the clip itself, not to genuine CFL instability -- but removing it
  # without also fixing the domain (2026-07-13) made things dramatically
  # worse, since the singularity was still physically present. Both issues
  # are resolved together by rho_min>0: see
  # concepts/coordinate-velocity-clipping in the project vault for the full
  # history.

  # entire coefficient preceding dT/dr in heat transport equations
  full_chi_face_ion = (
      geo.g1_over_vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
      * chi_face_ion_total
  )
  full_chi_face_el = (
      geo.g1_over_vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
      * chi_face_el_total
  )

  # entire coefficient preceding dne/dr in particle equation
  full_d_face_el = geo.g1_over_vpr_face * d_face_el_total
  full_v_face_el = geo.g0_face * v_face_el_total

  # density source terms. Initialize source matrix to zero
  source_mat_nn = jnp.zeros_like(geo.rho)

  # density source vector based both on original and updated core profiles
  source_n_e = merged_source_profiles.total_sources('n_e', geo)

  # Stage B fixed prescribed particle source (2026-07-22). A truly n_e-INDEPENDENT
  # inner-wall-peaked ionization source, the goal-1.2.3 "fixed exponential"
  # (Step-1) form, injected when profile_conditions.fixed_source_S0 > 0. Shape is
  # the neutral optical-depth attenuation exp(-tau) with tau built natively in
  # PHYSICAL rho_hat from the same h_rho used by the n_n advection above (neutrals
  # born at rho_hat=rho_min stream outward, ionizing as they penetrate). Amplitude
  # normalized via TORAX's own volume_integration over the RESTRICTED geometry, so
  # the total integrated source integral(vpr*S) == fixed_source_S0 natively on
  # [rho_min,1] (NOT truncated/renormalized from a [0,1] build). Scales linearly
  # with S0 => exactly inert (0) at the default S0=0, no branch needed. n_ref sets
  # the penetration depth (tau ~ n_ref*R_iz*h_rho/v_th_n). See Stage B in goal-1-2-6.
  # Stage 0 (2026-08-07, energy-equation-scoping Task 2): R_iz(T_e), computed
  # once and reused at all three call sites below (this one, and the n_n/n_e
  # ionisation-reaction terms further down). use_local_R_iz=False (default)
  # reproduces runtime_params.profile_conditions.R_iz exactly -- an exact
  # no-op regardless of core_profiles.T_e's value, preserving every existing
  # script's behaviour (T_e was previously an unused placeholder everywhere).
  _R_iz_cell = jnp.where(
      runtime_params.profile_conditions.use_local_R_iz,
      # core_profiles.T_e.value is in keV (state.py); Voronov's fit expects eV.
      _voronov_R_iz_H(core_profiles.T_e.value * 1e3),
      runtime_params.profile_conditions.R_iz,
  )
  _fs_S0 = runtime_params.profile_conditions.fixed_source_S0
  _fs_nref = runtime_params.profile_conditions.fixed_source_n_ref
  # Stage 1 (2026-08-19): self-consistent optical depth uses the CURRENT
  # n_e state instead of the fixed n_ref -- tau(rho_hat) = int n_e(s)*R_iz*
  # h_rho(s)/v_th_n ds becomes a functional of the solution itself. Default
  # (use_self_consistent_source=False) keeps the fixed_source_n_ref density,
  # an exact no-op for every prior script.
  _kappa_density = jnp.where(
      runtime_params.profile_conditions.use_self_consistent_source,
      core_profiles.n_e.value,
      _fs_nref,
  )
  _B_cell = jnp.sqrt(jnp.maximum(geo.gm5, 1e-10))
  # Same h_rho correction as the n_n advection above, applied to the fixed
  # source's optical depth so the two cannot disagree. Default False is a no-op.
  _h_rho_metric_cell = jnp.sqrt(
      jnp.maximum(geo.vpr / jnp.maximum(geo.g1_over_vpr, 1e-300), 0.0)
  )
  _h_rho_cell = jnp.where(
      runtime_params.profile_conditions.use_metric_h_rho,
      _h_rho_metric_cell,
      2.0 * psi_b * rho_hat_cell / jnp.maximum(_B_cell, 1e-10),
  )
  _kappa_cell = _kappa_density * _R_iz_cell * _h_rho_cell / (
      runtime_params.profile_conditions.v_th_n
  )
  # cumulative optical depth from the inner wall (physical rho_hat spacing),
  # shifted so tau=0 at the first cell (inner wall reference).
  _drho_hat = (1.0 - rho_min) * (geo.rho_norm[1] - geo.rho_norm[0])
  _tau_cell = (jnp.cumsum(_kappa_cell) - _kappa_cell[0]) * _drho_hat
  _fs_shape = jnp.exp(-_tau_cell)
  _fs_norm = jnp.maximum(math_utils.volume_integration(_fs_shape, geo), 1e-300)
  source_n_e = source_n_e + _fs_S0 * _fs_shape / _fs_norm

  # Neutral density (n_n) ionisation sink -- Phase 1 (2026-07-09).
  # Bilinear term n_e*R_iz*n_n, linearised by freezing n_e at its current
  # value (the standard treatment for a reaction term where only one of the
  # two factors is the variable being solved in THIS equation -- mirrors how
  # Qei's implicit_ii/ee terms use the OTHER temperature's current value).
  # This is the ('n_n','n_n') source_mat entry below -- diagonal (row=col),
  # so its implicit scaling ratio SCALING_FACTORS['n_n']/SCALING_FACTORS['n_n']
  # = 1 trivially; no correction needed (printed explicitly for the record).
  _nn_nn_scaling_ratio = (
      convertors.SCALING_FACTORS['n_n'] / convertors.SCALING_FACTORS['n_n']
  )
  source_mat_n_n_n_n = (
      -core_profiles.n_e.value
      * _R_iz_cell
      * geo.vpr
      * _nn_nn_scaling_ratio
  )
  source_n_n = jnp.zeros_like(geo.rho)

  # n_e ionisation source -- Phase 3 (2026-07-14). Symmetric cross-coupling
  # term completing the bilinear ionisation reaction n_e*R_iz*n_n: n_n's row
  # (above) freezes n_e and is linear in n_n (its own diagonal); n_e's row
  # mirrors this exact linearisation convention -- freeze n_e (the SAME
  # frozen value used above, for a single self-consistent linearisation
  # scheme) and be linear in n_n (the column variable) -- this is the
  # ('n_e','n_n') off-diagonal entry below. Only meaningful once
  # evolve_n_n=True AND evolve_density=True are both active (Phase 3); with
  # evolve_density=False (Phase 1/2), n_e is not in evolving_names so this
  # entry is never selected into source_mat_cell regardless of being defined
  # here.
  #
  # Scaling-factor correction (2026-07-15): the solver's own state vector is
  # x_scaled = n_physical/SCALING_FACTORS[name] (convertors.py). A
  # source_mat[(row,col)] entry multiplies x_col_scaled, so to represent the
  # PHYSICAL coefficient C correctly within row's own scaled equation it
  # must be rescaled by SCALING_FACTORS[col]/SCALING_FACTORS[row]. Every
  # source_mat entry before this one was either diagonal (ratio=1 trivially)
  # or (Qei's T_i/T_e cross terms) between variables sharing the same
  # SCALING_FACTORS=1.0 (ratio=1 again) -- this is the first cross-coupling
  # entry between variables with genuinely different scaling factors
  # (n_e:1e20, n_n:1e15). Without this factor the term was ~1e5x too large,
  # causing Phase 3 to diverge to NaN within the first step.
  _ne_nn_scaling_ratio = (
      convertors.SCALING_FACTORS['n_n'] / convertors.SCALING_FACTORS['n_e']
  )
  source_mat_n_e_n_n = (
      core_profiles.n_e.value
      * _R_iz_cell
      * geo.vpr
      * _ne_nn_scaling_ratio
  )

  # Pereverzev-Corrigan correction for heat and particle transport
  # (deals with stiff nonlinearity of transport coefficients)
  # TODO(b/311653933) this forces us to include value 0
  # convection terms in discrete system, slowing compilation down by ~10%.
  # See if can improve with a different pattern.
  (
      chi_face_per_ion,
      chi_face_per_el,
      v_heat_face_ion,
      v_heat_face_el,
      d_face_per_el,
      v_face_per_el,
  ) = jax.lax.cond(
      use_pereverzev,
      lambda: _calculate_pereverzev_flux(
          runtime_params,
          geo,
          core_profiles,
          pedestal_model_output,
      ),
      lambda: tuple([jnp.zeros_like(geo.rho_face)] * 6),
  )

  full_chi_face_ion += chi_face_per_ion
  full_chi_face_el += chi_face_per_el
  full_d_face_el += d_face_per_el
  full_v_face_el += v_face_per_el

  # Track B Step 2 (2026-08-11): physical electron heat convection from the
  # entropy closure (QLDipole_nU's v_heat_face_el, when use_entropy_chi_e is
  # set), added the SAME way Pereverzev/Phi_b_dot already add their own
  # contributions to v_heat_face_el below -- turbulent_transport.v_heat_face_el
  # is None for every transport model that doesn't populate it (structural,
  # not a traced value, so this check is trace-time-static and safe under
  # jit), an exact no-op for everything except this project's own model.
  if turbulent_transport.v_heat_face_el is not None:
    v_heat_face_el += turbulent_transport.v_heat_face_el

  # Add Phi_b_dot terms to heat transport convection
  v_heat_face_ion += (
      -3.0
      / 4.0
      * geo.Phi_b_dot
      / geo.Phi_b
      * geo.rho_face_norm
      * geo.vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
  )

  v_heat_face_el += (
      -3.0
      / 4.0
      * geo.Phi_b_dot
      / geo.Phi_b
      * geo.rho_face_norm
      * geo.vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
  )

  # Add Phi_b_dot terms to particle transport convection
  full_v_face_el += (
      -1.0 / 2.0 * geo.Phi_b_dot / geo.Phi_b * geo.rho_face_norm * geo.vpr_face
  )

  # Fill heat transport equation sources. Initialize source matrices to zero

  source_i = merged_source_profiles.total_sources('T_i', geo)
  source_e = merged_source_profiles.total_sources('T_e', geo)

  # Add the Qei effects.
  qei = merged_source_profiles.qei
  source_mat_ii = qei.implicit_ii * geo.vpr
  source_i += qei.explicit_i * geo.vpr
  source_mat_ee = qei.implicit_ee * geo.vpr
  source_e += qei.explicit_e * geo.vpr
  source_mat_ie = qei.implicit_ie * geo.vpr
  source_mat_ei = qei.implicit_ei * geo.vpr

  # Add effective Phi_b_dot heat source terms

  d_vpr53_rhon_n_e_drhon = jnp.gradient(
      geo.vpr ** (5.0 / 3.0) * geo.rho_norm * core_profiles.n_e.value,
      geo.rho_norm,
  )
  d_vpr53_rhon_n_i_drhon = jnp.gradient(
      geo.vpr ** (5.0 / 3.0) * geo.rho_norm * core_profiles.n_i.value,
      geo.rho_norm,
  )

  source_i += (
      3.0
      / 4.0
      * geo.vpr ** (-2.0 / 3.0)
      * d_vpr53_rhon_n_i_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.T_i.value
      * consts.keV_to_J
  )

  source_e += (
      3.0
      / 4.0
      * geo.vpr ** (-2.0 / 3.0)
      * d_vpr53_rhon_n_e_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.T_e.value
      * consts.keV_to_J
  )

  d_vpr_rhon_drhon = jnp.gradient(geo.vpr * geo.rho_norm, geo.rho_norm)

  # Add effective Phi_b_dot particle source terms
  source_n_e += (
      1.0
      / 2.0
      * d_vpr_rhon_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.n_e.value
  )

  # Add effective Phi_b_dot poloidal flux source term
  source_psi += (
      8.0
      * jnp.pi**2
      * consts.mu_0
      * geo.Phi_b_dot
      * geo.Phi_b
      * geo.rho_norm**2
      * conductivity.sigma
      / geo.F**2
      * core_profiles.psi.grad()
  )

  # Add pedestal source terms
  (
      source_i,
      source_e,
      source_n_e,
      source_mat_ii,
      source_mat_ee,
      source_mat_nn,
  ) = adaptive_source.apply_adaptive_source(
      source_T_i=source_i,
      source_T_e=source_e,
      source_n_e=source_n_e,
      source_mat_ii=source_mat_ii,
      source_mat_ee=source_mat_ee,
      source_mat_nn=source_mat_nn,
      runtime_params=runtime_params,
      pedestal_model_output=pedestal_model_output,
      geo=geo,
  )

  # Build arguments to solver based on which variables are evolving
  var_to_toc = {
      'T_i': toc_T_i,
      'T_e': toc_T_e,
      'psi': toc_psi,
      'n_e': toc_dens_el,
      'n_n': toc_n_n,
  }
  var_to_tic = {
      'T_i': tic_T_i,
      'T_e': tic_T_e,
      'psi': tic_psi,
      'n_e': tic_dens_el,
      'n_n': tic_n_n,
  }
  transient_out_cell = tuple(var_to_toc[var] for var in evolving_names)
  transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)

  var_to_d_face = {
      'T_i': full_chi_face_ion,
      'T_e': full_chi_face_el,
      'psi': d_face_psi,
      'n_e': full_d_face_el,
      'n_n': d_face_n_n,
  }
  d_face = tuple(var_to_d_face[var] for var in evolving_names)

  var_to_v_face = {
      'T_i': v_heat_face_ion,
      'T_e': v_heat_face_el,
      'psi': v_face_psi,
      'n_e': full_v_face_el,
      'n_n': full_v_face_n_n,
  }
  v_face = tuple(var_to_v_face.get(var) for var in evolving_names)

  # d maps (row var, col var) to the coefficient for that block of the matrix
  # (Can't use a descriptive name or the nested comprehension to build the
  # matrix gets too long)
  d = {
      ('T_i', 'T_i'): source_mat_ii,
      ('T_i', 'T_e'): source_mat_ie,
      ('T_e', 'T_i'): source_mat_ei,
      ('T_e', 'T_e'): source_mat_ee,
      ('n_e', 'n_e'): source_mat_nn,
      ('n_e', 'n_n'): source_mat_n_e_n_n,
      ('n_n', 'n_n'): source_mat_n_n_n_n,
      ('psi', 'psi'): source_mat_psi,
  }
  source_mat_cell = tuple(
      tuple(d.get((row_block, col_block)) for col_block in evolving_names)
      for row_block in evolving_names
  )

  # var_to_source ends up as a vector in the constructed PDE. Therefore any
  # scalings from CoreProfiles state variables to x must be applied here too.
  var_to_source = {
      'T_i': source_i / convertors.SCALING_FACTORS['T_i'],
      'T_e': source_e / convertors.SCALING_FACTORS['T_e'],
      'psi': source_psi / convertors.SCALING_FACTORS['psi'],
      'n_e': source_n_e / convertors.SCALING_FACTORS['n_e'],
      'n_n': source_n_n / convertors.SCALING_FACTORS['n_n'],
  }
  source_cell = tuple(var_to_source.get(var) for var in evolving_names)

  coeffs = block_1d_coeffs.Block1DCoeffs(
      transient_out_cell=transient_out_cell,
      transient_in_cell=transient_in_cell,
      d_face=d_face,
      v_face=v_face,
      source_mat_cell=source_mat_cell,
      source_cell=source_cell,
  )

  return coeffs


@functools.partial(
    jax.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _calc_coeffs_reduced(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates only the transient_in_cell terms in Block1DCoeffs."""

  # Only transient_in_cell is used for explicit terms if theta_implicit=1
  tic_T_i = core_profiles.n_i.value * geo.vpr ** (5.0 / 3.0)
  tic_T_e = core_profiles.n_e.value * geo.vpr ** (5.0 / 3.0)
  tic_psi = jnp.ones_like(geo.vpr)
  tic_dens_el = geo.vpr
  tic_n_n = geo.vpr

  var_to_tic = {
      'T_i': tic_T_i,
      'T_e': tic_T_e,
      'psi': tic_psi,
      'n_e': tic_dens_el,
      'n_n': tic_n_n,
  }
  transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)

  coeffs = block_1d_coeffs.Block1DCoeffs(
      transient_in_cell=transient_in_cell,
  )
  return coeffs
