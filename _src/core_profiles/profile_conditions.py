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

"""Profile condition parameters used throughout TORAX simulations."""
import dataclasses
import enum
from typing import Annotated, Callable, Final

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Self

# pylint: disable=invalid-name

# Order of magnitude validations to catch common config errors.
_MIN_IP_AMPS: Final[float] = 1e3
_MIN_DENSITY_M3: Final[float] = 1e10
_MAX_DENSITY_GW: Final[float] = 1e2
_MAX_TEMPERATURE_KEV: Final[float] = 1e3
_MAX_TEMPERATURE_BC_KEV: Final[float] = 5e1


class InitialPsiMode(enum.StrEnum):
  """How to calculate the initial psi value."""

  PROFILE_CONDITIONS = 'profile_conditions'
  GEOMETRY = 'geometry'
  J = 'j'


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParams:
  """Prescribed values and boundary conditions for the core profiles."""

  Ip: array_typing.FloatScalar
  v_loop_lcfs: array_typing.FloatScalar
  T_i_left_bc: array_typing.FloatScalar
  T_i_right_bc: array_typing.FloatScalar
  T_e_left_bc: array_typing.FloatScalar
  T_e_right_bc: array_typing.FloatScalar
  # Temperature profiles defined on the cell grid.
  T_e: array_typing.FloatVector
  T_i: array_typing.FloatVector
  # If provided as array, Psi profile defined on the cell grid.
  psi: array_typing.FloatVector | None
  psidot: array_typing.FloatVector | None
  toroidal_angular_velocity: array_typing.FloatVector | None
  toroidal_angular_velocity_right_bc: array_typing.FloatScalar | None
  # Electron density profile on the cell grid.
  n_e: array_typing.FloatVector
  nbar: array_typing.FloatScalar
  n_e_nbar_is_fGW: bool
  n_e_left_bc: array_typing.FloatScalar
  n_e_left_bc_is_fGW: bool
  n_e_right_bc: array_typing.FloatScalar
  n_e_right_bc_is_fGW: bool
  n_e_robin_bc_c_s: array_typing.FloatScalar
  # Outer sheath BC (2026-08-06, group-review directive): Gamma_out =
  # n(1)*c_s(T_e) at the LCFS, mirroring the inner Robin machinery above.
  # 0.0 (default, via ValidatedDefault fallback below) is an exact no-op --
  # the use_neumann_outer formula's denominator gets +0, reproducing the
  # existing zero-flux behaviour identically.
  n_e_sheath_outer_c_s: array_typing.FloatScalar
  # Inner grazing-recycling BC (2026-08-06, Track A): same Robin/mixed-BC
  # algebra as n_e_robin_bc_c_s, but the coefficient enters with a FLIPPED
  # SIGN (see getters.py) -- a net INFLUX proportional to n_face rather than
  # an outflux. Kept as a separate PositiveTimeVaryingScalar (a magnitude,
  # sign applied downstream) rather than loosening n_e_robin_bc_c_s's own
  # positivity constraint. 0.0 default is an exact no-op.
  n_e_graze_recycling_c_s: array_typing.FloatScalar
  # Track B Step 2 amendment, Task 3 (2026-08-11): sheath-like heat-loss
  # inner BC, q = gamma_sheath*n*c_s*T_e -- the thermal counterpart of the
  # particle grazing-recycling BC, testing the group's "heat flux out at the
  # inner boundary via grazing incidence" directive. gamma_sheath~7 is a
  # standard order-of-magnitude sheath heat-transmission coefficient
  # (dimensionless). 0.0 default is an exact no-op (see getters.py).
  gamma_sheath: array_typing.FloatScalar
  # Track B Step 2 amendment, Check 2 (2026-08-11): floor on the sheath-heat
  # BC's face value, testing whether the entropy closure's chi_e~T_e
  # feedback runaway (crashing T_e to sub-eV, outside the closure's derived
  # regime) is what produces the pressure-relocation finding, or whether it
  # survives a physically bounded wall temperature. 0.0 default is an exact
  # no-op (same [0, T_e0] clip as before).
  T_e_floor_keV: array_typing.FloatScalar
  n_n_left_bc: array_typing.FloatScalar
  v_th_n: array_typing.FloatScalar
  R_iz: array_typing.FloatScalar
  R_RECYCLE: array_typing.FloatScalar
  rho_min: array_typing.FloatScalar
  # Stage B (2026-07-22): fixed prescribed particle source for n_e. S0 is the
  # total integrated source (particles/s, = integral(vpr*S) drho_hat); n_ref
  # sets the optical-depth shape exp(-tau), tau built from n_ref*R_iz*h_rho/
  # v_th_n. Inert when S0=0 (default) -- source scales as S0 so no branch needed.
  fixed_source_S0: array_typing.FloatScalar
  fixed_source_n_ref: array_typing.FloatScalar
  # Stage 1 (2026-08-19, goal-5-realistic-neutrals-and-thermal): when True,
  # the optical-depth density in the Stage B source above is core_profiles.
  # n_e.value (the CURRENT, evolving state) instead of the fixed constant
  # fixed_source_n_ref -- makes S a functional of n, not a fixed profile.
  # Default False is an exact no-op (fixed_source_n_ref keeps its existing
  # role for every prior script). See calc_coeffs.py.
  use_self_consistent_source: bool = dataclasses.field(metadata={'static': True})
  current_profile_nu: float
  use_robin_bc: bool = dataclasses.field(metadata={'static': True})
  use_neumann_outer: bool = dataclasses.field(metadata={'static': True})
  use_sheath_outer: bool = dataclasses.field(metadata={'static': True})
  # 2026-09-10 (Jerome directive): the existing use_sheath_outer formula's
  # B_sheath_outer=c_s/drN*J term makes the effective boundary condition
  # progressively weaker relative to the diffusive term A_r=2*D_eff/drN^2 as
  # the grid is refined (A_r/B_sheath_outer ~ 1/drN -> infinity), so the
  # enforced flux is grid-dependent, not a fixed physical condition. This flag
  # selects a grid-independent alternative, derived from the continuum Robin
  # condition Gamma(L)=n(L)*c_s directly (see getters.py) rather than from an
  # equivalent-Dirichlet-value weighting whose balance point shifts with the
  # mesh. False (default) is an EXACT no-op, preserving the original
  # use_sheath_outer formula bit-for-bit -- see the getters.py regression
  # gate. Only meaningful when use_sheath_outer=True.
  use_sheath_outer_flux_form: bool = dataclasses.field(metadata={'static': True})
  use_graze_recycling: bool = dataclasses.field(metadata={'static': True})
  # Stage 0 (2026-08-07, energy-equation-scoping Task 2): both default False,
  # an exact no-op preserving every existing script's behaviour regardless of
  # what T_e happens to be configured to (previously always an unused
  # placeholder). When True, the corresponding quantity is computed from
  # core_profiles.T_e instead of the config-supplied constant -- R_iz via the
  # Voronov fit (calc_coeffs.py), the Robin-BC sheath c_s via sqrt(e*T_e/m_i)
  # (getters.py). n_e_robin_bc_c_s/R_iz remain valid explicit overrides in
  # either case; these flags only change what a NEW run can opt into.
  use_local_R_iz: bool = dataclasses.field(metadata={'static': True})
  use_local_robin_c_s: bool = dataclasses.field(metadata={'static': True})
  # Track B Step 2 (2026-08-11): zero-gradient (zero heat-flux) inner BC for
  # T_e, an alternative to the Dirichlet T_e_left_bc used everywhere so far.
  # Default False is an exact no-op (T_e_left_bc keeps its existing role).
  use_zero_heat_flux_inner: bool = dataclasses.field(metadata={'static': True})
  # Track B Step 2 amendment, Task 3 (2026-08-11): sheath-like heat-loss
  # inner BC, alternative to both T_e_left_bc (Dirichlet) and
  # use_zero_heat_flux_inner (pure Neumann). Default False is an exact
  # no-op. Takes priority over use_zero_heat_flux_inner when both are set
  # (mutually exclusive in practice; see getters.py).
  use_sheath_heat_inner: bool = dataclasses.field(metadata={'static': True})
  initial_j_is_total_current: bool = dataclasses.field(
      metadata={'static': True}
  )
  initial_psi_from_j: bool = dataclasses.field(metadata={'static': True})
  normalize_n_e_to_nbar: bool = dataclasses.field(metadata={'static': True})
  use_v_loop_lcfs_boundary_condition: bool = dataclasses.field(
      metadata={'static': True}
  )
  n_e_left_bc_is_absolute: bool = dataclasses.field(metadata={'static': True})
  n_e_right_bc_is_absolute: bool = dataclasses.field(metadata={'static': True})
  initial_psi_mode: InitialPsiMode = dataclasses.field(
      metadata={'static': True}
  )


class ProfileConditions(torax_pydantic.BaseModelFrozen):
  """Generic numeric parameters for the simulation.

  The `from_dict(...)` method can accept a dictionary defined by
  https://torax.readthedocs.io/en/latest/configuration.html#profile-conditions.

  Attributes:
    Ip: Total plasma current in A. Note that if Ip_from_parameters=False in
      geometry, then this Ip will be overwritten by values from the geometry
      data. If use_v_loop_lcfs_boundary_condition, only used as an initial
      condition.
    use_v_loop_lcfs_boundary_condition: Boundary condition at LCFS for Vloop ( =
      dspsi_lcfs/dt ). If use_v_loop_lcfs_boundary_condition is True, then the
      specified Vloop at the LCFS is used as the boundary condition for the psi
      equation; otherwise, Ip is used as the boundary condition.
    v_loop_lcfs: Boundary condition at LCFS for Vloop ( = dpsi_lcfs/dt ).
    T_i_right_bc: Temperature boundary conditions at r=a_minor. If this is
      `None` the boundary condition will instead be taken from `T_i` and `T_e`
      at rhon=1.
    T_e_right_bc: Temperature boundary conditions at r=a_minor. If this is
      `None` the boundary condition will instead be taken from `T_i` and `T_e`
      at rhon=1.
    T_i: Prescribed or evolving values for temperature at different times.
    T_e: Prescribed or evolving values for temperature at different times.
    psi: Initial values for psi. If provided, the initial psi will be taken from
      here. Otherwise, the initial psi will be calculated from either the
      geometry or the "current_profile_nu formula" dependant on the
      `initial_psi_from_j` field.
    psidot: Prescribed values for the time derivative of poloidal flux (loop
      voltage). If provided, and if `evolve_current` is False, this prescribed
      `psidot` will be used instead of the internally calculated one. This is
      useful for cases where an unphysical transient `psidot` from the initial
      `psi` condition needs to be overridden.
    toroidal_angular_velocity: Prescribed or evolving values for toroidal
      angular velocity. If None, toroidal_angular_velocity will be initialized
      to zero.
    toroidal_angular_velocity_right_bc: Toroidal velocity boundary condition for
      r=a_minor. If this is `None` the boundary condition will instead be taken
      from `toroidal_angular_velocity` at rhon=1. If `toroidal_angular_velocity
      is also `None`, then the boundary condition will be set to zero.
    n_e: Prescribed or evolving values for electron density at different times.
    normalize_n_e_to_nbar: Whether to renormalize the density profile to have
      the desired line averaged density `nbar`.
    nbar: Line averaged density. In units of m^-3 if n_e_nbar_is_fGW = False. In
      Greenwald fraction if n_e_nbar_is_fGW = True. nGW = Ip/(pi*a^2) with a in
      m, nGW in 10^20 m-3, Ip in MA
    n_e_nbar_is_fGW: Toggle units of nbar
    n_e_right_bc: Density boundary condition for r=a_minor. In units of m^-3 if
      n_e_right_bc_is_fGW = False. In Greenwald fraction if `n_e_right_bc_is_fGW
      = True`. If `n_e_right_bc` is `None` then the boundary condition will
      instead be taken from `n_e` at rho_norm=1. In this case,
      `n_e_right_bc_is_absolute` in the StaticRuntimeParams will be set to
      `False` and n_e_right_bc_is_fGW` will be set to `n_e_nbar_is_fGW`. If
      `n_e_right_bc` is not `None` then `n_e_right_bc_is_absolute` will be set
      to `True`.
    n_e_right_bc_is_fGW: Toggle units of n_e_right_bc.
    current_profile_nu: Peaking factor of "Ohmic" current: j_ohmic = j0*(1 -
      r^2/a^2)^current_profile_nu
    initial_j_is_total_current: Toggles if "Ohmic" current is treated as total
      current upon initialization, or if non-inductive current should be
      included in initial j_total calculation.
    initial_psi_from_j: Toggles if the initial psi calculation is based on the
      "current_profile_nu" current formula, or from the psi available in the
      numerical geometry file. This setting is ignored for the ad-hoc circular
      geometry, which has no numerical geometry.
    initial_psi_mode: Mode of the initial psi value. If `PROFILE_CONDITIONS` is
      used, then the initial psi is taken from the `psi` attribute. If
      `GEOMETRY` is used, then the initial psi is taken from the geometry. If
      `J` is used, then the psi calculation is based on the `current_profile_nu`
      current formula. For now if `PROFILE_CONDITIONS` is used, but `psi` is not
      provided then the `initial_psi_from_j` setting will be used to match the
      legacy behavior.
  """

  Ip: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(15e6)
  use_v_loop_lcfs_boundary_condition: Annotated[
      bool, torax_pydantic.JAX_STATIC
  ] = False
  v_loop_lcfs: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  T_i_left_bc: torax_pydantic.PositiveTimeVaryingScalar | None = None
  T_i_right_bc: torax_pydantic.PositiveTimeVaryingScalar | None = None
  T_e_left_bc: torax_pydantic.PositiveTimeVaryingScalar | None = None
  T_e_right_bc: torax_pydantic.PositiveTimeVaryingScalar | None = None
  T_i: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0: {0: 15.0, 1: 1.0}})
  )
  T_e: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0: {0: 15.0, 1: 1.0}})
  )
  psi: torax_pydantic.TimeVaryingArray | None = None
  psidot: torax_pydantic.TimeVaryingArray | None = None
  toroidal_angular_velocity: torax_pydantic.TimeVaryingArray | None = None
  toroidal_angular_velocity_right_bc: (
      torax_pydantic.TimeVaryingScalar | None
  ) = None
  n_e: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0: {0: 1.2e20, 1: 0.8e20}})
  )
  normalize_n_e_to_nbar: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  nbar: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.85e20
  )
  n_e_nbar_is_fGW: bool = False
  n_e_left_bc: torax_pydantic.TimeVaryingScalar | None = None
  n_e_right_bc: torax_pydantic.TimeVaryingScalar | None = None
  n_e_left_bc_is_fGW: bool = False
  n_e_right_bc_is_fGW: bool = False
  use_robin_bc: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  use_neumann_outer: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  n_e_robin_bc_c_s: torax_pydantic.PositiveTimeVaryingScalar | None = None
  # Stage 0 (2026-08-07): when True, the Robin BC's sheath-branch c_s is
  # computed from core_profiles.T_e at the inner face instead of reading
  # n_e_robin_bc_c_s -- see getters.py. Default False is an exact no-op:
  # n_e_robin_bc_c_s keeps its existing role (including the near-zero
  # "zeroflux" convention) for every script that doesn't set this.
  use_local_robin_c_s: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  # Track B Step 2 (2026-08-11): zero-gradient (zero heat-flux) inner BC for
  # T_e -- see getters.py's get_updated_electron_temperature. Default False
  # is an exact no-op (T_e_left_bc's existing Dirichlet role unchanged).
  use_zero_heat_flux_inner: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  # Track B Step 2 amendment, Task 3 (2026-08-11): sheath-like heat-loss
  # inner BC, q=gamma_sheath*n*c_s*T_e -- the thermal counterpart of the
  # inner Robin/grazing-recycling machinery above, testing the group's
  # "heat flux out via grazing incidence" directive. False (default) is an
  # exact no-op; gamma_sheath=0 with the flag True is ALSO an exact no-op
  # (reproduces use_zero_heat_flux_inner's pure-Neumann result), matching
  # the outer sheath BC's own tiny-value-reproduces-Neumann regression
  # pattern (Task 13).
  use_sheath_heat_inner: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  gamma_sheath: torax_pydantic.PositiveTimeVaryingScalar | None = None
  # Check 2 (2026-08-11): floor on the sheath-heat BC's face value, in keV.
  # 0.0 default is an exact no-op.
  T_e_floor_keV: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  # Outer sheath BC (2026-08-06): Gamma_out = n(1)*c_s(T_e) at the LCFS,
  # mirroring use_robin_bc/n_e_robin_bc_c_s above. use_sheath_outer=False
  # (default) is an exact no-op -- see getters.py's outer-BC block.
  use_sheath_outer: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  # 2026-09-10 (Jerome directive): grid-independent alternative to the above
  # formula's implicit grid-dependence. False (default) = exact no-op, see
  # RuntimeParams docstring above and getters.py for the full derivation.
  use_sheath_outer_flux_form: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  n_e_sheath_outer_c_s: torax_pydantic.PositiveTimeVaryingScalar | None = None
  # Inner grazing-recycling BC (2026-08-06): use_graze_recycling=True flips
  # the sign of the inner Robin coefficient (see getters.py) from outflux to
  # influx, using n_e_graze_recycling_c_s as the (always-positive) magnitude.
  # False (default) is an exact no-op.
  use_graze_recycling: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  n_e_graze_recycling_c_s: torax_pydantic.PositiveTimeVaryingScalar | None = None
  n_n_left_bc: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1e12)
  )
  # Neutral transport physics constants (Phase 1 scaffolding for n_n).
  # Defaults match T_neutral_eV=1, T_e=100eV, hydrogen mass -- see
  # ql_dipole_validation_v3.2.ipynb cell 18.
  v_th_n: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.385e4)
  )
  R_iz: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(3.17e-14)
  )
  # Stage 0 (2026-08-07): when True, R_iz is computed per-cell from
  # core_profiles.T_e via the Voronov (1997) fit instead of broadcasting this
  # config scalar -- see calc_coeffs.py. Default False is an exact no-op
  # (every existing script keeps the constant R_iz regardless of its T_e
  # placeholder value).
  use_local_R_iz: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  R_RECYCLE: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.99)
  )
  # Physical inner boundary of the computational domain, rho_hat=rho_min,
  # NOT a numerical parameter -- for the Junior equilibrium this should be
  # set to the first closed flux surface (fcfs), rho_min~0.2, below which
  # the domain is inside the magnet coil and non-physical. Default 0.0
  # preserves prior behaviour (domain starting at the magnetic axis) for
  # configs that don't set this explicitly.
  rho_min: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  # Stage B fixed prescribed n_e particle source (2026-07-22). Default 0.0 =>
  # inert (source scales linearly with S0), preserving all prior configs.
  fixed_source_S0: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  fixed_source_n_ref: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  # Stage 1 (2026-08-19): use core_profiles.n_e.value instead of
  # fixed_source_n_ref for the Stage B source's optical-depth density.
  # Default False is an exact no-op (every prior script keeps the fixed
  # n_ref).
  use_self_consistent_source: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  current_profile_nu: float = 1.0
  initial_j_is_total_current: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  # TODO(b/434175938): Remove this before the V2 API release in place of
  # initial_psi_source.
  initial_psi_from_j: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  initial_psi_mode: Annotated[InitialPsiMode, torax_pydantic.JAX_STATIC] = (
      InitialPsiMode.PROFILE_CONDITIONS
  )

  @pydantic.model_validator(mode='after')
  def after_validator(self) -> Self:
    error_messages = []

    def _sanity_check_profile_boundary_conditions(
        values,
        value_name,
        errors_list,
    ):
      """Check that the profile is defined at rho=1.0 for various cases."""
      if not values.right_boundary_conditions_defined:
        error_message = (
            f'As no right boundary condition was set for {value_name}, the'
            f' profile for {value_name} must include a rho=1.0 boundary'
            ' condition.'
        )
        errors_list.append(error_message)

    if self.T_i_right_bc is None:
      _sanity_check_profile_boundary_conditions(self.T_i, 'T_i', error_messages)
    if self.T_e_right_bc is None:
      _sanity_check_profile_boundary_conditions(self.T_e, 'T_e', error_messages)
    if self.n_e_right_bc is None:
      _sanity_check_profile_boundary_conditions(self.n_e, 'n_e', error_messages)

    # Validate plasma current input order of magnitude.
    if np.any(self.Ip.value < _MIN_IP_AMPS):
      failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
          self.Ip.value, self.Ip.time, lambda x: x < _MIN_IP_AMPS
      )
      error_messages.append(
          f'Plasma current Ip at time {time_at_fail}s is {failed_val:.2e} A,'
          f' which is below the minimum threshold of {_MIN_IP_AMPS:.0e} A.'
          ' Possible cause: erroneous input Ip in MA instead of A.'
      )

    # Validate plasma density inputs order of magnitude.
    if not self.n_e_nbar_is_fGW and self.normalize_n_e_to_nbar:
      if np.any(self.nbar.value < _MIN_DENSITY_M3):
        failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
            self.nbar.value, self.nbar.time, lambda x: x < _MIN_DENSITY_M3
        )
        error_messages.append(
            'Density inputs set in m^-3 units. Line-averaged density nbar at'
            f' time {time_at_fail}s is {failed_val:.2e} m^-3  which is below'
            f' the minimum threshold of {_MIN_DENSITY_M3:.0e} m^-3. Possible'
            ' cause: erroneous input nbar in normalized units instead of m^-3.'
        )

    if self.n_e_nbar_is_fGW and self.normalize_n_e_to_nbar:
      if np.any(self.nbar.value > _MAX_DENSITY_GW):
        failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
            self.nbar.value, self.nbar.time, lambda x: x > _MAX_DENSITY_GW
        )
        error_messages.append(
            'Density inputs set in Greenwald fraction. Line-averaged density'
            f' nbar time {time_at_fail}s is {failed_val:.2e} fGW which is'
            f' above the maximum threshold of {_MAX_DENSITY_GW:.0e} fGW.'
            ' Possible cause: erroneous input nbar in m^-3 instead of fGW.'
        )

    if not self.n_e_nbar_is_fGW and not self.normalize_n_e_to_nbar:
      for n_e_time, n_e_value in self.n_e.value.items():
        if np.any(n_e_value[1] < _MIN_DENSITY_M3):
          failed_val, rho_norm_at_fail = (
              _get_first_failing_value_and_time_or_rho(
                  n_e_value[1], n_e_value[0], lambda x: x < _MIN_DENSITY_M3
              )
          )
          error_messages.append(
              f'Density inputs set in m^-3 units. n_e at time {n_e_time}s and'
              f' rho_norm {rho_norm_at_fail} is {failed_val:.2e} m^-3  which is'
              f' below the minimum threshold of {_MIN_DENSITY_M3:.0e} m^-3.'
              ' Possible cause: erroneous input n_e in normalized units instead'
              ' of m^-3.'
          )
        break

    if self.n_e_nbar_is_fGW and not self.normalize_n_e_to_nbar:
      for n_e_time, n_e_value in self.n_e.value.items():
        if np.any(n_e_value[1] > _MAX_DENSITY_GW):
          (
              failed_val,
              rho_norm_at_fail,
          ) = _get_first_failing_value_and_time_or_rho(
              n_e_value[1], n_e_value[0], lambda x: x > _MAX_DENSITY_GW
          )
          error_messages.append(
              'Density inputs set in Greenwald fraction. n_e at time'
              f' {n_e_time}s and rho_norm {rho_norm_at_fail} is'
              f' {failed_val:.2e} m^-3  which is above the maximum threshold of'
              f' {_MAX_DENSITY_GW:.0e} fGW. Possible cause: erroneous input n_e'
              ' in m^-3 instead of fGW.'
          )
          break

    if self.n_e_right_bc is not None:
      if not self.n_e_right_bc_is_fGW:
        if np.any(self.n_e_right_bc.value < _MIN_DENSITY_M3):
          failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
              self.n_e_right_bc.value,
              self.n_e_right_bc.time,
              lambda x: x < _MIN_DENSITY_M3,
          )
          error_messages.append(
              'Density boundary condition inputs set in m^-3 units,'
              f' n_e_right_bc at time {time_at_fail}s is {failed_val:.2e} m^-3'
              f' which is below the minimum threshold of {_MIN_DENSITY_M3:.0e}'
              ' m^-3. Possible cause: erroneous input n_e_right_bc in'
              ' normalized units instead of m^-3.'
          )
      else:
        if np.any(self.n_e_right_bc.value > _MAX_DENSITY_GW):
          failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
              self.n_e_right_bc.value,
              self.n_e_right_bc.time,
              lambda x: x > _MAX_DENSITY_GW,
          )
          error_messages.append(
              'Density boundary condition inputs set in Greenwald fraction,'
              f' n_e_right_bc at time {time_at_fail}s is {failed_val:.2e} fGW'
              f' which is above the maximum threshold of {_MAX_DENSITY_GW:.0e}'
              ' fGW. Possible cause: erroneous input n_e_right_bc in m^-3'
              ' instead of fGW.'
          )

    # Validate temperature inputs order of magnitude.
    for T_e_time, T_e_value in self.T_e.value.items():
      if np.any(T_e_value[1] > _MAX_TEMPERATURE_KEV):
        (
            failed_val,
            rho_norm_at_fail,
        ) = _get_first_failing_value_and_time_or_rho(
            T_e_value[1], T_e_value[0], lambda x: x > _MAX_TEMPERATURE_KEV
        )
        error_messages.append(
            f'T_e at time {T_e_time}s and rho_norm {rho_norm_at_fail} is'
            f' {failed_val:.2e} which is above the maximum threshold of'
            f' {_MAX_TEMPERATURE_KEV:.0e} keV. Possible cause: erroneous'
            ' input T_e in eV instead of keV.'
        )
        break

    for T_i_time, T_i_value in self.T_i.value.items():
      if np.any(T_i_value[1] > _MAX_TEMPERATURE_KEV):
        (
            failed_val,
            rho_norm_at_fail,
        ) = _get_first_failing_value_and_time_or_rho(
            T_i_value[1], T_i_value[0], lambda x: x > _MAX_TEMPERATURE_KEV
        )
        error_messages.append(
            f'T_i at time {T_i_time}s and rho_norm {rho_norm_at_fail} is'
            f' {failed_val:.2e} which is above the maximum threshold of'
            f' {_MAX_TEMPERATURE_KEV:.0e} keV. Possible cause: erroneous'
            ' input T_i in eV instead of keV.'
        )
        break

    if self.T_e_right_bc is not None:
      if np.any(self.T_e_right_bc.value > _MAX_TEMPERATURE_BC_KEV):
        failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
            self.T_e_right_bc.value,
            self.T_e_right_bc.time,
            lambda x: x > _MAX_TEMPERATURE_BC_KEV,
        )
        error_messages.append(
            f'T_e_right_bc at time {time_at_fail}s is {failed_val:.2e} keV'
            ' which is above the maximum threshold of'
            f' {_MAX_TEMPERATURE_BC_KEV:.0e} keV. Possible cause: erroneous'
            ' input T_e_right_bc in eV instead of keV.'
        )

    if self.T_i_right_bc is not None:
      if np.any(self.T_i_right_bc.value > _MAX_TEMPERATURE_BC_KEV):
        failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
            self.T_i_right_bc.value,
            self.T_i_right_bc.time,
            lambda x: x > _MAX_TEMPERATURE_BC_KEV,
        )
        error_messages.append(
            f'T_i_right_bc at time {time_at_fail}s is {failed_val:.2e} keV'
            ' which is above the maximum threshold of'
            f' {_MAX_TEMPERATURE_BC_KEV:.0e} keV. Possible cause: erroneous'
            ' input T_i_right_bc in eV instead of keV.'
        )

    if error_messages:
      error_message_preamble = (
          f'{len(error_messages)} errors were found in profile conditions'
          ' config:\n\n'
      )
      final_error_message = '\n\n'.join(error_messages)
      raise ValueError(error_message_preamble + final_error_message)

    return self

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    """Builds a RuntimeParams object for time t."""

    runtime_params = {
        x.name: getattr(self, x.name)
        for x in dataclasses.fields(RuntimeParams)
        if x.name not in [
            'n_e_left_bc_is_absolute', 'n_e_right_bc_is_absolute',
            'n_e_robin_bc_c_s', 'n_e_sheath_outer_c_s',
            'n_e_graze_recycling_c_s', 'gamma_sheath',
        ]
    }

    if self.T_e_left_bc is None:
      runtime_params['T_e_left_bc'] = self.T_e.get_value(
          t, grid_type='face_left'
      )

    if self.T_e_right_bc is None:
      runtime_params['T_e_right_bc'] = self.T_e.get_value(
          t, grid_type='face_right'
      )

    if self.T_i_left_bc is None:
      runtime_params['T_i_left_bc'] = self.T_i.get_value(
          t, grid_type='face_left'
      )

    if self.T_i_right_bc is None:
      runtime_params['T_i_right_bc'] = self.T_i.get_value(
          t, grid_type='face_right'
      )

    if self.toroidal_angular_velocity_right_bc is None:
      if self.toroidal_angular_velocity is None:
        runtime_params['toroidal_angular_velocity_right_bc'] = 0.0
      else:
        runtime_params['toroidal_angular_velocity_right_bc'] = (
            self.toroidal_angular_velocity.get_value(t, grid_type='face_right')
        )

    if self.n_e_left_bc is None:
      runtime_params['n_e_left_bc'] = self.n_e.get_value(
          t, grid_type='face_left'
      )
      runtime_params['n_e_left_bc_is_fGW'] = self.n_e_nbar_is_fGW
      runtime_params['n_e_left_bc_is_absolute'] = False
    else:
      runtime_params['n_e_left_bc_is_absolute'] = True

    if self.n_e_right_bc is None:
      runtime_params['n_e_right_bc'] = self.n_e.get_value(
          t, grid_type='face_right'
      )
      runtime_params['n_e_right_bc_is_fGW'] = self.n_e_nbar_is_fGW
      runtime_params['n_e_right_bc_is_absolute'] = False
    else:
      runtime_params['n_e_right_bc_is_absolute'] = True

    runtime_params['n_e_robin_bc_c_s'] = (
        self.n_e_robin_bc_c_s.get_value(t)
        if self.n_e_robin_bc_c_s is not None
        else jnp.zeros(())
    )

    runtime_params['n_e_sheath_outer_c_s'] = (
        self.n_e_sheath_outer_c_s.get_value(t)
        if self.n_e_sheath_outer_c_s is not None
        else jnp.zeros(())
    )

    runtime_params['n_e_graze_recycling_c_s'] = (
        self.n_e_graze_recycling_c_s.get_value(t)
        if self.n_e_graze_recycling_c_s is not None
        else jnp.zeros(())
    )

    runtime_params['gamma_sheath'] = (
        self.gamma_sheath.get_value(t)
        if self.gamma_sheath is not None
        else jnp.zeros(())
    )

    def _get_value(x):
      if isinstance(
          x, (torax_pydantic.TimeVaryingScalar, torax_pydantic.TimeVaryingArray)
      ):
        return x.get_value(t)
      else:
        return x

    runtime_params = {k: _get_value(v) for k, v in runtime_params.items()}
    return RuntimeParams(**runtime_params)


def _get_first_failing_value_and_time_or_rho(
    values: np.ndarray,
    time_or_rho: np.ndarray,
    comparator: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, float]:
  """Returns the first failing value and time or rho for a given comparator.

  If values came from a TimeVaryingScalar, then time_or_rho will be the time.
  If values came from a TimeVaryingArray, then time_or_rho will be the rho.

  Args:
    values: The values to check.
    time_or_rho: The time or rho associated with the values.
    comparator: A function that takes an array of values and returns an array of
      bools indicating whether each value passes the check.

  Returns:
    A tuple of (first_failing_value, time_or_rho)
  """
  fail_indices = np.where(comparator(values))[0]
  if fail_indices.size == 0:
    raise ValueError(
        '_get_first_failing_time was called with no failing values.'
    )
  return values[fail_indices[0]], time_or_rho[fail_indices[0]]
