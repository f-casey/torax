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

"""Functions for getting updated CellVariable objects for CoreProfiles."""

import dataclasses
from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profile_conditions
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.core_profiles.plasma_composition import electron_density_ratios_zeff
from torax._src.core_profiles.plasma_composition import impurity_fractions
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import charge_states
from torax._src.physics import formulas
from torax._src.physics import psi_calculations
from torax._src import state

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Ions:
  """Helper container for holding ion attributes."""

  n_i: cell_variable.CellVariable
  n_impurity: cell_variable.CellVariable
  impurity_fractions: Mapping[str, array_typing.FloatVectorCell]
  main_ion_fractions: Mapping[str, array_typing.FloatScalar]
  Z_i: array_typing.FloatVectorCell
  Z_i_face: array_typing.FloatVectorFace
  Z_impurity: array_typing.FloatVectorCell
  Z_impurity_face: array_typing.FloatVectorFace
  A_i: array_typing.FloatScalar
  A_impurity: array_typing.FloatVectorCell
  A_impurity_face: array_typing.FloatVectorFace
  Z_eff: array_typing.FloatVectorCell
  Z_eff_face: array_typing.FloatVectorFace
  charge_state_info: charge_states.ChargeStateInfo
  charge_state_info_face: charge_states.ChargeStateInfo


def get_updated_ion_temperature(
    profile_conditions_params: profile_conditions.RuntimeParams,
    geo: geometry.Geometry,
    only_boundary_condition: bool = False,
    original_T_i_value: cell_variable.CellVariable | None = None,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed ion temperature profiles."""
  if only_boundary_condition:
    if original_T_i_value is None:
      raise ValueError(
          'original_T_i_value must be provided when only updating the boundary'
          ' condition.'
      )
    value = original_T_i_value.value
  else:
    value = profile_conditions_params.T_i
  T_i = cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      left_face_grad_constraint=None,
      left_face_constraint=profile_conditions_params.T_i_left_bc,
      right_face_grad_constraint=None,
      right_face_constraint=profile_conditions_params.T_i_right_bc,
  )
  return T_i


def get_updated_electron_temperature(
    profile_conditions_params: profile_conditions.RuntimeParams,
    geo: geometry.Geometry,
    only_boundary_condition: bool = False,
    original_T_e_value: cell_variable.CellVariable | None = None,
    transport_coeffs: state.CoreTransport | None = None,
    n_e: cell_variable.CellVariable | None = None,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron temperature profiles."""
  if only_boundary_condition:
    if original_T_e_value is None:
      raise ValueError(
          'original_T_e_value must be provided when only updating the boundary'
          ' condition.'
      )
    value = original_T_e_value.value
  else:
    value = profile_conditions_params.T_e

  # Track B Step 2 amendment, Task 3 (2026-08-11): sheath-like heat-loss
  # inner BC, q = gamma_sheath*n*c_s*T_e -- the thermal counterpart of the
  # inner particle Robin/grazing-recycling BC (this file's
  # get_updated_electron_density block), same Robin/mixed-BC derivation
  # pattern: solve for the equivalent Dirichlet T_e_face that reproduces the
  # heat-loss condition through the SAME diffusion+convection FVM algebra.
  #   q_left = 2*chi_eff*(T_face-T[0])/dr0 + v_heat*(2a-1)*T[0]
  #            + 2*v_heat*(1-a)*T_face  =  -gamma_sheath*n[0]*c_s(T[0])*T_face
  # c_s uses the INTERIOR cell's T[0] (known/frozen), not T_face itself,
  # keeping the equation LINEAR in T_face -- the same "freeze one factor"
  # linearisation already used for the n_e*R_iz*n_n bilinear term elsewhere
  # in this codebase. gamma_sheath=0 (default via ValidatedDefault) makes
  # the extra term vanish exactly, reproducing use_zero_heat_flux_inner's
  # pure-Neumann-equivalent result (mirrors the outer sheath BC's own
  # tiny-value-is-Neumann regression pattern, Task 13).
  #
  # UNIT/MAGNITUDE CAVEAT (2026-08-11): chi_eff and v_heat here come from
  # QLDipole_nU's entropy-closure output, whose overall scale is set by
  # D_tilde_s -- an EMPIRICALLY calibrated, uncalibrated-in-any-source free
  # parameter (see goal-1-2-6 Task 17/18). gamma_sheath's OWN absolute
  # effect on T_e therefore inherits that same scale uncertainty; only the
  # DIRECTION (does raising gamma_sheath cool the wall) and the REGRESSION
  # gate (gamma_sheath=0 is an exact no-op) are treated as reliable here.
  _sheath_heat_active = (
      profile_conditions_params.use_sheath_heat_inner
      and transport_coeffs is not None
      and n_e is not None
  )
  if _sheath_heat_active:
    chi_e_inner = transport_coeffs.chi_face_el[0]
    v_heat_inner = (
        transport_coeffs.v_heat_face_el[0]
        if transport_coeffs.v_heat_face_el is not None
        else jnp.zeros(())
    )
    dr0 = geo.rho_face_norm[1] - geo.rho_face_norm[0]
    n_e0 = n_e.value[0]
    _keV_to_J = 1e3 * 1.602176634e-19
    # BUG FIX (2026-08-12): calc_coeffs.py's OWN interior convention for the
    # heat equation is full_chi_face_el = g1_over_vpr_face*n_e_face*keV_to_J
    # *chi_face_el_total -- an EXTRA n_e*keV_to_J factor the PARTICLE
    # equation's D_eff does NOT carry (full_d_face_el = g1_over_vpr_face*
    # d_face_el_total only). The original version of this block mirrored the
    # particle equation's D_eff construction literally and omitted this
    # factor, making chi_eff (and hence A_h) ~871x too small at the tested
    # wall conditions -- this, combined with bug #2 below, is what caused
    # B_sheath_heat to dominate A_h by 10+ orders of magnitude at EVERY
    # gamma_sheath tested (0.01 through 7), instead of scaling gradually.
    # v_heat_face_el is NOT given the analogous treatment: calc_coeffs.py's
    # own full_v_face_el (particle) and v_heat_face_el (heat, Pereverzev/
    # Phi_b_dot contributions) both lack an n_e*keV_to_J factor too --
    # confirmed directly from calc_coeffs.py -- so only the diffusive term
    # needs this fix, not the convective one.
    chi_eff = geo.g1_over_vpr_face[0] * n_e0 * _keV_to_J * chi_e_inner
    v_heat_full = v_heat_inner  # NOT g0_face-multiplied, see calc_coeffs.py

    safe_chi = jnp.maximum(chi_eff, jnp.ones(()) * 1e-30)
    pe_h = -0.5 * dr0 * v_heat_full / safe_chi
    pe_eps = jnp.array(1e-3)
    pe_hs = jnp.where(jnp.abs(pe_h) < pe_eps, pe_eps, pe_h)
    alpha_h = jnp.where(
        jnp.abs(pe_h) < pe_eps,
        jnp.array(0.5),
        jnp.where(
            pe_h > 10.0, (pe_hs - 1.0) / pe_hs,
            jnp.where(
                pe_h > pe_eps,
                ((pe_hs - 1.0) + (1.0 - pe_hs / 10.0) ** 5) / pe_hs,
                jnp.where(
                    pe_h >= -10.0,
                    ((1.0 + pe_hs / 10.0) ** 5 - 1.0) / pe_hs,
                    -1.0 / pe_hs,
                ),
            ),
        ),
    )

    T_e0_keV = value[0]
    T_e0_eV = T_e0_keV * 1e3
    m_i_hydrogen = 1.67e-27
    c_s_T0 = jnp.sqrt(1.602e-19 * T_e0_eV / m_i_hydrogen)
    _jac_heat = 1.0 / (1.0 - profile_conditions_params.rho_min)
    A_h = 2.0 * chi_eff / (dr0 * dr0)
    # BUG FIX (2026-08-12): q_sheath[Joules-flux] = gamma_sheath*n_e0*c_s(T0)
    # *T_face[Joules], and T_face[Joules] = T_face[keV]*keV_to_J -- so the
    # coefficient multiplying the UNKNOWN T_face[keV] must carry keV_to_J
    # (the FIXED unit-conversion CONSTANT), not T_e0_keV (the INTERIOR
    # CELL's temperature, a state-dependent VARIABLE). The original version
    # used T_e0_keV here -- a genuine bug, not a modelling choice: it
    # confused "convert keV to Joules" with "multiply by the wall
    # temperature again". T_e0_keV~1e-4 (0.1eV) happens to still be a small
    # number, which is why the resulting saturation wasn't immediately
    # obvious as wrong, but it is ~6e11x larger than the correct keV_to_J
    # ~1.6e-16, and combined with A_h's own bug (above), together explain
    # why B_sheath_heat dominated A_h by 10+ orders of magnitude at every
    # gamma_sheath tested instead of scaling gradually with it.
    B_sheath_heat = (
        profile_conditions_params.gamma_sheath * n_e0 * c_s_T0 * _keV_to_J
        / dr0 * _jac_heat
    )
    numerator_h = A_h + v_heat_full * (1.0 - 2.0 * alpha_h) / dr0
    denominator_h = (
        A_h + 2.0 * v_heat_full * (1.0 - alpha_h) / dr0 + B_sheath_heat
    )
    T_e_face_robin = T_e0_keV * numerator_h / jnp.maximum(
        denominator_h, jnp.ones(()) * 1e-30
    )
    T_e_face_robin = jnp.clip(
        T_e_face_robin,
        profile_conditions_params.T_e_floor_keV,
        T_e0_keV,
    )

  # Track B Step 2 (2026-08-11): zero-gradient (zero heat-flux) inner BC,
  # an alternative to the Dirichlet T_e_left_bc used everywhere so far.
  # use_zero_heat_flux_inner is JAX_STATIC (a concrete Python bool, not a
  # traced value), so a plain Python conditional is correct here -- these are
  # STRUCTURAL choices (None vs an array), not a numeric jnp.where select.
  # Default False is an exact no-op -- same Dirichlet
  # left_face_constraint/left_face_grad_constraint=None pair as before. True
  # swaps to left_face_grad_constraint=0.0, left_face_constraint=None
  # (natural zero-flux, mirrors n_n's own outer zero-gradient BC pattern).
  # use_sheath_heat_inner (when active) takes priority over both.
  if _sheath_heat_active:
    _left_face_grad_constraint = None
    _left_face_constraint = T_e_face_robin
  elif profile_conditions_params.use_zero_heat_flux_inner:
    _left_face_grad_constraint = jnp.zeros(())
    _left_face_constraint = None
  else:
    _left_face_grad_constraint = None
    _left_face_constraint = profile_conditions_params.T_e_left_bc

  T_e = cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      left_face_grad_constraint=_left_face_grad_constraint,
      left_face_constraint=_left_face_constraint,
      right_face_grad_constraint=None,
      right_face_constraint=profile_conditions_params.T_e_right_bc,
  )
  return T_e


def get_updated_electron_density(
    profile_conditions_params: profile_conditions.RuntimeParams,
    geo: geometry.Geometry,
    transport_coeffs: state.CoreTransport | None = None,
    only_boundary_condition: bool = False,
    original_n_e_value: cell_variable.CellVariable | None = None,
    T_e: cell_variable.CellVariable | None = None,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron density profiles."""
  # Greenwald density in m^-3.
  # Ip in MA. a_minor in m.
  nGW = (
      profile_conditions_params.Ip
      / 1e6  # Convert to MA.
      / (jnp.pi * geo.a_minor**2)
      * 1e20
  )

  assert not profile_conditions_params.normalize_n_e_to_nbar

  n_e_left_bc = jnp.where(
      profile_conditions_params.n_e_left_bc_is_fGW,
      profile_conditions_params.n_e_left_bc * nGW,
      profile_conditions_params.n_e_left_bc,
  )

  # Calculate n_e_right_bc.
  n_e_right_bc = jnp.where(
      profile_conditions_params.n_e_right_bc_is_fGW,
      profile_conditions_params.n_e_right_bc * nGW,
      profile_conditions_params.n_e_right_bc,
  )

  n_e_value = jnp.where(
      profile_conditions_params.n_e_nbar_is_fGW,
      profile_conditions_params.n_e * nGW,
      profile_conditions_params.n_e,
  )

  if profile_conditions_params.normalize_n_e_to_nbar:
    face_left = n_e_value[0]  # Zero gradient boundary condition at left face.
    face_right = n_e_right_bc
    face_inner = math_utils.inner_face_values_from_cell_values(
        cell_values=n_e_value,
        face_centers=geo.rho_face_norm,
        cell_centers=geo.rho_norm,
    )
    n_e_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]],
    )
    # Find normalization factor such that desired line-averaged n_e is set.
    # Line-averaged electron density (nbar) is poorly defined. In general, the
    # definition is machine-dependent and even shot-dependent since it depends
    # on the usage of a specific interferometry chord. Furthermore, even if we
    # knew the specific chord used, its calculation would depend on magnetic
    # geometry information beyond what is available in StandardGeometry.
    # In lieu of a better solution, we use line-averaged electron density
    # defined on the outer midplane.
    a_minor_out = geo.R_out_face[-1] - geo.R_out_face[0]
    # find target nbar in absolute units
    target_nbar = jnp.where(
        profile_conditions_params.n_e_nbar_is_fGW,
        profile_conditions_params.nbar * nGW,
        profile_conditions_params.nbar,
    )
    if not profile_conditions_params.n_e_right_bc_is_absolute:
      # In this case, n_e_right_bc is taken from n_e and we also normalize it.
      C = target_nbar / (_trapz(n_e_face, geo.R_out_face) / a_minor_out)
      n_e_right_bc = C * n_e_right_bc
    else:
      # If n_e_right_bc is absolute, subtract off contribution from outer
      # face to get C we need to multiply the inner values with.
      nbar_from_n_e_face_inner = (
          _trapz(n_e_face[:-1], geo.R_out_face[:-1]) / a_minor_out
      )

      dr_edge = geo.R_out_face[-1] - geo.R_out_face[-2]

      C = (target_nbar - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
          nbar_from_n_e_face_inner + 0.5 * n_e_face[-2] * dr_edge / a_minor_out
      )
  else:
    C = 1

  n_e_value = C * n_e_value

  if only_boundary_condition:
    if original_n_e_value is None:
      raise ValueError(
          'original_n_e_value must be provided when only updating the boundary'
          ' condition.'
      )
    value = original_n_e_value.value
  else:
    value = n_e_value

  if profile_conditions_params.use_robin_bc and transport_coeffs is not None:
    # Equivalent Dirichlet BC for the sheath Robin condition Γ_total = n_face * c_s,
    # where Γ_total includes BOTH the FVM diffusion AND convection (QL pinch) fluxes.
    #
    # Derivation (ghost-cell Dirichlet mode, see diffusion_terms.py / convection_terms.py):
    #   The FVM flux at the left face (particles/s in the ρ-coordinate) is:
    #     Γ_left = 2*D_eff*(n_face-n[0])/dr0 + V*(2α-1)*n[0] + 2*V*(1-α)*n_face
    #   Sheath condition: Γ_left = -n_face * c_s  (NO geometric area factor — the
    #   QL dipole FVM flux D_eff*dn/dρ already absorbs all geometry via D_eff=D̃*U,
    #   so c_s enters without an additional g0 or a_minor multiplier).
    #   Solving for n_face:
    #     n_face = n[0] * (A + V*(1-2α)/dr0) / (A + 2*V*(1-α)/dr0 + c_s/dr0)
    #   where A = 2*D_eff/dr0², V = g0_face[0]*v_face_el[0] (= full_v_face_el[0]),
    #   α = peclet_to_alpha(-0.5*dr0*V/D_eff).  Reduces to diffusion-only when V=0.
    #   At steady state this gives n_face = N_EDGE*U[-1]/(U[0]+c_s/D̃), matching
    #   the QL dipole analytical target.
    D_e_inner = transport_coeffs.d_face_el[0]
    v_e_inner = transport_coeffs.v_face_el[0]
    # Inner grazing-recycling BC (2026-08-06, Track A, Task 2 candidate (a)):
    # SAME Robin/mixed-BC algebra, sign of the coefficient flipped -- a net
    # INFLUX proportional to n_face (ion grazing-incidence recycling off the
    # magnet, "opposite of a sheath") rather than the normal outflux. Reuses
    # 100% of the existing B_sheath/numerator/denominator algebra below;
    # only the sign of c_s_val changes. use_graze_recycling=False (default)
    # or n_e_graze_recycling_c_s=0 is an exact no-op.
    #
    # Stage 0 (2026-08-07, energy-equation-scoping Task 2): the sheath-branch
    # c_s can now be computed from the LOCAL T_e at the inner face instead of
    # the config-supplied n_e_robin_bc_c_s -- use_local_robin_c_s=False
    # (default) is an exact no-op, n_e_robin_bc_c_s keeps its existing role
    # (including the near-zero "zeroflux" convention used throughout Track
    # A) for every script that doesn't set this flag.
    # T_e.value is in keV (state.py); c_s formula expects eV, hence *1e3.
    _T_e_local_eV = (T_e.value[0] * 1e3) if T_e is not None else 100.0
    _c_s_local = jnp.sqrt(1.602e-19 * _T_e_local_eV / 1.67e-27)
    c_s_val = jnp.where(
        profile_conditions_params.use_graze_recycling,
        -profile_conditions_params.n_e_graze_recycling_c_s,
        jnp.where(
            profile_conditions_params.use_local_robin_c_s,
            _c_s_local,
            profile_conditions_params.n_e_robin_bc_c_s,
        ),
    )
    dr0 = geo.rho_face_norm[1] - geo.rho_face_norm[0]
    D_eff = geo.g1_over_vpr_face[0] * D_e_inner        # m³/s
    V_full = geo.g0_face[0] * v_e_inner                 # m³/s (full_v_face_el[0])

    # Power-law upwind weight (mirrors peclet_to_alpha in convection_terms.py).
    safe_D = jnp.maximum(D_eff, jnp.ones(()) * 1e-30)
    pe = -0.5 * dr0 * V_full / safe_D
    pe_eps = jnp.array(1e-3)
    pe_s = jnp.where(jnp.abs(pe) < pe_eps, pe_eps, pe)
    alpha = jnp.where(
        jnp.abs(pe) < pe_eps,
        jnp.array(0.5),
        jnp.where(
            pe > 10.0, (pe_s - 1.0) / pe_s,
            jnp.where(
                pe > pe_eps,
                ((pe_s - 1.0) + (1.0 - pe_s / 10.0) ** 5) / pe_s,
                jnp.where(
                    pe >= -10.0,
                    ((1.0 + pe_s / 10.0) ** 5 - 1.0) / pe_s,
                    -1.0 / pe_s,
                ),
            ),
        ),
    )

    # Coordinate Jacobian (2026-07-22, Stage B0 fix). D_e_inner/v_e_inner read
    # above are ALREADY the Jacobian-corrected coefficients (QLDipole_nU scales
    # d_face_el by 1/(1-rho_min)^2 and v_face_el by 1/(1-rho_min) at its output),
    # so D_eff = J^2*D_phys and V_full = J*V_phys here -- consistent with the
    # interior FVM, no mixing. With these, the reconstructed left-face flux
    # (A and V_full/dr0 terms, dr0 in MESH coordinate) equals TORAX's internal
    # flux Phi_torax = J*Phi_physical. The sheath condition is on the PHYSICAL
    # flux, Phi_physical = n_face*c_s, so balancing Phi_torax against the sheath
    # requires the c_s term to carry one factor of J as well:
    #   Phi_torax = J*Phi_physical = J*n_face*c_s  ->  c_s_eff = J*c_s.
    # (The Peclet number pe above comes out physically correct automatically:
    # pe = -0.5*dr0_mesh*V_full/D_eff = -0.5*dr0_mesh*V_phys/(J*D_phys), i.e. it
    # uses the physical half-cell length dr0_mesh/J.) EXACT no-op at rho_min=0
    # (J=1). NOTE ON GRADIENT-CONSTRAINT BCs: TORAX applies face_grad_constraint
    # values in the MESH coordinate (CellVariable.face_centers = geo.rho_face_norm,
    # [0,1]). Today's only grad constraint is n_n's outer zero-gradient (0 scales
    # to 0, safe). Any FUTURE nonzero gradient constraint on a rho_min>0 domain
    # must convert a physical dn/drho_hat to the mesh value via *(1-rho_min), i.e.
    # mesh_constraint = physical_gradient/J. See restricted-domain-migration-checklist.
    _jac_robin = 1.0 / (1.0 - profile_conditions_params.rho_min)
    A = 2.0 * D_eff / (dr0 * dr0)
    B_sheath = c_s_val / dr0 * _jac_robin
    numerator   = A + V_full * (1.0 - 2.0 * alpha) / dr0
    denominator = A + 2.0 * V_full * (1.0 - alpha) / dr0 + B_sheath
    n_face_robin = value[0] * numerator / jnp.maximum(denominator, jnp.ones(()) * 1e-30)
    # Clip to physical range to guard against transient blow-up when D_eff ≈ 0 at
    # startup. Sheath/outflux (use_graze_recycling=False): face density can only
    # be depleted relative to the interior, so clip to [0, value[0]] -- this is
    # the original guard. Grazing-recycling (use_graze_recycling=True): an INFLUX
    # BC, so the physically correct direction is n_face >= value[0] (recycling
    # enriches the face above the interior); clipping to value[0] here would
    # silently zero out the entire recycling effect (bug found 2026-08-07 --
    # every nonzero recycling strength was being clamped back to the zero-flux
    # face value). Use [value[0], large-finite] instead, the finite bound still
    # catching the same D_eff≈0 startup blow-up the sheath branch guards against.
    n_face_robin = jnp.where(
        profile_conditions_params.use_graze_recycling,
        jnp.clip(n_face_robin, value[0], jnp.ones(()) * 1e30),
        jnp.clip(n_face_robin, jnp.zeros(()), value[0]),
    )

    left_face_bc = n_face_robin
  else:
    left_face_bc = n_e_left_bc

  if (
      profile_conditions_params.use_neumann_outer
      or profile_conditions_params.use_sheath_outer
  ) and transport_coeffs is not None:
    # Zero-total-flux outer BC: Γ_diff + Γ_conv = 0 at ρ=1 (use_neumann_outer),
    # OR the outer SHEATH BC (2026-08-06, group-review directive): Γ_diff +
    # Γ_conv = -n_face*c_s_outer at ρ=1 (use_sheath_outer) -- the LCFS as a
    # parallel-loss channel for levitated operation, mirroring the inner
    # Robin BC's B_sheath term exactly (see this file's inner-BC block above).
    # Right-face analog of the Robin inner BC — solve for n_face_outer s.t.
    # net particle flux (diffusive + convective) crosses the outer face at
    # the (possibly zero) sheath rate.
    #   Γ_right = 2D(n_face-n[-1])/drN + V[(2α-1)n[-1] + 2(1-α)n_face]
    #           = -n_face * c_s_outer
    #   → n_face = n[-1] × (A + V(1-2α)/drN) / (A + 2V(1-α)/drN + B_sheath_outer)
    # where A = 2D/drN², B_sheath_outer = c_s_outer/drN * J (SAME Jacobian
    # convention as the inner Robin BC's B_sheath, J=1/(1-rho_min)), pe =
    # +0.5·drN·V/D (positive sign for right face: outward V→Pe>0→interior is
    # upwind). B_sheath_outer=0 (use_sheath_outer=False, or c_s_outer=0) is an
    # EXACT no-op, reproducing the pure-Neumann formula identically. When
    # V·drN/(2D) > 1 the formula gives n_face < 0 (convection too strong to
    # balance); clips to 0 (absorbing-wall limit).
    D_e_outer = transport_coeffs.d_face_el[-1]
    v_e_outer = transport_coeffs.v_face_el[-1]
    drN = geo.rho_face_norm[-1] - geo.rho_face_norm[-2]
    D_eff_r = geo.g1_over_vpr_face[-1] * D_e_outer
    V_full_r = geo.g0_face[-1] * v_e_outer

    safe_D_r = jnp.maximum(D_eff_r, jnp.ones(()) * 1e-30)
    pe_r = 0.5 * drN * V_full_r / safe_D_r
    pe_eps = jnp.array(1e-3)
    pe_s_r = jnp.where(jnp.abs(pe_r) < pe_eps, pe_eps, pe_r)
    alpha_r = jnp.where(
        jnp.abs(pe_r) < pe_eps,
        jnp.array(0.5),
        jnp.where(
            pe_r > 10.0, (pe_s_r - 1.0) / pe_s_r,
            jnp.where(
                pe_r > pe_eps,
                ((pe_s_r - 1.0) + (1.0 - pe_s_r / 10.0) ** 5) / pe_s_r,
                jnp.where(
                    pe_r >= -10.0,
                    ((1.0 + pe_s_r / 10.0) ** 5 - 1.0) / pe_s_r,
                    -1.0 / pe_s_r,
                ),
            ),
        ),
    )

    _jac_sheath_outer = 1.0 / (1.0 - profile_conditions_params.rho_min)

    if profile_conditions_params.use_sheath_outer_flux_form:
      # 2026-09-10 (Jerome directive): grid-independent flux condition,
      # derived directly from the continuum Robin BC Gamma_phys(L)=n(L)*c_s
      # (not from an equivalent-Dirichlet-value weighting between A_r and
      # B_sheath_outer, whose balance point drifts with drN -- see the
      # RuntimeParams docstring). Derivation: with D_phys=D_eff_r/J^2 and
      # V_phys=V_full_r/J (undoing the Stage-B0 mesh-Jacobian pre-scaling),
      # a first-order Taylor extrapolation from the last cell centre to the
      # face, n_face = n[-1] + (dn/drho_hat)|_L * (drN/(2J)), combined with
      # the Robin condition (dn/drho_hat)|_L = -n_face*(c_s_outer-V_phys)/
      # D_phys, gives n_face EXACTLY satisfying Gamma_phys(L)=n_face*c_s_outer
      # for ANY drN by construction (verified: the algebra solves for n_face
      # such that this holds identically, not just in a limit) -- only the
      # accuracy of n_face as an approximation to the true continuum n(L)
      # depends on drN (standard first-order truncation error), not the
      # boundary condition itself.
      denom_flux = 2.0 * D_eff_r + (
          profile_conditions_params.n_e_sheath_outer_c_s * _jac_sheath_outer
          - V_full_r
      ) * drN
      n_face_outer = value[-1] * 2.0 * D_eff_r / jnp.maximum(denom_flux, jnp.ones(()) * 1e-30)
      n_face_outer = jnp.clip(n_face_outer, jnp.zeros(()), value[-1])
    else:
      A_r = 2.0 * D_eff_r / (drN * drN)
      B_sheath_outer = (
          profile_conditions_params.n_e_sheath_outer_c_s / drN * _jac_sheath_outer
      )
      numerator_r   = A_r + V_full_r * (1.0 - 2.0 * alpha_r) / drN
      denominator_r = A_r + 2.0 * V_full_r * (1.0 - alpha_r) / drN + B_sheath_outer
      n_face_outer = value[-1] * numerator_r / jnp.maximum(denominator_r, jnp.ones(()) * 1e-30)
      n_face_outer = jnp.clip(n_face_outer, jnp.zeros(()), value[-1])
    right_face_bc = n_face_outer
  else:
    right_face_bc = n_e_right_bc

  n_e = cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      left_face_grad_constraint=None,
      left_face_constraint=left_face_bc,
      right_face_grad_constraint=None,
      right_face_constraint=right_face_bc,
  )
  return n_e


def get_updated_neutral_density(
    profile_conditions_params: profile_conditions.RuntimeParams,
    geo: geometry.Geometry,
    only_boundary_condition: bool = False,
    original_n_n_value: cell_variable.CellVariable | None = None,
    n_e: cell_variable.CellVariable | None = None,
    T_e: cell_variable.CellVariable | None = None,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed neutral density profiles.

  Initial profile is a flat 1e12 m^-3 -- not yet a user-configurable
  profile_conditions field (only the boundary conditions are exposed at this
  stage; mirrors the simple T_i/T_e pattern rather than n_e's Greenwald/nbar
  machinery, which doesn't apply here).

  Inner BC is the recycling condition n_n(0) = R_RECYCLE * n_e(0) * c_s /
  v_th_n, recomputed from the CURRENT n_e each call (mirrors what the
  external operator-split loop did, now inside TORAX). c_s uses the LOCAL
  T_e at cell 0 (Stage 0, 2026-08-07 -- previously a hardcoded 100eV
  placeholder) and hydrogen mass -- matching the notebook's own standalone
  c_s convention exactly, NOT core_profiles.A_i (which reflects the
  configured main_ion='D', deuterium, ~2x hydrogen mass) -- consistency
  with the notebook's external validation (Run C_RS, evolve_neutral_pde)
  matters more here than using the "real" configured ion species, since
  that's what we're being validated against. If `n_e` is not provided
  (e.g. during initialization, before any n_e state exists yet), falls back
  to the plain n_n_left_bc Dirichlet default (1e12). If `T_e` is not
  provided (e.g. an older caller), falls back to the previous 100eV
  placeholder.

  Outer BC is zero-gradient (outflow): no face-value constraint, matching
  the natural upwind exit for pure advection (n_n has D=0, per Jerome).
  """
  if only_boundary_condition:
    if original_n_n_value is None:
      raise ValueError(
          'original_n_n_value must be provided when only updating the '
          'boundary condition.'
      )
    value = original_n_n_value.value
  else:
    value = jnp.full_like(geo.rho, 1e12)

  if n_e is not None:
    # Stage 0 (2026-08-07, energy-equation-scoping Task 2): use the actual
    # local T_e at cell 0 (matching n_e_inner's own indexing) instead of a
    # hardcoded placeholder. No override/backward-compat gate needed here --
    # unlike n_e_robin_bc_c_s, this was never a user-configurable value, and
    # n_n's own BC value doesn't feed back into n_e's evolution when
    # evolve_n_n=False (every currently-existing config), so this change
    # carries no regression risk regardless of T_e's configured value.
    # T_e.value is in keV (state.py); c_s formula expects eV, hence *1e3.
    m_i_hydrogen = 1.67e-27
    T_e_local_eV = (T_e.value[0] * 1e3) if T_e is not None else 100.0
    c_s = jnp.sqrt(1.602e-19 * T_e_local_eV / m_i_hydrogen)
    # WHICH DENSITY SETS THE RECYCLING INFLOW (2026-10-02).
    # The default uses n_e.value[0], the first CELL-CENTRE value. That makes the
    # neutral inflow MESH-DEPENDENT: the first cell centre sits further from the
    # wall on a coarser mesh, and where n_e rises steeply off the wall the inflow
    # is inflated by exactly n_e[0]/n_e(wall). Measured in the Day 3 coupled
    # runs: n_e[0]/n_0 = 1.005 / 1.011 / 1.052 at Pe = 10 / 100 / 1000 on fitted
    # meshes, and 2.105 at Pe = 1000 on a coarse one -- and dividing it out
    # recovered the analytic reference to 0.01-0.04%, i.e. it accounted for the
    # WHOLE discrepancy.
    # use_metric_h_rho=True selects the wall FACE density instead, which is the
    # C1 condition inflow = n_0 c_s and is mesh-independent. Default False keeps
    # n_e.value[0] exactly, so every prior script is unchanged.
    n_e_inner = jnp.where(
        profile_conditions_params.use_metric_h_rho,
        profile_conditions_params.n_e_left_bc,
        n_e.value[0],
    )
    _density_driven_bc = (
        profile_conditions_params.R_RECYCLE
        * n_e_inner
        * c_s
        / profile_conditions_params.v_th_n
    )
    # FLUX-DRIVEN INFLOW (2026-10-08). Both branches above build the inflow from
    # a DENSITY times c_s, which presumes the ion flux into the coil is the Bohm
    # flux. With a pinch the ion flux is whatever the transport solution gives,
    # and Jerome's recycling-consistent wall asks for inflow = R * Gamma_w with
    # that computed flux. use_flux_driven_n_n_bc=True hands n_n_left_bc straight
    # through so the caller can supply R*max(Gamma_w,0)/v_th_n itself; R is then
    # the caller's, not R_RECYCLE's. jnp.where (not a Python if) because
    # n_n_left_bc is traced; the flag itself is static, so XLA prunes the dead
    # branch and the flag-off path is bit-identical.
    left_face_bc = jnp.where(
        profile_conditions_params.use_flux_driven_n_n_bc,
        profile_conditions_params.n_n_left_bc,
        _density_driven_bc,
    )
  else:
    left_face_bc = profile_conditions_params.n_n_left_bc

  n_n = cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      left_face_grad_constraint=None,
      left_face_constraint=left_face_bc,
      right_face_grad_constraint=jnp.zeros(()),
      right_face_constraint=None,
  )
  return n_n


def get_updated_psi(
    profile_conditions_params: profile_conditions.RuntimeParams,
    geo: geometry.Geometry,
    dt: array_typing.FloatScalar,
    theta: array_typing.FloatScalar,
    only_boundary_condition: bool,
    original_psi: cell_variable.CellVariable,
) -> cell_variable.CellVariable:
  """Gets psi boundary conditions and maybe the prescribed profile."""
  if only_boundary_condition or profile_conditions_params.psi is None:
    value = original_psi.value
  else:
    value = profile_conditions_params.psi

  right_face_grad_constraint = (
      psi_calculations.calculate_psi_grad_constraint_from_Ip(  # pylint: disable=g-long-ternary
          Ip=profile_conditions_params.Ip,
          geo=geo,
      )
      if not profile_conditions_params.use_v_loop_lcfs_boundary_condition
      else None
  )
  right_face_constraint = (
      psi_calculations.calculate_psi_value_constraint_from_v_loop(  # pylint: disable=g-long-ternary
          dt=dt,
          v_loop_lcfs_t=profile_conditions_params.v_loop_lcfs,
          v_loop_lcfs_t_plus_dt=profile_conditions_params.v_loop_lcfs,
          psi_lcfs_t=original_psi.right_face_constraint,
          theta=theta,
      )
      if profile_conditions_params.use_v_loop_lcfs_boundary_condition
      else None
  )
  return cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=right_face_grad_constraint,
      right_face_constraint=right_face_constraint,
  )


def get_updated_toroidal_angular_velocity(
    profile_conditions_params: profile_conditions.RuntimeParams,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed toroidal velocity profiles."""
  if profile_conditions_params.toroidal_angular_velocity is None:
    value = jnp.zeros_like(geo.rho)
  else:
    value = profile_conditions_params.toroidal_angular_velocity
  toroidal_angular_velocity = cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=None,
      right_face_constraint=profile_conditions_params.toroidal_angular_velocity_right_bc,
  )
  return toroidal_angular_velocity


@dataclasses.dataclass(frozen=True)
class _IonProperties:
  """Helper container for holding ion calculation outputs."""

  A_impurity: array_typing.FloatVectorCell
  A_impurity_face: array_typing.FloatVectorFace
  Z_impurity: array_typing.FloatVectorCell
  Z_impurity_face: array_typing.FloatVectorFace
  Z_eff: array_typing.FloatVectorCell
  dilution_factor: array_typing.FloatVectorCell
  dilution_factor_inner_edge: array_typing.FloatScalar
  dilution_factor_outer_edge: array_typing.FloatScalar
  impurity_fractions: Mapping[str, array_typing.FloatVectorCell]
  charge_state_info: charge_states.ChargeStateInfo
  charge_state_info_face: charge_states.ChargeStateInfo


def _get_ion_properties_from_fractions(
    impurity_params: impurity_fractions.RuntimeParams,
    T_e: cell_variable.CellVariable,
    Z_i: array_typing.FloatVectorCell,
    Z_i_face: array_typing.FloatVectorFace,
    Z_eff_from_config: array_typing.FloatVectorCell,
    Z_eff_face_from_config: array_typing.FloatVectorFace,
) -> _IonProperties:
  """Calculates ion properties when impurity content is defined by fractions."""

  charge_state_info = charge_states.get_average_charge_state(
      T_e=T_e.value,
      fractions=impurity_params.fractions,
      Z_override=impurity_params.Z_override,
  )
  Z_impurity = charge_state_info.Z_mixture

  charge_state_info_face = charge_states.get_average_charge_state(
      T_e=T_e.face_value(),
      fractions=impurity_params.fractions_face,
      Z_override=impurity_params.Z_override,
  )
  Z_impurity_face = charge_state_info_face.Z_mixture

  Z_eff = Z_eff_from_config
  Z_eff_inner_edge = Z_eff_face_from_config[0]
  Z_eff_outer_edge = Z_eff_face_from_config[-1]

  dilution_factor = jnp.where(
      Z_eff == 1.0,
      1.0,
      formulas.calculate_main_ion_dilution_factor(Z_i, Z_impurity, Z_eff),
  )
  dilution_factor_inner_edge = jnp.where(
      Z_eff_inner_edge == 1.0,
      1.0,
      formulas.calculate_main_ion_dilution_factor(
          Z_i_face[0], Z_impurity_face[0], Z_eff_inner_edge
      ),
  )
  dilution_factor_outer_edge = jnp.where(
      Z_eff_outer_edge == 1.0,
      1.0,
      formulas.calculate_main_ion_dilution_factor(
          Z_i_face[-1], Z_impurity_face[-1], Z_eff_outer_edge
      ),
  )
  return _IonProperties(
      A_impurity=impurity_params.A_avg,
      A_impurity_face=impurity_params.A_avg_face,
      Z_impurity=Z_impurity,
      Z_impurity_face=Z_impurity_face,
      Z_eff=Z_eff,
      dilution_factor=dilution_factor,
      dilution_factor_inner_edge=dilution_factor_inner_edge,
      dilution_factor_outer_edge=dilution_factor_outer_edge,
      impurity_fractions=impurity_params.fractions,
      charge_state_info=charge_state_info,
      charge_state_info_face=charge_state_info_face,
  )


def _get_ion_properties_from_n_e_ratios(
    impurity_params: electron_density_ratios.RuntimeParams,
    T_e: cell_variable.CellVariable,
    Z_i: array_typing.FloatVectorCell,
    Z_i_face: array_typing.FloatVectorFace,
) -> _IonProperties:
  """Calculates ion properties when impurity content is defined by n_e ratios."""
  average_charge_state = charge_states.get_average_charge_state(
      T_e=T_e.value,
      fractions=impurity_params.fractions,
      Z_override=impurity_params.Z_override,
  )
  average_charge_state_face = charge_states.get_average_charge_state(
      T_e=T_e.face_value(),
      fractions=impurity_params.fractions_face,
      Z_override=impurity_params.Z_override,
  )
  Z_impurity = average_charge_state.Z_mixture
  Z_impurity_face = average_charge_state_face.Z_mixture
  dilution_factor = (
      1
      - jnp.sum(
          jnp.array([
              average_charge_state.Z_per_species[ion] * n_e_ratio
              for ion, n_e_ratio in impurity_params.n_e_ratios.items()
          ]),
          axis=0,
      )
      / Z_i
  )
  dilution_factor_inner_edge = (
      1
      - jnp.sum(
          jnp.array([
              average_charge_state_face.Z_per_species[ion][0] * n_e_ratio[0]
              for ion, n_e_ratio in impurity_params.n_e_ratios_face.items()
          ]),
          axis=0,
      )
      / Z_i_face[0]
  )
  dilution_factor_outer_edge = (
      1
      - jnp.sum(
          jnp.array([
              average_charge_state_face.Z_per_species[ion][-1] * n_e_ratio[-1]
              for ion, n_e_ratio in impurity_params.n_e_ratios_face.items()
          ]),
          axis=0,
      )
      / Z_i_face[-1]
  )
  Z_eff = dilution_factor * Z_i**2 + jnp.sum(
      jnp.array([
          average_charge_state.Z_per_species[ion] ** 2 * n_e_ratio
          for ion, n_e_ratio in impurity_params.n_e_ratios.items()
      ]),
      axis=0,
  )

  return _IonProperties(
      A_impurity=impurity_params.A_avg,
      A_impurity_face=impurity_params.A_avg_face,
      Z_impurity=Z_impurity,
      Z_impurity_face=Z_impurity_face,
      Z_eff=Z_eff,
      dilution_factor=dilution_factor,
      dilution_factor_inner_edge=dilution_factor_inner_edge,
      dilution_factor_outer_edge=dilution_factor_outer_edge,
      impurity_fractions=impurity_params.fractions,
      charge_state_info=average_charge_state,
      charge_state_info_face=average_charge_state_face,
  )


# TODO(b/440666091): Refactor this function by breaking it down to several
# smaller helper functions
def _get_ion_properties_from_n_e_ratios_Z_eff(
    impurity_params: electron_density_ratios_zeff.RuntimeParams,
    T_e: cell_variable.CellVariable,
    Z_i: array_typing.FloatVectorCell,
    Z_i_face: array_typing.FloatVectorFace,
    Z_eff_from_config: array_typing.FloatVectorCell,
    Z_eff_face_from_config: array_typing.FloatVectorFace,
) -> _IonProperties:
  """Calculates ion properties when one impurity is constrained by Z_eff.

  We solve for the unknown impurity species n_e_ratio and the main ion
  n_e_ratio (dilution factor) from quasi-neutrality and Z_eff equations:

  ne = Z_i * n_i + sum(Z_impurity * n_impurity)
  Z_eff = (Z_i**2 * n_i + sum(Z_impurity**2 * n_impurity)) / n_e

  This defines a 2x2 system of equations

  x * Z_i + y * Z_unknown = 1 - sum(Z_known * n_known / n_e)
  x * Z_i**2 + y * Z_unknown**2 = Z_eff - sum(Z_known**2 * n_known / n_e)

  Where x = n_i / n_e = dilution, y = n_unknown / n_e , and we define "known"
  and "unknown" to refer to impurity species with known and unknown densities.

  Args:
    impurity_params: Impurity parameters.
    T_e: Electron temperature profile.
    Z_i: Average charge state of main ion on cell grid.
    Z_i_face: Average charge state of main ion on face grid.
    Z_eff_from_config: Z_eff profile from config.
    Z_eff_face_from_config: Z_eff profile on face grid from config.

  Returns:
    _IonProperties container with calculated ion properties.
  """
  # --- Vectorized charge state calculation ---
  # This is JIT-compatible because impurity_symbols is a static tuple, so the
  # list comprehension is unrolled during compilation.
  impurity_symbols = tuple(impurity_params.n_e_ratios.keys())
  Z_per_species = jnp.stack([
      charge_states.calculate_average_charge_state_single_species(
          T_e.value, symbol
      )
      for symbol in impurity_symbols
  ])
  Z_per_species_face = jnp.stack([
      charge_states.calculate_average_charge_state_single_species(
          T_e.face_value(), symbol
      )
      for symbol in impurity_symbols
  ])

  unknown_species_index = impurity_symbols.index(
      impurity_params.unknown_species
  )
  Z_unknown = Z_per_species[unknown_species_index]
  Z_unknown_face = Z_per_species_face[unknown_species_index]

  # Create arrays of known ratios, with 0 for the unknown species.
  n_e_ratios_known = jnp.array([
      impurity_params.n_e_ratios[symbol]
      if symbol != impurity_params.unknown_species
      else jnp.zeros_like(T_e.value)
      for symbol in impurity_symbols
  ])
  n_e_ratios_known_face = jnp.array([
      impurity_params.n_e_ratios_face[symbol]
      if symbol != impurity_params.unknown_species
      else jnp.zeros_like(T_e.face_value())
      for symbol in impurity_symbols
  ])

  sum_Z_n_ratio = jnp.sum(n_e_ratios_known * Z_per_species, axis=0)
  sum_Z2_n_ratio = jnp.sum(n_e_ratios_known * Z_per_species**2, axis=0)
  sum_Z_n_ratio_face = jnp.sum(
      n_e_ratios_known_face * Z_per_species_face, axis=0
  )
  sum_Z2_n_ratio_face = jnp.sum(
      n_e_ratios_known_face * Z_per_species_face**2, axis=0
  )

  # Solve the 2x2 system for dilution and the unknown n_e_ratio on both grids

  # x * Z_i + y * Z_unknown = - sum(Z_known * n_known / n_e)
  # x * Z_i**2 + y * Z_unknown**2 = Z_eff - sum(Z_known**2 * n_known / n_e)

  def _solve_system(a1, a2, b1, b2, c1, c2):
    """Solves a 2x2 system of the form a1*x + b1*y = c1, a2*x + b2*y = c2."""
    det_A = a1 * b2 - a2 * b1
    # Add a small epsilon to avoid division by zero if det_A is zero
    det_A = jnp.where(
        jnp.abs(det_A) < constants.CONSTANTS.eps, constants.CONSTANTS.eps, det_A
    )
    # Use Cramer's rule to solve the system
    x = (b2 * c1 - b1 * c2) / det_A
    y = (a1 * c2 - a2 * c1) / det_A
    return x, y

  dilution_factor, r_unknown = _solve_system(
      a1=Z_i,
      b1=Z_unknown,
      a2=Z_i**2,
      b2=Z_unknown**2,
      c1=1.0 - sum_Z_n_ratio,
      c2=Z_eff_from_config - sum_Z2_n_ratio,
  )
  dilution_factor_face, r_unknown_face = _solve_system(
      a1=Z_i_face,
      b1=Z_unknown_face,
      a2=Z_i_face**2,
      b2=Z_unknown_face**2,
      c1=1.0 - sum_Z_n_ratio_face,
      c2=Z_eff_face_from_config - sum_Z2_n_ratio_face,
  )

  # Now update the row for the unknown species with its calculated profile
  n_e_ratios_all_species = n_e_ratios_known.at[unknown_species_index, :].set(
      r_unknown
  )
  n_e_ratios_all_species_face = n_e_ratios_known_face.at[
      unknown_species_index, :
  ].set(r_unknown_face)

  n_e_ratios_mapping = {
      symbol: n_e_ratios_all_species[i]
      for i, symbol in enumerate(impurity_symbols)
  }
  n_e_ratios_mapping_face = {
      symbol: n_e_ratios_all_species_face[i]
      for i, symbol in enumerate(impurity_symbols)
  }

  fractions = electron_density_ratios.calculate_fractions_from_ratios(
      n_e_ratios_mapping
  )
  fractions_face = electron_density_ratios.calculate_fractions_from_ratios(
      n_e_ratios_mapping_face
  )

  # Build the final ion mixture and calculate properties

  if not impurity_params.A_override:
    A_avg = jnp.sum(
        jnp.array([
            constants.ION_PROPERTIES_DICT[ion].A * fraction
            for ion, fraction in fractions.items()
        ]),
        axis=0,
    )
    A_avg_face = jnp.sum(
        jnp.array([
            constants.ION_PROPERTIES_DICT[ion].A * fraction
            for ion, fraction in fractions_face.items()
        ]),
        axis=0,
    )
  else:
    A_avg = jnp.ones_like(T_e.value) * impurity_params.A_override
    A_avg_face = (
        jnp.ones_like(T_e.face_value()) * impurity_params.A_override_face
    )

  charge_state_info = charge_states.get_average_charge_state(
      T_e=T_e.value,
      fractions=fractions,
      Z_override=impurity_params.Z_override,
  )
  Z_impurity = charge_state_info.Z_mixture

  charge_state_info_face = charge_states.get_average_charge_state(
      T_e=T_e.face_value(),
      fractions=fractions_face,
      Z_override=impurity_params.Z_override,
  )
  Z_impurity_face = charge_state_info_face.Z_mixture

  return _IonProperties(
      A_impurity=A_avg,
      A_impurity_face=A_avg_face,
      Z_impurity=Z_impurity,
      Z_impurity_face=Z_impurity_face,
      Z_eff=Z_eff_from_config,
      dilution_factor=dilution_factor,
      dilution_factor_inner_edge=dilution_factor_face[0],
      dilution_factor_outer_edge=dilution_factor_face[-1],
      impurity_fractions=fractions,
      charge_state_info=charge_state_info,
      charge_state_info_face=charge_state_info_face,
  )


# jitted since also used outside the solver
@jax.jit
def get_updated_ions(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
) -> Ions:
  """Updated ion density, charge state, and mass based on state and config.

  Main ion and impurities are each treated as a single effective ion, but could
  be comparised of multiple species within an IonMixture. The main ion and
  impurity densities are calculated depending on the Z_eff constraint,
  quasineutrality, and the average impurity charge state which may be
  temperature dependent.

  Z_eff = (Z_i**2 * n_i + Z_impurity**2 * n_impurity)/n_e  ;
  n_impurity*Z_impurity + n_i*Z_i = n_e

  Args:
    runtime_params: Runtime parameters.
    geo: Geometry of the tokamak.
    n_e: Electron density profile [m^-3].
    T_e: Electron temperature profile [keV].

  Returns:
    Ion container with the following attributes:
      n_i: Ion density profile [m^-3].
      n_impurity: Impurity density profile [m^-3].
      Z_i: Average charge state of main ion on cell grid [dimensionless].
        Typically just the average of the atomic numbers since these are
        normally low Z ions and can be assumed to be fully ionized.
      Z_i_face: Average charge state of main ion on face grid [dimensionless].
      Z_impurity: Average charge state of impurities on cell grid
      [dimensionless].
      Z_impurity_face: Average charge state of impurities on face grid
      [dimensionless].
      A_i: Average atomic number of main ion [amu].
      A_impurity: Average atomic number of impurities on cell grid [amu].
      A_impurity_face: Average atomic number of impurities on face grid [amu].
  """

  Z_i = charge_states.get_average_charge_state(
      T_e=T_e.value,
      fractions=runtime_params.plasma_composition.main_ion.fractions,
      Z_override=runtime_params.plasma_composition.main_ion.Z_override,
  ).Z_mixture
  Z_i_face = charge_states.get_average_charge_state(
      T_e=T_e.face_value(),
      fractions=runtime_params.plasma_composition.main_ion.fractions,
      Z_override=runtime_params.plasma_composition.main_ion.Z_override,
  ).Z_mixture

  impurity_params = runtime_params.plasma_composition.impurity

  match impurity_params:
    case impurity_fractions.RuntimeParams():
      ion_properties = _get_ion_properties_from_fractions(
          impurity_params,
          T_e,
          Z_i,
          Z_i_face,
          runtime_params.plasma_composition.Z_eff,
          runtime_params.plasma_composition.Z_eff_face,
      )

    case electron_density_ratios.RuntimeParams():
      ion_properties = _get_ion_properties_from_n_e_ratios(
          impurity_params,
          T_e,
          Z_i,
          Z_i_face,
      )
    case electron_density_ratios_zeff.RuntimeParams():
      ion_properties = _get_ion_properties_from_n_e_ratios_Z_eff(
          impurity_params,
          T_e,
          Z_i,
          Z_i_face,
          runtime_params.plasma_composition.Z_eff,
          runtime_params.plasma_composition.Z_eff_face,
      )
    case _:
      # Not expected to be reached but needed to avoid linter errors.
      raise ValueError('Unknown impurity mode.')

  # Use right_face_value (not right_face_constraint) so this works for both
  # Dirichlet and Neumann outer BCs. For Dirichlet: equals right_face_constraint.
  # For Neumann (use_neumann_outer): extrapolates from the last cell value.
  n_e_right_face_val = n_e.right_face_value[..., 0]

  n_i = cell_variable.CellVariable(
      value=n_e.value * ion_properties.dilution_factor,
      face_centers=geo.rho_face_norm,
      left_face_grad_constraint=None,
      left_face_constraint=n_e.left_face_constraint * ion_properties.dilution_factor_inner_edge,
      right_face_grad_constraint=None,
      right_face_constraint=n_e_right_face_val
      * ion_properties.dilution_factor_outer_edge,
  )

  n_impurity_value = jnp.where(
      ion_properties.dilution_factor == 1.0,
      0.0,
      (n_e.value - n_i.value * Z_i) / ion_properties.Z_impurity,
  )

  n_impurity_left_face_constraint = jnp.where(
      ion_properties.dilution_factor_inner_edge == 1.0,
      0.0,
      (n_e.left_face_constraint - n_i.left_face_constraint * Z_i_face[0])
      / ion_properties.Z_impurity_face[0],
  )

  n_impurity_right_face_constraint = jnp.where(
      ion_properties.dilution_factor_outer_edge == 1.0,
      0.0,
      (n_e_right_face_val - n_i.right_face_constraint * Z_i_face[-1])
      / ion_properties.Z_impurity_face[-1],
  )

  n_impurity = cell_variable.CellVariable(
      value=n_impurity_value,
      face_centers=geo.rho_face_norm,
      left_face_grad_constraint=None,
      left_face_constraint=n_impurity_left_face_constraint,
      right_face_grad_constraint=None,
      right_face_constraint=n_impurity_right_face_constraint,
  )

  # Z_eff from plasma composition is imposed and can be passed to CoreProfiles.
  # However, we must recalculate Z_eff_face from the updated densities and
  # charge states since linearly interpolated Z_eff (which is what plasma
  # composition Z_eff_face is) would not be physically consistent.
  Z_eff_face = _calculate_Z_eff(
      Z_i_face,
      ion_properties.Z_impurity_face,
      n_i.face_value(),
      n_impurity.face_value(),
      n_e.face_value(),
  )

  # Convert array of fractions to a mapping from symbol to fraction profile.
  # Ensure that output is always a full radial profile for consistency across
  # all impurity modes.
  impurity_fractions_dict = {}
  for symbol in runtime_params.plasma_composition.impurity_names:
    fraction = ion_properties.impurity_fractions[symbol]
    if fraction.ndim == 0:
      impurity_fractions_dict[symbol] = jnp.full_like(n_e.value, fraction)
    else:
      impurity_fractions_dict[symbol] = fraction

  # Populate main ion fractions (which are time varying scalars) from
  # plasma composition.
  main_ion_fractions_dict = {}
  for symbol in runtime_params.plasma_composition.main_ion_names:
    fraction = runtime_params.plasma_composition.main_ion.fractions[symbol]
    main_ion_fractions_dict[symbol] = fraction

  return Ions(
      n_i=n_i,
      n_impurity=n_impurity,
      impurity_fractions=impurity_fractions_dict,
      main_ion_fractions=main_ion_fractions_dict,
      Z_i=Z_i,
      Z_i_face=Z_i_face,
      Z_impurity=ion_properties.Z_impurity,
      Z_impurity_face=ion_properties.Z_impurity_face,
      A_i=runtime_params.plasma_composition.main_ion.A_avg,
      A_impurity=ion_properties.A_impurity,
      A_impurity_face=ion_properties.A_impurity_face,
      Z_eff=ion_properties.Z_eff,
      Z_eff_face=Z_eff_face,
      charge_state_info=ion_properties.charge_state_info,
      charge_state_info_face=ion_properties.charge_state_info_face,
  )


def _calculate_Z_eff(
    Z_i: array_typing.FloatVector,
    Z_impurity: array_typing.FloatVector,
    n_i: array_typing.FloatVector,
    n_impurity: array_typing.FloatVector,
    n_e: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Calculates Z_eff based on single effective impurity and main_ion."""
  return (Z_i**2 * n_i + Z_impurity**2 * n_impurity) / n_e
