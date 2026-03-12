"""
LENR Physics Engine — 3 modes of calculation.
Maxwell (standard), Coulomb Original (1785), Cherepanov (photon mass).
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal
from lenr_constants import (
    NUCLEAR, SCREENING_EXPERIMENTAL, LATTICE, DIFFUSION, LOADING,
    EQPET_SCREENING, TSC_PARAMS, BARRIER_FACTORS,
    gamow_penetration, cross_section_DD, screened_cross_section,
    enhancement_factor, diffusion_coefficient,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PhysicsMode = Literal['maxwell', 'coulomb_original', 'cherepanov']

kB_eV = 8.617e-5  # eV/K


@dataclass
class BarrierResult:
    """Result of barrier calculation for one physics mode."""
    mode: PhysicsMode
    barrier_keV: float
    effective_barrier_keV: float
    penetration_probability: float
    reaction_rate_relative: float
    screening_eV: float
    notes: str


class PhysicsEngine:
    """Core physics engine with 3 calculation modes."""

    _cherepanov_engine = None  # class-level cache

    def __init__(self, mode: PhysicsMode = 'maxwell'):
        self.mode = mode

    def set_mode(self, mode: PhysicsMode):
        self.mode = mode

    def get_screening(self, material: str) -> float:
        """Get experimental screening energy for material."""
        entry = SCREENING_EXPERIMENTAL.get(material)
        if entry:
            return entry['Us_eV']
        # Fallback: estimate from electron density
        lat = LATTICE.get(material)
        if lat:
            return lat.get('e_density_A3', 0.1) * 1000
        return 50.0

    def calculate_barrier(
        self,
        material: str,
        E_cm_keV: float,
        T_K: float = 300.0,
        D_loading: float = 0.5,
        defect_concentration: float = 0.0,
        B_field_T: float = 0.0,
    ) -> BarrierResult:
        """Calculate effective barrier and reaction parameters.

        Parameters
        ----------
        defect_concentration : float
            Fraction of lattice sites with defects (0-1). Used by Cherepanov mode.
        B_field_T : float
            External magnetic field in Tesla. Used by Cherepanov mode.
        """
        Us = self.get_screening(material)

        if self.mode == 'maxwell':
            return self._barrier_maxwell(material, E_cm_keV, T_K, D_loading, Us)
        elif self.mode == 'coulomb_original':
            return self._barrier_coulomb_original(material, E_cm_keV, T_K, D_loading, Us)
        else:
            return self._barrier_cherepanov(
                material, E_cm_keV, T_K, D_loading, Us,
                defect_concentration=defect_concentration,
                B_field_T=B_field_T,
            )

    def _barrier_maxwell(self, material, E_cm_keV, T_K, D_loading, Us):
        """Standard electromagnetic: Coulomb barrier with electron screening."""
        barrier = NUCLEAR['coulomb_barrier_vacuum_keV']
        Us_keV = Us / 1000.0
        eff_barrier = barrier - Us_keV

        P = gamow_penetration(E_cm_keV + Us_keV)
        loading_boost = (D_loading / 0.84) ** 4 if D_loading > 0.84 else 0.01
        rate = P * loading_boost

        return BarrierResult(
            mode='maxwell',
            barrier_keV=barrier,
            effective_barrier_keV=eff_barrier,
            penetration_probability=P,
            reaction_rate_relative=rate,
            screening_eV=Us,
            notes=f'Standard Coulomb {barrier} keV, screening {Us} eV',
        )

    def _barrier_coulomb_original(self, material, E_cm_keV, T_K, D_loading, Us):
        """Coulomb original: charge = mass of electricity, density interaction."""
        lat = LATTICE.get(material.replace('_Raiola', '').replace('_Huke', ''), {})
        e_density = lat.get('e_density_A3', 0.1)
        mass_density_factor = e_density * 10

        barrier = NUCLEAR['coulomb_barrier_vacuum_keV'] * (1 - mass_density_factor * 0.1)
        Us_keV = Us * (1 + mass_density_factor * 0.5) / 1000.0
        eff_barrier = max(barrier - Us_keV * mass_density_factor, 1.0)

        P = gamow_penetration(E_cm_keV + Us_keV * mass_density_factor)
        loading_boost = (D_loading / 0.7) ** 6 if D_loading > 0.7 else 0.01
        rate = P * loading_boost

        return BarrierResult(
            mode='coulomb_original',
            barrier_keV=barrier,
            effective_barrier_keV=eff_barrier,
            penetration_probability=P,
            reaction_rate_relative=rate,
            screening_eV=Us * (1 + mass_density_factor * 0.5),
            notes=f'Mass density rho_e={e_density:.3f}, barrier={barrier:.0f} keV',
        )

    def _barrier_cherepanov(self, material, E_cm_keV, T_K, D_loading, Us,
                            defect_concentration=0.0, B_field_T=0.0):
        """Cherepanov: no charge, photon mass, magnetic flux interactions.

        Delegates to CherepanovEngine for real physics calculation.
        Uses class-level cached engine instance to avoid re-creation overhead.
        """
        if PhysicsEngine._cherepanov_engine is None:
            from cherepanov_engine import CherepanovEngine
            PhysicsEngine._cherepanov_engine = CherepanovEngine()
        return PhysicsEngine._cherepanov_engine.calculate_barrier(
            material, E_cm_keV, T_K, D_loading,
            defect_concentration=defect_concentration,
            B_field_T=B_field_T,
        )

    def cross_section_bare(self, E_cm_keV: float) -> float:
        return cross_section_DD(E_cm_keV)

    def cross_section_screened(self, E_cm_keV: float, material: str) -> float:
        Us = self.get_screening(material)
        return screened_cross_section(E_cm_keV, Us)

    def enhancement(self, E_cm_keV: float, material: str) -> float:
        Us = self.get_screening(material)
        return enhancement_factor(E_cm_keV, Us)

    def diffusion(self, material: str, T_K: float) -> float:
        base = material.split('_')[0]
        return diffusion_coefficient(base, T_K)

    def mckubre_excess_power(
        self,
        current_A_cm2: float,
        loading: float,
        dxdt: float = 0.001,
    ) -> float:
        """McKubre excess power formula: P_ex = M*(i-i0)*(x-x0)*|dx/dt|"""
        M = LOADING['McKubre_M']
        i0 = LOADING['McKubre_i0']
        x0 = LOADING['McKubre_x0']
        if current_A_cm2 <= i0 or loading <= x0:
            return 0.0
        return M * (current_A_cm2 - i0) * (loading - x0) * abs(dxdt)


def compare_modes(
    material: str = 'Pd',
    E_cm_keV: float = 2.5,
    T_K: float = 340.0,
    D_loading: float = 0.9,
) -> dict[str, BarrierResult]:
    """Run all 3 physics modes and return comparison."""
    results = {}
    for mode in ('maxwell', 'coulomb_original', 'cherepanov'):
        engine = PhysicsEngine(mode)
        results[mode] = engine.calculate_barrier(material, E_cm_keV, T_K, D_loading)
    return results


if __name__ == '__main__':
    print("=== Mode Comparison: Pd, E=2.5 keV, T=340K, D/Pd=0.9 ===\n")
    results = compare_modes()
    for mode, r in results.items():
        print(f"[{mode}]")
        print(f"  Barrier:       {r.barrier_keV:.1f} keV")
        print(f"  Effective:     {r.effective_barrier_keV:.1f} keV")
        print(f"  P(penetrate):  {r.penetration_probability:.2e}")
        print(f"  Rate (rel):    {r.reaction_rate_relative:.2e}")
        print(f"  Notes:         {r.notes}")
        print()
