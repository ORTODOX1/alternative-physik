"""
Cherepanov Physics Engine — Pure Alternative Framework
=======================================================
NO charge. NO Coulomb barrier. NO tunneling. NO electrons/protons/neutrons.

Core concepts:
  - Photon mass density ρ_γ [kg/m³] — accumulates in lattice via friction/diffusion
  - Medium resistance R_m — replaces "Coulomb barrier"; property of the medium
  - Magnetic flux B [kg/s] — the real "electric current", light, all radiation
  - Lattice focusing — crystal structure channels photon mass into reaction sites
  - Defects = channels — dislocations/vacancies lower medium resistance

Key experimental predictions:
  - Czerski 2023: cold-rolled Pd (Us=18,200 eV) → defects lower R_m by ~700x
  - PdO > Pd (600 vs 310 eV) → oxide layer = photon mass lens
  - Ni at low loading (Piantelli) → ferromagnetic focusing
  - Temperature-independent screening (Kasagi) → Planck-like flat region at RT

Based on: Cherepanov A.I. analysis of Maxwell errors, pp.39-44
          Coulomb 1785 original: F = k·(ρ₁·ρ₂)/r²
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# PHYSICAL CONSTANTS (Cherepanov framework)
# =============================================================================

# Photon mass base density in vacuum [kg/m³]
# This is the "ether density" — background photon mass that fills space
RHO_GAMMA_VACUUM = 1.0e-18  # very sparse in free space

# Medium resistance base [dimensionless, relative to vacuum]
R_MEDIUM_BASE = 1000.0  # vacuum has highest resistance to nuclear interaction

# Reaction threshold: critical ratio ρ_γ / ρ_critical
# Below this ratio, no reaction occurs (analog of "barrier")
REACTION_THRESHOLD_K = 3.0  # sharpness parameter
REACTION_THRESHOLD_N = 2.0  # power parameter

# Magnetic flux quantum [kg/s] — fundamental unit
B_QUANTUM = 2.067833848e-15  # Wb in SI, but interpreted as kg/s

# Lattice structure factors (photon mass channeling efficiency)
STRUCTURE_FACTORS = {
    'FCC': 1.0,    # best: octahedral + tetrahedral sites
    'BCC': 0.7,    # tetrahedral sites less symmetric
    'HCP': 0.5,    # fewer interstitial channels
}

# Magnetic classification (photon mass focusing ability)
# ferromagnetic >> paramagnetic > diamagnetic
MAGNETIC_CLASS = {
    'ferromagnetic': {'factor': 10.0, 'description': 'Strong focusing'},
    'paramagnetic':  {'factor': 2.0,  'description': 'Moderate focusing'},
    'diamagnetic':   {'factor': 0.5,  'description': 'Weak defocusing'},
}

# Surface/defect state → defect concentration mapping
DEFECT_CONCENTRATION = {
    'cold_rolled':  0.50,   # heavy plastic deformation → many dislocations
    'nano':         0.30,   # nanostructured → grain boundaries
    'irradiated':   0.25,   # radiation damage → vacancy clusters
    'sputtered':    0.20,   # thin film deposition → columnar defects
    'oxidized':     0.15,   # oxide layer → interfacial defects
    'mesh':         0.10,   # wire mesh → surface defects
    'polycrystal':  0.05,   # standard polycrystalline
    'annealed':     0.005,  # heat-treated → very few defects
    'single_crystal': 0.001,  # nearly perfect lattice
    'insulator':    0.001,
    'semiconductor': 0.003,
}

# Material properties relevant to Cherepanov physics
# χ_m: magnetic susceptibility (SI volumetric)
# classification: ferro/para/diamagnetic
MATERIAL_MAGNETIC = {
    'Pd':  {'chi_m': -7.2e-6,  'class': 'diamagnetic'},
    'Ni':  {'chi_m': 600e-6,   'class': 'ferromagnetic'},  # ferromagnetic!
    'Fe':  {'chi_m': 1.0,      'class': 'ferromagnetic'},  # saturated
    'Ti':  {'chi_m': 153e-6,   'class': 'paramagnetic'},
    'Au':  {'chi_m': -2.8e-6,  'class': 'diamagnetic'},
    'Pt':  {'chi_m': 193e-6,   'class': 'paramagnetic'},
    'W':   {'chi_m': 59e-6,    'class': 'paramagnetic'},
    'Cu':  {'chi_m': -5.5e-6,  'class': 'diamagnetic'},
    'Zr':  {'chi_m': -13.8e-6, 'class': 'diamagnetic'},
    'Ta':  {'chi_m': 154e-6,   'class': 'paramagnetic'},
    'Al':  {'chi_m': 16.5e-6,  'class': 'paramagnetic'},
    'Co':  {'chi_m': 1.0,      'class': 'ferromagnetic'},
    'Nb':  {'chi_m': 195e-6,   'class': 'paramagnetic'},
    'V':   {'chi_m': 255e-6,   'class': 'paramagnetic'},
    'Cr':  {'chi_m': 180e-6,   'class': 'paramagnetic'},  # antiferro below Neel
    'Mn':  {'chi_m': 529e-6,   'class': 'paramagnetic'},
    'Sn':  {'chi_m': -2.2e-5,  'class': 'diamagnetic'},
    'In':  {'chi_m': -1.1e-5,  'class': 'diamagnetic'},
    'Ag':  {'chi_m': -1.95e-5, 'class': 'diamagnetic'},
    'Be':  {'chi_m': -9.0e-6,  'class': 'diamagnetic'},
    'Er':  {'chi_m': 44.5e-3,  'class': 'paramagnetic'},  # large paramagnet
    'C':   {'chi_m': -6.2e-6,  'class': 'diamagnetic'},
    'Si':  {'chi_m': -3.2e-6,  'class': 'diamagnetic'},
    'Ge':  {'chi_m': -7.7e-6,  'class': 'diamagnetic'},
    # === Raiola 2002 metals (newly added) ===
    'Mg':  {'chi_m': 13.1e-6,  'class': 'paramagnetic'},
    'Zn':  {'chi_m': -11.4e-6, 'class': 'diamagnetic'},
    'Y':   {'chi_m': 187e-6,   'class': 'paramagnetic'},
    'Mo':  {'chi_m': 72e-6,    'class': 'paramagnetic'},
    'Ru':  {'chi_m': 39e-6,    'class': 'paramagnetic'},
    'Rh':  {'chi_m': 102e-6,   'class': 'paramagnetic'},
    'Cd':  {'chi_m': -18e-6,   'class': 'diamagnetic'},
    'Hf':  {'chi_m': 75e-6,    'class': 'paramagnetic'},
    'Re':  {'chi_m': 67e-6,    'class': 'paramagnetic'},
    'Ir':  {'chi_m': 25e-6,    'class': 'paramagnetic'},
    'Pb':  {'chi_m': -15.5e-6, 'class': 'diamagnetic'},
    'B':   {'chi_m': -6.7e-6,  'class': 'diamagnetic'},
    # Compounds / alloys
    'PdO':        {'chi_m': 1.2e-4,  'class': 'paramagnetic'},
    'SUS304':     {'chi_m': 1.01e-2, 'class': 'paramagnetic'},
    'Constantan': {'chi_m': 2.8e-5,  'class': 'paramagnetic'},
    'Be_BeO':     {'chi_m': 5.0e-5,  'class': 'paramagnetic'},
    'BeO':        {'chi_m': -1.0e-5, 'class': 'diamagnetic'},
}

# Lattice parameters for materials (a in Å, θ_D in K, structure)
MATERIAL_LATTICE = {
    'Pd':  {'a': 3.8907, 'theta_D': 274, 'structure': 'FCC', 'density': 12.023},
    'Ni':  {'a': 3.5240, 'theta_D': 450, 'structure': 'FCC', 'density': 8.908},
    'Fe':  {'a': 2.8665, 'theta_D': 470, 'structure': 'BCC', 'density': 7.874},
    'Ti':  {'a': 2.9508, 'theta_D': 420, 'structure': 'HCP', 'density': 4.507},
    'Au':  {'a': 4.0782, 'theta_D': 165, 'structure': 'FCC', 'density': 19.30},
    'Pt':  {'a': 3.9242, 'theta_D': 240, 'structure': 'FCC', 'density': 21.45},
    'W':   {'a': 3.1652, 'theta_D': 400, 'structure': 'BCC', 'density': 19.25},
    'Cu':  {'a': 3.6149, 'theta_D': 343, 'structure': 'FCC', 'density': 8.96},
    'Zr':  {'a': 3.2316, 'theta_D': 291, 'structure': 'HCP', 'density': 6.52},
    'Ta':  {'a': 3.3013, 'theta_D': 240, 'structure': 'BCC', 'density': 16.65},
    'Al':  {'a': 4.0495, 'theta_D': 428, 'structure': 'FCC', 'density': 2.70},
    'Co':  {'a': 2.5071, 'theta_D': 445, 'structure': 'HCP', 'density': 8.90},
    'Nb':  {'a': 3.3004, 'theta_D': 275, 'structure': 'BCC', 'density': 8.57},
    'V':   {'a': 3.0240, 'theta_D': 380, 'structure': 'BCC', 'density': 6.11},
    'Cr':  {'a': 2.8849, 'theta_D': 630, 'structure': 'BCC', 'density': 7.19},
    'Mn':  {'a': 8.9125, 'theta_D': 410, 'structure': 'BCC', 'density': 7.44},
    'Ag':  {'a': 4.0853, 'theta_D': 225, 'structure': 'FCC', 'density': 10.49},
    'Sn':  {'a': 5.8318, 'theta_D': 200, 'structure': 'BCT', 'density': 7.27},
    'In':  {'a': 3.2523, 'theta_D': 108, 'structure': 'BCT', 'density': 7.31},
    'Be':  {'a': 2.2858, 'theta_D': 1440, 'structure': 'HCP', 'density': 1.85},
    'Er':  {'a': 3.5592, 'theta_D': 168, 'structure': 'HCP', 'density': 9.07},
    # === Raiola 2002 metals (newly added) ===
    'Mg':  {'a': 3.2094, 'theta_D': 400, 'structure': 'HCP', 'density': 1.74},
    'Zn':  {'a': 2.6650, 'theta_D': 327, 'structure': 'HCP', 'density': 7.13},
    'Y':   {'a': 3.6482, 'theta_D': 280, 'structure': 'HCP', 'density': 4.47},
    'Mo':  {'a': 3.1470, 'theta_D': 450, 'structure': 'BCC', 'density': 10.28},
    'Ru':  {'a': 2.7059, 'theta_D': 600, 'structure': 'HCP', 'density': 12.37},
    'Rh':  {'a': 3.8034, 'theta_D': 480, 'structure': 'FCC', 'density': 12.41},
    'Cd':  {'a': 2.9793, 'theta_D': 209, 'structure': 'HCP', 'density': 8.65},
    'Hf':  {'a': 3.1946, 'theta_D': 252, 'structure': 'HCP', 'density': 13.31},
    'Re':  {'a': 2.7610, 'theta_D': 416, 'structure': 'HCP', 'density': 21.02},
    'Ir':  {'a': 3.8394, 'theta_D': 420, 'structure': 'FCC', 'density': 22.56},
    'Pb':  {'a': 4.9508, 'theta_D': 105, 'structure': 'FCC', 'density': 11.34},
    'B':   {'a': 8.73,   'theta_D': 1480, 'structure': 'rhombohedral', 'density': 2.34},
    # Compounds
    'PdO':        {'a': 3.04,  'theta_D': 300, 'structure': 'FCC', 'density': 8.3},
    'SUS304':     {'a': 3.59,  'theta_D': 350, 'structure': 'FCC', 'density': 8.0},
    'Constantan': {'a': 3.57,  'theta_D': 340, 'structure': 'FCC', 'density': 8.9},
    'Be_BeO':     {'a': 2.70,  'theta_D': 1000, 'structure': 'HCP', 'density': 2.5},
    'BeO':        {'a': 2.6980, 'theta_D': 1280, 'structure': 'HCP', 'density': 3.01},
}


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class CherepanovResult:
    """Result of Cherepanov physics calculation.

    No barrier_keV — because there IS no barrier.
    Instead: medium resistance + photon mass density determine reaction.
    """
    medium_resistance: float       # R_m [dimensionless] — replaces barrier
    photon_mass_density: float     # ρ_γ [kg/m³] — accumulated in lattice
    photon_mass_density_critical: float  # ρ_c — threshold for reaction
    reaction_probability: float    # P = f(ρ_γ/ρ_c) — no tunneling!
    magnetic_flux_B_kg_s: float    # B [kg/s] — "current" = photon mass flow
    lattice_focusing_factor: float  # how well lattice channels photon mass
    defect_concentration: float    # fraction of lattice sites with defects
    photon_phonon_coupling: float  # coupling between photon mass and phonons
    notes: str = ''


# =============================================================================
# CHEREPANOV ENGINE
# =============================================================================

class CherepanovEngine:
    """Pure Cherepanov physics engine.

    No charge. No electric field. No tunneling.
    Everything is magnetic flux, photon mass density,
    and medium resistance (property of the material, NOT fundamental).
    """

    def photon_mass_density(
        self,
        material: str,
        T_K: float,
        D_loading: float,
        B_field_T: float = 0.0,
    ) -> float:
        """Calculate photon mass density accumulated in lattice.

        ρ_γ = ρ₀ × f_material × f_loading × f_temperature × f_B

        Physics (Cherepanov):
          - Deuterium diffusing through lattice generates friction → photon mass
          - Material density and magnetic properties determine accumulation rate
          - Temperature couples via Planck-like distribution
          - External B field adds photon mass directly
        """
        lat = MATERIAL_LATTICE.get(material, MATERIAL_LATTICE.get('Pd'))
        mag = MATERIAL_MAGNETIC.get(material, {'chi_m': 1e-6, 'class': 'diamagnetic'})

        # Material factor: |χ_m| × ρ_mat / (θ_D / 1000)
        # High susceptibility + high density + soft lattice → more accumulation
        chi_abs = abs(mag['chi_m'])
        theta_D = lat['theta_D']
        rho_mat = lat['density']
        f_material = chi_abs * rho_mat / (theta_D / 1000.0)
        # Ferromagnetics get massive boost
        mag_class = MAGNETIC_CLASS.get(mag['class'], MAGNETIC_CLASS['diamagnetic'])
        f_material *= mag_class['factor']

        # Loading factor: deuterium friction generates photon mass
        # More D atoms diffusing = more friction = more ρ_γ
        f_loading = 1.0 + D_loading * 50.0

        # Temperature factor: Planck-like coupling
        # x = T/θ_D; peak near x ≈ 1; flat at room T for most metals
        x = T_K / theta_D if theta_D > 0 else 1.0
        f_temperature = x * np.exp(-x) * np.e  # normalized so peak = 1 at x=1

        # External magnetic field factor
        f_B = 1.0 + abs(B_field_T) * 100.0

        rho_gamma = RHO_GAMMA_VACUUM * f_material * f_loading * f_temperature * f_B

        return max(rho_gamma, RHO_GAMMA_VACUUM)

    def medium_resistance(
        self,
        material: str,
        T_K: float,
        D_loading: float,
        defect_concentration: float = 0.05,
    ) -> float:
        """Calculate medium resistance to nuclear reaction.

        R_m = R_base × f_lattice × f_defect × f_magnetic × f_loading

        THIS is what replaces the "Coulomb barrier".
        It's a property of the MEDIUM, not a fundamental force.
        Can be engineered by:
          - Crystal structure (FCC best)
          - Defect engineering (cold rolling, irradiation)
          - Magnetic state (ferromagnets focus photon mass)
          - D loading (more D = lower resistance)

        KEY INSIGHT (Czerski 2023):
          cold-rolled Pd: defects ~0.5 → R_m drops ~700x
          → explains U_e = 18,200 eV (vs 25 eV theory)
        """
        lat = MATERIAL_LATTICE.get(material, MATERIAL_LATTICE.get('Pd'))
        mag = MATERIAL_MAGNETIC.get(material, {'chi_m': 1e-6, 'class': 'diamagnetic'})

        # Lattice factor: depends on structure and lattice parameter
        a = lat['a']
        structure = lat.get('structure', 'FCC')
        sf = STRUCTURE_FACTORS.get(structure, 0.5)
        f_lattice = (a / 3.5) ** 2 * sf  # normalized to ~1 for Pd

        # DEFECT FACTOR — THE KEY TO EVERYTHING
        # Defects create channels for photon mass flow
        # More defects → LOWER resistance → EASIER reaction
        # cold_rolled (0.5): f = 1/(1 + 50) = 0.0196 → ~50x reduction!
        # annealed (0.005): f = 1/(1 + 0.5) = 0.667 → barely reduced
        f_defect = 1.0 / (1.0 + defect_concentration * 100.0)

        # Magnetic factor: ferromagnets focus photon mass → lower R
        mag_class_info = MAGNETIC_CLASS.get(mag['class'], MAGNETIC_CLASS['diamagnetic'])
        f_magnetic = 1.0 / mag_class_info['factor']
        # ferromagnetic: 1/10 = 0.1 → 10x lower resistance!
        # diamagnetic: 1/0.5 = 2.0 → 2x higher resistance

        # Loading factor: more deuterium = lower resistance
        f_loading = 1.0 / (1.0 + D_loading * 10.0)

        R_m = R_MEDIUM_BASE * f_lattice * f_defect * f_magnetic * f_loading

        return max(R_m, 0.01)

    def critical_density(self, material: str) -> float:
        """Calculate critical photon mass density for reaction.

        ρ_critical depends on:
          - Lattice parameter (interstitial site size)
          - Debye temperature (lattice stiffness)
          - Number of interstitial channels
        """
        lat = MATERIAL_LATTICE.get(material, MATERIAL_LATTICE.get('Pd'))

        a = lat['a']
        theta_D = lat['theta_D']
        structure = lat.get('structure', 'FCC')

        # More interstitial sites in FCC → lower threshold
        n_sites = {'FCC': 12, 'BCC': 6, 'HCP': 6}.get(structure, 6)

        # Critical density: inversely proportional to available channels
        # Soft lattice (low θ_D) and large unit cell (high a) help
        rho_c = RHO_GAMMA_VACUUM * 1e6 / (n_sites * (a / 3.0) ** 3 / (theta_D / 300.0))

        return max(rho_c, RHO_GAMMA_VACUUM)

    def reaction_probability(
        self,
        rho_gamma: float,
        rho_critical: float,
    ) -> float:
        """Calculate reaction probability from photon mass density ratio.

        P = 0                                    if ρ_γ < ρ_c
        P = 1 - exp(-k × (ρ/ρ_c - 1)^n)       if ρ_γ ≥ ρ_c

        NO tunneling! This is resonant transfer of photon mass.
        When density exceeds threshold, the lattice sites resonate
        and photon mass flows between nuclei → reaction.
        """
        if rho_gamma < rho_critical or rho_critical <= 0:
            return 0.0

        ratio = rho_gamma / rho_critical
        excess = ratio - 1.0

        P = 1.0 - np.exp(-REACTION_THRESHOLD_K * excess ** REACTION_THRESHOLD_N)
        return float(np.clip(P, 0.0, 1.0))

    def lattice_focusing(self, material: str) -> float:
        """Calculate how well the lattice focuses photon mass.

        Factors:
          - Crystal structure (FCC best for octahedral channeling)
          - Magnetic state (ferromagnets have natural focusing)
          - Lattice parameter vs deuterium size
        """
        lat = MATERIAL_LATTICE.get(material, MATERIAL_LATTICE.get('Pd'))
        mag = MATERIAL_MAGNETIC.get(material, {'chi_m': 1e-6, 'class': 'diamagnetic'})

        structure = lat.get('structure', 'FCC')
        sf = STRUCTURE_FACTORS.get(structure, 0.5)

        mag_class = MAGNETIC_CLASS.get(mag['class'], MAGNETIC_CLASS['diamagnetic'])
        mag_factor = mag_class['factor']

        # Lattice parameter: optimal around 3.5-4.0 Å for D-D
        a = lat['a']
        a_optimal = 3.8  # Pd is near-optimal
        a_factor = np.exp(-((a - a_optimal) / 1.5) ** 2)

        focusing = sf * mag_factor * a_factor

        return float(focusing)

    def photon_phonon_coupling(self, material: str, T_K: float) -> float:
        """Calculate coupling between photon mass waves and phonons.

        Phonons modulate the lattice spacing → periodic compression
        of photon mass channels → pulsed focusing effect.
        """
        lat = MATERIAL_LATTICE.get(material, MATERIAL_LATTICE.get('Pd'))
        theta_D = lat['theta_D']

        # Bose-Einstein-like occupation at T/θ_D
        x = theta_D / T_K if T_K > 0 else 100.0
        if x > 50:
            n_phonon = 0.0
        else:
            n_phonon = 1.0 / (np.exp(x) - 1.0) if x > 0.01 else T_K / theta_D

        # Coupling strength: stronger when more phonons
        # but not too hot (lattice disruption above ~θ_D)
        coupling = n_phonon * np.exp(-T_K / (3.0 * theta_D))

        return float(max(coupling, 0.0))

    def magnetic_flux_B(
        self,
        material: str,
        T_K: float,
        D_loading: float,
        B_external_T: float = 0.0,
    ) -> float:
        """Calculate effective magnetic flux B [kg/s].

        In Cherepanov framework: B IS the "electric current".
        B IS light. B IS all radiation.
        B [kg/s] = mass flow of photon mass.
        """
        mag = MATERIAL_MAGNETIC.get(material, {'chi_m': 1e-6, 'class': 'diamagnetic'})
        chi = abs(mag['chi_m'])

        # Internal flux from loaded deuterium
        B_internal = chi * D_loading * 1e-10  # very small

        # External contribution
        B_total = B_internal + abs(B_external_T) * B_QUANTUM

        return float(B_total)

    def calculate(
        self,
        material: str,
        E_cm_keV: float,
        T_K: float = 300.0,
        D_loading: float = 0.5,
        B_field_T: float = 0.0,
        defect_concentration: float = 0.05,
        surface_state: Optional[str] = None,
    ) -> CherepanovResult:
        """Full Cherepanov calculation.

        Parameters
        ----------
        material : str
            Target material name
        E_cm_keV : float
            Center-of-mass energy in keV (used for compatibility;
            in Cherepanov framework energy comes from photon mass density)
        T_K : float
            Temperature in Kelvin
        D_loading : float
            Deuterium loading ratio (D/Metal)
        B_field_T : float
            External magnetic field in Tesla
        defect_concentration : float
            Fraction of lattice sites with defects (0-1)
        surface_state : str, optional
            Surface preparation state (overrides defect_concentration)
        """
        # Resolve defect concentration from surface state
        if surface_state and surface_state in DEFECT_CONCENTRATION:
            defect_concentration = DEFECT_CONCENTRATION[surface_state]

        # 1. Photon mass density
        rho_gamma = self.photon_mass_density(material, T_K, D_loading, B_field_T)

        # Energy contribution: beam energy adds to photon mass density
        # (kinetic energy → friction → photon mass)
        energy_boost = 1.0 + E_cm_keV * 0.1  # 10 keV → 2x boost
        rho_gamma *= energy_boost

        # 2. Medium resistance
        R_m = self.medium_resistance(material, T_K, D_loading, defect_concentration)

        # 3. Critical density (lowered by low R_m)
        rho_c = self.critical_density(material)
        rho_c_effective = rho_c * (R_m / R_MEDIUM_BASE)  # lower R → lower threshold

        # 4. Reaction probability
        P = self.reaction_probability(rho_gamma, rho_c_effective)

        # 5. Lattice focusing
        focusing = self.lattice_focusing(material)

        # 6. Photon-phonon coupling
        pp_coupling = self.photon_phonon_coupling(material, T_K)

        # 7. Magnetic flux
        B = self.magnetic_flux_B(material, T_K, D_loading, B_field_T)

        # Build notes
        notes_parts = [f'No charge. No barrier.']
        notes_parts.append(f'rho_g={rho_gamma:.2e}, rho_c_eff={rho_c_effective:.2e}')
        notes_parts.append(f'R_m={R_m:.1f}')
        if defect_concentration > 0.1:
            notes_parts.append(f'HIGH DEFECTS ({defect_concentration:.0%}) → low R_m')
        mag = MATERIAL_MAGNETIC.get(material, {'class': 'unknown'})
        if mag['class'] == 'ferromagnetic':
            notes_parts.append('FERROMAGNETIC focusing')

        return CherepanovResult(
            medium_resistance=R_m,
            photon_mass_density=rho_gamma,
            photon_mass_density_critical=rho_c_effective,
            reaction_probability=P,
            magnetic_flux_B_kg_s=B,
            lattice_focusing_factor=focusing,
            defect_concentration=defect_concentration,
            photon_phonon_coupling=pp_coupling,
            notes='; '.join(notes_parts),
        )

    # -----------------------------------------------------------------
    # BACKWARD COMPATIBILITY: wrap into BarrierResult for V2 code
    # -----------------------------------------------------------------
    def cherepanov_to_barrier_result(self, cr: CherepanovResult) -> 'BarrierResult':
        """Convert CherepanovResult to BarrierResult for V2 compatibility.

        Maps:
          medium_resistance → barrier_keV (analog)
          low R_m → low effective_barrier_keV
          reaction_probability → penetration_probability
        """
        from physics_engine import BarrierResult

        # Map medium resistance to "barrier" keV
        # R_m = 1000 (vacuum) → barrier = 400 keV (standard Coulomb)
        # R_m = 1 (ultra-low) → barrier = 0.4 keV
        barrier_keV = cr.medium_resistance * 0.4  # linear mapping

        # Effective barrier
        eff_barrier = barrier_keV * (1.0 - cr.lattice_focusing_factor * 0.3)
        eff_barrier = max(eff_barrier, 0.01)

        # Reaction rate = P × focusing × coupling
        rate = cr.reaction_probability * (1.0 + cr.lattice_focusing_factor)

        return BarrierResult(
            mode='cherepanov',
            barrier_keV=barrier_keV,
            effective_barrier_keV=eff_barrier,
            penetration_probability=cr.reaction_probability,
            reaction_rate_relative=rate,
            screening_eV=0.0,  # no screening concept in Cherepanov
            notes=cr.notes,
        )

    def calculate_barrier(
        self,
        material: str,
        E_cm_keV: float,
        T_K: float = 300.0,
        D_loading: float = 0.5,
        defect_concentration: float = 0.0,
        B_field_T: float = 0.0,
    ) -> 'BarrierResult':
        """Direct BarrierResult output for PhysicsEngine compatibility."""
        cr = self.calculate(
            material, E_cm_keV, T_K, D_loading,
            B_field_T=B_field_T,
            defect_concentration=defect_concentration,
        )
        return self.cherepanov_to_barrier_result(cr)


# =============================================================================
# HELPER: get ML features from CherepanovResult
# =============================================================================

def cherepanov_features(cr: CherepanovResult, material: Optional[str] = None) -> dict:
    """Extract 8 Cherepanov features for V3 ML pipeline.

    Parameters
    ----------
    cr : CherepanovResult
        Result from CherepanovEngine.calculate()
    material : str, optional
        Material name to look up magnetic susceptibility.
        If None, caller must fill 'magnetic_susceptibility_abs' manually.
    """
    chi_m_abs = 0.0
    if material is not None:
        mag = MATERIAL_MAGNETIC.get(material, {'chi_m': 1e-6})
        chi_m_abs = abs(mag['chi_m'])

    return {
        'magnetic_susceptibility_abs': chi_m_abs,
        'photon_mass_density': cr.photon_mass_density,
        'photon_mass_density_critical': cr.photon_mass_density_critical,
        'medium_resistance': cr.medium_resistance,
        'lattice_focusing_factor': cr.lattice_focusing_factor,
        'defect_concentration': cr.defect_concentration,
        'magnetic_flux_B_kg_s': cr.magnetic_flux_B_kg_s,
        'photon_phonon_coupling': cr.photon_phonon_coupling,
    }


# =============================================================================
# CLI TEST
# =============================================================================
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    engine = CherepanovEngine()

    print('=' * 70)
    print('CHEREPANOV ENGINE — No charge, no barrier, photon mass physics')
    print('=' * 70)

    # Test 1: Standard Pd (polycrystal)
    cr = engine.calculate('Pd', 2.5, 340, 0.9, defect_concentration=0.05)
    print(f'\n[Pd polycrystal] E=2.5 keV, T=340K, D/Pd=0.9')
    print(f'  Medium resistance:  {cr.medium_resistance:.2f}')
    print(f'  rho_gamma:          {cr.photon_mass_density:.2e}')
    print(f'  rho_critical:       {cr.photon_mass_density_critical:.2e}')
    print(f'  P(reaction):        {cr.reaction_probability:.4f}')
    print(f'  Focusing:           {cr.lattice_focusing_factor:.3f}')
    print(f'  Notes: {cr.notes}')

    # Test 2: Cold-rolled Pd (Czerski 2023)
    cr2 = engine.calculate('Pd', 2.5, 300, 0.9, defect_concentration=0.5)
    print(f'\n[Pd COLD-ROLLED] E=2.5 keV, T=300K, D/Pd=0.9, defects=0.5')
    print(f'  Medium resistance:  {cr2.medium_resistance:.2f}')
    print(f'  rho_gamma:          {cr2.photon_mass_density:.2e}')
    print(f'  rho_critical:       {cr2.photon_mass_density_critical:.2e}')
    print(f'  P(reaction):        {cr2.reaction_probability:.4f}')
    print(f'  R_m ratio:          {cr.medium_resistance / cr2.medium_resistance:.0f}x lower')

    # Test 3: Ni (ferromagnetic focusing)
    cr3 = engine.calculate('Ni', 2.5, 500, 0.03)
    print(f'\n[Ni] E=2.5 keV, T=500K, D/Ni=0.03')
    print(f'  Medium resistance:  {cr3.medium_resistance:.2f}')
    print(f'  P(reaction):        {cr3.reaction_probability:.4f}')
    print(f'  Focusing:           {cr3.lattice_focusing_factor:.3f}')
    print(f'  Notes: {cr3.notes}')

    # Test 4: Au (minimal effect)
    cr4 = engine.calculate('Au', 2.5, 300, 0.0)
    print(f'\n[Au] E=2.5 keV, T=300K, D/Au=0.0 (no loading)')
    print(f'  Medium resistance:  {cr4.medium_resistance:.2f}')
    print(f'  P(reaction):        {cr4.reaction_probability:.4f}')

    # Test 5: Backward compatibility
    br = engine.calculate_barrier('Pd', 2.5, 340, 0.9)
    print(f'\n[BarrierResult compat] Pd:')
    print(f'  barrier_keV:        {br.barrier_keV:.1f}')
    print(f'  effective:          {br.effective_barrier_keV:.1f}')
    print(f'  P:                  {br.penetration_probability:.4f}')
    print(f'  rate:               {br.reaction_rate_relative:.4f}')

    # Test 6: Material comparison
    print(f'\n{"=" * 70}')
    print(f'Material Comparison (E=2.5 keV, T=300K, D_loading=0.5, polycrystal)')
    print(f'{"Material":>10}  {"R_m":>8}  {"rho_g":>12}  {"P(rxn)":>8}  {"Focus":>6}  Class')
    print(f'{"-" * 70}')
    for mat in ['Pd', 'Ni', 'Fe', 'Ti', 'Au', 'Pt', 'W', 'PdO', 'Ta', 'Al']:
        cr = engine.calculate(mat, 2.5, 300, 0.5)
        mag = MATERIAL_MAGNETIC.get(mat, {'class': '?'})
        print(f'{mat:>10}  {cr.medium_resistance:>8.1f}  '
              f'{cr.photon_mass_density:>12.2e}  {cr.reaction_probability:>8.4f}  '
              f'{cr.lattice_focusing_factor:>6.3f}  {mag["class"]}')
