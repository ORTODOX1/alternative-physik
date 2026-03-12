"""
LENR Alternative Physics ML Simulation — Physical Constants & Data
All values from peer-reviewed sources + Takahashi TSC theory + Cherepanov framework
"""

import numpy as np

# =============================================================================
# NUCLEAR REFERENCE CONSTANTS
# =============================================================================
NUCLEAR = {
    'gamow_energy_DD_keV': 986.0,
    'coulomb_barrier_vacuum_keV': 400.0,
    'Q_DpT_MeV': 4.033,           # D+D → T+p
    'Q_Dn3He_MeV': 3.269,         # D+D → ³He+n
    'Q_D4He_gamma_MeV': 23.847,   # D+D → ⁴He+γ
    'Q_4D_8Be_MeV': 47.6,         # 4D → ⁸Be* → 2α
    'He4_binding_MeV': 28.296,
    'Be8_above_2alpha_keV': 91.84,
    'Be8_width_eV': 6.0,
    'Be8_halflife_s': 8.19e-17,
    'S0_DpT_keVb': 55.0,
    'S0_Dn3He_keVb': 52.0,
    'fine_structure_alpha': 1.0 / 137.036,
    'deuteron_mass_MeV': 1875.613,
    'reduced_mass_DD_MeV': 937.807,
    'branch_ratio_DpT': 0.50,
    'branch_ratio_Dn3He': 0.50,
    'branch_ratio_D4He_gamma': 1e-7,
}

# =============================================================================
# SCREENING ENERGIES (eV) — EXPERIMENTAL
# =============================================================================
# Source: Kasagi (Tohoku, 2002), Raiola (Bochum), Huke (Berlin), NASA TP-2020
SCREENING_EXPERIMENTAL = {
    'PdO':  {'Us_eV': 600, 'error_eV': 60,  'enhancement_2_5keV': 50.0, 'source': 'Kasagi'},
    'Pd':   {'Us_eV': 310, 'error_eV': 30,  'enhancement_2_5keV': 10.0, 'source': 'Kasagi'},
    'Pd_Raiola': {'Us_eV': 800, 'error_eV': 90, 'source': 'Raiola (Bochum)'},
    'Pd_Huke':   {'Us_eV': 313, 'error_eV': 2,  'source': 'Huke (Berlin, lower limit)'},
    'Fe':   {'Us_eV': 200, 'error_eV': 20,  'enhancement_2_5keV': 5.0,  'source': 'Kasagi'},
    'Au':   {'Us_eV': 70,  'error_eV': 10,  'enhancement_2_5keV': 1.5,  'source': 'Kasagi'},
    'Ti':   {'Us_eV': 65,  'error_eV': 10,  'enhancement_2_5keV': 1.2,  'source': 'Kasagi'},
    'Ta':   {'Us_eV': 309, 'error_eV': 12,  'source': 'Raiola'},
    'Zr':   {'Us_eV': 297, 'error_eV': 8,   'source': 'Huke (100-600 eV dep. on vacancies)'},
    'Ni':   {'Us_eV': 420, 'error_eV': 50,  'source': 'Raiola'},
    'Al':   {'Us_eV': 190, 'error_eV': 15,  'source': 'Huke'},
    'BeO':  {'Us_eV': 180, 'error_eV': 40,  'source': 'NASA TP-2020'},
    'Pt':   {'Us_eV': 122, 'error_eV': 20,  'source': 'Raiola'},
    'C':    {'Us_eV': 25,  'error_eV': 5,   'source': 'Huke (no enhancement)'},
}

# =============================================================================
# TAKAHASHI TSC / EQPET THEORY PARAMETERS
# =============================================================================
EQPET_SCREENING = {
    # e*(m,Z): {Us, b0, R_dd, trapping_depth}
    '(1,1)_electron':    {'Us_eV': 36,    'b0_pm': 40,   'Rdd_pm': 101,  'trap_eV': -15.4},
    '(1,1)x2_D2':        {'Us_eV': 72,    'b0_pm': 20,   'Rdd_pm': 73,   'trap_eV': -37.8},
    '(2,2)_Cooper':      {'Us_eV': 360,   'b0_pm': 4,    'Rdd_pm': 33.8, 'trap_eV': -259},
    '(4,4)_quadruplet':  {'Us_eV': 4000,  'b0_pm': 0.36, 'Rdd_pm': 15.1, 'trap_eV': -2460},
    '(6,6)':             {'Us_eV': 9600,  'b0_pm': 0.15, 'Rdd_pm': None, 'trap_eV': None},
    '(8,8)_octal':       {'Us_eV': 22154, 'b0_pm': 0.065,'Rdd_pm': None, 'trap_eV': None},
}

TSC_PARAMS = {
    'initial_dd_distance_pm': 74,       # D₂ ground state
    'tsc_radius_start_pm': 45.8,        # (√3/2) × Bohr radius
    'tsc_radius_min_fm': 20,            # before nuclear reaction
    'condensation_time_4D_fs': 1.4,
    'condensation_time_4H_fs': 1.0,
    'max_tsc_density_per_cm3': 1e22,
    'max_fusion_rate_MW_per_cm3': 46,
    'energy_per_Pd_atom_keV': 23,
    'neutron_to_He4_ratio': 1e-12,
    'S_4D_extrapolated_keVb': 1e11,     # PEF scaling
}

# Barrier factors at E_d = 0.22 eV (Bloch potential depth in Pd)
BARRIER_FACTORS = {
    '(1,1)':  {'bf_2D': 1e-125, 'bf_4D': 1e-250, 'rate_4D': 1e-252},
    '(2,2)':  {'bf_2D': 1e-7,   'bf_4D': 1e-15,  'rate_4D': 1e-17},
    '(4,4)':  {'bf_2D': 3e-4,   'bf_4D': 1e-7,   'rate_4D': 1e-9},
    '(8,8)':  {'bf_2D': 4e-1,   'bf_4D': 1e-1,   'rate_4D': 1e-3},
}

# =============================================================================
# MATERIAL PROPERTIES
# =============================================================================
LATTICE = {
    'Pd': {'structure': 'FCC', 'a_A': 3.8907, 'r_atom_A': 1.376, 'debye_K': 274,
            'oct_site_A': 0.570, 'tet_site_A': 0.310, 'e_density_A3': 0.34},
    'Ni': {'structure': 'FCC', 'a_A': 3.5240, 'r_atom_A': 1.246, 'debye_K': 450,
            'oct_site_A': 0.516, 'tet_site_A': 0.280, 'e_density_A3': 0.16},
    'Ti': {'structure': 'HCP', 'a_A': 2.9508, 'c_A': 4.6855, 'r_atom_A': 1.462, 'debye_K': 420,
            'e_density_A3': 0.051},
    'Fe': {'structure': 'BCC', 'a_A': 2.8665, 'r_atom_A': 1.241, 'debye_K': 470,
            'tet_site_A': 0.361, 'e_density_A3': 0.170},
    'Au': {'structure': 'FCC', 'a_A': 4.0782, 'r_atom_A': 1.442, 'debye_K': 165,
            'e_density_A3': 0.059},
    'Pt': {'structure': 'FCC', 'a_A': 3.9242, 'r_atom_A': 1.387, 'debye_K': 240,
            'e_density_A3': 0.066},
    'W':  {'structure': 'BCC', 'a_A': 3.1652, 'r_atom_A': 1.367, 'debye_K': 400,
            'e_density_A3': 0.063},
    'Cu': {'structure': 'FCC', 'a_A': 3.6149, 'r_atom_A': 1.278, 'debye_K': 343,
            'e_density_A3': 0.085},
    'Ag': {'structure': 'FCC', 'a_A': 4.0853, 'r_atom_A': 1.445, 'debye_K': 225,
            'e_density_A3': 0.059},
}

# Deuterium diffusion: D(T) = D0 * exp(-Ea / kB*T)
DIFFUSION = {
    'Pd': {'D0_cm2s': 2.0e-3, 'Ea_eV': 0.230, 'D_300K': 1e-7},
    'Ni': {'D0_cm2s': 2.4e-2, 'Ea_eV': 0.457, 'D_300K': 5e-10},
    'Fe': {'D0_cm2s': 7.4e-4, 'Ea_eV': 0.041, 'D_300K': 1.5e-5},
    'Ti': {'D0_cm2s': 2e-3,   'Ea_eV': 0.34,  'D_300K': 3e-9},
}

# Loading parameters
LOADING = {
    'Pd_max_1atm_RT': 0.70,
    'Pd_max_electrolysis': 0.92,
    'Pd_max_highP_77K': 0.97,
    'LENR_threshold_McKubre': 0.84,
    'LENR_threshold_Storms': 0.90,
    'McKubre_M': 2.33e5,       # V/cm
    'McKubre_i0': 0.4,         # A/cm²
    'McKubre_x0': 0.832,
    'Pd_alpha_beta_low': 0.017,
    'Pd_alpha_beta_high': 0.58,
    'Pd_critical_T_C': 276,
    'Pd_lattice_expansion_pct': 3.4,  # at β-PdD0.6
}

# =============================================================================
# EXCESS HEAT EXPERIMENTS (structured for ML)
# =============================================================================
EXCESS_HEAT_DATA = [
    {'lab': 'Fleischmann-Pons', 'material': 'Pd', 'method': 'electrolysis',
     'excess_W': 150, 'COP': 40, 'duration_days': 14, 'DPd': 0.95,
     'temperature_K': 340, 'pressure_Pa': 1e5, 'reproducibility': 0.30},
    {'lab': 'McKubre/SRI', 'material': 'Pd', 'method': 'electrolysis',
     'excess_W': 2.1, 'COP': 1.38, 'duration_days': 60, 'DPd': 0.90,
     'temperature_K': 340, 'pressure_Pa': 1e5, 'reproducibility': 0.17},
    {'lab': 'Kitamura/Kobe', 'material': 'PdNi_ZrO2', 'method': 'gas_loading',
     'excess_W': 24, 'COP': 2.0, 'duration_days': 42, 'DPd': 0.80,
     'temperature_K': 500, 'pressure_Pa': 1e5, 'reproducibility': 1.0},
    {'lab': 'Li_XZ_China', 'material': 'Pd', 'method': 'gas_loading',
     'excess_W': 87, 'COP': 1.1, 'duration_days': 40, 'DPd': 0.12,
     'temperature_K': 350, 'pressure_Pa': 9e4, 'reproducibility': 1.0},
    {'lab': 'Iwamura/Clean_Planet', 'material': 'NiCu', 'method': 'gas_permeation',
     'excess_W': 5, 'COP': 1.5, 'duration_days': 589, 'DPd': 0.80,
     'temperature_K': 343, 'pressure_Pa': 5e4, 'reproducibility': 1.0},
    {'lab': 'Storms', 'material': 'Pd', 'method': 'electrolysis',
     'excess_W': 7.5, 'COP': 1.2, 'duration_days': 0.5, 'DPd': 0.82,
     'temperature_K': 340, 'pressure_Pa': 1e5, 'reproducibility': 0.30},
    {'lab': 'Arata/Zhang', 'material': 'nano_Pd', 'method': 'gas_loading',
     'excess_W': 24, 'COP': 1.5, 'duration_days': 21, 'DPd': 3.0,
     'temperature_K': 300, 'pressure_Pa': 5e5, 'reproducibility': 0.80},
    {'lab': 'Piantelli', 'material': 'Ni', 'method': 'gas_loading_H',
     'excess_W': 38.9, 'COP': 1.38, 'duration_days': 278, 'DPd': 0.50,
     'temperature_K': 500, 'pressure_Pa': 1e5, 'reproducibility': 0.60},
    {'lab': 'Brillouin', 'material': 'Ni', 'method': 'gas_loading_H',
     'excess_W': 60, 'COP': 2.25, 'duration_days': 30, 'DPd': 0.50,
     'temperature_K': 500, 'pressure_Pa': 1e5, 'reproducibility': 0.80},
    {'lab': 'Constantan_2025', 'material': 'Constantan', 'method': 'gas_loading',
     'excess_W': 209, 'COP': 3.91, 'duration_days': 0.001, 'DPd': None,
     'temperature_K': 400, 'pressure_Pa': 1e5, 'reproducibility': 0.50},
]

# =============================================================================
# PHYSICS ENGINE MODES
# =============================================================================
PHYSICS_MODES = {
    'maxwell': {
        'description': 'Standard electromagnetic theory',
        'charge_unit': '[L^(3/2) * T^(-1) * M^(1/2)]',
        'force_law': 'F = k * q1 * q2 / r^2',
        'barrier_type': 'coulomb_electrostatic',
        'field_exists': True,
        'particles': ['electron', 'proton', 'neutron', 'deuteron'],
    },
    'coulomb_original': {
        'description': 'Original Coulomb - mass of electricity',
        'charge_unit': '[M * L^(-2)]  (mass density)',
        'force_law': 'F = k * rho1 * rho2 / r^2',
        'barrier_type': 'mass_density_interaction',
        'field_exists': False,
        'particles': ['mass_elements'],
    },
    'cherepanov': {
        'description': 'Cherepanov - photon mass, no charge',
        'charge_unit': 'None (no charge exists)',
        'force_law': 'Magnetic flux interactions B[kg/s]',
        'barrier_type': 'medium_resistance',
        'field_exists': False,
        'particles': ['photon_mass'],
        'notes': 'Radioactive element = photon mass accumulator',
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def gamow_penetration(E_cm_keV, E_G_keV=986.0):
    """Gamow penetration probability for D-D fusion."""
    eta = np.sqrt(E_G_keV / (4 * E_cm_keV))
    return np.exp(-2 * np.pi * eta)


def cross_section_DD(E_cm_keV, S0_keVb=110.0, E_G_keV=986.0):
    """D-D fusion cross-section in barns (Bosch-Hale approx)."""
    return S0_keVb / (E_cm_keV * np.exp(np.sqrt(E_G_keV / E_cm_keV)))


def screened_cross_section(E_cm_keV, Us_eV, S0_keVb=110.0, E_G_keV=986.0):
    """D-D cross-section with electron screening."""
    Us_keV = Us_eV / 1000.0
    E_eff = E_cm_keV + Us_keV
    enhancement = cross_section_DD(E_eff, S0_keVb, E_G_keV) / cross_section_DD(E_cm_keV, S0_keVb, E_G_keV)
    return cross_section_DD(E_cm_keV, S0_keVb, E_G_keV) * enhancement


def enhancement_factor(E_cm_keV, Us_eV, E_G_keV=986.0):
    """Enhancement factor due to screening at energy E."""
    Us_keV = Us_eV / 1000.0
    sigma_screened = cross_section_DD(E_cm_keV + Us_keV)
    sigma_bare = cross_section_DD(E_cm_keV)
    return sigma_screened / sigma_bare if sigma_bare > 0 else np.inf


def diffusion_coefficient(metal, T_K):
    """Deuterium diffusion coefficient at temperature T."""
    kB_eV = 8.617e-5  # eV/K
    d = DIFFUSION[metal]
    return d['D0_cm2s'] * np.exp(-d['Ea_eV'] / (kB_eV * T_K))


if __name__ == '__main__':
    print("=== D-D Cross-sections (bare) ===")
    for E in [1, 2.5, 5, 10, 25, 50]:
        sigma = cross_section_DD(E)
        P = gamow_penetration(E)
        print(f"E={E:5.1f} keV: σ = {sigma:.2e} b, P_Gamow = {P:.2e}")

    print("\n=== Enhancement in PdO (Us=600 eV) ===")
    for E in [1, 2.5, 5, 10]:
        ef = enhancement_factor(E, 600)
        print(f"E={E:5.1f} keV: enhancement = {ef:.1f}×")

    print("\n=== Diffusion at 300K ===")
    for metal in ['Pd', 'Ni', 'Fe', 'Ti']:
        D = diffusion_coefficient(metal, 300)
        print(f"{metal}: D(300K) = {D:.2e} cm²/s")
