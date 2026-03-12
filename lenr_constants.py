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
    # Additional screening data (Raiola 2004, Czerski 2001-2004)
    'V':    {'Us_eV': 316, 'error_eV': 25,  'source': 'Raiola'},
    'Nb':   {'Us_eV': 284, 'error_eV': 20,  'source': 'Raiola'},
    'Sn':   {'Us_eV': 130, 'error_eV': 18,  'source': 'Raiola'},
    'In':   {'Us_eV': 165, 'error_eV': 22,  'source': 'Raiola'},
    'Mn':   {'Us_eV': 246, 'error_eV': 30,  'source': 'Raiola'},
    'Co':   {'Us_eV': 240, 'error_eV': 30,  'source': 'Raiola'},
    'Cr':   {'Us_eV': 181, 'error_eV': 20,  'source': 'Raiola'},
    'Cu':   {'Us_eV': 43,  'error_eV': 10,  'source': 'Czerski 2001'},
    'W':    {'Us_eV': 74,  'error_eV': 12,  'source': 'Raiola'},
    'Ag':   {'Us_eV': 91,  'error_eV': 15,  'source': 'Raiola'},
    'Constantan': {'Us_eV': 230, 'error_eV': 40, 'source': 'Estimated (Cu55-Ni45 weighted)'},
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
    # SUS304 (AISI 304) stainless steel — Mizuno neutron experiments
    'SUS304': {'structure': 'FCC', 'a_A': 3.5950, 'r_atom_A': 1.27, 'debye_K': 400,
               'e_density_A3': 0.14, 'composition': 'Fe69-Cr19-Ni10-Mn2'},
    # Additional materials for expanded dataset
    'Zr': {'structure': 'HCP', 'a_A': 3.2316, 'c_A': 5.1477, 'r_atom_A': 1.602, 'debye_K': 291,
            'e_density_A3': 0.043},
    'Ta': {'structure': 'BCC', 'a_A': 3.3013, 'r_atom_A': 1.430, 'debye_K': 240,
            'e_density_A3': 0.055},
    'Constantan': {'structure': 'FCC', 'a_A': 3.57, 'r_atom_A': 1.27, 'debye_K': 350,
                   'e_density_A3': 0.10, 'composition': 'Cu55-Ni45'},
    'PdO': {'structure': 'tetragonal', 'a_A': 3.043, 'c_A': 5.337, 'debye_K': 300,
            'e_density_A3': 0.40},
}

# Deuterium diffusion: D(T) = D0 * exp(-Ea / kB*T)
DIFFUSION = {
    'Pd': {'D0_cm2s': 2.0e-3, 'Ea_eV': 0.230, 'D_300K': 1e-7},
    'Ni': {'D0_cm2s': 2.4e-2, 'Ea_eV': 0.457, 'D_300K': 5e-10},
    'Fe': {'D0_cm2s': 7.4e-4, 'Ea_eV': 0.041, 'D_300K': 1.5e-5},
    'Ti': {'D0_cm2s': 2e-3,   'Ea_eV': 0.34,  'D_300K': 3e-9},
    'Zr': {'D0_cm2s': 1e-3,   'Ea_eV': 0.40,  'D_300K': 2e-10},
    'Ta': {'D0_cm2s': 4.4e-4, 'Ea_eV': 0.14,  'D_300K': 2e-6},
    'Cu': {'D0_cm2s': 1e-2,   'Ea_eV': 0.40,  'D_300K': 2e-9},
    'W':  {'D0_cm2s': 4.1e-3, 'Ea_eV': 0.39,  'D_300K': 3e-10},
    'Au': {'D0_cm2s': 5e-3,   'Ea_eV': 0.23,  'D_300K': 1.2e-7},
    'Pt': {'D0_cm2s': 3e-3,   'Ea_eV': 0.26,  'D_300K': 1e-8},
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
# MIZUNO R19 REACTOR DATA (55 tests, Feb-May 2019)
# Source: ICCF-22 "Increased Excess Heat from Palladium Deposited on Nickel"
# Reactor: Ni-mesh 54g (180 mesh) + Pd coating 45-60mg, D₂ gas
# Calorimeter: Air flow, accuracy ±5W
# =============================================================================
MIZUNO_R19_DATA = [
    # (date, pressure_Pa, input_W, output_input_ratio, excess_W, reactor_temp_C, heat_per_g_Ni, D_Ni_ratio)
    {'date': '2/20',  'pressure_Pa': 5412, 'input_W': 100.0, 'COP': 1.39, 'excess_W': 38.63, 'temp_C': 238.9, 'heat_per_g': 0.7154, 'D_Ni': 0.000051},
    {'date': '2/21',  'pressure_Pa': 6320, 'input_W': 197.4, 'COP': 1.40, 'excess_W': 78.31, 'temp_C': 386.0, 'heat_per_g': 1.4501, 'D_Ni': 0.000115},
    {'date': '2/26',  'pressure_Pa': 5949, 'input_W': 50.1,  'COP': 1.21, 'excess_W': 10.42, 'temp_C': 145.0, 'heat_per_g': 0.1929, 'D_Ni': 0.000128},
    {'date': '3/22',  'pressure_Pa': 5421, 'input_W': 99.3,  'COP': 1.42, 'excess_W': 41.84, 'temp_C': 234.0, 'heat_per_g': 0.7749, 'D_Ni': 0.000312},
    {'date': '3/23',  'pressure_Pa': 600,  'input_W': 98.6,  'COP': 1.42, 'excess_W': 41.06, 'temp_C': 229.0, 'heat_per_g': 0.7604, 'D_Ni': 0.000269},
    {'date': '3/25',  'pressure_Pa': 120,  'input_W': 98.2,  'COP': 1.40, 'excess_W': 39.64, 'temp_C': 232.0, 'heat_per_g': 0.7340, 'D_Ni': 0.000250},
    {'date': '3/26',  'pressure_Pa': 78,   'input_W': 98.0,  'COP': 1.43, 'excess_W': 42.30, 'temp_C': 232.7, 'heat_per_g': 0.7834, 'D_Ni': 0.000232},
    {'date': '3/27',  'pressure_Pa': 46,   'input_W': 98.0,  'COP': 1.41, 'excess_W': 40.46, 'temp_C': 232.6, 'heat_per_g': 0.7493, 'D_Ni': 0.000222},
    {'date': '3/28',  'pressure_Pa': 35,   'input_W': 97.9,  'COP': 1.42, 'excess_W': 40.89, 'temp_C': 232.0, 'heat_per_g': 0.7573, 'D_Ni': 0.000214},
    {'date': '3/29',  'pressure_Pa': 28,   'input_W': 97.7,  'COP': 1.40, 'excess_W': 39.20, 'temp_C': 231.5, 'heat_per_g': 0.7260, 'D_Ni': 0.000208},
    {'date': '3/30',  'pressure_Pa': 25,   'input_W': 97.5,  'COP': 1.39, 'excess_W': 38.02, 'temp_C': 230.1, 'heat_per_g': 0.7040, 'D_Ni': 0.000201},
    {'date': '4/1',   'pressure_Pa': 17,   'input_W': 97.4,  'COP': 1.41, 'excess_W': 40.07, 'temp_C': 231.96,'heat_per_g': 0.7419, 'D_Ni': 0.000197},
    {'date': '4/2',   'pressure_Pa': 20,   'input_W': 97.5,  'COP': 1.40, 'excess_W': 39.43, 'temp_C': 232.36,'heat_per_g': 0.7301, 'D_Ni': 0.000193},
    {'date': '4/3',   'pressure_Pa': 4644, 'input_W': 97.3,  'COP': 1.40, 'excess_W': 39.03, 'temp_C': 231.83,'heat_per_g': 0.7228, 'D_Ni': 0.000280},
    {'date': '4/4',   'pressure_Pa': 4553, 'input_W': 97.2,  'COP': 1.42, 'excess_W': 40.37, 'temp_C': 233.1, 'heat_per_g': 0.7476, 'D_Ni': 0.000310},
    {'date': '4/5',   'pressure_Pa': 4092, 'input_W': 152.3, 'COP': 1.50, 'excess_W': 76.17, 'temp_C': 320.0, 'heat_per_g': 1.4105, 'D_Ni': 0.000502},
    {'date': '4/6',   'pressure_Pa': 1932, 'input_W': 200.9, 'COP': 1.51, 'excess_W': 102.05,'temp_C': 382.7, 'heat_per_g': 1.8898, 'D_Ni': 0.001005},
    {'date': '4/8',   'pressure_Pa': 3751, 'input_W': 200.8, 'COP': 1.49, 'excess_W': 97.62, 'temp_C': 383.3, 'heat_per_g': 1.8078, 'D_Ni': 0.001670},
    {'date': '4/9',   'pressure_Pa': 3670, 'input_W': 200.8, 'COP': 1.49, 'excess_W': 97.94, 'temp_C': 382.6, 'heat_per_g': 1.8137, 'D_Ni': 0.002280},
    {'date': '4/10',  'pressure_Pa': 3268, 'input_W': 201.0, 'COP': 1.49, 'excess_W': 97.64, 'temp_C': 384.5, 'heat_per_g': 1.8081, 'D_Ni': 0.002825},
    {'date': '4/11',  'pressure_Pa': 3199, 'input_W': 200.9, 'COP': 1.48, 'excess_W': 96.74, 'temp_C': 383.93,'heat_per_g': 1.7914, 'D_Ni': 0.003373},
    {'date': '4/12',  'pressure_Pa': 3173, 'input_W': 200.9, 'COP': 1.48, 'excess_W': 96.04, 'temp_C': 383.1, 'heat_per_g': 1.7784, 'D_Ni': 0.003959},
    {'date': '4/13',  'pressure_Pa': 2581, 'input_W': 200.9, 'COP': 1.48, 'excess_W': 96.22, 'temp_C': 383.4, 'heat_per_g': 1.7818, 'D_Ni': 0.004640},
    {'date': '4/15',  'pressure_Pa': 3042, 'input_W': 200.7, 'COP': 1.51, 'excess_W': 101.67,'temp_C': 381.0, 'heat_per_g': 1.8829, 'D_Ni': 0.005211},
    {'date': '4/16',  'pressure_Pa': 3207, 'input_W': 200.6, 'COP': 1.47, 'excess_W': 94.24, 'temp_C': 380.8, 'heat_per_g': 1.7451, 'D_Ni': 0.005758},
    {'date': '4/17',  'pressure_Pa': 3058, 'input_W': 200.6, 'COP': 1.46, 'excess_W': 91.95, 'temp_C': 381.9, 'heat_per_g': 1.7028, 'D_Ni': 0.006624},
    {'date': '4/18',  'pressure_Pa': 2646, 'input_W': 200.6, 'COP': 1.47, 'excess_W': 93.22, 'temp_C': 382.57,'heat_per_g': 1.7263, 'D_Ni': 0.007206},
    {'date': '4/19',  'pressure_Pa': 2546, 'input_W': 200.6, 'COP': 1.46, 'excess_W': 91.97, 'temp_C': 380.7, 'heat_per_g': 1.7031, 'D_Ni': 0.007875},
    {'date': '4/20',  'pressure_Pa': 2676, 'input_W': 200.5, 'COP': 1.47, 'excess_W': 93.42, 'temp_C': 378.16,'heat_per_g': 1.7300, 'D_Ni': 0.008453},
    {'date': '4/21',  'pressure_Pa': 2903, 'input_W': 200.5, 'COP': 1.47, 'excess_W': 93.39, 'temp_C': 377.56,'heat_per_g': 1.7295, 'D_Ni': 0.008973},
    {'date': '4/22',  'pressure_Pa': 2863, 'input_W': 200.5, 'COP': 1.45, 'excess_W': 91.09, 'temp_C': 377.4, 'heat_per_g': 1.6868, 'D_Ni': 0.009531},
    {'date': '4/23',  'pressure_Pa': 2771, 'input_W': 200.5, 'COP': 1.47, 'excess_W': 94.49, 'temp_C': 377.0, 'heat_per_g': 1.7499, 'D_Ni': 0.010095},
    {'date': '4/24',  'pressure_Pa': 2773, 'input_W': 200.5, 'COP': 1.47, 'excess_W': 93.44, 'temp_C': 377.0, 'heat_per_g': 1.7303, 'D_Ni': 0.010680},
    {'date': '4/25',  'pressure_Pa': 2761, 'input_W': 200.4, 'COP': 1.46, 'excess_W': 92.74, 'temp_C': 376.0, 'heat_per_g': 1.7173, 'D_Ni': 0.011250},
    {'date': '4/26',  'pressure_Pa': 2831, 'input_W': 200.5, 'COP': 1.47, 'excess_W': 94.42, 'temp_C': 377.0, 'heat_per_g': 1.7485, 'D_Ni': 0.011834},
    {'date': '4/27',  'pressure_Pa': 2111, 'input_W': 200.4, 'COP': 1.46, 'excess_W': 92.17, 'temp_C': 375.5, 'heat_per_g': 1.7068, 'D_Ni': 0.012600},
    {'date': '4/29',  'pressure_Pa': 1996, 'input_W': 200.4, 'COP': 1.46, 'excess_W': 91.15, 'temp_C': 377.8, 'heat_per_g': 1.6879, 'D_Ni': 0.013380},
    {'date': '5/1',   'pressure_Pa': 1156, 'input_W': 200.5, 'COP': 1.47, 'excess_W': 94.23, 'temp_C': 378.3, 'heat_per_g': 1.7451, 'D_Ni': 0.013397},
    {'date': '5/3',   'pressure_Pa': 1152, 'input_W': 200.5, 'COP': 1.45, 'excess_W': 90.48, 'temp_C': 377.0, 'heat_per_g': 1.6755, 'D_Ni': 0.013400},
    {'date': '5/4',   'pressure_Pa': 2537, 'input_W': 200.4, 'COP': 1.44, 'excess_W': 88.44, 'temp_C': 377.6, 'heat_per_g': 1.6378, 'D_Ni': 0.014008},
    {'date': '5/5',   'pressure_Pa': 2129, 'input_W': 200.4, 'COP': 1.45, 'excess_W': 89.27, 'temp_C': 379.0, 'heat_per_g': 1.6532, 'D_Ni': 0.014744},
    {'date': '5/7',   'pressure_Pa': 2560, 'input_W': 200.4, 'COP': 1.45, 'excess_W': 90.04, 'temp_C': 378.68,'heat_per_g': 1.6675, 'D_Ni': 0.015345},
    {'date': '5/8',   'pressure_Pa': 2726, 'input_W': 200.4, 'COP': 1.46, 'excess_W': 91.57, 'temp_C': 377.6, 'heat_per_g': 1.6956, 'D_Ni': 0.015911},
    {'date': '5/9',   'pressure_Pa': 2800, 'input_W': 200.4, 'COP': 1.45, 'excess_W': 90.33, 'temp_C': 377.6, 'heat_per_g': 1.6728, 'D_Ni': 0.016462},
    {'date': '5/10',  'pressure_Pa': 42,   'input_W': 200.3, 'COP': 1.45, 'excess_W': 89.61, 'temp_C': 374.0, 'heat_per_g': 1.6594, 'D_Ni': 0.016463},
    {'date': '5/11',  'pressure_Pa': 2,    'input_W': 200.4, 'COP': 1.44, 'excess_W': 88.53, 'temp_C': 373.5, 'heat_per_g': 1.6394, 'D_Ni': 0.016464},
    {'date': '5/13',  'pressure_Pa': 5,    'input_W': 100.3, 'COP': 1.35, 'excess_W': 35.09, 'temp_C': 235.0, 'heat_per_g': 0.6498, 'D_Ni': 0.016465},
    {'date': '5/14',  'pressure_Pa': 6,    'input_W': 100.0, 'COP': 1.34, 'excess_W': 33.97, 'temp_C': 236.0, 'heat_per_g': 0.6291, 'D_Ni': 0.016464},
    {'date': '5/15',  'pressure_Pa': 6,    'input_W': 99.7,  'COP': 1.37, 'excess_W': 36.41, 'temp_C': 236.0, 'heat_per_g': 0.6743, 'D_Ni': 0.016464},
    {'date': '5/16',  'pressure_Pa': 6,    'input_W': 99.5,  'COP': 1.35, 'excess_W': 34.32, 'temp_C': 236.25,'heat_per_g': 0.6355, 'D_Ni': 0.016463},
    {'date': '5/18',  'pressure_Pa': 7,    'input_W': 0.0,   'COP': None, 'excess_W': 2.11,  'temp_C': 23.64, 'heat_per_g': 0.0390, 'D_Ni': 0.016463},
    {'date': '5/20',  'pressure_Pa': 8,    'input_W': 0.0,   'COP': None, 'excess_W': 2.89,  'temp_C': 23.07, 'heat_per_g': 0.0534, 'D_Ni': 0.016463},
    {'date': '5/21',  'pressure_Pa': 5082, 'input_W': 98.3,  'COP': 1.36, 'excess_W': 35.28, 'temp_C': 232.0, 'heat_per_g': 0.6534, 'D_Ni': 0.016591},
    {'date': '5/22',  'pressure_Pa': 4778, 'input_W': 98.2,  'COP': 1.34, 'excess_W': 33.82, 'temp_C': 232.0, 'heat_per_g': 0.6262, 'D_Ni': 0.016664},
    {'date': '5/23',  'pressure_Pa': 4571, 'input_W': 98.1,  'COP': 1.35, 'excess_W': 34.49, 'temp_C': 232.2, 'heat_per_g': 0.6387, 'D_Ni': 0.016711},
]

# Mizuno R20 high-power tests (outside calorimeter)
MIZUNO_R20_DATA = [
    {'reactor': 'R13', 'input_W': 100,  'output_W': 112,  'COP': 1.12, 'excess_W': 12,   'notes': 'ICCF21 baseline'},
    {'reactor': 'R19', 'input_W': 200,  'output_W': 290,  'COP': 1.45, 'excess_W': 90,   'notes': 'Table 1 average'},
    {'reactor': 'R20', 'input_W': 50,   'output_W': 300,  'COP': 6.00, 'excess_W': 250,  'notes': 'Low power in calorimeter'},
    {'reactor': 'R20', 'input_W': 300,  'output_W': 2250, 'COP': 7.50, 'excess_W': 1950, 'notes': 'Room heater test'},
]

# Heat-After-Death event (1991): see detailed MIZUNO_HEAT_AFTER_DEATH below (line ~373)

# Mizuno empirical formulas (from R19 data fitting)
MIZUNO_EMPIRICAL = {
    'temp_dependence': {
        'formula': 'Excess_W/g = A * exp(B * T_abs)',
        'A': 1.2e-6,
        'B_per_K': 0.008,
        'R2': 0.959,
    },
    'inverse_loading': {
        'formula': 'Heat_W/g = C - D * (D/Ni)',
        'C': 1.85,
        'D': 12,
    },
    'pressure_dependence': 'No strong correlation (100-6000 Pa)',
    'optimal_pressure_Pa': (100, 300),
}

# Mizuno reactor material properties
MIZUNO_MATERIALS = {
    'Ni_mesh': {'purity_pct': 99.6, 'wire_diameter_mm': 0.055, 'mesh_count': 180,
                'weight_per_mesh_g': 18, 'total_Ni_g': 54},
    'Pd_coating': {'method': 'rubbing', 'mass_per_layer_mg': 17.5,
                   'layers': 3, 'total_mg': 52.5},
    'D2_gas': {'loading_max_cm3': 176, 'optimal_D_Ni': 0.016,
               'optimal_pressure_Pa': 200},
}

# =============================================================================
# WIDOM-LARSEN THEORY PARAMETERS
# Source: Widom & Larsen, Eur. Phys. J. C, 2006; Mizuno 2025 neutron data
# =============================================================================
WIDOM_LARSEN = {
    'surface_EM_field_V_per_nm': 140,   # on Pd surface
    'heavy_electron_energy_MeV': 0.78,
    'ULMN_energy_MeV': 0.7,             # Ultra-Low-Momentum Neutrons (observed by Mizuno 2025)
    'classical_DD_neutron_MeV': 2.45,   # NOT observed in LENR
    'gamma_suppression': 'local_IR_conversion',  # micron scale
    'charge_screening': 'neutral_ULMN',           # no Coulomb barrier for neutral particles
}

# Neutron observations (European J. Applied Physics, Aug 2025)
NEUTRON_DATA = {
    'classical_DD_2_45_MeV': {'observed': False, 'notes': 'Expected from D+D->3He+n, NOT seen in any LENR experiment'},
    'ULMN_0_7_MeV': {'observed': True, 'notes': 'Ultra-Low-Momentum neutrons detected at 0.7 MeV peak'},
    'implication': 'Mechanism is NOT classical D-D fusion',
    'SUS304_neutrons': {
        'gas': 'H2',  # hydrogen, not deuterium — important distinction
        'threshold_C': 440,
        'peak_energy_MeV': 0.7,
        'rate_driver': 'dT/dt (temperature change rate)',
        'notes': 'Neutrons from H₂+metal (no deuterium!) → supports Widom-Larsen ULMN theory',
    },
}

# Screening potential ranges (for ML feature engineering)
SCREENING_RANGES = {
    'gas_D2_eV': (20, 30),           # baseline, no enhancement
    'Pd_metal_low_loading_eV': (200, 300),
    'Pd_metal_optimal_eV': (400, 750),
    'cluster_fusion_enhancement': 1e26,  # coherent deuteron motion (Takahashi)
}

# =============================================================================
# MIZUNO NEUTRON EXPERIMENTS (SUS304 + H₂, 2020-2025)
# Source: "Neutrons Produced by Heating Processed Metals"
# European J. Applied Physics, Aug 2025
# DOI: eu-opensci.org/index.php/ejphysics/article/view/11383
# Reactor: SUS304 stainless steel cylinder (10cm×40cm, 12.7kg), H₂ gas
# Detector: NE213 liquid scintillator, rise time discrimination
# =============================================================================
MIZUNO_NEUTRON_EXPERIMENTS = [
    # (date, input_W, temp_C, neutron_count_per_min)
    {'date': '2020-11-29', 'input_W': 724, 'temp_C': 764, 'neutron_cpm': 0.803},
    {'date': '2024-08-05', 'input_W': 478, 'temp_C': 589, 'neutron_cpm': 0.415},
    {'date': '2024-10-21', 'input_W': 584, 'temp_C': 727, 'neutron_cpm': 0.803},
    {'date': '2024-10-29', 'input_W': 699, 'temp_C': 783, 'neutron_cpm': 1.972},
    {'date': '2024-11-19', 'input_W': 547, 'temp_C': 562, 'neutron_cpm': 0.250},
    {'date': '2024-11-20', 'input_W': 720, 'temp_C': 622, 'neutron_cpm': 1.554},
    {'date': '2024-11-25', 'input_W': 725, 'temp_C': 783, 'neutron_cpm': 0.826},
    {'date': '2024-11-29', 'input_W': 724, 'temp_C': 764, 'neutron_cpm': 1.064},
    {'date': '2025-04-14', 'input_W': 720, 'temp_C': 760, 'neutron_cpm': 1.060},
    {'date': '2025-04-15', 'input_W': 729, 'temp_C': 758, 'neutron_cpm': 1.198},
]

MIZUNO_NEUTRON_REACTOR = {
    'material': 'SUS304',  # AISI 304 stainless steel
    'composition': {'Fe': 0.69, 'Cr': 0.19, 'Ni': 0.10, 'Mn': 0.02},
    'dimensions_cm': {'diameter': 10, 'length': 40, 'thickness': 0.2},
    'total_mass_kg': 12.7,
    'gas': 'H2',  # hydrogen (NOT deuterium)
    'initial_pressure_Pa': 500,
    'heater_W': 600,  # 100V, 600W sheath heater
    'surface_treatment': 'mesh_400_buffed',
    'excess_heat_W': 200,      # reported excess
    'excess_heat_error_W': 2,  # ±2W
    'output_ratio': 1.25,      # output/input = 1.25
}

MIZUNO_NEUTRON_PHYSICS = {
    'peak_energy_MeV': 0.7,           # observed neutron peak
    'classical_DD_MeV': 2.45,         # NOT observed
    'threshold_temp_C': 440,          # neutrons start appearing
    'detector': 'NE213_liquid_scintillator',
    'detector_size_cm': 12.7,         # diameter = thickness
    'background_cph': 5,              # counts per hour
    'foreground_600C_cph': 24,        # counts per hour at 600°C
    'max_rate_cpm': 5,                # max observed counts/min
    'key_finding': 'rate correlates with dT/dt (temperature change rate), not just T',
    'calibration_source': '252Cf_96.5_mCi',
}

# =============================================================================
# HEAT-AFTER-DEATH DETAILED CHRONOLOGY (1991)
# Source: Mizuno, "Nuclear Transmutation: The Reality of Cold Fusion"
# Translation: Jed Rothwell, Infinite Energy Press
# =============================================================================
MIZUNO_HEAT_AFTER_DEATH = {
    'date': '1991-04-22',
    'total_energy_MJ': 85,
    'Pd_mass_g': 100,
    'energy_density_kJg': 850,  # 27x gasoline equivalent
    'water_evaporated_L': 37.5,
    'duration_days': 15,
    'total_experiment_energy_MJ': 114,  # entire experiment including electrolysis
    'energy_vs_gasoline': 27,           # times more energy than equivalent mass of gasoline
    'chronology': [
        {'date': '1991-03-XX', 'event': 'experiment_start', 'notes': 'New closed cell experiment begins'},
        {'date': '1991-04-XX', 'event': 'excess_heat_detected', 'notes': 'Small but significant excess heat'},
        {'date': '1991-04-22', 'event': 'electrolysis_stopped', 'notes': 'External power disconnected'},
        {'date': '1991-04-25', 'event': 'heat_after_death_confirmed',
         'energy_since_shutoff_J': 1.2e7, 'temp_above_ambient': True,
         'notes': 'Cell moved from underground lab, temp >100°C'},
        {'date': '1991-04-26', 'event': 'bucket_15L', 'notes': 'Cell in 15L bucket, temperature not declining'},
        {'date': '1991-04-27', 'event': 'evaporation_10L', 'water_evaporated_L': 10,
         'notes': 'Transferred to 20L bucket, submerged in 15L water'},
        {'date': '1991-04-30', 'event': 'evaporation_10L', 'water_evaporated_L': 10,
         'notes': 'Water replenished to 15L'},
        {'date': '1991-05-01', 'event': 'water_added', 'water_added_L': 5},
        {'date': '1991-05-02', 'event': 'water_added', 'water_added_L': 5},
        {'date': '1991-05-07', 'event': 'cell_cool', 'water_remaining_L': 7.5,
         'notes': 'Cell finally cool. Total evaporation: 37.5L'},
    ],
    'stasis_effect': {
        'description': 'After external heater turned off, heat-after-death increased '
                       'to compensate for loss of external heat, maintaining constant temperature',
        'notes': 'Pons observed similar "memory" effect — temperature returns to fixed level after perturbation',
        'storms_hypothesis': 'Special configuration of matter with densely packed deuterium; '
                             'diffusion rate slower at high T than usual beta-phase PdD',
    },
    'heat_of_vaporization_J_per_g': 2268,  # 540 cal/g
}

# =============================================================================
# NUCLEAR TRANSMUTATION DATA
# Source: Mizuno, "Nuclear Transmutation" book; ICCF papers
# =============================================================================
TRANSMUTATION = {
    'host_metal_reactions': {
        'description': 'Cathode metal itself undergoes nuclear reactions (fission + fusion)',
        'Pd_products': ['Cu', 'Zn', 'Fe', 'Cr', 'Ca'],  # observed transmutation products
        'isotope_anomaly': True,  # non-natural isotopic distributions observed
        'notes': 'Anomalous isotopes found around eruption sites on cathode surface',
    },
    'surface_eruptions': {
        'description': 'Lily-shaped eruptions on cathode surface from subsurface reactions',
        'materials_observed': ['Au', 'Pd'],  # Ohmori gold cathodes, Mizuno Pd cathodes
        'mechanism': 'Violent reaction under metal surface vaporizes metal, spews it out',
        'transmutation_locus': True,  # eruption sites are where transmutations occur
    },
    'tritium_production': {
        'observed': True,
        'notes': 'Tritium definitively observed in Mizuno experiments (published 1990-1991)',
    },
    'blackened_cathode': {
        'description': 'Black film on Pd cathode after successful run — later recognized as '
                       'microscopic erupted structures similar to Ohmori gold cathodes',
        'initially_mistaken_for': 'contamination',
    },
}

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
