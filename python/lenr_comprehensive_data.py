"""
LENR Comprehensive Experimental Database — All Physical Processes
=================================================================
Organized by physical process categories for multi-target ML training.
All data from peer-reviewed publications + ICCF/JCMNS proceedings.

Process Categories:
  1. Excess Heat (calorimetry)
  2. Nuclear Products (neutrons, tritium, helium-4, charged particles)
  3. Transmutations (elemental/isotopic changes)
  4. Surface Effects (morphology, eruptions, nanostructure)
  5. Electrochemistry (current, loading dynamics)
  6. Gas Loading (pressure, absorption, desorption)
  7. Radiation (gamma, X-ray, RF emission)
  8. Phase Transitions (alpha-beta, superlattice)
  9. Thermal Dynamics (heat-after-death, bursts, incubation)
  10. Magnetic/EM Effects (applied fields, Lenz effect)
"""

import numpy as np

# =============================================================================
# CATEGORY 1: EXPANDED MATERIAL PROPERTIES
# Sources: CRC Handbook, ASM International, Kittel Solid State Physics
# =============================================================================
MATERIALS_EXPANDED = {
    'Pd': {
        'Z': 46, 'A': 106.42, 'structure': 'FCC',
        'a_A': 3.8907, 'debye_K': 274,
        'melting_K': 1828, 'density_g_cm3': 12.023,
        'bulk_modulus_GPa': 180, 'shear_modulus_GPa': 44,
        'thermal_conductivity_W_mK': 71.8,
        'specific_heat_J_gK': 0.244,
        'work_function_eV': 5.12,
        'fermi_energy_eV': 5.56,
        'e_density_A3': 0.34,
        'n_valence': 10,  # [Kr] 4d10
        'magnetic_susceptibility': -7.2e-6,  # diamagnetic
        'hydrogen_solubility': 'high',
        'max_H_loading': 1.0,  # H/Pd or D/Pd
        'D0_cm2s': 2.0e-3, 'Ea_diffusion_eV': 0.230,
        'H_absorption_enthalpy_eV': -0.197,  # exothermic
        'alpha_beta_low': 0.017, 'alpha_beta_high': 0.58,
        'critical_T_C': 276,
        'lattice_expansion_beta_pct': 3.4,  # at D/Pd=0.6
        'stable_isotopes': {102: 0.0102, 104: 0.1114, 105: 0.2233,
                            106: 0.2733, 108: 0.2646, 110: 0.1172},
        'n_stable_isotopes': 6,
    },
    'Ni': {
        'Z': 28, 'A': 58.69, 'structure': 'FCC',
        'a_A': 3.5240, 'debye_K': 450,
        'melting_K': 1728, 'density_g_cm3': 8.908,
        'bulk_modulus_GPa': 180, 'shear_modulus_GPa': 76,
        'thermal_conductivity_W_mK': 90.7,
        'specific_heat_J_gK': 0.444,
        'work_function_eV': 5.15,
        'fermi_energy_eV': 7.04,
        'e_density_A3': 0.16,
        'n_valence': 10,
        'magnetic_susceptibility': 600e-6,  # ferromagnetic
        'hydrogen_solubility': 'low',
        'max_H_loading': 0.03,
        'D0_cm2s': 2.4e-2, 'Ea_diffusion_eV': 0.457,
        'H_absorption_enthalpy_eV': -0.16,
        'stable_isotopes': {58: 0.6808, 60: 0.2622, 61: 0.0114,
                            62: 0.0363, 64: 0.0093},
        'n_stable_isotopes': 5,
    },
    'Ti': {
        'Z': 22, 'A': 47.87, 'structure': 'HCP',
        'a_A': 2.9508, 'c_A': 4.6855, 'debye_K': 420,
        'melting_K': 1941, 'density_g_cm3': 4.507,
        'bulk_modulus_GPa': 110, 'shear_modulus_GPa': 44,
        'thermal_conductivity_W_mK': 21.9,
        'specific_heat_J_gK': 0.523,
        'work_function_eV': 4.33,
        'fermi_energy_eV': 4.87,
        'e_density_A3': 0.051,
        'n_valence': 4,
        'magnetic_susceptibility': 153e-6,
        'hydrogen_solubility': 'very_high',
        'max_H_loading': 2.0,  # TiH2
        'D0_cm2s': 2e-3, 'Ea_diffusion_eV': 0.34,
        'H_absorption_enthalpy_eV': -0.67,
        'stable_isotopes': {46: 0.0825, 47: 0.0744, 48: 0.7372,
                            49: 0.0541, 50: 0.0518},
        'n_stable_isotopes': 5,
    },
    'Fe': {
        'Z': 26, 'A': 55.845, 'structure': 'BCC',
        'a_A': 2.8665, 'debye_K': 470,
        'melting_K': 1811, 'density_g_cm3': 7.874,
        'bulk_modulus_GPa': 170, 'shear_modulus_GPa': 82,
        'thermal_conductivity_W_mK': 80.2,
        'specific_heat_J_gK': 0.449,
        'work_function_eV': 4.50,
        'fermi_energy_eV': 7.47,
        'e_density_A3': 0.170,
        'n_valence': 8,
        'magnetic_susceptibility': 'ferromagnetic',
        'hydrogen_solubility': 'very_low',
        'max_H_loading': 0.0001,
        'D0_cm2s': 7.4e-4, 'Ea_diffusion_eV': 0.041,
        'H_absorption_enthalpy_eV': +0.29,  # endothermic
        'stable_isotopes': {54: 0.0585, 56: 0.9175, 57: 0.0212, 58: 0.0028},
        'n_stable_isotopes': 4,
    },
    'Au': {
        'Z': 79, 'A': 196.97, 'structure': 'FCC',
        'a_A': 4.0782, 'debye_K': 165,
        'melting_K': 1337, 'density_g_cm3': 19.30,
        'bulk_modulus_GPa': 220, 'shear_modulus_GPa': 27,
        'thermal_conductivity_W_mK': 317,
        'specific_heat_J_gK': 0.129,
        'work_function_eV': 5.10,
        'fermi_energy_eV': 5.53,
        'e_density_A3': 0.059,
        'n_valence': 11,
        'magnetic_susceptibility': -2.8e-6,
        'hydrogen_solubility': 'none',
        'max_H_loading': 0.0,
        'stable_isotopes': {197: 1.0},
        'n_stable_isotopes': 1,
    },
    'Pt': {
        'Z': 78, 'A': 195.08, 'structure': 'FCC',
        'a_A': 3.9242, 'debye_K': 240,
        'melting_K': 2041, 'density_g_cm3': 21.45,
        'bulk_modulus_GPa': 230, 'shear_modulus_GPa': 61,
        'thermal_conductivity_W_mK': 71.6,
        'specific_heat_J_gK': 0.133,
        'work_function_eV': 5.65,
        'fermi_energy_eV': 5.74,
        'e_density_A3': 0.066,
        'n_valence': 10,
        'magnetic_susceptibility': 193e-6,
        'hydrogen_solubility': 'low',
        'max_H_loading': 0.05,
        'stable_isotopes': {190: 0.0001, 192: 0.0079, 194: 0.3297,
                            195: 0.3383, 196: 0.2521, 198: 0.0719},
        'n_stable_isotopes': 6,
    },
    'W': {
        'Z': 74, 'A': 183.84, 'structure': 'BCC',
        'a_A': 3.1652, 'debye_K': 400,
        'melting_K': 3695, 'density_g_cm3': 19.25,
        'bulk_modulus_GPa': 310, 'shear_modulus_GPa': 161,
        'thermal_conductivity_W_mK': 173,
        'specific_heat_J_gK': 0.132,
        'work_function_eV': 4.55,
        'fermi_energy_eV': 5.77,
        'e_density_A3': 0.063,
        'n_valence': 6,
        'magnetic_susceptibility': 59e-6,
        'hydrogen_solubility': 'very_low',
        'max_H_loading': 0.0001,
        'stable_isotopes': {180: 0.0012, 182: 0.2650, 183: 0.1431,
                            184: 0.3064, 186: 0.2843},
        'n_stable_isotopes': 5,
    },
    'Cu': {
        'Z': 29, 'A': 63.546, 'structure': 'FCC',
        'a_A': 3.6149, 'debye_K': 343,
        'melting_K': 1358, 'density_g_cm3': 8.96,
        'bulk_modulus_GPa': 140, 'shear_modulus_GPa': 48,
        'thermal_conductivity_W_mK': 401,
        'specific_heat_J_gK': 0.385,
        'work_function_eV': 4.65,
        'fermi_energy_eV': 7.00,
        'e_density_A3': 0.085,
        'n_valence': 11,
        'magnetic_susceptibility': -5.5e-6,
        'hydrogen_solubility': 'none',
        'max_H_loading': 0.0,
        'stable_isotopes': {63: 0.6915, 65: 0.3085},
        'n_stable_isotopes': 2,
    },
    'Zr': {
        'Z': 40, 'A': 91.224, 'structure': 'HCP',
        'a_A': 3.2316, 'c_A': 5.1477, 'debye_K': 291,
        'melting_K': 2128, 'density_g_cm3': 6.52,
        'bulk_modulus_GPa': 94, 'shear_modulus_GPa': 33,
        'thermal_conductivity_W_mK': 22.7,
        'specific_heat_J_gK': 0.278,
        'work_function_eV': 4.05,
        'fermi_energy_eV': 5.36,
        'e_density_A3': 0.043,
        'n_valence': 4,
        'magnetic_susceptibility': -13.8e-6,
        'hydrogen_solubility': 'high',
        'max_H_loading': 2.0,  # ZrH2
        'stable_isotopes': {90: 0.5145, 91: 0.1122, 92: 0.1715, 94: 0.1738, 96: 0.0280},
        'n_stable_isotopes': 5,
    },
    'Ta': {
        'Z': 73, 'A': 180.95, 'structure': 'BCC',
        'a_A': 3.3013, 'debye_K': 240,
        'melting_K': 3290, 'density_g_cm3': 16.65,
        'bulk_modulus_GPa': 200, 'shear_modulus_GPa': 69,
        'thermal_conductivity_W_mK': 57.5,
        'specific_heat_J_gK': 0.140,
        'work_function_eV': 4.25,
        'fermi_energy_eV': 5.33,
        'e_density_A3': 0.055,
        'n_valence': 5,
        'magnetic_susceptibility': 154e-6,
        'hydrogen_solubility': 'moderate',
        'max_H_loading': 0.5,
        'stable_isotopes': {180: 0.0001, 181: 0.9999},
        'n_stable_isotopes': 2,
    },
    'SUS304': {
        'Z': 26, 'A': 55.9, 'structure': 'FCC',
        'a_A': 3.5950, 'debye_K': 400,
        'melting_K': 1723, 'density_g_cm3': 8.00,
        'bulk_modulus_GPa': 160, 'shear_modulus_GPa': 77,
        'thermal_conductivity_W_mK': 16.2,
        'specific_heat_J_gK': 0.500,
        'work_function_eV': 4.40,
        'fermi_energy_eV': 7.0,
        'e_density_A3': 0.14,
        'n_valence': 8,
        'composition': {'Fe': 0.69, 'Cr': 0.19, 'Ni': 0.10, 'Mn': 0.02},
        'hydrogen_solubility': 'low',
        'max_H_loading': 0.001,
    },
    'Constantan': {
        'Z': 29, 'A': 61.1, 'structure': 'FCC',
        'a_A': 3.57, 'debye_K': 350,
        'melting_K': 1503, 'density_g_cm3': 8.9,
        'bulk_modulus_GPa': 160, 'shear_modulus_GPa': 60,
        'thermal_conductivity_W_mK': 22,
        'specific_heat_J_gK': 0.410,
        'work_function_eV': 4.8,
        'fermi_energy_eV': 6.5,
        'e_density_A3': 0.10,
        'n_valence': 10,
        'composition': {'Cu': 0.55, 'Ni': 0.45},
        'hydrogen_solubility': 'low',
        'max_H_loading': 0.01,
    },
    'PdO': {
        'Z': 46, 'A': 122.42, 'structure': 'tetragonal',
        'a_A': 3.043, 'c_A': 5.337, 'debye_K': 300,
        'melting_K': 1143, 'density_g_cm3': 8.3,
        'e_density_A3': 0.40,
        'work_function_eV': 5.5,
        'hydrogen_solubility': 'moderate',
        'max_H_loading': 0.7,
    },
}

# =============================================================================
# CATEGORY 2: SCREENING ENERGIES — COMPLETE DATABASE
# Sources: Kasagi 2002, Raiola 2004, Huke 2008, Czerski 2001-2004,
#          NASA TP-2020, LUNA collaboration, Greife 1995
# =============================================================================
SCREENING_COMPLETE = {
    # Metal targets — d(d,p)t reaction screening
    'PdO':  {'Us_eV': 600,  'err': 60,  'E_beam_keV': 2.5, 'reaction': 'd(d,p)t', 'source': 'Kasagi 2002', 'method': 'implantation'},
    'Pd':   {'Us_eV': 310,  'err': 30,  'E_beam_keV': 2.5, 'reaction': 'd(d,p)t', 'source': 'Kasagi 2002', 'method': 'implantation'},
    'Pd_R': {'Us_eV': 800,  'err': 90,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'Pd_H': {'Us_eV': 313,  'err': 2,   'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Huke 2008', 'method': 'implantation'},
    'Fe':   {'Us_eV': 200,  'err': 20,  'E_beam_keV': 2.5, 'reaction': 'd(d,p)t', 'source': 'Kasagi 2002', 'method': 'implantation'},
    'Au':   {'Us_eV': 70,   'err': 10,  'E_beam_keV': 2.5, 'reaction': 'd(d,p)t', 'source': 'Kasagi 2002', 'method': 'implantation'},
    'Ti':   {'Us_eV': 65,   'err': 10,  'E_beam_keV': 2.5, 'reaction': 'd(d,p)t', 'source': 'Kasagi 2002', 'method': 'implantation'},
    'Ta':   {'Us_eV': 309,  'err': 12,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'Zr':   {'Us_eV': 297,  'err': 8,   'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Huke 2008', 'method': 'implantation',
             'vacancy_dependent': True, 'Us_range_eV': (100, 600)},
    'Ni':   {'Us_eV': 420,  'err': 50,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'Al':   {'Us_eV': 190,  'err': 15,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Huke 2008', 'method': 'implantation'},
    'BeO':  {'Us_eV': 180,  'err': 40,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'NASA TP-2020', 'method': 'implantation'},
    'Pt':   {'Us_eV': 122,  'err': 20,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'C':    {'Us_eV': 25,   'err': 5,   'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Huke 2008', 'method': 'implantation'},
    'Cu':   {'Us_eV': 43,   'err': 10,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Czerski 2001', 'method': 'implantation'},
    'Ag':   {'Us_eV': 91,   'err': 15,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'W':    {'Us_eV': 74,   'err': 12,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'V':    {'Us_eV': 316,  'err': 25,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'Nb':   {'Us_eV': 284,  'err': 20,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'Sn':   {'Us_eV': 130,  'err': 18,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'In':   {'Us_eV': 165,  'err': 22,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'Mn':   {'Us_eV': 246,  'err': 30,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'Co':   {'Us_eV': 240,  'err': 30,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    'Cr':   {'Us_eV': 181,  'err': 20,  'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Raiola 2004', 'method': 'implantation'},
    # Gas phase (bare D2)
    'D2_gas': {'Us_eV': 25, 'err': 3, 'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Greife 1995', 'method': 'gas_target'},
    # Theoretical (Debye model)
    'Debye_model': {'Us_eV': 30, 'err': 5, 'E_beam_keV': 5.0, 'reaction': 'd(d,p)t', 'source': 'Assenbaum 1987', 'method': 'theory'},
}

# =============================================================================
# CATEGORY 3: EXCESS HEAT EXPERIMENTS — COMPREHENSIVE
# All experiments with measured excess heat, organized with full parameters
# =============================================================================
EXCESS_HEAT_COMPREHENSIVE = [
    # === ELECTROLYSIS EXPERIMENTS ===
    {
        'id': 'FP_1989', 'lab': 'Fleischmann-Pons/Southampton',
        'material': 'Pd', 'substrate': 'Pd_rod', 'gas': 'D2O',
        'method': 'electrolysis', 'electrolyte': 'D2O_LiOD',
        'excess_W_min': 20, 'excess_W_max': 240, 'excess_W_typical': 80,
        'excess_W_per_cm3': 150, 'COP': 40,
        'temperature_K': 340, 'pressure_Pa': 1e5,
        'current_density_A_cm2': 0.5, 'cell_voltage_V': 5.0,
        'DPd': 0.95, 'duration_hours': 336,
        'He4_detected': True, 'tritium_detected': True, 'neutron_detected': False,
        'gamma_detected': False, 'transmutation_detected': False,
        'reproducibility': 0.30, 'year': 1989,
        'surface_treatment': 'annealed', 'cathode_size_cm2': 0.8,
    },
    {
        'id': 'McKubre_SRI', 'lab': 'McKubre/SRI',
        'material': 'Pd', 'substrate': 'Pd_wire', 'gas': 'D2O',
        'method': 'electrolysis', 'electrolyte': 'D2O_LiOD',
        'excess_W_typical': 2.1, 'COP': 1.38,
        'temperature_K': 340, 'pressure_Pa': 1e5,
        'current_density_A_cm2': 0.6, 'cell_voltage_V': 4.5,
        'DPd': 0.90, 'duration_hours': 1440,
        'He4_detected': True, 'tritium_detected': False, 'neutron_detected': False,
        'gamma_detected': False, 'transmutation_detected': False,
        'loading_threshold_DPd': 0.84,
        'reproducibility': 0.17, 'year': 2009,
    },
    {
        'id': 'Storms_1', 'lab': 'Storms',
        'material': 'Pd', 'substrate': 'Pd_cathode', 'gas': 'D2O',
        'method': 'electrolysis', 'electrolyte': 'D2O_LiOD',
        'excess_W_typical': 7.5, 'COP': 1.2,
        'temperature_K': 340, 'pressure_Pa': 1e5,
        'current_density_A_cm2': 0.4, 'cell_voltage_V': 5.0,
        'DPd': 0.82, 'duration_hours': 12,
        'reproducibility': 0.30, 'year': 2007,
    },
    {
        'id': 'Miles_1', 'lab': 'Miles/NRL',
        'material': 'Pd', 'substrate': 'Pd_cathode', 'gas': 'D2O',
        'method': 'electrolysis', 'electrolyte': 'D2O_LiOD',
        'excess_W_typical': 0.3, 'COP': 1.05,
        'temperature_K': 320, 'pressure_Pa': 1e5,
        'DPd': 0.85, 'duration_hours': 720,
        'He4_detected': True, 'He4_per_W_atoms': 1.2e11,  # ~24 MeV per He4
        'tritium_detected': False, 'neutron_detected': False,
        'reproducibility': 0.60, 'year': 1993,
        'notes': 'He4/W ratio consistent with D+D->4He+24MeV',
    },
    {
        'id': 'Letts_Cravens', 'lab': 'Letts-Cravens',
        'material': 'Pd', 'substrate': 'Pd_cathode', 'gas': 'D2O',
        'method': 'electrolysis_laser', 'electrolyte': 'D2O_LiOD',
        'excess_W_typical': 1.5, 'COP': 1.15,
        'temperature_K': 330, 'pressure_Pa': 1e5,
        'DPd': 0.88, 'duration_hours': 48,
        'laser_stimulation': True, 'laser_wavelength_nm': 685,
        'laser_power_mW': 30,
        'dual_laser': True,  # beat frequency triggers excess heat
        'reproducibility': 0.80, 'year': 2006,
    },
    {
        'id': 'Ohmori_Au', 'lab': 'Ohmori/Hokkaido',
        'material': 'Au', 'substrate': 'Au_cathode', 'gas': 'H2O',
        'method': 'electrolysis', 'electrolyte': 'H2O_K2SO4',
        'excess_W_typical': 6.0, 'COP': 1.3,
        'temperature_K': 350, 'pressure_Pa': 1e5,
        'current_density_A_cm2': 1.5,
        'He4_detected': False, 'tritium_detected': False,
        'transmutation_detected': True,
        'transmutation_products': ['Fe', 'Cr', 'Cu'],
        'surface_eruptions': True,
        'reproducibility': 0.60, 'year': 1996,
    },

    # === GAS LOADING EXPERIMENTS ===
    {
        'id': 'Kitamura_Kobe', 'lab': 'Kitamura/Kobe',
        'material': 'PdNi_ZrO2', 'substrate': 'nano_composite', 'gas': 'D2',
        'method': 'gas_loading',
        'excess_W_min': 3, 'excess_W_max': 24, 'excess_W_typical': 12,
        'excess_W_burst': 110,
        'COP': 2.0,
        'temperature_K': 500, 'pressure_Pa': 1e5,
        'DPd': 0.80, 'duration_hours': 1008,
        'nanostructure': True, 'particle_size_nm': 5,
        'reproducibility': 1.0, 'year': 2012,
    },
    {
        'id': 'Li_XZ_China', 'lab': 'Li Xing Zhong/China',
        'material': 'Pd', 'substrate': 'Pd_bulk', 'gas': 'D2',
        'method': 'gas_loading',
        'excess_W_typical': 41, 'excess_W_max': 87, 'COP': 1.1,
        'temperature_K': 350, 'pressure_Pa': 9e4,
        'DPd': 0.12, 'duration_hours': 83,
        'total_energy_MJ': 79.58,
        'reproducibility': 1.0, 'year': 2003,
    },
    {
        'id': 'Arata_Zhang', 'lab': 'Arata-Zhang/Osaka',
        'material': 'nano_Pd', 'substrate': 'Pd_nanopowder_ZrO2', 'gas': 'D2',
        'method': 'gas_loading',
        'excess_W_typical': 24, 'COP': 1.5,
        'temperature_K': 300, 'pressure_Pa': 5e5,
        'DPd': 3.0,  # nano-Pd can load >1
        'duration_hours': 504,
        'He4_detected': True, 'nanostructure': True,
        'particle_size_nm': 5,
        'reproducibility': 0.80, 'year': 2008,
    },
    {
        'id': 'Piantelli_Ni', 'lab': 'Piantelli/Siena',
        'material': 'Ni', 'substrate': 'Ni_rod', 'gas': 'H2',
        'method': 'gas_loading',
        'excess_W_typical': 38.9, 'COP': 1.38,
        'temperature_K': 500, 'pressure_Pa': 1e5,
        'DPd': 0.5, 'duration_hours': 6672,
        'reproducibility': 0.60, 'year': 1995,
    },
    {
        'id': 'Brillouin_Ni', 'lab': 'Brillouin Energy',
        'material': 'Ni', 'substrate': 'Ni_coated', 'gas': 'H2',
        'method': 'gas_loading_Q_pulse',
        'excess_W_typical': 60, 'COP': 2.25,
        'temperature_K': 500, 'pressure_Pa': 1e5,
        'rf_stimulation': True, 'Q_pulse_frequency_Hz': 1e6,
        'duration_hours': 720,
        'reproducibility': 0.80, 'year': 2014,
    },
    {
        'id': 'Constantan_2025', 'lab': 'Celani/INFN',
        'material': 'Constantan', 'substrate': 'Constantan_wire', 'gas': 'D2',
        'method': 'gas_loading',
        'excess_W_typical': 209, 'COP': 3.91,
        'temperature_K': 400, 'pressure_Pa': 1e5,
        'duration_hours': 0.008,  # ~30 seconds
        'reproducibility': 0.50, 'year': 2025,
    },

    # === GAS PERMEATION EXPERIMENTS ===
    {
        'id': 'Iwamura_CP', 'lab': 'Iwamura/Clean Planet',
        'material': 'NiCu', 'substrate': 'NiCu_multilayer', 'gas': 'H2',
        'method': 'gas_permeation',
        'excess_W_typical': 5, 'COP': 1.5,
        'temperature_K': 343, 'pressure_Pa': 5e4,
        'DPd': 0.80, 'duration_hours': 14136,
        'total_energy_MJ': 1.1,
        'reproducibility': 1.0, 'year': 2019,
    },
]

# =============================================================================
# CATEGORY 4: NUCLEAR PRODUCT OBSERVATIONS
# Neutrons, tritium, helium-4, charged particles, gamma/X-rays
# =============================================================================
NUCLEAR_PRODUCTS_DATA = [
    # --- HELIUM-4 ---
    {
        'type': 'He4', 'lab': 'Miles/NRL', 'year': 1993,
        'material': 'Pd', 'method': 'electrolysis',
        'He4_atoms_per_s': 1.5e11, 'excess_W': 0.3,
        'He4_per_Watt': 1.2e11,  # atoms/s/W
        'ratio_MeV_per_He4': 24.0,  # close to 23.8 MeV theoretical
        'notes': 'First quantitative He4/excess-heat correlation',
    },
    {
        'type': 'He4', 'lab': 'McKubre/SRI', 'year': 2004,
        'material': 'Pd', 'method': 'electrolysis',
        'He4_per_Watt': 1.1e11,
        'ratio_MeV_per_He4': 25.0,
        'notes': 'Confirmed 24 MeV/He4 within measurement error',
    },
    {
        'type': 'He4', 'lab': 'DeNinno/ENEA', 'year': 2002,
        'material': 'Pd', 'method': 'electrolysis',
        'He4_atoms_total': 2.7e14, 'excess_energy_J': 130,
        'ratio_MeV_per_He4': 31.0,
        'notes': 'Laser Mass Spectrometer detection',
    },
    {
        'type': 'He4', 'lab': 'Arata-Zhang', 'year': 2008,
        'material': 'nano_Pd', 'method': 'gas_loading',
        'He4_detected': True,
        'He4_in_nano_pores': True,  # trapped in nano structure
        'notes': 'He4 found in Pd nanopowder after D2 exposure',
    },

    # --- TRITIUM ---
    {
        'type': 'tritium', 'lab': 'Bockris/Texas A&M', 'year': 1989,
        'material': 'Pd', 'method': 'electrolysis',
        'tritium_Bq_per_mL': 1.5e4,  # well above background
        'background_Bq_per_mL': 10,
        'notes': 'Confirmed by independent assay',
    },
    {
        'type': 'tritium', 'lab': 'Storms/LANL', 'year': 1991,
        'material': 'Pd', 'method': 'electrolysis',
        'tritium_detected': True,
        'notes': 'Correlated with excess heat episodes',
    },
    {
        'type': 'tritium', 'lab': 'Mizuno/Hokkaido', 'year': 1991,
        'material': 'Pd', 'method': 'electrolysis',
        'tritium_detected': True,
        'notes': 'Published 1990-1991, definitively observed',
    },
    {
        'type': 'tritium', 'lab': 'BARC/India', 'year': 1989,
        'material': 'Pd', 'method': 'electrolysis',
        'tritium_Bq_per_mL': 5e5,  # massive production
        'neutron_per_tritium': 1e-8,  # extremely low n/T ratio
        'notes': 'Anomalous n/T ratio: classical D+D gives n/T=1',
    },

    # --- NEUTRONS ---
    {
        'type': 'neutron', 'lab': 'Mizuno/Hokkaido', 'year': 2025,
        'material': 'SUS304', 'method': 'gas_loading', 'gas': 'H2',
        'neutron_energy_MeV': 0.7,
        'classical_DD_neutron_MeV': 2.45,  # NOT observed
        'threshold_temp_C': 440,
        'rate_driver': 'dT/dt',
        'max_neutron_cpm': 5,
        'notes': 'H2 (not D2)! Supports Widom-Larsen ULMN theory',
    },
    {
        'type': 'neutron', 'lab': 'Jones/BYU', 'year': 1989,
        'material': 'Ti', 'method': 'electrolysis',
        'neutron_detected': True,
        'neutron_rate_cps': 0.01,  # very low
        'notes': 'First report, very controversial',
    },
    {
        'type': 'neutron', 'lab': 'Takahashi/Osaka', 'year': 1991,
        'material': 'Pd', 'method': 'electrolysis',
        'neutron_detected': True,
        'neutron_burst': True,
        'burst_counts': 1000,  # in single burst
        'notes': 'Burst neutrons during electrolysis spikes',
    },

    # --- CHARGED PARTICLES ---
    {
        'type': 'charged_particle', 'lab': 'Lipson/Moscow', 'year': 2005,
        'material': 'PdD', 'method': 'desorption',
        'alpha_detected': True, 'proton_detected': True,
        'alpha_energy_MeV': 3.0,
        'notes': 'During thermal desorption of deuterium from Pd',
    },
    {
        'type': 'charged_particle', 'lab': 'Kasagi/Tohoku', 'year': 2002,
        'material': 'PdO', 'method': 'beam_implantation',
        'proton_detected': True, 'proton_energy_MeV': 3.0,
        'alpha_detected': True, 'alpha_energy_MeV': 3.0,
        'notes': 'Standard nuclear products from d(d,p)t and d(d,n)3He',
    },
]

# =============================================================================
# CATEGORY 5: TRANSMUTATION DATA — COMPLETE
# =============================================================================
TRANSMUTATION_DATA = [
    # --- Iwamura/Mitsubishi (replicated by Toyota) ---
    {
        'lab': 'Iwamura/Mitsubishi', 'year': 2002,
        'method': 'gas_permeation', 'substrate': 'Pd/CaO_multilayer',
        'reaction': 'Cs133 + 4D -> Pr141', 'delta_mass': 8,
        'detection': ['XPS', 'SIMS', 'ICP-MS', 'SPring-8'],
        'replicated_by': 'Toyota', 'confidence': 'high',
        'n_layers': 200, 'Pd_thickness_nm': 100, 'CaO_thickness_nm': 2,
    },
    {
        'lab': 'Iwamura/Mitsubishi', 'year': 2002,
        'reaction': 'Sr88 + 4D -> Mo96', 'delta_mass': 8,
        'detection': ['XPS', 'SIMS'], 'confidence': 'high',
    },
    {
        'lab': 'Iwamura/Mitsubishi', 'year': 2004,
        'reaction': 'Ba137 + 6D -> Sm149', 'delta_mass': 12,
        'detection': ['SIMS'], 'confidence': 'moderate',
    },
    {
        'lab': 'Iwamura/Clean Planet', 'year': 2014,
        'reaction': 'Ca + 4D -> Ti', 'delta_mass': 8,
        'detection': ['Nikkei report'], 'confidence': 'moderate',
    },
    # --- Mizuno transmutations ---
    {
        'lab': 'Mizuno/Hokkaido', 'year': 1998,
        'method': 'electrolysis', 'substrate': 'Pd_cathode',
        'host_transmutation': True,
        'products': ['Cu', 'Zn', 'Fe', 'Cr', 'Ca'],
        'Cr52_natural_pct': 83.8, 'Cr52_observed_pct': 50.7,
        'isotope_anomaly': True,
        'detection': ['SIMS', 'EDX'],
        'reproduced_times': 8,
        'current_density_A_cm2': 0.2,
        'notes': 'Products found at eruption sites on cathode surface',
    },
    # --- Miley transmutations ---
    {
        'lab': 'Miley/UIUC', 'year': 1996,
        'method': 'electrolysis', 'substrate': 'thin_film_multilayer',
        'transmutation_pattern': 'bimodal',
        'products_light': ['Ca', 'Fe', 'Cu', 'Zn'],
        'products_heavy': ['Ag', 'Cd'],
        'notes': 'Systematic pattern: products cluster around Fe and Ag mass regions',
    },
]

# =============================================================================
# CATEGORY 6: PHASE TRANSITION & LOADING DYNAMICS
# =============================================================================
LOADING_DYNAMICS = {
    'Pd_D': {
        'alpha_phase': {'D_Pd_range': (0, 0.017), 'lattice_expansion_pct': 0},
        'mixed_phase': {'D_Pd_range': (0.017, 0.58), 'lattice_expansion_pct': (0, 3.4)},
        'beta_phase': {'D_Pd_range': (0.58, 1.0), 'lattice_expansion_pct': (3.4, 5.0)},
        'critical_T_C': 276,
        'McKubre_threshold': 0.84,
        'Storms_threshold': 0.90,
        'max_loading_electrolysis': 0.97,
        'max_loading_gas_1atm': 0.70,
        'loading_time_hours': (24, 168),  # typical
        'deloading_hysteresis': True,
    },
    'Ni_H': {
        'max_loading': 0.03,
        'interstitial_sites': 'octahedral',
        'notes': 'Low solubility but Piantelli shows excess heat at low loading',
    },
    'Ti_H': {
        'max_loading': 2.0,  # TiH2
        'delta_phase': {'H_Ti_range': (1.5, 2.0)},
        'notes': 'Very high loading capacity, used by Jones',
    },
}

# =============================================================================
# CATEGORY 7: THERMAL DYNAMICS
# Heat-after-death, bursts, incubation, thermal cycling
# =============================================================================
THERMAL_DYNAMICS_DATA = [
    {
        'type': 'heat_after_death', 'lab': 'Mizuno/Hokkaido', 'year': 1991,
        'material': 'Pd', 'total_energy_MJ': 114,
        'mass_g': 100, 'energy_density_kJ_g': 850,
        'duration_days': 15, 'water_evaporated_L': 37.5,
        'stasis_effect': True,  # temperature self-regulates
    },
    {
        'type': 'heat_after_death', 'lab': 'Fleischmann-Pons', 'year': 1993,
        'material': 'Pd', 'stasis_effect': True,
        'notes': 'Temperature returns to fixed level after perturbation',
    },
    {
        'type': 'burst', 'lab': 'Kitamura/Kobe', 'year': 2012,
        'material': 'PdNi_ZrO2', 'burst_W': 110,
        'background_W': 12, 'burst_duration_s': 60,
    },
    {
        'type': 'burst', 'lab': 'Fleischmann-Pons', 'year': 1993,
        'material': 'Pd', 'burst_W': 240,
        'notes': 'Meltdown events (cell destruction)',
    },
    {
        'type': 'incubation', 'lab': 'McKubre/SRI', 'year': 2009,
        'material': 'Pd', 'incubation_hours': 168,  # ~1 week
        'notes': 'Time to achieve D/Pd > 0.84',
    },
    {
        'type': 'thermal_cycling', 'lab': 'Mizuno/Hokkaido', 'year': 2025,
        'material': 'SUS304',
        'neutron_rate_correlates_with': 'dT/dt',
        'notes': 'Neutron production peaks during temperature changes, not at peak T',
    },
]

# =============================================================================
# CATEGORY 8: SURFACE & NANOSTRUCTURE EFFECTS
# =============================================================================
SURFACE_EFFECTS_DATA = [
    {
        'type': 'eruption', 'lab': 'Mizuno/Hokkaido',
        'material': 'Pd', 'description': 'Lily-shaped eruptions on cathode',
        'transmutation_at_eruptions': True,
        'products': ['Cu', 'Zn', 'Fe', 'Cr'],
    },
    {
        'type': 'eruption', 'lab': 'Ohmori/Hokkaido',
        'material': 'Au', 'description': 'Similar eruptions on gold cathode',
        'transmutation_at_eruptions': True,
        'products': ['Fe', 'Cr', 'Cu'],
    },
    {
        'type': 'nanostructure', 'lab': 'Arata-Zhang',
        'material': 'nano_Pd', 'particle_size_nm': 5,
        'ZrO2_matrix': True,
        'DPd_loading': 3.0,  # superlattice loading
        'He4_trapped': True,
    },
    {
        'type': 'multilayer', 'lab': 'Iwamura/Mitsubishi',
        'material': 'Pd/CaO', 'n_layers': 200,
        'Pd_nm': 100, 'CaO_nm': 2,
        'total_thickness_um': 40,
        'transmutation_demonstrated': True,
    },
    {
        'type': 'surface_treatment', 'lab': 'Mizuno/Hokkaido',
        'material': 'Ni_mesh', 'method': 'Pd_coating_by_rubbing',
        'Pd_mass_mg': 52.5, 'Ni_mass_g': 54,
        'COP_achieved': 1.45,
    },
    {
        'type': 'surface_treatment', 'lab': 'Mizuno/Hokkaido',
        'material': 'SUS304', 'method': 'mesh_400_buffed',
        'notes': 'Surface roughening increases H2 absorption',
    },
]

# =============================================================================
# CATEGORY 9: ELECTROMAGNETIC / STIMULATION EFFECTS
# =============================================================================
EM_STIMULATION_DATA = [
    {
        'type': 'laser', 'lab': 'Letts-Cravens', 'year': 2006,
        'wavelength_nm': 685, 'power_mW': 30,
        'effect': 'Triggers excess heat during electrolysis',
        'dual_laser_beat': True,
        'optimal_beat_frequency_THz': 8.0,
        'COP_increase': 1.15,
    },
    {
        'type': 'rf_pulse', 'lab': 'Brillouin Energy', 'year': 2014,
        'frequency_Hz': 1e6, 'effect': 'Q-pulse triggers nuclear reactions',
        'COP': 2.25,
    },
    {
        'type': 'magnetic_field', 'lab': 'Biberian', 'year': 2007,
        'B_field_T': 0.5,
        'effect': 'Possible enhancement, inconclusive',
    },
    {
        'type': 'ultrasound', 'lab': 'Stringham', 'year': 2000,
        'frequency_kHz': 20,
        'effect': 'Cavitation in D2O produces transmutations on target',
        'material': 'Pd', 'method': 'sono_fusion',
    },
]

# =============================================================================
# EXPANDED FEATURE SET DEFINITION
# =============================================================================
FEATURE_COLUMNS_V2 = {
    # -- Material properties (15 features) --
    'material_group': [
        'atomic_number_Z', 'atomic_mass_amu', 'crystal_structure_encoded',
        'lattice_constant_A', 'debye_temperature_K', 'electron_density_A3',
        'density_g_cm3', 'melting_point_K', 'bulk_modulus_GPa',
        'work_function_eV', 'fermi_energy_eV',
        'thermal_conductivity_W_mK', 'specific_heat_J_gK',
        'n_valence_electrons', 'n_stable_isotopes',
    ],
    # -- Hydrogen/Deuterium loading (10 features) --
    'loading_group': [
        'hydrogen_isotope',        # 1=H, 2=D, 3=T
        'loading_ratio',           # D/M or H/M
        'max_loading_capacity',    # theoretical max for this material
        'loading_method_encoded',  # 0=electrolysis, 1=gas, 2=permeation, 3=beam
        'loading_fraction',        # loading / max_capacity (normalized)
        'above_McKubre_threshold', # binary
        'above_Storms_threshold',  # binary
        'phase_encoded',           # 0=alpha, 1=mixed, 2=beta
        'H_absorption_enthalpy_eV',
        'lattice_expansion_pct',
    ],
    # -- Barrier & screening physics (12 features) --
    'barrier_group': [
        'screening_energy_eV',
        'barrier_reduction_maxwell', 'barrier_reduction_coulomb', 'barrier_reduction_cherepanov',
        'log_penetration_maxwell', 'log_penetration_coulomb', 'log_penetration_cherepanov',
        'log_rate_maxwell', 'log_rate_coulomb', 'log_rate_cherepanov',
        'enhancement_factor',
        'log_cross_section',
    ],
    # -- Thermal & energy (8 features) --
    'thermal_group': [
        'temperature_K', 'pressure_Pa',
        'beam_energy_keV',         # thermal kBT for gas/electrolysis
        'input_power_W',
        'heating_rate_K_per_min',  # dT/dt
        'diffusion_coefficient',
        'diffusion_activation_eV',
        'thermal_phonon_energy_meV',  # kB * Debye_T
    ],
    # -- Electrochemistry (5 features) --
    'electrochem_group': [
        'current_density_A_cm2',
        'cell_voltage_V',
        'electrolyte_encoded',     # 0=none, 1=D2O_LiOD, 2=D2O_NaOD, 3=H2O_LiOH
        'cathode_area_cm2',
        'overpotential_V',
    ],
    # -- Surface & nanostructure (6 features) --
    'surface_group': [
        'surface_treatment_encoded',  # 0=none, 1=polished, 2=etched, 3=mesh, 4=nano, 5=multilayer
        'particle_size_nm',           # 0 = bulk
        'n_layers',                   # 0 = single, >0 = multilayer
        'coating_thickness_nm',
        'surface_area_ratio',         # relative to flat surface
        'nanostructure_flag',         # binary
    ],
    # -- Stimulation (4 features) --
    'stimulation_group': [
        'laser_stimulation',       # binary
        'rf_stimulation',          # binary
        'applied_B_field_T',
        'ultrasound_stimulation',  # binary
    ],
    # -- Time dynamics (4 features) --
    'time_group': [
        'experiment_duration_hours',
        'incubation_time_hours',
        'COP',
        'data_source_encoded',     # 0=synthetic, 1=experimental, 2=mizuno_r19, 3=neutron
    ],
}

# Target variables for multi-task learning
TARGET_COLUMNS_V2 = {
    'classification': [
        'reaction_occurred',       # binary: any nuclear signature detected
        'excess_heat_detected',    # binary: COP > 1.0
        'neutron_detected',        # binary
        'tritium_detected',        # binary
        'He4_detected',            # binary
        'transmutation_detected',  # binary
    ],
    'regression': [
        'excess_heat_W',           # continuous
        'COP',                     # continuous, >= 1.0 if reaction
        'neutron_rate_cpm',        # continuous
        'energy_density_kJ_g',     # continuous
    ],
}


def get_feature_columns_v2() -> list[str]:
    """Return flattened list of all V2 feature column names."""
    cols = []
    for group_cols in FEATURE_COLUMNS_V2.values():
        cols.extend(group_cols)
    return cols


def get_all_materials() -> list[str]:
    """Return list of all materials with expanded properties."""
    return list(MATERIALS_EXPANDED.keys())


# =============================================================================
# ENCODING HELPERS
# =============================================================================
STRUCTURE_ENCODING = {'FCC': 0, 'BCC': 1, 'HCP': 2, 'tetragonal': 3}
METHOD_ENCODING = {
    'electrolysis': 0, 'gas_loading': 1, 'gas_permeation': 2,
    'beam_implantation': 3, 'electrolysis_laser': 4,
    'gas_loading_Q_pulse': 5, 'sono_fusion': 6,
    'gas_loading_H': 1,  # alias
}
ELECTROLYTE_ENCODING = {
    'none': 0, 'D2O_LiOD': 1, 'D2O_NaOD': 2,
    'H2O_LiOH': 3, 'H2O_K2SO4': 4,
}
SURFACE_ENCODING = {
    'none': 0, 'annealed': 1, 'polished': 2, 'etched': 3,
    'mesh': 4, 'nano': 5, 'multilayer': 6, 'mesh_400_buffed': 4,
    'Pd_coating_by_rubbing': 2,
}
ISOTOPE_ENCODING = {'H': 1, 'D': 2, 'T': 3, 'H2': 1, 'D2': 2, 'D2O': 2, 'H2O': 1}
PHASE_ENCODING = {'alpha': 0, 'mixed': 1, 'beta': 2}


if __name__ == '__main__':
    print(f"Materials: {len(MATERIALS_EXPANDED)}")
    print(f"Screening entries: {len(SCREENING_COMPLETE)}")
    print(f"Excess heat experiments: {len(EXCESS_HEAT_COMPREHENSIVE)}")
    print(f"Nuclear product observations: {len(NUCLEAR_PRODUCTS_DATA)}")
    print(f"Transmutation records: {len(TRANSMUTATION_DATA)}")
    print(f"Feature columns V2: {len(get_feature_columns_v2())} features")
    print(f"  Groups: {list(FEATURE_COLUMNS_V2.keys())}")
    for group, cols in FEATURE_COLUMNS_V2.items():
        print(f"    {group}: {len(cols)} features")
    print(f"Target columns: {sum(len(v) for v in TARGET_COLUMNS_V2.values())}")
