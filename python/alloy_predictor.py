"""
LENR Alloy & Composite Predictor
=================================

Predicts properties and LENR activity for:
- Binary alloys (Pd-Ni, Cu-Ni, Fe-Cr, etc.)
- Ternary alloys (Pd-Ni-Cu, Fe-Cr-Ni, etc.)
- Metal-oxide composites (Pd/CeO2, Ni/ZrO2, etc.)
- Multilayer structures (Pd/CaO, Pd/Ni/Cu)

Uses Vegard's law, rule of mixtures, and physics-aware
corrections for magnetic, electronic, and lattice properties.

Key innovation: predicts LENR screening energy, Cherepanov
medium resistance, and reaction probability for UNTESTED
material combinations.

Author: LENR ML Project
Date: 2026-03
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
import itertools
import warnings

logger = logging.getLogger(__name__)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class AlloyComponent:
    """Single component of an alloy."""
    symbol: str
    fraction: float  # molar/atomic fraction (0-1)

    def __post_init__(self):
        if not 0 < self.fraction <= 1.0:
            raise ValueError(f"Fraction must be (0, 1], got {self.fraction}")


@dataclass
class AlloyProperties:
    """Predicted properties of an alloy or composite."""
    name: str
    components: List[AlloyComponent]

    # Basic
    Z_eff: float = 0.0
    A_eff: float = 0.0
    density_g_cm3: float = 0.0

    # Lattice
    a_eff_A: float = 0.0
    structure: str = 'FCC'
    debye_K: float = 0.0

    # Electronic
    e_density_A3: float = 0.0
    work_function_eV: float = 0.0
    fermi_energy_eV: float = 0.0
    n_valence_eff: float = 0.0

    # Magnetic (KEY for Cherepanov)
    chi_m_eff: float = 0.0
    magnetic_class: str = 'diamagnetic'

    # Thermal
    melting_K: float = 0.0
    thermal_conductivity: float = 0.0
    bulk_modulus_GPa: float = 0.0

    # Hydrogen/Deuterium
    max_loading: float = 0.0
    D0_eff: float = 0.0
    Ea_diffusion_eV: float = 0.0
    H_absorption_eV: float = 0.0

    # LENR predictions
    predicted_screening_eV: float = 0.0
    predicted_medium_resistance: float = 0.0
    predicted_reaction_probability: float = 0.0
    predicted_excess_heat_W: float = 0.0
    predicted_COP: float = 0.0

    # Confidence
    data_quality: str = 'predicted'  # 'measured', 'interpolated', 'predicted'
    confidence: float = 0.0  # 0-1

    # Notes
    notes: str = ''


@dataclass
class PredictionResult:
    """Result of LENR prediction for a material."""
    material_name: str
    components: str

    # Scores (0-100)
    lenr_score: float = 0.0           # Overall LENR potential
    screening_score: float = 0.0      # Screening energy potential
    loading_score: float = 0.0        # Hydrogen loading capacity
    magnetic_score: float = 0.0       # Magnetic focusing (Cherepanov)
    defect_score: float = 0.0         # Defect channel potential
    thermal_score: float = 0.0        # Thermal stability

    # Predicted values
    predicted_Us_eV: float = 0.0
    predicted_excess_W: float = 0.0
    predicted_COP: float = 1.0

    # Risk/confidence
    confidence: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    advantages: List[str] = field(default_factory=list)

    # Full alloy properties (for downstream use)
    alloy_properties: Optional['AlloyProperties'] = None

    def __repr__(self):
        return (f"PredictionResult({self.material_name}: "
                f"LENR={self.lenr_score:.0f}/100, "
                f"Us={self.predicted_Us_eV:.0f} eV, "
                f"COP={self.predicted_COP:.2f})")


# ============================================================
# COMPREHENSIVE MATERIAL DATABASE
# ============================================================

ELEMENT_DB = {
    # Format: symbol -> {properties}
    # All values from standard materials science references

    'Pd': {
        'Z': 46, 'A': 106.42, 'structure': 'FCC',
        'a_A': 3.8907, 'debye_K': 274, 'melting_K': 1828,
        'density': 12.023, 'bulk_mod': 180, 'shear_mod': 44,
        'thermal_k': 71.8, 'specific_heat': 0.244,
        'work_fn': 5.12, 'fermi_E': 5.56, 'e_density': 0.34,
        'n_valence': 10, 'chi_m': -7.2e-6,
        'max_loading': 1.0, 'D0': 2.0e-3, 'Ea_diff': 0.230,
        'H_enthalpy': -0.197, 'Us_measured': 310,
    },
    'Ni': {
        'Z': 28, 'A': 58.69, 'structure': 'FCC',
        'a_A': 3.5240, 'debye_K': 450, 'melting_K': 1728,
        'density': 8.908, 'bulk_mod': 180, 'shear_mod': 76,
        'thermal_k': 90.9, 'specific_heat': 0.444,
        'work_fn': 5.15, 'fermi_E': 7.04, 'e_density': 0.40,
        'n_valence': 10, 'chi_m': 600e-6,
        'max_loading': 0.03, 'D0': 2.4e-2, 'Ea_diff': 0.457,
        'H_enthalpy': -0.16, 'Us_measured': 420,
    },
    'Ti': {
        'Z': 22, 'A': 47.87, 'structure': 'HCP',
        'a_A': 2.9508, 'debye_K': 420, 'melting_K': 1941,
        'density': 4.507, 'bulk_mod': 110, 'shear_mod': 44,
        'thermal_k': 21.9, 'specific_heat': 0.523,
        'work_fn': 4.33, 'fermi_E': 5.23, 'e_density': 0.28,
        'n_valence': 4, 'chi_m': 153e-6,
        'max_loading': 2.0, 'D0': 2.0e-3, 'Ea_diff': 0.34,
        'H_enthalpy': -0.55, 'Us_measured': 65,
    },
    'Fe': {
        'Z': 26, 'A': 55.845, 'structure': 'BCC',
        'a_A': 2.8665, 'debye_K': 470, 'melting_K': 1811,
        'density': 7.874, 'bulk_mod': 170, 'shear_mod': 82,
        'thermal_k': 80.4, 'specific_heat': 0.449,
        'work_fn': 4.50, 'fermi_E': 11.1, 'e_density': 0.42,
        'n_valence': 8, 'chi_m': 1.0,
        'max_loading': 0.0001, 'D0': 7.4e-4, 'Ea_diff': 0.041,
        'H_enthalpy': 0.29, 'Us_measured': 200,
    },
    'Au': {
        'Z': 79, 'A': 196.97, 'structure': 'FCC',
        'a_A': 4.0782, 'debye_K': 165, 'melting_K': 1337,
        'density': 19.32, 'bulk_mod': 220, 'shear_mod': 27,
        'thermal_k': 318, 'specific_heat': 0.129,
        'work_fn': 5.10, 'fermi_E': 5.53, 'e_density': 0.29,
        'n_valence': 11, 'chi_m': -2.8e-6,
        'max_loading': 0.0, 'D0': 1e-5, 'Ea_diff': 0.85,
        'H_enthalpy': 0.84, 'Us_measured': 70,
    },
    'Pt': {
        'Z': 78, 'A': 195.08, 'structure': 'FCC',
        'a_A': 3.9242, 'debye_K': 240, 'melting_K': 2041,
        'density': 21.45, 'bulk_mod': 230, 'shear_mod': 61,
        'thermal_k': 71.6, 'specific_heat': 0.133,
        'work_fn': 5.65, 'fermi_E': 5.93, 'e_density': 0.33,
        'n_valence': 10, 'chi_m': 193e-6,
        'max_loading': 0.02, 'D0': 1e-4, 'Ea_diff': 0.26,
        'H_enthalpy': 0.40, 'Us_measured': 122,
    },
    'W': {
        'Z': 74, 'A': 183.84, 'structure': 'BCC',
        'a_A': 3.1652, 'debye_K': 400, 'melting_K': 3695,
        'density': 19.25, 'bulk_mod': 310, 'shear_mod': 161,
        'thermal_k': 173, 'specific_heat': 0.132,
        'work_fn': 4.55, 'fermi_E': 5.77, 'e_density': 0.31,
        'n_valence': 6, 'chi_m': 59e-6,
        'max_loading': 0.0001, 'D0': 1e-4, 'Ea_diff': 0.39,
        'H_enthalpy': 1.04, 'Us_measured': 74,
    },
    'Cu': {
        'Z': 29, 'A': 63.546, 'structure': 'FCC',
        'a_A': 3.6149, 'debye_K': 343, 'melting_K': 1358,
        'density': 8.96, 'bulk_mod': 140, 'shear_mod': 48,
        'thermal_k': 401, 'specific_heat': 0.385,
        'work_fn': 4.65, 'fermi_E': 7.00, 'e_density': 0.42,
        'n_valence': 11, 'chi_m': -5.5e-6,
        'max_loading': 0.0, 'D0': 1e-5, 'Ea_diff': 0.40,
        'H_enthalpy': 0.44, 'Us_measured': 45,
    },
    'Zr': {
        'Z': 40, 'A': 91.224, 'structure': 'HCP',
        'a_A': 3.2320, 'debye_K': 291, 'melting_K': 2128,
        'density': 6.506, 'bulk_mod': 94, 'shear_mod': 33,
        'thermal_k': 22.6, 'specific_heat': 0.278,
        'work_fn': 4.05, 'fermi_E': 5.03, 'e_density': 0.26,
        'n_valence': 4, 'chi_m': -13e-6,
        'max_loading': 2.0, 'D0': 1e-3, 'Ea_diff': 0.45,
        'H_enthalpy': -0.57, 'Us_measured': 297,
    },
    'Ta': {
        'Z': 73, 'A': 180.95, 'structure': 'BCC',
        'a_A': 3.3058, 'debye_K': 240, 'melting_K': 3290,
        'density': 16.65, 'bulk_mod': 200, 'shear_mod': 69,
        'thermal_k': 57.5, 'specific_heat': 0.140,
        'work_fn': 4.25, 'fermi_E': 5.30, 'e_density': 0.28,
        'n_valence': 5, 'chi_m': 154e-6,
        'max_loading': 0.50, 'D0': 4.4e-4, 'Ea_diff': 0.14,
        'H_enthalpy': -0.36, 'Us_measured': 309,
    },
    'Al': {
        'Z': 13, 'A': 26.98, 'structure': 'FCC',
        'a_A': 4.0495, 'debye_K': 428, 'melting_K': 933,
        'density': 2.70, 'bulk_mod': 76, 'shear_mod': 26,
        'thermal_k': 237, 'specific_heat': 0.897,
        'work_fn': 4.28, 'fermi_E': 11.7, 'e_density': 0.27,
        'n_valence': 3, 'chi_m': 16.5e-6,
        'max_loading': 0.0001, 'D0': 1e-5, 'Ea_diff': 0.52,
        'H_enthalpy': 0.62, 'Us_measured': 190,
    },
    'Be': {
        'Z': 4, 'A': 9.012, 'structure': 'HCP',
        'a_A': 2.2860, 'debye_K': 1440, 'melting_K': 1560,
        'density': 1.85, 'bulk_mod': 130, 'shear_mod': 132,
        'thermal_k': 200, 'specific_heat': 1.825,
        'work_fn': 4.98, 'fermi_E': 14.3, 'e_density': 0.49,
        'n_valence': 2, 'chi_m': -9.0e-6,
        'max_loading': 0.0, 'D0': 1e-6, 'Ea_diff': 0.70,
        'H_enthalpy': 0.50, 'Us_measured': 180,
    },
    'Co': {
        'Z': 27, 'A': 58.93, 'structure': 'HCP',
        'a_A': 2.5071, 'debye_K': 445, 'melting_K': 1768,
        'density': 8.90, 'bulk_mod': 180, 'shear_mod': 75,
        'thermal_k': 100, 'specific_heat': 0.421,
        'work_fn': 5.00, 'fermi_E': 7.10, 'e_density': 0.41,
        'n_valence': 9, 'chi_m': 250e-6,
        'max_loading': 0.001, 'D0': 1e-3, 'Ea_diff': 0.38,
        'H_enthalpy': 0.30, 'Us_measured': 150,
    },
    'Cr': {
        'Z': 24, 'A': 52.00, 'structure': 'BCC',
        'a_A': 2.8845, 'debye_K': 630, 'melting_K': 2180,
        'density': 7.19, 'bulk_mod': 160, 'shear_mod': 115,
        'thermal_k': 93.9, 'specific_heat': 0.449,
        'work_fn': 4.50, 'fermi_E': 7.23, 'e_density': 0.38,
        'n_valence': 6, 'chi_m': 180e-6,
        'max_loading': 0.001, 'D0': 1e-4, 'Ea_diff': 0.25,
        'H_enthalpy': 0.61, 'Us_measured': 80,
    },
    'Mn': {
        'Z': 25, 'A': 54.94, 'structure': 'BCC',
        'a_A': 8.9125, 'debye_K': 410, 'melting_K': 1519,
        'density': 7.21, 'bulk_mod': 120, 'shear_mod': 80,
        'thermal_k': 7.81, 'specific_heat': 0.479,
        'work_fn': 4.10, 'fermi_E': 6.80, 'e_density': 0.35,
        'n_valence': 7, 'chi_m': 529e-6,
        'max_loading': 0.001, 'D0': 1e-4, 'Ea_diff': 0.35,
        'H_enthalpy': 0.15, 'Us_measured': None,
    },
    'V': {
        'Z': 23, 'A': 50.94, 'structure': 'BCC',
        'a_A': 3.0240, 'debye_K': 380, 'melting_K': 2183,
        'density': 6.11, 'bulk_mod': 160, 'shear_mod': 47,
        'thermal_k': 30.7, 'specific_heat': 0.489,
        'work_fn': 4.30, 'fermi_E': 5.80, 'e_density': 0.30,
        'n_valence': 5, 'chi_m': 255e-6,
        'max_loading': 0.50, 'D0': 3.0e-4, 'Ea_diff': 0.045,
        'H_enthalpy': -0.33, 'Us_measured': 120,
    },
    'Nb': {
        'Z': 41, 'A': 92.91, 'structure': 'BCC',
        'a_A': 3.3007, 'debye_K': 275, 'melting_K': 2750,
        'density': 8.57, 'bulk_mod': 170, 'shear_mod': 38,
        'thermal_k': 53.7, 'specific_heat': 0.265,
        'work_fn': 4.30, 'fermi_E': 5.32, 'e_density': 0.28,
        'n_valence': 5, 'chi_m': 195e-6,
        'max_loading': 0.80, 'D0': 5.0e-4, 'Ea_diff': 0.106,
        'H_enthalpy': -0.35, 'Us_measured': 90,
    },
    'Mo': {
        'Z': 42, 'A': 95.95, 'structure': 'BCC',
        'a_A': 3.1470, 'debye_K': 450, 'melting_K': 2896,
        'density': 10.28, 'bulk_mod': 230, 'shear_mod': 120,
        'thermal_k': 138, 'specific_heat': 0.251,
        'work_fn': 4.60, 'fermi_E': 6.82, 'e_density': 0.35,
        'n_valence': 6, 'chi_m': 72e-6,
        'max_loading': 0.001, 'D0': 1e-4, 'Ea_diff': 0.25,
        'H_enthalpy': 0.28, 'Us_measured': None,
    },
    'Ag': {
        'Z': 47, 'A': 107.87, 'structure': 'FCC',
        'a_A': 4.0862, 'debye_K': 225, 'melting_K': 1235,
        'density': 10.50, 'bulk_mod': 100, 'shear_mod': 30,
        'thermal_k': 429, 'specific_heat': 0.235,
        'work_fn': 4.26, 'fermi_E': 5.49, 'e_density': 0.29,
        'n_valence': 11, 'chi_m': -19.5e-6,
        'max_loading': 0.001, 'D0': 1e-5, 'Ea_diff': 0.39,
        'H_enthalpy': 0.68, 'Us_measured': 52,
    },
    'Sn': {
        'Z': 50, 'A': 118.71, 'structure': 'tetragonal',
        'a_A': 5.8318, 'debye_K': 200, 'melting_K': 505,
        'density': 7.27, 'bulk_mod': 58, 'shear_mod': 18,
        'thermal_k': 66.8, 'specific_heat': 0.228,
        'work_fn': 4.42, 'fermi_E': 10.2, 'e_density': 0.37,
        'n_valence': 4, 'chi_m': -24.0e-6,
        'max_loading': 0.0, 'D0': 1e-6, 'Ea_diff': 0.50,
        'H_enthalpy': 0.30, 'Us_measured': 120,
    },
    'Pb': {
        'Z': 82, 'A': 207.2, 'structure': 'FCC',
        'a_A': 4.9502, 'debye_K': 105, 'melting_K': 601,
        'density': 11.34, 'bulk_mod': 46, 'shear_mod': 5.6,
        'thermal_k': 35.3, 'specific_heat': 0.129,
        'work_fn': 4.25, 'fermi_E': 9.47, 'e_density': 0.33,
        'n_valence': 4, 'chi_m': -15.5e-6,
        'max_loading': 0.0, 'D0': 1e-6, 'Ea_diff': 0.60,
        'H_enthalpy': 0.50, 'Us_measured': 60,
    },
    'Hf': {
        'Z': 72, 'A': 178.49, 'structure': 'HCP',
        'a_A': 3.1946, 'debye_K': 252, 'melting_K': 2506,
        'density': 13.31, 'bulk_mod': 110, 'shear_mod': 30,
        'thermal_k': 23, 'specific_heat': 0.144,
        'work_fn': 3.90, 'fermi_E': 5.10, 'e_density': 0.25,
        'n_valence': 4, 'chi_m': 75e-6,
        'max_loading': 1.80, 'D0': 1e-3, 'Ea_diff': 0.50,
        'H_enthalpy': -0.60, 'Us_measured': None,
    },
    'Y': {
        'Z': 39, 'A': 88.91, 'structure': 'HCP',
        'a_A': 3.6482, 'debye_K': 280, 'melting_K': 1799,
        'density': 4.47, 'bulk_mod': 41, 'shear_mod': 26,
        'thermal_k': 17.2, 'specific_heat': 0.298,
        'work_fn': 3.10, 'fermi_E': 4.80, 'e_density': 0.22,
        'n_valence': 3, 'chi_m': 187e-6,
        'max_loading': 2.0, 'D0': 1e-3, 'Ea_diff': 0.40,
        'H_enthalpy': -0.90, 'Us_measured': None,
    },
    'Ce': {
        'Z': 58, 'A': 140.12, 'structure': 'FCC',
        'a_A': 5.1610, 'debye_K': 179, 'melting_K': 1068,
        'density': 6.77, 'bulk_mod': 22, 'shear_mod': 14,
        'thermal_k': 11.3, 'specific_heat': 0.192,
        'work_fn': 2.90, 'fermi_E': 4.50, 'e_density': 0.20,
        'n_valence': 3, 'chi_m': 2500e-6,
        'max_loading': 2.5, 'D0': 1e-3, 'Ea_diff': 0.35,
        'H_enthalpy': -1.0, 'Us_measured': None,
    },
    'La': {
        'Z': 57, 'A': 138.91, 'structure': 'HCP',
        'a_A': 3.7700, 'debye_K': 142, 'melting_K': 1193,
        'density': 6.16, 'bulk_mod': 28, 'shear_mod': 14,
        'thermal_k': 13.4, 'specific_heat': 0.195,
        'work_fn': 3.50, 'fermi_E': 4.60, 'e_density': 0.21,
        'n_valence': 3, 'chi_m': 95.9e-6,
        'max_loading': 2.87, 'D0': 1e-3, 'Ea_diff': 0.40,
        'H_enthalpy': -0.96, 'Us_measured': None,
    },
    'Sc': {
        'Z': 21, 'A': 44.96, 'structure': 'HCP',
        'a_A': 3.3090, 'debye_K': 360, 'melting_K': 1814,
        'density': 2.99, 'bulk_mod': 57, 'shear_mod': 29,
        'thermal_k': 15.8, 'specific_heat': 0.568,
        'work_fn': 3.50, 'fermi_E': 5.20, 'e_density': 0.25,
        'n_valence': 3, 'chi_m': 295e-6,
        'max_loading': 2.0, 'D0': 1e-3, 'Ea_diff': 0.45,
        'H_enthalpy': -0.80, 'Us_measured': None,
    },
    # Carbon (graphite) for composites
    'C': {
        'Z': 6, 'A': 12.011, 'structure': 'HCP',
        'a_A': 2.4612, 'debye_K': 2230, 'melting_K': 3823,
        'density': 2.267, 'bulk_mod': 33, 'shear_mod': 4.1,
        'thermal_k': 119, 'specific_heat': 0.709,
        'work_fn': 5.00, 'fermi_E': 5.00, 'e_density': 0.35,
        'n_valence': 4, 'chi_m': -12.0e-6,
        'max_loading': 0.0, 'D0': 1e-8, 'Ea_diff': 0.90,
        'H_enthalpy': 0.80, 'Us_measured': None,
    },
    'Si': {
        'Z': 14, 'A': 28.085, 'structure': 'FCC',
        'a_A': 5.4310, 'debye_K': 645, 'melting_K': 1687,
        'density': 2.33, 'bulk_mod': 98, 'shear_mod': 67,
        'thermal_k': 149, 'specific_heat': 0.705,
        'work_fn': 4.85, 'fermi_E': 5.50, 'e_density': 0.30,
        'n_valence': 4, 'chi_m': -3.12e-6,
        'max_loading': 0.0, 'D0': 1e-8, 'Ea_diff': 1.0,
        'H_enthalpy': 0.60, 'Us_measured': None,
    },
}

# Known oxide properties for composites
OXIDE_DB = {
    'ZrO2': {
        'density': 5.68, 'debye_K': 500, 'melting_K': 2988,
        'thermal_k': 2.0, 'chi_m': -13.8e-6,
        'notes': 'Nanostructured matrix for Pd particles (Arata-Zhang)',
        'lenr_bonus': 1.5,  # Confinement effect
    },
    'CeO2': {
        'density': 7.22, 'debye_K': 395, 'melting_K': 2673,
        'thermal_k': 12, 'chi_m': 26e-6,
        'notes': 'Oxygen vacancy conductor, proton conductor',
        'lenr_bonus': 1.8,  # Oxygen vacancy channels
    },
    'CaO': {
        'density': 3.34, 'debye_K': 648, 'melting_K': 2886,
        'thermal_k': 30, 'chi_m': -15e-6,
        'notes': 'Iwamura multilayer (Pd/CaO) for transmutations',
        'lenr_bonus': 2.0,  # Phonon mismatch = scattering centers
    },
    'TiO2': {
        'density': 4.23, 'debye_K': 760, 'melting_K': 2116,
        'thermal_k': 8.5, 'chi_m': 5.9e-6,
        'notes': 'Photocatalytic + high dielectric',
        'lenr_bonus': 1.3,
    },
    'Al2O3': {
        'density': 3.95, 'debye_K': 1042, 'melting_K': 2345,
        'thermal_k': 35, 'chi_m': -13e-6,
        'notes': 'Insulating barrier, thermal stabilizer',
        'lenr_bonus': 0.8,
    },
    'SiO2': {
        'density': 2.65, 'debye_K': 470, 'melting_K': 1986,
        'thermal_k': 1.3, 'chi_m': -4.5e-6,
        'notes': 'Piezoelectric support',
        'lenr_bonus': 0.5,
    },
    'Y2O3': {
        'density': 5.01, 'debye_K': 450, 'melting_K': 2698,
        'thermal_k': 27, 'chi_m': 44.4e-6,
        'notes': 'Proton conductor at high T, YSZ stabilizer',
        'lenr_bonus': 1.4,
    },
    'PdO': {
        'density': 8.3, 'debye_K': 300, 'melting_K': 1143,
        'thermal_k': 3.5, 'chi_m': -7.66e-6,
        'notes': 'Native oxide = photon mass lens (Kasagi: 600 eV)',
        'lenr_bonus': 2.5,
    },
    'NiO': {
        'density': 6.67, 'debye_K': 600, 'melting_K': 2228,
        'thermal_k': 23, 'chi_m': 660e-6,
        'notes': 'Antiferromagnetic, high defect density',
        'lenr_bonus': 1.6,
    },
    'Fe2O3': {
        'density': 5.24, 'debye_K': 500, 'melting_K': 1838,
        'thermal_k': 5.0, 'chi_m': 3586e-6,
        'notes': 'Ferrimagnetic, strong photon mass interaction',
        'lenr_bonus': 1.7,
    },
}


# ============================================================
# KNOWN ALLOYS (from experiments)
# ============================================================

KNOWN_ALLOYS = {
    'Constantan': {
        'components': {'Cu': 0.55, 'Ni': 0.45},
        'Us_measured': None,
        'excess_heat_W': 209,
        'COP': 3.91,
        'notes': '2025 record: 209W, COP 3.91 in ~30 seconds',
    },
    'NiCu_Iwamura': {
        'components': {'Ni': 0.50, 'Cu': 0.50},
        'Us_measured': None,
        'excess_heat_W': 5,
        'COP': 1.5,
        'notes': 'Iwamura/Clean Planet 589 days continuous',
    },
    'SUS304': {
        'components': {'Fe': 0.70, 'Cr': 0.19, 'Ni': 0.10, 'Mn': 0.01},
        'Us_measured': None,
        'excess_heat_W': 0,
        'COP': 1.0,
        'notes': 'Mizuno 2025: neutrons with H2, not D2',
    },
    'PdAg': {
        'components': {'Pd': 0.77, 'Ag': 0.23},
        'Us_measured': 250,
        'notes': 'Standard H-filter membrane',
    },
    'PdNi_nano': {
        'components': {'Pd': 0.90, 'Ni': 0.10},
        'Us_measured': None,
        'notes': 'Nano-alloy for enhanced loading',
    },
}


# ============================================================
# MAIN ENGINE
# ============================================================

class AlloyPredictor:
    """
    Predicts LENR properties for arbitrary metal alloys and composites.

    Methods:
        calculate_alloy_properties(components) -> AlloyProperties
        predict_lenr_potential(properties) -> PredictionResult
        scan_binary_alloys() -> DataFrame of all binary combinations
        scan_ternary_alloys(base_metals) -> DataFrame
        find_optimal_composition(metal1, metal2, n_steps) -> best fraction
        generate_prediction_matrix() -> full heatmap data
    """

    def __init__(self):
        self.element_db = ELEMENT_DB
        self.oxide_db = OXIDE_DB
        self.known_alloys = KNOWN_ALLOYS

        # OLS-fitted screening coefficients (fit on init from measured data)
        self._screening_coeffs = {}
        self._screening_model: Optional[LinearRegression] = None
        self._screening_features: Optional[np.ndarray] = None
        self._screening_targets: Optional[np.ndarray] = None
        self._screening_loo_r2: float = 0.0

        # Fit screening model from measured data
        self._fit_screening_model()

        # Calibrate excess heat against experimental data
        self._excess_heat_coeffs = {}
        self._calibrate_excess_heat_model()

    # --------------------------------------------------------
    # MODEL FITTING (OLS from measured screening data)
    # --------------------------------------------------------

    # Feature names for diagnostics
    _SCREENING_FEATURE_NAMES = [
        'log_chi_m', 'is_ferromagnetic', 'lattice_ratio',
        'log_loading_plus1', 'e_density', 'neg_H_enthalpy',
        'debye_K_scaled', 'n_valence_scaled',
    ]

    def _build_screening_features(self, elem: str, props: dict) -> np.ndarray:
        """
        Build feature vector for screening energy prediction.

        Log-transforms for wide-range features. Target is log(Us).
        Features (Cherepanov-inspired):
          0: log(|chi_m| + 1e-7)  — magnetic susceptibility (log scale)
          1: is_ferromagnetic     — binary (1 if |chi_m| > 0.01)
          2: a / (theta_D/1000)   — lattice focusing ratio
          3: log(loading + 0.001) — hydrogen capacity (log scale)
          4: e_density            — electron density at interstitials
          5: -H_enthalpy          — neg. absorption energy (positive = exothermic = good)
          6: debye_K / 1000       — lattice stiffness
          7: n_valence / 10       — valence electron count
        """
        chi_abs = abs(props['chi_m'])
        is_ferro = 1.0 if chi_abs > 0.01 else 0.0
        lattice_ratio = props['a_A'] / (props['debye_K'] / 1000.0)
        loading = props.get('max_loading', 0.0)
        e_dens = props.get('e_density', 0.0)
        h_enth = props.get('H_enthalpy', 0.0)
        debye = props.get('debye_K', 300.0)
        n_val = props.get('n_valence', 4)

        return np.array([
            np.log(chi_abs + 1e-7),    # 0: log scale for chi_m
            is_ferro,                    # 1
            lattice_ratio,               # 2
            np.log(loading + 0.001),     # 3: log scale for loading
            e_dens,                      # 4
            -h_enth,                     # 5: negate so positive = exothermic
            debye / 1000.0,              # 6
            n_val / 10.0,                # 7
        ])

    def _fit_screening_model(self) -> None:
        """
        Fit Ridge regression on measured screening energies.

        Uses all elements in ELEMENT_DB with 'Us_measured' != None.
        Log-transforms the target: model predicts log(Us), then exp() back.
        Ridge regularization prevents overfitting on small dataset (19 points, 8 features).
        Alpha is auto-tuned by maximizing LOO R² (on original scale).
        """
        X_list, y_list = [], []
        elem_order = []

        for elem, props in self.element_db.items():
            us = props.get('Us_measured')
            if us is not None and us > 0:
                features = self._build_screening_features(elem, props)
                X_list.append(features)
                y_list.append(us)
                elem_order.append(elem)

        if len(X_list) < 3:
            warnings.warn("Too few measured screening points for Ridge fit, using fallback")
            self._screening_coeffs = {}
            return

        X = np.array(X_list)
        y = np.array(y_list, dtype=float)
        self._screening_features = X
        self._screening_targets = y
        self._screening_elements = elem_order

        # Log-transform target for better linearity
        y_log = np.log(y)
        self._screening_log_targets = y_log

        # Standardize features for proper Ridge regularization
        self._screening_scaler = StandardScaler()
        X_scaled = self._screening_scaler.fit_transform(X)

        # Auto-tune alpha via LOO R² (evaluated on original scale)
        # Include very low alphas to let the model capture real signal
        best_alpha, best_loo_r2 = 1.0, -np.inf
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            loo_r2 = self._compute_loo_r2_ridge_log(X_scaled, y_log, y, alpha)
            if loo_r2 > best_loo_r2:
                best_alpha = alpha
                best_loo_r2 = loo_r2

        # Fit final model with best alpha on log(Us)
        model = Ridge(alpha=best_alpha)
        model.fit(X_scaled, y_log)
        self._screening_model = model
        self._screening_alpha = best_alpha
        self._screening_loo_r2 = best_loo_r2

        # Store named coefficients for introspection
        self._screening_coeffs = {'intercept_log': model.intercept_}
        for name, coef in zip(self._SCREENING_FEATURE_NAMES, model.coef_):
            self._screening_coeffs[name] = coef

        # Full-fit R² (on original scale)
        y_pred_log = model.predict(X_scaled)
        y_pred_full = np.exp(y_pred_log)
        self._screening_fit_r2 = r2_score(y, y_pred_full)
        self._screening_fit_rmse = np.sqrt(mean_squared_error(y, y_pred_full))

    def _compute_loo_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        """Leave-One-Out cross-validation R² for screening model (OLS)."""
        loo = LeaveOneOut()
        y_pred_loo = np.zeros_like(y)

        for train_idx, test_idx in loo.split(X):
            model_cv = LinearRegression()
            model_cv.fit(X[train_idx], y[train_idx])
            y_pred_loo[test_idx] = model_cv.predict(X[test_idx])

        return r2_score(y, y_pred_loo)

    def _compute_loo_r2_ridge(self, X: np.ndarray, y: np.ndarray, alpha: float) -> float:
        """Leave-One-Out cross-validation R² for Ridge model."""
        loo = LeaveOneOut()
        y_pred_loo = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in loo.split(X):
            model_cv = Ridge(alpha=alpha)
            model_cv.fit(X[train_idx], y[train_idx])
            y_pred_loo[test_idx] = model_cv.predict(X[test_idx])

        return r2_score(y, y_pred_loo)

    def _compute_loo_r2_ridge_log(
        self, X: np.ndarray, y_log: np.ndarray, y_orig: np.ndarray, alpha: float
    ) -> float:
        """
        LOO R² for Ridge model fitted on log(Us), evaluated on original scale.
        This ensures we optimize for accuracy in eV, not in log-eV.
        """
        loo = LeaveOneOut()
        y_pred_loo = np.zeros_like(y_orig, dtype=float)

        for train_idx, test_idx in loo.split(X):
            model_cv = Ridge(alpha=alpha)
            model_cv.fit(X[train_idx], y_log[train_idx])
            y_pred_log = model_cv.predict(X[test_idx])
            y_pred_loo[test_idx] = np.exp(y_pred_log)

        return r2_score(y_orig, y_pred_loo)

    def get_screening_diagnostics(self) -> dict:
        """
        Return diagnostics for the screening energy model.

        Returns dict with:
            coefficients: fitted OLS coefficients
            fit_r2: R² on training data
            loo_r2: Leave-One-Out R² (honest generalization estimate)
            fit_rmse: RMSE on training data
            n_points: number of measured data points used
            per_element: dict of {element: (measured, predicted, residual)}
        """
        diag = {
            'coefficients': dict(self._screening_coeffs),
            'fit_r2': getattr(self, '_screening_fit_r2', None),
            'loo_r2': self._screening_loo_r2,
            'fit_rmse': getattr(self, '_screening_fit_rmse', None),
            'n_points': len(self._screening_targets) if self._screening_targets is not None else 0,
        }

        if self._screening_model is not None and self._screening_features is not None:
            X_scaled = self._screening_scaler.transform(self._screening_features)
            y_pred_log = self._screening_model.predict(X_scaled)
            y_pred = np.exp(y_pred_log)
            per_elem = {}
            for i, elem in enumerate(self._screening_elements):
                measured = self._screening_targets[i]
                predicted = y_pred[i]
                per_elem[elem] = {
                    'measured': float(measured),
                    'predicted': round(float(predicted), 1),
                    'residual': round(float(measured - predicted), 1),
                    'pct_error': round(float(abs(measured - predicted) / max(measured, 1) * 100), 1),
                }
            diag['per_element'] = per_elem

        return diag

    def _calibrate_excess_heat_model(self) -> None:
        """
        Calibrate excess heat scaling factor against all EXCESS_HEAT_DATA experiments.

        Uses median-based robust estimation:
          excess_W = k × (Us / 100) × loading
        where k is calibrated from experimental data via median of
        k_i = excess_measured_i / ((Us_i / 100) × loading_i).

        Median is robust to outliers (Fleischmann COP=40, Constantan burst).
        """
        try:
            from lenr_constants import EXCESS_HEAT_DATA
            experiments = EXCESS_HEAT_DATA
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from lenr_constants import EXCESS_HEAT_DATA
            experiments = EXCESS_HEAT_DATA

        # Map material names to screening energies
        material_to_us = {
            'Pd': 310, 'Ni': 420, 'NiCu': 230,
            'PdNi_ZrO2': 400, 'nano_Pd': 310,
            'Constantan': 230,
        }

        k_values = []
        for exp in experiments:
            mat = exp['material']
            us = material_to_us.get(mat, 200)
            loading = exp.get('DPd') or 0.5
            excess = exp['excess_W']

            denominator = (us / 100.0) * max(loading, 0.01)
            if excess > 0 and denominator > 0.01:
                k_values.append(excess / denominator)

        if k_values:
            self._excess_heat_k = float(np.median(k_values))
            # Store individual k values for diagnostics
            self._excess_heat_k_range = (min(k_values), max(k_values))
            self._excess_heat_k_values = k_values
        else:
            self._excess_heat_k = 10.0  # fallback
            self._excess_heat_k_range = (10.0, 10.0)

        # Store for diagnostics
        self._excess_heat_coeffs = {
            'k_median': self._excess_heat_k,
            'k_min': self._excess_heat_k_range[0],
            'k_max': self._excess_heat_k_range[1],
            'n_experiments': len(k_values),
        }

    # --------------------------------------------------------
    # ALLOY PROPERTY CALCULATION
    # --------------------------------------------------------

    def calculate_alloy_properties(
        self,
        components: Dict[str, float],
        oxide_matrix: Optional[str] = None,
        oxide_fraction: float = 0.0,
        defect_concentration: float = 0.05,
        surface_treatment: str = 'none',
    ) -> AlloyProperties:
        """
        Calculate effective properties of an alloy or composite.

        Args:
            components: dict of {element: fraction}, fractions must sum to ~1.0
            oxide_matrix: optional oxide name from OXIDE_DB (e.g. 'ZrO2', 'CeO2')
            oxide_fraction: fraction of oxide in composite (0-0.5 typical)
            defect_concentration: 0-1, from surface treatment or processing
            surface_treatment: 'annealed', 'cold_rolled', 'nano', etc.

        Returns:
            AlloyProperties with all predicted values
        """
        # Validate
        total = sum(components.values())
        if abs(total - 1.0) > 0.05:
            warnings.warn(f"Component fractions sum to {total:.3f}, normalizing to 1.0")
            components = {k: v/total for k, v in components.items()}

        # Check all elements exist
        for elem in components:
            if elem not in self.element_db:
                raise ValueError(f"Unknown element: {elem}. Available: {list(self.element_db.keys())}")

        # Build name
        name_parts = []
        for elem, frac in sorted(components.items(), key=lambda x: -x[1]):
            if frac >= 0.05:
                name_parts.append(f"{elem}{int(frac*100)}")
        name = '-'.join(name_parts)
        if oxide_matrix:
            name += f'/{oxide_matrix}'

        comp_list = [AlloyComponent(sym, frac) for sym, frac in components.items()]

        props = AlloyProperties(name=name, components=comp_list)

        # ---- Rule of mixtures / Vegard's law ----

        # Linear mixing for most properties
        props.Z_eff = self._weighted_avg(components, 'Z')
        props.A_eff = self._weighted_avg(components, 'A')
        props.density_g_cm3 = self._weighted_avg(components, 'density')
        props.melting_K = self._weighted_avg(components, 'melting_K')
        props.thermal_conductivity = self._weighted_avg(components, 'thermal_k')
        props.bulk_modulus_GPa = self._weighted_avg(components, 'bulk_mod')
        props.work_function_eV = self._weighted_avg(components, 'work_fn')
        props.fermi_energy_eV = self._weighted_avg(components, 'fermi_E')
        props.e_density_A3 = self._weighted_avg(components, 'e_density')
        props.n_valence_eff = self._weighted_avg(components, 'n_valence')

        # Vegard's law for lattice parameter
        props.a_eff_A = self._weighted_avg(components, 'a_A')

        # Debye temperature (geometric mean is more physical)
        log_debye = sum(
            frac * np.log(self.element_db[elem]['debye_K'])
            for elem, frac in components.items()
        )
        props.debye_K = np.exp(log_debye)

        # Crystal structure from majority component
        majority = max(components.items(), key=lambda x: x[1])
        props.structure = self.element_db[majority[0]]['structure']

        # ---- Magnetic properties (NON-LINEAR!) ----
        # Ferromagnetic components dominate even at low fractions
        props.chi_m_eff = self._calculate_effective_chi(components)
        props.magnetic_class = self._classify_magnetic(props.chi_m_eff, components)

        # ---- Hydrogen/Deuterium properties ----
        # Loading: maximum of components (best path wins)
        loadings = [
            self.element_db[elem]['max_loading'] * frac
            for elem, frac in components.items()
        ]
        # Synergistic: alloy can exceed individual loading
        props.max_loading = max(
            self._weighted_avg(components, 'max_loading'),
            max(loadings) * 1.2  # 20% synergy bonus
        )

        # Diffusion: fastest path dominates (parallel paths)
        D_values = []
        for elem, frac in components.items():
            if frac > 0.05:
                D_300K = (self.element_db[elem]['D0'] *
                         np.exp(-self.element_db[elem]['Ea_diff'] / (8.617e-5 * 300)))
                D_values.append(D_300K)
        props.D0_eff = max(D_values) if D_values else 1e-10
        props.Ea_diffusion_eV = self._weighted_avg(components, 'Ea_diff')
        props.H_absorption_eV = self._weighted_avg(components, 'H_enthalpy')

        # ---- Oxide matrix correction ----
        if oxide_matrix and oxide_matrix in self.oxide_db:
            oxide = self.oxide_db[oxide_matrix]
            metal_frac = 1.0 - oxide_fraction

            # Density correction
            props.density_g_cm3 = (props.density_g_cm3 * metal_frac +
                                   oxide['density'] * oxide_fraction)
            # Thermal conductivity reduction
            props.thermal_conductivity *= metal_frac
            # Debye temperature shift
            props.debye_K = props.debye_K * metal_frac + oxide['debye_K'] * oxide_fraction

        # ---- LENR predictions ----
        props.predicted_screening_eV = self._predict_screening(
            props, components, defect_concentration, oxide_matrix, oxide_fraction
        )
        props.predicted_medium_resistance = self._predict_resistance(
            props, defect_concentration, components
        )
        props.predicted_reaction_probability = self._predict_reaction_prob(
            props, defect_concentration
        )
        props.predicted_excess_heat_W = self._predict_excess_heat(
            props, defect_concentration
        )
        props.predicted_COP = self._predict_COP(props)

        # Confidence based on data availability
        n_measured = sum(
            1 for elem in components
            if self.element_db[elem].get('Us_measured') is not None
        )
        props.confidence = min(1.0, n_measured / len(components) * 0.7 + 0.3)
        if oxide_matrix:
            props.confidence *= 0.8  # Less certain for composites

        props.data_quality = 'predicted'
        if name in ['Pd100', 'Ni100', 'Fe100']:
            props.data_quality = 'measured'

        notes = []
        if props.magnetic_class == 'ferromagnetic':
            notes.append('Ferromagnetic: strong photon mass focusing (Cherepanov)')
        if props.max_loading > 0.5:
            notes.append(f'High D loading capacity: {props.max_loading:.2f}')
        if oxide_matrix:
            notes.append(f'Composite with {oxide_matrix}: interface defects enhance channels')
        if defect_concentration > 0.1:
            notes.append(f'High defect density ({defect_concentration:.2f}): enhanced channels')
        props.notes = '; '.join(notes)

        return props

    # --------------------------------------------------------
    # LENR PREDICTION
    # --------------------------------------------------------

    def predict_lenr_potential(
        self,
        components: Dict[str, float],
        oxide_matrix: Optional[str] = None,
        oxide_fraction: float = 0.0,
        defect_concentration: float = 0.05,
        temperature_K: float = 340,
        D_loading: float = 0.85,
    ) -> PredictionResult:
        """
        Full LENR potential prediction for a material combination.

        Returns PredictionResult with scores 0-100 and risk analysis.
        """
        props = self.calculate_alloy_properties(
            components, oxide_matrix, oxide_fraction, defect_concentration
        )

        # ---- Individual scores ----

        # Screening score (0-100)
        Us = props.predicted_screening_eV
        screening_score = min(100, Us / 10.0)  # 1000 eV = 100

        # Loading score
        loading_score = min(100, props.max_loading / 0.01)  # 1.0 loading = 100
        if props.max_loading < 0.001:
            loading_score = 0  # No hydrogen = no fusion

        # Magnetic score (Cherepanov: ferromagnets >> paramagnets >> diamagnets)
        chi_abs = abs(props.chi_m_eff)
        if props.magnetic_class == 'ferromagnetic':
            magnetic_score = min(100, 60 + chi_abs * 40)
        elif props.magnetic_class == 'paramagnetic':
            magnetic_score = min(60, chi_abs * 1e4 * 20)
        else:
            magnetic_score = max(5, 20 - chi_abs * 1e6)

        # Defect score
        defect_score = min(100, defect_concentration * 200)

        # Thermal stability
        thermal_score = min(100, (props.melting_K - 300) / 30.0)
        if props.melting_K < 600:
            thermal_score *= 0.3  # Penalty for low melting

        # ---- Combined LENR score ----
        # Weighted combination based on Cherepanov physics
        weights = {
            'screening': 0.30,
            'loading': 0.25,
            'magnetic': 0.20,
            'defect': 0.15,
            'thermal': 0.10,
        }

        lenr_score = (
            weights['screening'] * screening_score +
            weights['loading'] * loading_score +
            weights['magnetic'] * magnetic_score +
            weights['defect'] * defect_score +
            weights['thermal'] * thermal_score
        )

        # ---- Risk factors & advantages ----
        risks = []
        advantages = []

        if props.max_loading < 0.01:
            risks.append("Very low H/D loading capacity")
        if props.max_loading < 0.001:
            risks.append("CRITICAL: No hydrogen absorption")
        if props.melting_K < 600:
            risks.append("Low melting point limits operating T")
        if props.H_absorption_eV > 0.5:
            risks.append("Endothermic H absorption (needs pressure)")
        if props.thermal_conductivity < 10:
            risks.append("Low thermal conductivity (hotspot risk)")

        if props.predicted_screening_eV > 300:
            advantages.append(f"High screening energy: {props.predicted_screening_eV:.0f} eV")
        if props.magnetic_class == 'ferromagnetic':
            advantages.append("Ferromagnetic focusing of photon mass")
        if props.max_loading > 0.5:
            advantages.append(f"Excellent D loading: {props.max_loading:.2f}")
        if props.H_absorption_eV < -0.3:
            advantages.append("Exothermic H absorption (spontaneous loading)")
        if defect_concentration > 0.2:
            advantages.append(f"High defect channel density: {defect_concentration:.0%}")
        if oxide_matrix:
            advantages.append(f"Interface with {oxide_matrix}: phonon mismatch channels")

        comp_str = ' + '.join(
            f"{elem}({frac:.0%})" for elem, frac in
            sorted(components.items(), key=lambda x: -x[1])
        )
        if oxide_matrix:
            comp_str += f" / {oxide_matrix}({oxide_fraction:.0%})"

        return PredictionResult(
            material_name=props.name,
            components=comp_str,
            lenr_score=lenr_score,
            screening_score=screening_score,
            loading_score=loading_score,
            magnetic_score=magnetic_score,
            defect_score=defect_score,
            thermal_score=thermal_score,
            predicted_Us_eV=props.predicted_screening_eV,
            predicted_excess_W=props.predicted_excess_heat_W,
            predicted_COP=props.predicted_COP,
            confidence=props.confidence,
            risk_factors=risks,
            advantages=advantages,
            alloy_properties=props,
        )

    # --------------------------------------------------------
    # SCANNING: ALL BINARY COMBINATIONS
    # --------------------------------------------------------

    def scan_binary_alloys(
        self,
        elements: Optional[List[str]] = None,
        fractions: Optional[List[float]] = None,
        defect_concentration: float = 0.05,
    ) -> pd.DataFrame:
        """
        Scan all binary alloy combinations and rank by LENR potential.

        Args:
            elements: list of elements to scan (default: all)
            fractions: list of fractions for metal_1 (default: [0.1, 0.25, 0.5, 0.75, 0.9])
            defect_concentration: defect level for all predictions

        Returns:
            DataFrame sorted by LENR score with columns for all subscores
        """
        if elements is None:
            elements = list(self.element_db.keys())
        if fractions is None:
            fractions = [0.25, 0.50, 0.75]

        results = []
        pairs = list(itertools.combinations(elements, 2))

        for m1, m2 in pairs:
            for f1 in fractions:
                f2 = 1.0 - f1
                try:
                    pred = self.predict_lenr_potential(
                        {m1: f1, m2: f2},
                        defect_concentration=defect_concentration
                    )
                    results.append({
                        'alloy': pred.material_name,
                        'metal_1': m1, 'frac_1': f1,
                        'metal_2': m2, 'frac_2': f2,
                        'lenr_score': pred.lenr_score,
                        'screening_score': pred.screening_score,
                        'loading_score': pred.loading_score,
                        'magnetic_score': pred.magnetic_score,
                        'defect_score': pred.defect_score,
                        'thermal_score': pred.thermal_score,
                        'predicted_Us_eV': pred.predicted_Us_eV,
                        'predicted_COP': pred.predicted_COP,
                        'predicted_excess_W': pred.predicted_excess_W,
                        'confidence': pred.confidence,
                        'n_risks': len(pred.risk_factors),
                        'n_advantages': len(pred.advantages),
                    })
                except Exception as e:
                    logger.debug("Binary scan skipped %s-%s (%.0f%%): %s",
                                 m1, m2, f1 * 100, e)

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('lenr_score', ascending=False).reset_index(drop=True)
        return df

    def scan_with_oxides(
        self,
        base_metals: Optional[List[str]] = None,
        oxides: Optional[List[str]] = None,
        defect_concentration: float = 0.15,
    ) -> pd.DataFrame:
        """
        Scan metal + oxide composite combinations.
        """
        if base_metals is None:
            base_metals = ['Pd', 'Ni', 'Ti', 'Fe', 'Zr', 'Cu', 'Co', 'V', 'Nb', 'Ta']
        if oxides is None:
            oxides = list(self.oxide_db.keys())

        results = []
        oxide_fracs = [0.05, 0.10, 0.20, 0.30]

        for metal in base_metals:
            for oxide in oxides:
                for ox_frac in oxide_fracs:
                    try:
                        pred = self.predict_lenr_potential(
                            {metal: 1.0},
                            oxide_matrix=oxide,
                            oxide_fraction=ox_frac,
                            defect_concentration=defect_concentration,
                        )
                        results.append({
                            'composite': f"{metal}/{oxide}({ox_frac:.0%})",
                            'metal': metal,
                            'oxide': oxide,
                            'oxide_fraction': ox_frac,
                            'lenr_score': pred.lenr_score,
                            'predicted_Us_eV': pred.predicted_Us_eV,
                            'predicted_COP': pred.predicted_COP,
                            'predicted_excess_W': pred.predicted_excess_W,
                            'magnetic_score': pred.magnetic_score,
                            'loading_score': pred.loading_score,
                            'confidence': pred.confidence,
                        })
                    except Exception as e:
                        logger.debug("Oxide scan skipped %s/%s(%.0f%%): %s",
                                     metal, oxide, ox_frac * 100, e)

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('lenr_score', ascending=False).reset_index(drop=True)
        return df

    def scan_ternary_alloys(
        self,
        base_metals: Optional[List[str]] = None,
        defect_concentration: float = 0.05,
    ) -> pd.DataFrame:
        """
        Scan promising ternary alloy combinations.
        Uses limited element set to keep computation feasible.
        """
        if base_metals is None:
            base_metals = ['Pd', 'Ni', 'Ti', 'Fe', 'Cu', 'Zr', 'Co', 'V', 'Ta', 'Nb']

        # Ternary fractions: (a, b, c) with a+b+c=1
        frac_sets = [
            (0.60, 0.30, 0.10),
            (0.50, 0.30, 0.20),
            (0.50, 0.25, 0.25),
            (0.34, 0.33, 0.33),
        ]

        results = []
        triples = list(itertools.combinations(base_metals, 3))

        for m1, m2, m3 in triples:
            for f1, f2, f3 in frac_sets:
                try:
                    pred = self.predict_lenr_potential(
                        {m1: f1, m2: f2, m3: f3},
                        defect_concentration=defect_concentration,
                    )
                    results.append({
                        'alloy': pred.material_name,
                        'metal_1': m1, 'frac_1': f1,
                        'metal_2': m2, 'frac_2': f2,
                        'metal_3': m3, 'frac_3': f3,
                        'lenr_score': pred.lenr_score,
                        'predicted_Us_eV': pred.predicted_Us_eV,
                        'predicted_COP': pred.predicted_COP,
                        'predicted_excess_W': pred.predicted_excess_W,
                        'confidence': pred.confidence,
                    })
                except Exception as e:
                    logger.debug("Ternary scan skipped %s-%s-%s: %s", m1, m2, m3, e)

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('lenr_score', ascending=False).reset_index(drop=True)
        return df

    def find_optimal_composition(
        self,
        metal1: str,
        metal2: str,
        n_steps: int = 20,
        defect_concentration: float = 0.05,
    ) -> pd.DataFrame:
        """
        Find optimal binary alloy composition by scanning fractions.
        """
        fractions = np.linspace(0.05, 0.95, n_steps)
        results = []

        for f1 in fractions:
            pred = self.predict_lenr_potential(
                {metal1: f1, metal2: 1.0 - f1},
                defect_concentration=defect_concentration,
            )
            results.append({
                f'{metal1}_fraction': f1,
                'lenr_score': pred.lenr_score,
                'predicted_Us_eV': pred.predicted_Us_eV,
                'predicted_COP': pred.predicted_COP,
                'screening_score': pred.screening_score,
                'loading_score': pred.loading_score,
                'magnetic_score': pred.magnetic_score,
            })

        return pd.DataFrame(results)

    def generate_heatmap_data(
        self,
        elements: Optional[List[str]] = None,
        metric: str = 'lenr_score',
        fraction: float = 0.50,
    ) -> pd.DataFrame:
        """
        Generate NxN matrix for heatmap of binary alloys at 50/50 composition.

        Returns:
            DataFrame with elements as index and columns, values = metric
        """
        if elements is None:
            elements = ['Pd', 'Ni', 'Ti', 'Fe', 'Cu', 'Zr', 'Co',
                        'V', 'Ta', 'Nb', 'Au', 'Pt', 'W', 'Al', 'Cr']

        n = len(elements)
        matrix = np.zeros((n, n))

        for i, m1 in enumerate(elements):
            for j, m2 in enumerate(elements):
                if i == j:
                    # Pure metal
                    pred = self.predict_lenr_potential({m1: 1.0})
                    matrix[i, j] = getattr(pred, metric, 0)
                else:
                    pred = self.predict_lenr_potential(
                        {m1: fraction, m2: 1.0 - fraction}
                    )
                    matrix[i, j] = getattr(pred, metric, 0)

        return pd.DataFrame(matrix, index=elements, columns=elements)

    def get_top_predictions(
        self,
        n_top: int = 30,
        include_ternary: bool = True,
        include_composites: bool = True,
        defect_conc: float = 0.1,
    ) -> pd.DataFrame:
        """
        Get top N predictions across all material types.
        Returns unified ranked table.
        """
        all_results = []

        # Binary alloys (key metals)
        key_metals = ['Pd', 'Ni', 'Ti', 'Fe', 'Cu', 'Zr', 'Co',
                      'V', 'Ta', 'Nb', 'Cr', 'Mn', 'Hf', 'La', 'Ce', 'Sc', 'Y']

        binary_df = self.scan_binary_alloys(
            elements=key_metals,
            fractions=[0.25, 0.50, 0.75],
            defect_concentration=defect_conc,
        )
        if len(binary_df) > 0:
            binary_df['type'] = 'binary_alloy'
            all_results.append(binary_df[['alloy', 'lenr_score', 'predicted_Us_eV',
                                          'predicted_COP', 'predicted_excess_W',
                                          'confidence', 'type']])

        # Pure metals for reference
        for metal in key_metals:
            pred = self.predict_lenr_potential(
                {metal: 1.0}, defect_concentration=defect_conc
            )
            all_results.append(pd.DataFrame([{
                'alloy': metal,
                'lenr_score': pred.lenr_score,
                'predicted_Us_eV': pred.predicted_Us_eV,
                'predicted_COP': pred.predicted_COP,
                'predicted_excess_W': pred.predicted_excess_W,
                'confidence': pred.confidence,
                'type': 'pure_metal',
            }]))

        # Composites with oxides
        if include_composites:
            comp_df = self.scan_with_oxides(
                base_metals=['Pd', 'Ni', 'Ti', 'Fe', 'Zr', 'Co', 'V', 'Nb'],
                defect_concentration=0.15,
            )
            if len(comp_df) > 0:
                comp_df['type'] = 'metal_oxide_composite'
                comp_df = comp_df.rename(columns={'composite': 'alloy'})
                all_results.append(comp_df[['alloy', 'lenr_score', 'predicted_Us_eV',
                                            'predicted_COP', 'predicted_excess_W',
                                            'confidence', 'type']])

        # Ternary (limited)
        if include_ternary:
            tern_df = self.scan_ternary_alloys(
                base_metals=['Pd', 'Ni', 'Ti', 'Fe', 'Cu', 'Zr', 'Co', 'V'],
                defect_concentration=defect_conc,
            )
            if len(tern_df) > 0:
                tern_df['type'] = 'ternary_alloy'
                all_results.append(tern_df[['alloy', 'lenr_score', 'predicted_Us_eV',
                                            'predicted_COP', 'predicted_excess_W',
                                            'confidence', 'type']])

        # Combine and rank
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values('lenr_score', ascending=False).reset_index(drop=True)
        combined['rank'] = range(1, len(combined) + 1)

        return combined.head(n_top)

    # --------------------------------------------------------
    # PRIVATE: PROPERTY CALCULATIONS
    # --------------------------------------------------------

    def _weighted_avg(self, components: Dict[str, float], prop: str) -> float:
        """Weighted average of element property."""
        return sum(
            frac * self.element_db[elem].get(prop, 0)
            for elem, frac in components.items()
        )

    def _calculate_effective_chi(self, components: Dict[str, float]) -> float:
        """
        Effective magnetic susceptibility.
        Non-linear for ferromagnetic components!
        Even 10% Ni in Cu dramatically changes magnetic behavior.
        """
        # Check for ferromagnetic components
        ferro_frac = sum(
            frac for elem, frac in components.items()
            if abs(self.element_db[elem]['chi_m']) > 0.01  # Fe, Co, Ni
        )

        if ferro_frac > 0.05:
            # Percolation-like: ferromagnetic fraction dominates
            chi_ferro = sum(
                frac * self.element_db[elem]['chi_m']
                for elem, frac in components.items()
                if abs(self.element_db[elem]['chi_m']) > 0.01
            )
            chi_para = sum(
                frac * self.element_db[elem]['chi_m']
                for elem, frac in components.items()
                if abs(self.element_db[elem]['chi_m']) <= 0.01
            )
            # Non-linear mixing: ferromagnet wins
            return chi_ferro * (ferro_frac ** 0.5) + chi_para
        else:
            # Linear mixing for weak magnets
            return sum(
                frac * self.element_db[elem]['chi_m']
                for elem, frac in components.items()
            )

    def _classify_magnetic(self, chi_m: float, components: Dict[str, float]) -> str:
        """Classify magnetic behavior."""
        # Check if any component is ferromagnetic
        for elem, frac in components.items():
            if abs(self.element_db[elem]['chi_m']) > 0.01 and frac > 0.05:
                if chi_m > 0.001:
                    return 'ferromagnetic'

        if chi_m > 100e-6:
            return 'paramagnetic'
        elif chi_m < -1e-6:
            return 'diamagnetic'
        else:
            return 'paramagnetic'

    def _predict_screening(
        self,
        props: AlloyProperties,
        components: Dict[str, float],
        defect_conc: float,
        oxide_matrix: Optional[str],
        oxide_fraction: float,
    ) -> float:
        """
        Predict screening energy using OLS-fitted Cherepanov-inspired model.

        Model is fitted at __init__ from 16+ measured screening energies via OLS.
        Features include nonlinear terms (chi_m × e_density, lattice_ratio²).

        For alloys: uses alloy effective properties → OLS prediction,
        then anchored to measured values of component elements.
        Defect concentration applied as multiplicative boost.
        """
        # Build alloy-level feature vector
        alloy_features = self._build_alloy_screening_features(props)

        # Ridge prediction: model predicts log(Us), we exp() back
        if self._screening_model is not None and hasattr(self, '_screening_scaler'):
            alloy_scaled = self._screening_scaler.transform(
                alloy_features.reshape(1, -1)
            )
            log_Us = float(self._screening_model.predict(alloy_scaled)[0])
            Us_predicted = np.exp(log_Us)
        else:
            # Fallback if model not fitted
            Us_predicted = 100.0

        # Defect boost: defects = channels for photon mass flow
        # Czerski 2023: cold-rolled Pd (defects~0.5) → 18200 eV vs 310 eV baseline
        # This gives a multiplicative factor: 1 + defect_conc × defect_sensitivity
        defect_sensitivity = 400.0  # eV per unit defect concentration
        Us_predicted += defect_conc * defect_sensitivity

        # Oxide bonus (interface defects, phonon mismatch)
        oxide_multiplier = 1.0
        if oxide_matrix and oxide_matrix in self.oxide_db:
            oxide_multiplier = self.oxide_db[oxide_matrix].get('lenr_bonus', 1.0)

        # Validate against known measurements for alloy anchoring
        measured_us = []
        for elem, frac in components.items():
            us_m = self.element_db[elem].get('Us_measured')
            if us_m is not None:
                measured_us.append((us_m, frac))

        if measured_us:
            # Anchor to measured values — blend OLS prediction with data
            weighted_measured = sum(us * f for us, f in measured_us)
            total_frac = sum(f for _, f in measured_us)
            measured_avg = weighted_measured / max(total_frac, 0.01)

            if total_frac > 0.8:
                # Mostly measured components: heavily anchor
                Us = 0.20 * Us_predicted + 0.80 * measured_avg
            elif total_frac > 0.5:
                Us = 0.35 * Us_predicted + 0.65 * measured_avg
            else:
                Us = 0.55 * Us_predicted + 0.45 * measured_avg
        else:
            Us = Us_predicted

        # Apply oxide multiplier after anchoring
        Us *= oxide_multiplier

        # Physical bounds: screening energy 10-2000 eV for non-exotic materials
        # Czerski cold-rolled Pd max = 18200 eV but that's extreme defect case
        max_Us = 2000 if defect_conc < 0.3 else 5000
        return max(10, min(Us, max_Us))

    def _build_alloy_screening_features(self, props: AlloyProperties) -> np.ndarray:
        """Build feature vector from AlloyProperties for Ridge screening model."""
        chi_abs = abs(props.chi_m_eff)
        is_ferro = 1.0 if props.magnetic_class == 'ferromagnetic' else 0.0
        lattice_ratio = props.a_eff_A / (props.debye_K / 1000.0)

        return np.array([
            np.log(chi_abs + 1e-7),
            is_ferro,
            lattice_ratio,
            np.log(props.max_loading + 0.001),
            props.e_density_A3,
            -props.H_absorption_eV,
            props.debye_K / 1000.0,
            props.n_valence_eff / 10.0,
        ])

    def _predict_resistance(
        self,
        props: AlloyProperties,
        defect_conc: float,
        components: Dict[str, float],
    ) -> float:
        """
        Predict Cherepanov medium resistance.
        Lower = better for LENR.
        """
        R_base = 1000.0  # Base resistance

        # Lattice factor
        struct_factors = {'FCC': 1.0, 'BCC': 0.7, 'HCP': 0.5, 'tetragonal': 0.4}
        f_lattice = (props.a_eff_A / 3.5) ** 2 * struct_factors.get(props.structure, 0.5)

        # Defect factor (KEY): defects = channels for photon mass
        f_defect = 1.0 / (1.0 + defect_conc * 100.0)

        # Magnetic factor
        if props.magnetic_class == 'ferromagnetic':
            f_magnetic = 0.1
        elif props.magnetic_class == 'paramagnetic':
            f_magnetic = 0.5
        else:
            f_magnetic = 1.0

        # Loading factor
        f_loading = 1.0 / (1.0 + props.max_loading * 10.0)

        R_m = R_base * f_lattice * f_defect * f_magnetic * f_loading

        return R_m

    def _predict_reaction_prob(self, props: AlloyProperties, defect_conc: float) -> float:
        """Predict reaction probability (0-1)."""
        R_m = props.predicted_medium_resistance

        # Guard: zero or negative resistance → reaction is certain
        if R_m <= 0:
            return 1.0

        # Critical resistance threshold
        R_critical = 50.0  # Below this, reaction becomes likely

        if R_m > R_critical * 10:
            return 0.0
        elif R_m < R_critical:
            return 1.0 - np.exp(-2.0 * (R_critical / R_m - 1.0))
        else:
            ratio = R_critical / R_m
            return ratio ** 2

    def _predict_excess_heat(self, props: AlloyProperties, defect_conc: float) -> float:
        """
        Predict excess heat in watts using median-calibrated model.

        Formula: excess_W = P_reaction × (Us / 100) × loading × k_median
        where k_median is calibrated from 10 experimental points via median.
        """
        P_reaction = props.predicted_reaction_probability

        if P_reaction < 0.01:
            return 0.0

        Us = props.predicted_screening_eV
        loading = props.max_loading

        # Median-calibrated formula
        k = self._excess_heat_k
        excess_W = P_reaction * (Us / 100.0) * loading * k

        # Loading threshold: below 0.1, very little excess heat
        if loading < 0.1:
            excess_W *= loading / 0.1

        # Cap at reasonable physical limits (Constantan burst was 209W)
        return min(500, max(0, excess_W))

    def _predict_COP(self, props: AlloyProperties) -> float:
        """
        Predict Coefficient of Performance.

        COP = 1 + excess_W / estimated_input_power

        Input power estimated as typical for LENR experiments:
          Electrolysis: 50-200 W input (current × voltage)
          Gas loading: 10-50 W input (heating + pressure)

        Calibrated against experimental COP values:
          McKubre: 2.1W excess / 56W input → COP=1.38
          Kitamura: 24W excess / 24W input → COP=2.0
          Brillouin: 60W excess / 48W input → COP=2.25
          Piantelli: 38.9W excess / 100W input → COP=1.38
          Constantan: 209W excess / 72W input → COP=3.91
        """
        excess_W = props.predicted_excess_heat_W

        if excess_W <= 0:
            return 1.0

        # Estimate typical input power based on loading conditions
        # Higher loading requires more input energy (electrolysis current)
        # Base input: ~50W for electrolysis, ~20W for gas loading
        loading = props.max_loading
        base_input = 50.0  # watts (typical electrolysis cell)

        # Loading correction: higher loading needs more input
        input_W = base_input * max(0.5, loading)

        COP = 1.0 + excess_W / input_W

        # Physical cap: most reproducible experiments show COP 1.1-4.0
        # Allow up to 10 for exceptional configurations
        return min(COP, 10.0)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    predictor = AlloyPredictor()

    print("=" * 70)
    print("  LENR ALLOY & COMPOSITE PREDICTOR")
    print("=" * 70)

    # 1. Known alloys validation
    print("\n--- KNOWN ALLOY VALIDATION ---")
    known_tests = [
        ('Pd pure', {'Pd': 1.0}, None),
        ('Ni pure', {'Ni': 1.0}, None),
        ('Fe pure', {'Fe': 1.0}, None),
        ('Constantan (Cu55-Ni45)', {'Cu': 0.55, 'Ni': 0.45}, None),
        ('NiCu Iwamura (50-50)', {'Ni': 0.50, 'Cu': 0.50}, None),
        ('SUS304', {'Fe': 0.70, 'Cr': 0.19, 'Ni': 0.10, 'Mn': 0.01}, None),
        ('Pd/CeO2 nanocomposite', {'Pd': 1.0}, 'CeO2'),
        ('Pd/ZrO2 Arata-Zhang', {'Pd': 1.0}, 'ZrO2'),
        ('Ni/CeO2 catalyst', {'Ni': 1.0}, 'CeO2'),
    ]

    for name, comp, oxide in known_tests:
        pred = predictor.predict_lenr_potential(
            comp,
            oxide_matrix=oxide,
            oxide_fraction=0.10 if oxide else 0.0,
            defect_concentration=0.1,
        )
        print(f"  {name:35s} | LENR={pred.lenr_score:5.1f}/100 | "
              f"Us={pred.predicted_Us_eV:6.0f} eV | COP={pred.predicted_COP:.2f}")

    # 2. Binary alloy scan
    print("\n--- TOP-20 BINARY ALLOYS ---")
    key_metals = ['Pd', 'Ni', 'Ti', 'Fe', 'Cu', 'Zr', 'Co', 'V', 'Ta', 'Nb', 'Cr']
    binary_df = predictor.scan_binary_alloys(
        elements=key_metals,
        fractions=[0.25, 0.50, 0.75],
        defect_concentration=0.1,
    )
    if len(binary_df) > 0:
        for _, row in binary_df.head(20).iterrows():
            print(f"  {row['alloy']:20s} | LENR={row['lenr_score']:5.1f} | "
                  f"Us={row['predicted_Us_eV']:6.0f} eV | COP={row['predicted_COP']:.2f}")

    # 3. Composite scan
    print("\n--- TOP-10 METAL/OXIDE COMPOSITES ---")
    comp_df = predictor.scan_with_oxides(defect_concentration=0.15)
    if len(comp_df) > 0:
        for _, row in comp_df.head(10).iterrows():
            print(f"  {row['composite']:30s} | LENR={row['lenr_score']:5.1f} | "
                  f"Us={row['predicted_Us_eV']:6.0f} eV")

    # 4. Novel combinations
    print("\n--- NOVEL PREDICTIONS (never tested in LENR) ---")
    novel = [
        ('Pd-Ni-W (Pd60-Ni30-W10)', {'Pd': 0.60, 'Ni': 0.30, 'W': 0.10}),
        ('Pd-Co-Zr (Pd50-Co30-Zr20)', {'Pd': 0.50, 'Co': 0.30, 'Zr': 0.20}),
        ('Ni-Fe-Cr-C (graphite steel)', {'Ni': 0.10, 'Fe': 0.70, 'Cr': 0.15, 'C': 0.05}),
        ('Pd-La-Ce (rare earth alloy)', {'Pd': 0.50, 'La': 0.25, 'Ce': 0.25}),
        ('Ti-Zr-Hf (group IV hydride)', {'Ti': 0.34, 'Zr': 0.33, 'Hf': 0.33}),
        ('Ni-V-Nb (BCC hydride former)', {'Ni': 0.40, 'V': 0.30, 'Nb': 0.30}),
        ('Co-Fe-Cr (ferromagnetic steel)', {'Co': 0.50, 'Fe': 0.30, 'Cr': 0.20}),
    ]

    for name, comp in novel:
        pred = predictor.predict_lenr_potential(
            comp, defect_concentration=0.2
        )
        print(f"\n  {name}")
        print(f"    LENR Score: {pred.lenr_score:.1f}/100")
        print(f"    Predicted Us: {pred.predicted_Us_eV:.0f} eV")
        print(f"    Predicted COP: {pred.predicted_COP:.2f}")
        if pred.advantages:
            print(f"    Advantages: {'; '.join(pred.advantages[:2])}")
        if pred.risk_factors:
            print(f"    Risks: {'; '.join(pred.risk_factors[:2])}")

    # 5. Summary stats
    print("\n--- SUMMARY ---")
    print(f"  Elements in database: {len(ELEMENT_DB)}")
    print(f"  Oxides in database: {len(OXIDE_DB)}")
    print(f"  Known alloys: {len(KNOWN_ALLOYS)}")
    if len(binary_df) > 0:
        print(f"  Binary combinations scanned: {len(binary_df)}")
        print(f"  Best binary: {binary_df.iloc[0]['alloy']} "
              f"(LENR={binary_df.iloc[0]['lenr_score']:.1f})")
