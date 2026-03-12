"""
LENR Data Generator V2 — Multi-Process Feature Generation
==========================================================
Generates training data with 64+ features covering ALL physical processes:
  - Material properties (15 features)
  - Loading & diffusion (10 features)
  - Barrier & screening (12 features)
  - Thermal & energy (8 features)
  - Electrochemistry (5 features)
  - Surface & nanostructure (6 features)
  - Stimulation effects (4 features)
  - Time dynamics (4 features)

Supports multi-task learning with 10 targets:
  - 6 classification (reaction, heat, neutron, tritium, He4, transmutation)
  - 4 regression (excess_heat_W, COP, neutron_rate, energy_density)
"""

import numpy as np
import pandas as pd
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physics_engine import PhysicsEngine, PhysicsMode
from lenr_constants import (
    NUCLEAR, LOADING, MIZUNO_R19_DATA, MIZUNO_NEUTRON_EXPERIMENTS,
    MIZUNO_NEUTRON_REACTOR, gamow_penetration, cross_section_DD,
    enhancement_factor, diffusion_coefficient,
)
from lenr_comprehensive_data import (
    MATERIALS_EXPANDED, SCREENING_COMPLETE, EXCESS_HEAT_COMPREHENSIVE,
    NUCLEAR_PRODUCTS_DATA, TRANSMUTATION_DATA, LOADING_DYNAMICS,
    THERMAL_DYNAMICS_DATA, SURFACE_EFFECTS_DATA, EM_STIMULATION_DATA,
    FEATURE_COLUMNS_V2, TARGET_COLUMNS_V2,
    STRUCTURE_ENCODING, METHOD_ENCODING, ELECTROLYTE_ENCODING,
    SURFACE_ENCODING, ISOTOPE_ENCODING, PHASE_ENCODING,
    get_feature_columns_v2, get_all_materials,
)

kB_eV = 8.617e-5  # eV/K
kB_meV = kB_eV * 1000  # meV/K


class LENRDataGeneratorV2:
    """Generate comprehensive LENR training data covering all physical processes."""

    # Materials with known LENR activity
    ACTIVE_MATERIALS = ['Pd', 'Ni', 'Ti', 'Fe', 'PdO', 'SUS304', 'Constantan', 'Zr']
    INACTIVE_MATERIALS = ['Au', 'Pt', 'W', 'Cu', 'Ta']
    ALL_MATERIALS = ACTIVE_MATERIALS + INACTIVE_MATERIALS

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.engines = {
            mode: PhysicsEngine(mode)
            for mode in ('maxwell', 'coulomb_original', 'cherepanov')
        }

    # =========================================================================
    # MATERIAL FEATURE EXTRACTION
    # =========================================================================
    def _get_material_features(self, material: str) -> dict:
        """Extract material-group features for a given material."""
        mat = MATERIALS_EXPANDED.get(material, MATERIALS_EXPANDED.get('Pd'))
        return {
            'atomic_number_Z': mat.get('Z', 46),
            'atomic_mass_amu': mat.get('A', 106),
            'crystal_structure_encoded': STRUCTURE_ENCODING.get(mat.get('structure', 'FCC'), 0),
            'lattice_constant_A': mat.get('a_A', 3.89),
            'debye_temperature_K': mat.get('debye_K', 274),
            'electron_density_A3': mat.get('e_density_A3', 0.1),
            'density_g_cm3': mat.get('density_g_cm3', 10.0),
            'melting_point_K': mat.get('melting_K', 1800),
            'bulk_modulus_GPa': mat.get('bulk_modulus_GPa', 180),
            'work_function_eV': mat.get('work_function_eV', 5.0),
            'fermi_energy_eV': mat.get('fermi_energy_eV', 5.5),
            'thermal_conductivity_W_mK': mat.get('thermal_conductivity_W_mK', 70),
            'specific_heat_J_gK': mat.get('specific_heat_J_gK', 0.25),
            'n_valence_electrons': mat.get('n_valence', 10),
            'n_stable_isotopes': mat.get('n_stable_isotopes', 5),
        }

    def _get_screening(self, material: str) -> float:
        """Get screening energy for material."""
        scr = SCREENING_COMPLETE.get(material)
        if scr:
            return scr['Us_eV']
        # Fallback: use base material
        base = material.replace('O', '').split('_')[0]
        scr = SCREENING_COMPLETE.get(base)
        if scr:
            return scr['Us_eV']
        return 50.0

    def _get_max_loading(self, material: str) -> float:
        """Get max loading capacity."""
        mat = MATERIALS_EXPANDED.get(material, {})
        return mat.get('max_H_loading', 0.5)

    def _get_diffusion_params(self, material: str) -> tuple:
        """Get diffusion D0 and Ea for material."""
        mat = MATERIALS_EXPANDED.get(material, {})
        D0 = mat.get('D0_cm2s', 1e-3)
        Ea = mat.get('Ea_diffusion_eV', 0.3)
        return D0, Ea

    def _calc_diffusion(self, material: str, T_K: float) -> float:
        """Calculate diffusion coefficient at temperature T."""
        D0, Ea = self._get_diffusion_params(material)
        return D0 * np.exp(-Ea / (kB_eV * T_K))

    def _determine_phase(self, material: str, loading: float) -> int:
        """Determine PdD phase: 0=alpha, 1=mixed, 2=beta."""
        if material in ('Pd', 'PdO'):
            if loading < 0.017:
                return 0
            elif loading < 0.58:
                return 1
            else:
                return 2
        return 1  # default mixed for other materials

    def _lattice_expansion(self, material: str, loading: float) -> float:
        """Estimate lattice expansion percentage from loading."""
        if material in ('Pd', 'PdO'):
            if loading < 0.017:
                return 0.0
            elif loading < 0.58:
                return loading / 0.58 * 3.4
            else:
                return 3.4 + (loading - 0.58) * 3.8
        elif material == 'Ti':
            return loading * 5.0  # large expansion
        return loading * 2.0  # generic

    # =========================================================================
    # BARRIER & SCREENING CALCULATIONS
    # =========================================================================
    def _calc_barrier_features(self, material: str, E_cm_keV: float,
                                T_K: float, D_loading: float) -> dict:
        """Calculate barrier/screening features for all 3 physics modes."""
        Us = self._get_screening(material)
        enh = enhancement_factor(E_cm_keV, Us) if E_cm_keV > 0.01 else 1.0
        sigma = cross_section_DD(E_cm_keV) if E_cm_keV > 0 else 1e-50
        sigma_log = np.log10(max(sigma, 1e-50))

        features = {
            'screening_energy_eV': Us,
            'enhancement_factor': enh,
            'log_cross_section': sigma_log,
        }

        # Use base material for physics engine (strip suffixes)
        base_mat = material
        for suffix in ('_R', '_H', 'O', '_Raiola', '_Huke'):
            base_mat = base_mat.replace(suffix, '')
        if base_mat not in ('Pd', 'Ni', 'Fe', 'Ti', 'Au', 'Pt', 'W', 'Cu'):
            base_mat = 'Pd'  # fallback

        # Map engine key → short column name (consistent with V1 and feature spec)
        _col_name = {
            'maxwell': 'maxwell',
            'coulomb_original': 'coulomb',
            'cherepanov': 'cherepanov',
        }

        for mode_name, engine in self.engines.items():
            try:
                br = engine.calculate_barrier(base_mat, E_cm_keV, T_K, D_loading)
            except Exception:
                br = engine.calculate_barrier('Pd', E_cm_keV, T_K, D_loading)

            col = _col_name.get(mode_name, mode_name)
            barrier_keV = max(br.barrier_keV, 1.0)
            features[f'barrier_reduction_{col}'] = br.effective_barrier_keV / barrier_keV
            features[f'log_penetration_{col}'] = np.log10(max(br.penetration_probability, 1e-300))
            features[f'log_rate_{col}'] = np.log10(max(br.reaction_rate_relative, 1e-300))

        return features

    # =========================================================================
    # REACTION PROBABILITY ESTIMATION (MULTI-TARGET)
    # =========================================================================
    def _estimate_reactions(
        self, material: str, E_cm_keV: float, T_K: float,
        D_loading: float, Us_eV: float, pressure_Pa: float,
        method: str = 'gas_loading', gas: str = 'D2',
        surface: str = 'none', nanostructure: bool = False,
        n_layers: int = 0, laser: bool = False, rf: bool = False,
    ) -> dict:
        """Estimate probabilities for all nuclear signatures."""
        prob = {}

        # Base reaction probability
        p_base = 0.0

        # Loading effect
        max_load = self._get_max_loading(material)
        load_frac = D_loading / max(max_load, 0.01) if max_load > 0 else 0.0
        if material in ('Pd', 'PdO') and D_loading > 0.84:
            p_base += 0.3 * ((D_loading - 0.84) / 0.16) ** 2
        elif material in ('Ni', 'Constantan', 'SUS304'):
            # Ni-based: works at low loading (Piantelli, Mizuno)
            if D_loading > 0.001:
                p_base += 0.15
        if D_loading > 0.90:
            p_base += 0.15

        # Screening
        p_base += min(Us_eV / 2000, 0.25)

        # Material bonus
        material_bonus = {
            'Pd': 0.10, 'PdO': 0.15, 'Ni': 0.08, 'Ti': 0.03,
            'Fe': 0.02, 'SUS304': 0.05, 'Constantan': 0.12,
            'Zr': 0.04, 'Au': 0.01, 'Pt': 0.01,
        }
        p_base += material_bonus.get(material, 0.02)

        # Temperature
        if 300 < T_K < 600:
            p_base += 0.05
        if T_K > 600:
            p_base += 0.08  # Mizuno SUS304 works at high T

        # Surface / nanostructure boost
        if nanostructure:
            p_base += 0.12
        if n_layers > 10:
            p_base += 0.10  # multilayer Iwamura effect

        # Stimulation boost
        if laser:
            p_base += 0.08
        if rf:
            p_base += 0.06

        # Gas type
        if gas in ('D2', 'D2O'):
            p_base += 0.05
        elif gas in ('H2', 'H2O'):
            p_base += 0.02  # H works too (Piantelli, Mizuno SUS304)

        p_base = np.clip(p_base, 0.001, 0.95)

        # Individual nuclear signatures
        prob['reaction_occurred'] = p_base

        # Excess heat: most common signature
        prob['excess_heat_detected'] = p_base * 0.90

        # Helium-4: correlates with excess heat in Pd experiments
        if material in ('Pd', 'PdO', 'nano_Pd') and gas in ('D2', 'D2O'):
            prob['He4_detected'] = p_base * 0.60
        else:
            prob['He4_detected'] = p_base * 0.10

        # Tritium: anomalous n/T ratio
        if gas in ('D2', 'D2O'):
            prob['tritium_detected'] = p_base * 0.15
        else:
            prob['tritium_detected'] = p_base * 0.01

        # Neutrons: rare in LENR, more common in SUS304+H2
        if material == 'SUS304' and gas in ('H2', 'H2O') and T_K > 700:
            prob['neutron_detected'] = 0.70
        else:
            prob['neutron_detected'] = p_base * 0.05

        # Transmutations: mainly Pd/CaO multilayer or high-current electrolysis
        if n_layers > 10:
            prob['transmutation_detected'] = 0.60
        elif method == 'electrolysis' and material in ('Pd', 'Au'):
            prob['transmutation_detected'] = p_base * 0.15
        else:
            prob['transmutation_detected'] = p_base * 0.03

        return prob

    def _estimate_excess_heat_W(
        self, material: str, D_loading: float, T_K: float,
        Us_eV: float, reaction_prob: float, input_W: float = 0,
    ) -> float:
        """Estimate excess heat output."""
        if reaction_prob < 0.05:
            return 0.0

        base_W = 5.0
        # Loading factor
        max_load = self._get_max_loading(material)
        if material in ('Pd', 'PdO') and D_loading > 0.84:
            load_f = ((D_loading - 0.84) / 0.16) ** 3
        elif material in ('Ni', 'SUS304', 'Constantan'):
            load_f = max(D_loading / 0.01, 0.1)  # low loading still works
        else:
            load_f = max(D_loading / max(max_load, 0.01), 0.01)

        screen_f = Us_eV / 300
        mat_factor = {
            'Pd': 1.0, 'PdO': 1.5, 'Ni': 0.8, 'Ti': 0.2,
            'Fe': 0.3, 'SUS304': 0.5, 'Constantan': 2.0,
            'Zr': 0.3, 'Au': 0.1, 'Pt': 0.15, 'Cu': 0.05,
        }
        mf = mat_factor.get(material, 0.3)

        return base_W * load_f * screen_f * mf * reaction_prob * 10

    # =========================================================================
    # MAIN GENERATION: SYNTHETIC DATA
    # =========================================================================
    def generate_synthetic(
        self,
        n_samples: int = 10000,
        noise_level: float = 0.05,
    ) -> pd.DataFrame:
        """Generate synthetic training data by sweeping full parameter space."""
        records = []

        for _ in range(n_samples):
            material = self.rng.choice(self.ALL_MATERIALS)
            mat_props = self._get_material_features(material)

            # Sample conditions
            T_K = float(self.rng.uniform(200, 1100))
            E_cm_keV = float(max(kB_eV * T_K / 1000.0, 0.01))  # thermal energy
            max_load = self._get_max_loading(material)
            D_loading = float(self.rng.uniform(0.0, min(max_load * 1.1, 3.0)))
            pressure_Pa = float(10 ** self.rng.uniform(1, 6))

            # Method & gas
            method = self.rng.choice(['electrolysis', 'gas_loading', 'gas_permeation'])
            gas = self.rng.choice(['D2', 'H2']) if method != 'electrolysis' else 'D2O'
            isotope = ISOTOPE_ENCODING.get(gas, 2)

            # Surface
            surface = self.rng.choice(['none', 'annealed', 'polished', 'mesh', 'nano', 'multilayer'],
                                       p=[0.3, 0.15, 0.15, 0.15, 0.15, 0.10])
            nanostructure = surface == 'nano'
            n_layers = int(self.rng.integers(50, 300)) if surface == 'multilayer' else 0
            particle_nm = float(self.rng.uniform(2, 50)) if nanostructure else 0.0
            coating_nm = float(self.rng.uniform(10, 500)) if surface in ('multilayer', 'mesh') else 0.0

            # Stimulation
            laser = bool(self.rng.random() < 0.05)
            rf = bool(self.rng.random() < 0.05)
            B_field = float(self.rng.uniform(0, 2.0)) if self.rng.random() < 0.1 else 0.0

            # Electrochemistry (only for electrolysis)
            if method == 'electrolysis':
                current_density = float(self.rng.uniform(0.05, 2.0))
                cell_voltage = float(self.rng.uniform(2.0, 8.0))
                electrolyte = self.rng.choice(['D2O_LiOD', 'D2O_NaOD', 'H2O_LiOH'])
                cathode_area = float(self.rng.uniform(0.1, 10.0))
                overpotential = cell_voltage - 1.23  # vs thermodynamic
            else:
                current_density = 0.0
                cell_voltage = 0.0
                electrolyte = 'none'
                cathode_area = 0.0
                overpotential = 0.0

            # Input power
            input_W = float(self.rng.uniform(0, 300)) if method != 'electrolysis' \
                else current_density * cell_voltage * cathode_area

            # Screening with noise
            Us_eV = self._get_screening(material)
            Us_noisy = Us_eV + self.rng.normal(0, Us_eV * noise_level)

            # Loading features
            phase = self._determine_phase(material, D_loading)
            lat_exp = self._lattice_expansion(material, D_loading)
            H_enthalpy = MATERIALS_EXPANDED.get(material, {}).get('H_absorption_enthalpy_eV', -0.2)

            # Barrier features
            barrier_feats = self._calc_barrier_features(material, E_cm_keV, T_K, D_loading)

            # Diffusion
            try:
                D_coeff = self._calc_diffusion(material, T_K)
            except Exception:
                D_coeff = 1e-10
            _, Ea_diff = self._get_diffusion_params(material)

            # Thermal phonon energy
            debye_K = mat_props['debye_temperature_K']
            phonon_meV = kB_meV * debye_K  # meV

            # Heating rate (random for synthetic)
            heating_rate = float(self.rng.uniform(0, 50))

            # Time dynamics
            duration_hours = float(10 ** self.rng.uniform(0, 4))
            incubation_hours = float(self.rng.uniform(0, duration_hours * 0.3))

            # Estimate all reaction probabilities
            probs = self._estimate_reactions(
                material, E_cm_keV, T_K, D_loading, Us_noisy, pressure_Pa,
                method, gas, surface, nanostructure, n_layers, laser, rf,
            )

            # Add threshold noise for deterministic labeling
            threshold_noise = self.rng.normal(0, 0.05)
            labels = {}
            for key, p in probs.items():
                labels[key] = int(p > (0.30 + threshold_noise))

            # Continuous targets
            excess_W = self._estimate_excess_heat_W(
                material, D_loading, T_K, Us_noisy,
                probs['reaction_occurred'], input_W,
            )
            excess_W *= (1 + self.rng.normal(0, noise_level))
            excess_W = max(excess_W, 0.0)

            COP_val = (1.0 + excess_W / max(input_W, 1.0)) if labels['excess_heat_detected'] else 1.0
            neutron_cpm = probs['neutron_detected'] * self.rng.uniform(0.1, 5) if labels['neutron_detected'] else 0.0
            energy_density = excess_W * duration_hours * 3.6 / max(
                mat_props['density_g_cm3'] * 10, 1)  # kJ/g rough estimate

            # Build record
            record = {
                'material': material,
                # Material properties
                **mat_props,
                # Loading
                'hydrogen_isotope': isotope,
                'loading_ratio': D_loading,
                'max_loading_capacity': max_load,
                'loading_method_encoded': METHOD_ENCODING.get(method, 1),
                'loading_fraction': D_loading / max(max_load, 0.001),
                'above_McKubre_threshold': int(D_loading > 0.84),
                'above_Storms_threshold': int(D_loading > 0.90),
                'phase_encoded': phase,
                'H_absorption_enthalpy_eV': H_enthalpy,
                'lattice_expansion_pct': lat_exp,
                # Barrier
                **barrier_feats,
                # Thermal
                'temperature_K': T_K,
                'pressure_Pa': pressure_Pa,
                'beam_energy_keV': E_cm_keV,
                'input_power_W': input_W,
                'heating_rate_K_per_min': heating_rate,
                'diffusion_coefficient': D_coeff,
                'diffusion_activation_eV': Ea_diff,
                'thermal_phonon_energy_meV': phonon_meV,
                # Electrochemistry
                'current_density_A_cm2': current_density,
                'cell_voltage_V': cell_voltage,
                'electrolyte_encoded': ELECTROLYTE_ENCODING.get(electrolyte, 0),
                'cathode_area_cm2': cathode_area,
                'overpotential_V': overpotential,
                # Surface
                'surface_treatment_encoded': SURFACE_ENCODING.get(surface, 0),
                'particle_size_nm': particle_nm,
                'n_layers': n_layers,
                'coating_thickness_nm': coating_nm,
                'surface_area_ratio': (1000 / max(particle_nm, 1)) if nanostructure else 1.0,
                'nanostructure_flag': int(nanostructure),
                # Stimulation
                'laser_stimulation': int(laser),
                'rf_stimulation': int(rf),
                'applied_B_field_T': B_field,
                'ultrasound_stimulation': 0,
                # Time
                'experiment_duration_hours': duration_hours,
                'incubation_time_hours': incubation_hours,
                'COP': COP_val,
                'data_source_encoded': 0,  # synthetic
                # Labels
                **labels,
                'excess_heat_W': excess_W,
                'neutron_rate_cpm': neutron_cpm,
                'energy_density_kJ_g': energy_density,
                # Metadata
                'data_source': 'synthetic',
                'method': method,
                'gas': gas,
            }
            records.append(record)

        return pd.DataFrame(records)

    # =========================================================================
    # REAL DATA: EXCESS HEAT EXPERIMENTS
    # =========================================================================
    def generate_real_excess_heat(self) -> pd.DataFrame:
        """Convert comprehensive real excess heat data to V2 features."""
        records = []

        for exp in EXCESS_HEAT_COMPREHENSIVE:
            material = exp['material']
            base_mat = material.replace('nano_', '').replace('PdNi_', '').replace('NiCu', 'Ni')
            mat_props = self._get_material_features(
                base_mat if base_mat in MATERIALS_EXPANDED else 'Pd'
            )

            T_K = exp.get('temperature_K', 340)
            E_cm_keV = max(kB_eV * T_K / 1000.0, 0.01)
            D_loading = exp.get('DPd', 0.5)
            pressure_Pa = exp.get('pressure_Pa', 1e5)
            excess_W = exp.get('excess_W_typical', exp.get('excess_W_min', 0))
            COP = exp.get('COP', 1.0) or 1.0

            method = exp.get('method', 'gas_loading')
            gas = exp.get('gas', 'D2')
            isotope = ISOTOPE_ENCODING.get(gas, 2)
            electrolyte = exp.get('electrolyte', 'none')

            Us_eV = self._get_screening(base_mat)
            barrier_feats = self._calc_barrier_features(base_mat, E_cm_keV, T_K, D_loading)

            try:
                D_coeff = self._calc_diffusion(base_mat, T_K)
            except Exception:
                D_coeff = 1e-10
            _, Ea_diff = self._get_diffusion_params(base_mat)
            debye_K = mat_props['debye_temperature_K']

            # Surface
            substrate = exp.get('substrate', '')
            nano = 'nano' in material or 'nano' in substrate
            multilayer = 'multilayer' in substrate or 'CaO' in substrate
            n_layers_val = exp.get('n_layers', 200 if multilayer else 0)
            surf_enc = 5 if nano else (6 if multilayer else 0)

            duration = exp.get('duration_hours', 100)
            input_W_est = excess_W / max(COP - 1, 0.01) if COP > 1 else excess_W * 5

            record = {
                'material': material,
                **mat_props,
                'hydrogen_isotope': isotope,
                'loading_ratio': D_loading,
                'max_loading_capacity': self._get_max_loading(base_mat),
                'loading_method_encoded': METHOD_ENCODING.get(method, 1),
                'loading_fraction': D_loading / max(self._get_max_loading(base_mat), 0.01),
                'above_McKubre_threshold': int(D_loading > 0.84),
                'above_Storms_threshold': int(D_loading > 0.90),
                'phase_encoded': self._determine_phase(base_mat, D_loading),
                'H_absorption_enthalpy_eV': MATERIALS_EXPANDED.get(base_mat, {}).get(
                    'H_absorption_enthalpy_eV', -0.2),
                'lattice_expansion_pct': self._lattice_expansion(base_mat, D_loading),
                **barrier_feats,
                'temperature_K': T_K,
                'pressure_Pa': pressure_Pa,
                'beam_energy_keV': E_cm_keV,
                'input_power_W': input_W_est,
                'heating_rate_K_per_min': 0,
                'diffusion_coefficient': D_coeff,
                'diffusion_activation_eV': Ea_diff,
                'thermal_phonon_energy_meV': kB_meV * debye_K,
                'current_density_A_cm2': exp.get('current_density_A_cm2', 0),
                'cell_voltage_V': exp.get('cell_voltage_V', 0),
                'electrolyte_encoded': ELECTROLYTE_ENCODING.get(electrolyte, 0),
                'cathode_area_cm2': exp.get('cathode_area_cm2', 0),
                'overpotential_V': max(exp.get('cell_voltage_V', 0) - 1.23, 0),
                'surface_treatment_encoded': surf_enc,
                'particle_size_nm': exp.get('particle_size_nm', 0),
                'n_layers': n_layers_val,
                'coating_thickness_nm': 0,
                'surface_area_ratio': 200 if nano else 1.0,
                'nanostructure_flag': int(nano),
                'laser_stimulation': int(exp.get('laser_stimulation', False)),
                'rf_stimulation': int(exp.get('rf_stimulation', False)),
                'applied_B_field_T': 0,
                'ultrasound_stimulation': 0,
                'experiment_duration_hours': duration,
                'incubation_time_hours': 0,
                'COP': COP,
                'data_source_encoded': 1,
                # Labels
                'reaction_occurred': 1,
                'excess_heat_detected': 1,
                'He4_detected': int(exp.get('He4_detected', False)),
                'tritium_detected': int(exp.get('tritium_detected', False)),
                'neutron_detected': int(exp.get('neutron_detected', False)),
                'transmutation_detected': int(exp.get('transmutation_detected', False)),
                'excess_heat_W': excess_W,
                'neutron_rate_cpm': 0,
                'energy_density_kJ_g': exp.get('total_energy_MJ', excess_W * duration * 3.6e-3) * 1000 /
                                        max(mat_props['density_g_cm3'] * 10, 1),
                'data_source': 'experimental',
                'method': method,
                'gas': gas,
            }
            records.append(record)

        return pd.DataFrame(records)

    # =========================================================================
    # REAL DATA: MIZUNO R19
    # =========================================================================
    def generate_mizuno_r19(self) -> pd.DataFrame:
        """Convert Mizuno R19 data (55 points) to V2 features."""
        records = []
        mat_props = self._get_material_features('Ni')
        Us_Ni = self._get_screening('Ni')

        for m in MIZUNO_R19_DATA:
            if m['input_W'] == 0:
                continue

            T_K = m['temp_C'] + 273.15
            E_cm_keV = max(kB_eV * T_K / 1000.0, 0.01)
            D_loading = m['D_Ni']
            pressure_Pa = m['pressure_Pa']

            barrier_feats = self._calc_barrier_features('Ni', E_cm_keV, T_K, D_loading)
            try:
                D_coeff = self._calc_diffusion('Ni', T_K)
            except Exception:
                D_coeff = 1e-10

            COP = m['COP'] or 1.0

            record = {
                'material': 'NiPd',
                **mat_props,
                'hydrogen_isotope': 2,  # D2
                'loading_ratio': D_loading,
                'max_loading_capacity': 0.03,
                'loading_method_encoded': METHOD_ENCODING['gas_loading'],
                'loading_fraction': D_loading / 0.03,
                'above_McKubre_threshold': 0,
                'above_Storms_threshold': 0,
                'phase_encoded': 1,
                'H_absorption_enthalpy_eV': -0.16,
                'lattice_expansion_pct': D_loading * 2.0,
                **barrier_feats,
                'temperature_K': T_K,
                'pressure_Pa': pressure_Pa,
                'beam_energy_keV': E_cm_keV,
                'input_power_W': m['input_W'],
                'heating_rate_K_per_min': 0,
                'diffusion_coefficient': D_coeff,
                'diffusion_activation_eV': 0.457,
                'thermal_phonon_energy_meV': kB_meV * 450,
                'current_density_A_cm2': 0,
                'cell_voltage_V': 0,
                'electrolyte_encoded': 0,
                'cathode_area_cm2': 0,
                'overpotential_V': 0,
                'surface_treatment_encoded': SURFACE_ENCODING['mesh'],
                'particle_size_nm': 0,
                'n_layers': 0,
                'coating_thickness_nm': 17.5,  # Pd coating
                'surface_area_ratio': 50,  # mesh has high surface area
                'nanostructure_flag': 0,
                'laser_stimulation': 0,
                'rf_stimulation': 0,
                'applied_B_field_T': 0,
                'ultrasound_stimulation': 0,
                'experiment_duration_hours': 24,
                'incubation_time_hours': 0,
                'COP': COP,
                'data_source_encoded': 2,
                # Labels
                'reaction_occurred': 1,
                'excess_heat_detected': 1,
                'He4_detected': 0,
                'tritium_detected': 0,
                'neutron_detected': 0,
                'transmutation_detected': 0,
                'excess_heat_W': m['excess_W'],
                'neutron_rate_cpm': 0,
                'energy_density_kJ_g': m['heat_per_g'] * 3.6,
                'data_source': 'mizuno_r19',
                'method': 'gas_loading',
                'gas': 'D2',
            }
            records.append(record)

        return pd.DataFrame(records)

    # =========================================================================
    # REAL DATA: MIZUNO NEUTRON SUS304
    # =========================================================================
    def generate_mizuno_neutron(self) -> pd.DataFrame:
        """Convert Mizuno SUS304 neutron experiments to V2 features."""
        records = []
        mat_props = self._get_material_features('SUS304')
        Us_Fe = self._get_screening('Fe')

        for exp in MIZUNO_NEUTRON_EXPERIMENTS:
            T_K = exp['temp_C'] + 273.15
            E_cm_keV = max(kB_eV * T_K / 1000.0, 0.01)
            pressure_Pa = MIZUNO_NEUTRON_REACTOR['initial_pressure_Pa']

            barrier_feats = self._calc_barrier_features('Fe', E_cm_keV, T_K, 0.001)
            try:
                D_coeff = self._calc_diffusion('Fe', T_K)
            except Exception:
                D_coeff = 1e-10

            record = {
                'material': 'SUS304',
                **mat_props,
                'hydrogen_isotope': 1,  # H2, not D2
                'loading_ratio': 0.001,
                'max_loading_capacity': 0.001,
                'loading_method_encoded': METHOD_ENCODING['gas_loading'],
                'loading_fraction': 1.0,
                'above_McKubre_threshold': 0,
                'above_Storms_threshold': 0,
                'phase_encoded': 1,
                'H_absorption_enthalpy_eV': 0.29,  # Fe: endothermic
                'lattice_expansion_pct': 0.01,
                **barrier_feats,
                'temperature_K': T_K,
                'pressure_Pa': pressure_Pa,
                'beam_energy_keV': E_cm_keV,
                'input_power_W': exp['input_W'],
                'heating_rate_K_per_min': 10,  # estimated
                'diffusion_coefficient': D_coeff,
                'diffusion_activation_eV': 0.041,
                'thermal_phonon_energy_meV': kB_meV * 400,
                'current_density_A_cm2': 0,
                'cell_voltage_V': 0,
                'electrolyte_encoded': 0,
                'cathode_area_cm2': 0,
                'overpotential_V': 0,
                'surface_treatment_encoded': SURFACE_ENCODING['mesh_400_buffed'],
                'particle_size_nm': 0,
                'n_layers': 0,
                'coating_thickness_nm': 0,
                'surface_area_ratio': 10,
                'nanostructure_flag': 0,
                'laser_stimulation': 0,
                'rf_stimulation': 0,
                'applied_B_field_T': 0,
                'ultrasound_stimulation': 0,
                'experiment_duration_hours': 12,
                'incubation_time_hours': 2,
                'COP': MIZUNO_NEUTRON_REACTOR['output_ratio'],
                'data_source_encoded': 3,
                # Labels
                'reaction_occurred': 1,
                'excess_heat_detected': 1,
                'He4_detected': 0,
                'tritium_detected': 0,
                'neutron_detected': 1,
                'transmutation_detected': 0,
                'excess_heat_W': MIZUNO_NEUTRON_REACTOR['excess_heat_W'],
                'neutron_rate_cpm': exp['neutron_cpm'],
                'energy_density_kJ_g': 0,
                'data_source': 'mizuno_neutron',
                'method': 'gas_loading',
                'gas': 'H2',
            }
            records.append(record)

        return pd.DataFrame(records)

    # =========================================================================
    # COMBINED DATASET
    # =========================================================================
    def generate_combined(
        self,
        n_synthetic: int = 10000,
        noise_level: float = 0.05,
        include_mizuno_r19: bool = True,
        include_mizuno_neutron: bool = True,
        include_real_excess_heat: bool = True,
    ) -> pd.DataFrame:
        """Generate combined dataset with all sources.

        Returns DataFrame with 64+ features and 10 targets.
        """
        frames = []

        # Synthetic
        print(f"Generating {n_synthetic} synthetic samples...")
        synthetic = self.generate_synthetic(n_synthetic, noise_level)
        frames.append(synthetic)

        # Real excess heat
        if include_real_excess_heat:
            print(f"Adding {len(EXCESS_HEAT_COMPREHENSIVE)} excess heat experiments...")
            real_heat = self.generate_real_excess_heat()
            frames.append(real_heat)

        # Mizuno R19
        if include_mizuno_r19:
            print("Adding Mizuno R19 data (55 points)...")
            mizuno = self.generate_mizuno_r19()
            frames.append(mizuno)

        # Mizuno neutron
        if include_mizuno_neutron:
            print("Adding Mizuno SUS304 neutron data (10 points)...")
            neutron = self.generate_mizuno_neutron()
            frames.append(neutron)

        # Find common columns and merge
        common_cols = set(frames[0].columns)
        for f in frames[1:]:
            common_cols &= set(f.columns)
        common_cols = sorted(list(common_cols))

        combined = pd.concat([f[common_cols] for f in frames], ignore_index=True)
        print(f"\nCombined dataset: {combined.shape[0]} rows, {combined.shape[1]} columns")
        print(f"  Synthetic: {len(synthetic)}")
        if include_real_excess_heat:
            print(f"  Real excess heat: {len(real_heat)}")
        if include_mizuno_r19:
            print(f"  Mizuno R19: {len(mizuno)}")
        if include_mizuno_neutron:
            print(f"  Mizuno neutron: {len(neutron)}")

        return combined

    # =========================================================================
    # FEATURE COLUMNS HELPERS
    # =========================================================================
    @staticmethod
    def get_feature_columns() -> list[str]:
        """Return all V2 feature column names (64 features)."""
        return get_feature_columns_v2()

    @staticmethod
    def get_classification_targets() -> list[str]:
        """Return classification target column names."""
        return TARGET_COLUMNS_V2['classification']

    @staticmethod
    def get_regression_targets() -> list[str]:
        """Return regression target column names."""
        return TARGET_COLUMNS_V2['regression']


# =============================================================================
# CLI
# =============================================================================
if __name__ == '__main__':
    gen = LENRDataGeneratorV2(seed=42)

    print("=" * 60)
    print("LENR Data Generator V2 — Multi-Process Features")
    print("=" * 60)

    # Generate combined dataset
    df = gen.generate_combined(n_synthetic=2000)

    print(f"\nFeature columns ({len(gen.get_feature_columns())}):")
    for group, cols in FEATURE_COLUMNS_V2.items():
        print(f"  {group} ({len(cols)}): {cols}")

    print(f"\nClassification targets: {gen.get_classification_targets()}")
    print(f"Regression targets: {gen.get_regression_targets()}")

    print(f"\nDataset statistics:")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")

    # Stats per target
    for target in gen.get_classification_targets():
        if target in df.columns:
            print(f"  {target}: {df[target].mean():.1%} positive")

    for target in gen.get_regression_targets():
        if target in df.columns:
            vals = df[df[target] > 0][target]
            if len(vals) > 0:
                print(f"  {target}: mean={vals.mean():.2f}, max={vals.max():.2f}")

    print(f"\nMaterials: {df['material'].value_counts().to_dict()}")
    print(f"Data sources: {df['data_source'].value_counts().to_dict()}")
