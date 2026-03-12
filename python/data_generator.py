"""
Synthetic data generator for LENR ML training.
Generates labeled datasets from the physics engine across parameter space.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
import sys, os

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physics_engine import PhysicsEngine, PhysicsMode
from lenr_constants import (
    SCREENING_EXPERIMENTAL, LATTICE, DIFFUSION, LOADING,
    EXCESS_HEAT_DATA, EQPET_SCREENING, BARRIER_FACTORS,
    MIZUNO_R19_DATA, MIZUNO_EMPIRICAL,
    MIZUNO_NEUTRON_EXPERIMENTS, MIZUNO_NEUTRON_REACTOR,
    diffusion_coefficient, enhancement_factor, cross_section_DD,
)


# Materials available for simulation
SIMULATION_MATERIALS = ['Pd', 'Ni', 'Fe', 'Ti', 'Au', 'Pt', 'PdO']

# Oxide → base element mapping (safe alternative to .replace('O', ''))
_OXIDE_TO_BASE = {'PdO': 'Pd', 'BeO': 'Be', 'TiO2': 'Ti', 'ZrO2': 'Zr'}


def _strip_oxide(material: str) -> str:
    """Get base element from material, safely handling oxides.

    Unlike .replace('O', ''), this does NOT corrupt 'Co' → 'C' or 'Mo' → 'M'.
    """
    return _OXIDE_TO_BASE.get(material, material)


class LENRDataGenerator:
    """Generate synthetic training data for LENR ML models."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.engines = {
            mode: PhysicsEngine(mode)
            for mode in ('maxwell', 'coulomb_original', 'cherepanov')
        }

    def generate_parameter_sweep(
        self,
        n_samples: int = 5000,
        noise_level: float = 0.05,
    ) -> pd.DataFrame:
        """Generate synthetic data by sweeping parameter space."""
        records = []

        for _ in range(n_samples):
            # Sample parameters
            material = self.rng.choice(SIMULATION_MATERIALS)
            E_cm_keV = float(self.rng.uniform(0.1, 50.0))
            T_K = float(self.rng.uniform(200, 800))
            D_loading = float(self.rng.uniform(0.0, 1.0))
            pressure_Pa = float(10 ** self.rng.uniform(4, 6))  # 10kPa to 1MPa

            # Material properties
            lat = LATTICE.get(_strip_oxide(material), LATTICE.get('Pd'))
            diff_data = DIFFUSION.get(_strip_oxide(material))
            scr = SCREENING_EXPERIMENTAL.get(material, {'Us_eV': 50, 'error_eV': 10})

            lattice_a = lat['a_A'] if lat else 3.89
            debye_K = lat['debye_K'] if lat else 274
            e_density = lat.get('e_density_A3', 0.1) if lat else 0.1
            structure = lat['structure'] if lat else 'FCC'

            Us_eV = scr['Us_eV']
            Us_error = scr.get('error_eV', Us_eV * 0.1)
            # Add measurement noise
            Us_noisy = Us_eV + self.rng.normal(0, Us_error * noise_level)

            # Diffusion
            D_coeff = 0.0
            if diff_data:
                try:
                    D_coeff = diffusion_coefficient(_strip_oxide(material), T_K)
                except Exception as e:
                    logger.debug("Diffusion calc failed for %s at %g K: %s", material, T_K, e)
                    D_coeff = 1e-10

            # Enhancement
            enh = enhancement_factor(E_cm_keV, Us_noisy) if E_cm_keV > 0.01 else 1.0

            # Cross-sections
            sigma_bare = cross_section_DD(E_cm_keV) if E_cm_keV > 0 else 0
            sigma_log = np.log10(max(sigma_bare, 1e-50))

            # Calculate barriers for all 3 modes
            barrier_results = {}
            for mode_name, engine in self.engines.items():
                br = engine.calculate_barrier(material, E_cm_keV, T_K, D_loading)
                barrier_results[mode_name] = br

            # Determine if reaction is likely (label)
            # Based on multiple criteria from experimental data
            reaction_probability = self._estimate_reaction_prob(
                material, E_cm_keV, T_K, D_loading, Us_noisy, pressure_Pa,
            )
            # Deterministic threshold with small noise (not random coin flip!)
            # This makes labels a learnable function of features
            threshold_noise = self.rng.normal(0, 0.05)
            is_reaction = int(reaction_probability > (0.30 + threshold_noise))

            # Excess heat estimation (for regression)
            excess_heat_W = self._estimate_excess_heat(
                material, D_loading, T_K, Us_noisy, reaction_probability,
            )
            # Add noise
            excess_heat_W *= (1 + self.rng.normal(0, noise_level))
            excess_heat_W = max(excess_heat_W, 0)

            record = {
                # Features
                'material': material,
                'material_encoded': hash(material) % 100,
                'structure': structure,
                'lattice_constant_A': lattice_a,
                'debye_temperature_K': debye_K,
                'electron_density_A3': e_density,
                'screening_energy_eV': Us_noisy,
                'beam_energy_keV': E_cm_keV,
                'temperature_K': T_K,
                'deuterium_loading': D_loading,
                'pressure_Pa': pressure_Pa,
                'diffusion_coefficient': D_coeff,
                'enhancement_factor': enh,
                'log_cross_section': sigma_log,
                'barrier_reduction_maxwell': barrier_results['maxwell'].effective_barrier_keV / max(barrier_results['maxwell'].barrier_keV, 1),
                'barrier_reduction_coulomb': barrier_results['coulomb_original'].effective_barrier_keV / max(barrier_results['coulomb_original'].barrier_keV, 1),
                'barrier_reduction_cherepanov': barrier_results['cherepanov'].effective_barrier_keV / max(barrier_results['cherepanov'].barrier_keV, 1),
                'penetration_maxwell': barrier_results['maxwell'].penetration_probability,
                'penetration_coulomb': barrier_results['coulomb_original'].penetration_probability,
                'penetration_cherepanov': barrier_results['cherepanov'].penetration_probability,
                'rate_maxwell': barrier_results['maxwell'].reaction_rate_relative,
                'rate_coulomb': barrier_results['coulomb_original'].reaction_rate_relative,
                'rate_cherepanov': barrier_results['cherepanov'].reaction_rate_relative,
                'log_rate_maxwell': np.log10(max(barrier_results['maxwell'].reaction_rate_relative, 1e-300)),
                'log_rate_coulomb': np.log10(max(barrier_results['coulomb_original'].reaction_rate_relative, 1e-300)),
                'log_rate_cherepanov': np.log10(max(barrier_results['cherepanov'].reaction_rate_relative, 1e-300)),
                'above_loading_threshold': int(D_loading > LOADING['LENR_threshold_McKubre']),
                'above_storms_threshold': int(D_loading > LOADING['LENR_threshold_Storms']),

                # Labels
                'reaction_occurred': is_reaction,
                'reaction_probability': reaction_probability,
                'excess_heat_W': excess_heat_W,
            }
            records.append(record)

        return pd.DataFrame(records)

    def _estimate_reaction_prob(
        self, material, E_cm_keV, T_K, D_loading, Us_eV, pressure_Pa,
    ) -> float:
        """Estimate reaction probability from physics + experimental priors."""
        prob = 0.0

        # Loading threshold effect (McKubre: D/Pd > 0.84)
        if D_loading > 0.84:
            prob += 0.3 * ((D_loading - 0.84) / 0.16) ** 2
        if D_loading > 0.90:
            prob += 0.2

        # Screening energy effect (higher = better)
        prob += min(Us_eV / 2000, 0.3)

        # Energy effect
        if E_cm_keV > 1:
            prob += min(E_cm_keV / 100, 0.15)

        # Material bonus (Pd, PdO known to work)
        if material in ('Pd', 'PdO'):
            prob += 0.1
        if material == 'PdO':
            prob += 0.05
        if material == 'Ni':
            prob += 0.05

        # Temperature (slight positive at moderate T)
        if 300 < T_K < 600:
            prob += 0.05

        # High pressure helps
        if pressure_Pa > 1e5:
            prob += 0.05

        return np.clip(prob, 0.001, 0.95)

    def _estimate_excess_heat(
        self, material, D_loading, T_K, Us_eV, reaction_prob,
    ) -> float:
        """Estimate excess heat output based on conditions."""
        if reaction_prob < 0.1:
            return 0.0

        # Base from experimental observations
        base_W = 5.0  # Typical modest excess heat

        # Scale by loading
        if D_loading > 0.84:
            loading_factor = ((D_loading - 0.84) / 0.16) ** 3
        else:
            loading_factor = 0.01

        # Scale by screening
        screening_factor = Us_eV / 300  # Pd baseline

        # Material factor
        mat_factor = {'Pd': 1.0, 'PdO': 1.5, 'Ni': 0.8, 'NiCu': 0.6, 'Fe': 0.3, 'Ti': 0.2, 'Au': 0.1, 'Pt': 0.15}
        mf = mat_factor.get(material, 0.5)

        return base_W * loading_factor * screening_factor * mf * reaction_prob * 10

    def generate_mizuno_dataframe(self) -> pd.DataFrame:
        """Convert Mizuno R19 real data (55 points) to ML-ready DataFrame.

        Fills in physics-engine features for each Mizuno measurement.
        Material = Ni (mesh) + Pd (coating) → use Ni screening/lattice.
        """
        records = []
        # Mizuno uses Ni mesh with Pd coating, D₂ gas loading
        Us_Ni = SCREENING_EXPERIMENTAL.get('Ni', {}).get('Us_eV', 420)
        lat = LATTICE.get('Ni', LATTICE['Pd'])

        for m in MIZUNO_R19_DATA:
            if m['input_W'] == 0:
                continue  # skip heat-after-death points (no input)

            T_K = m['temp_C'] + 273.15
            D_loading = m['D_Ni']  # D/Ni ratio (very low, ~0.001-0.017)
            pressure_Pa = m['pressure_Pa']

            # Use low energy (~thermal) for Mizuno gas loading
            # Thermal energy at T_K in keV
            E_cm_keV = max(8.617e-5 * T_K / 1000.0, 0.01)  # kB*T in keV

            # Diffusion
            try:
                D_coeff = diffusion_coefficient('Ni', T_K)
            except Exception as e:
                logger.debug("Diffusion calc failed for Ni at %g K: %s", T_K, e)
                D_coeff = 1e-10

            # Enhancement & cross-section
            enh = enhancement_factor(E_cm_keV, Us_Ni)
            sigma_bare = cross_section_DD(E_cm_keV)
            sigma_log = np.log10(max(sigma_bare, 1e-50))

            # Barrier calculations
            barrier_results = {}
            for mode_name, engine in self.engines.items():
                br = engine.calculate_barrier('Ni', E_cm_keV, T_K, D_loading)
                barrier_results[mode_name] = br

            record = {
                'material': 'NiPd',
                'material_encoded': hash('NiPd') % 100,
                'structure': lat['structure'],
                'lattice_constant_A': lat['a_A'],
                'debye_temperature_K': lat['debye_K'],
                'electron_density_A3': lat.get('e_density_A3', 0.16),
                'screening_energy_eV': Us_Ni,
                'beam_energy_keV': E_cm_keV,
                'temperature_K': T_K,
                'deuterium_loading': D_loading,
                'pressure_Pa': pressure_Pa,
                'diffusion_coefficient': D_coeff,
                'enhancement_factor': enh,
                'log_cross_section': sigma_log,
                'barrier_reduction_maxwell': barrier_results['maxwell'].effective_barrier_keV / max(barrier_results['maxwell'].barrier_keV, 1),
                'barrier_reduction_coulomb': barrier_results['coulomb_original'].effective_barrier_keV / max(barrier_results['coulomb_original'].barrier_keV, 1),
                'barrier_reduction_cherepanov': barrier_results['cherepanov'].effective_barrier_keV / max(barrier_results['cherepanov'].barrier_keV, 1),
                'penetration_maxwell': barrier_results['maxwell'].penetration_probability,
                'penetration_coulomb': barrier_results['coulomb_original'].penetration_probability,
                'penetration_cherepanov': barrier_results['cherepanov'].penetration_probability,
                'rate_maxwell': barrier_results['maxwell'].reaction_rate_relative,
                'rate_coulomb': barrier_results['coulomb_original'].reaction_rate_relative,
                'rate_cherepanov': barrier_results['cherepanov'].reaction_rate_relative,
                'log_rate_maxwell': np.log10(max(barrier_results['maxwell'].reaction_rate_relative, 1e-300)),
                'log_rate_coulomb': np.log10(max(barrier_results['coulomb_original'].reaction_rate_relative, 1e-300)),
                'log_rate_cherepanov': np.log10(max(barrier_results['cherepanov'].reaction_rate_relative, 1e-300)),
                'above_loading_threshold': 0,  # Mizuno D/Ni << 0.84 but still works!
                'above_storms_threshold': 0,
                # Labels
                'reaction_occurred': 1,  # all Mizuno tests show excess heat
                'reaction_probability': 1.0,
                'excess_heat_W': m['excess_W'],
                # Mizuno-specific
                'input_power_W': m['input_W'],
                'COP': m['COP'],
                'heat_per_g_Ni': m['heat_per_g'],
                'data_source': 'mizuno_r19',
            }
            records.append(record)

        return pd.DataFrame(records)

    def generate_neutron_dataframe(self) -> pd.DataFrame:
        """Convert Mizuno SUS304+H₂ neutron experiments (10 points) to ML-ready DataFrame.

        Key difference from R19: uses hydrogen (not deuterium), SUS304 steel (not Ni+Pd).
        Neutrons at 0.7 MeV — supports Widom-Larsen ULMN theory.
        """
        records = []
        lat = LATTICE.get('SUS304', LATTICE['Fe'])
        # SUS304 is Fe-Cr-Ni alloy — use Fe screening as approximation
        Us_Fe = SCREENING_EXPERIMENTAL.get('Fe', {}).get('Us_eV', 200)

        for exp in MIZUNO_NEUTRON_EXPERIMENTS:
            T_K = exp['temp_C'] + 273.15
            # Thermal energy in keV (gas loading, no beam)
            E_cm_keV = max(8.617e-5 * T_K / 1000.0, 0.01)
            pressure_Pa = MIZUNO_NEUTRON_REACTOR['initial_pressure_Pa']

            # Diffusion (use Fe as proxy for SUS304)
            try:
                D_coeff = diffusion_coefficient('Fe', T_K)
            except Exception as e:
                logger.debug("Diffusion calc failed for Fe at %g K: %s", T_K, e)
                D_coeff = 1e-10

            enh = enhancement_factor(E_cm_keV, Us_Fe)
            sigma_bare = cross_section_DD(E_cm_keV)
            sigma_log = np.log10(max(sigma_bare, 1e-50))

            barrier_results = {}
            for mode_name, engine in self.engines.items():
                br = engine.calculate_barrier('Fe', E_cm_keV, T_K, 0.01)
                barrier_results[mode_name] = br

            record = {
                'material': 'SUS304',
                'material_encoded': hash('SUS304') % 100,
                'structure': lat['structure'],
                'lattice_constant_A': lat['a_A'],
                'debye_temperature_K': lat['debye_K'],
                'electron_density_A3': lat.get('e_density_A3', 0.14),
                'screening_energy_eV': Us_Fe,
                'beam_energy_keV': E_cm_keV,
                'temperature_K': T_K,
                'deuterium_loading': 0.01,  # H₂ loading, very low
                'pressure_Pa': pressure_Pa,
                'diffusion_coefficient': D_coeff,
                'enhancement_factor': enh,
                'log_cross_section': sigma_log,
                'barrier_reduction_maxwell': barrier_results['maxwell'].effective_barrier_keV / max(barrier_results['maxwell'].barrier_keV, 1),
                'barrier_reduction_coulomb': barrier_results['coulomb_original'].effective_barrier_keV / max(barrier_results['coulomb_original'].barrier_keV, 1),
                'barrier_reduction_cherepanov': barrier_results['cherepanov'].effective_barrier_keV / max(barrier_results['cherepanov'].barrier_keV, 1),
                'penetration_maxwell': barrier_results['maxwell'].penetration_probability,
                'penetration_coulomb': barrier_results['coulomb_original'].penetration_probability,
                'penetration_cherepanov': barrier_results['cherepanov'].penetration_probability,
                'rate_maxwell': barrier_results['maxwell'].reaction_rate_relative,
                'rate_coulomb': barrier_results['coulomb_original'].reaction_rate_relative,
                'rate_cherepanov': barrier_results['cherepanov'].reaction_rate_relative,
                'log_rate_maxwell': np.log10(max(barrier_results['maxwell'].reaction_rate_relative, 1e-300)),
                'log_rate_coulomb': np.log10(max(barrier_results['coulomb_original'].reaction_rate_relative, 1e-300)),
                'log_rate_cherepanov': np.log10(max(barrier_results['cherepanov'].reaction_rate_relative, 1e-300)),
                'above_loading_threshold': 0,
                'above_storms_threshold': 0,
                # Labels — neutron experiments show nuclear activity
                'reaction_occurred': 1,
                'reaction_probability': 1.0,
                'excess_heat_W': MIZUNO_NEUTRON_REACTOR['excess_heat_W'],
                # Neutron-specific
                'input_power_W': exp['input_W'],
                'neutron_cpm': exp['neutron_cpm'],
                'data_source': 'mizuno_neutron_SUS304',
            }
            records.append(record)

        return pd.DataFrame(records)

    def generate_experimental_dataframe(self) -> pd.DataFrame:
        """Convert real experimental data to DataFrame."""
        records = []
        for exp in EXCESS_HEAT_DATA:
            mat = exp['material']
            base_mat = mat.replace('nano_', '').replace('PdNi_ZrO2', 'Pd')
            Us = SCREENING_EXPERIMENTAL.get(base_mat, SCREENING_EXPERIMENTAL.get('Pd', {})).get('Us_eV', 300)

            records.append({
                'lab': exp['lab'],
                'material': mat,
                'method': exp['method'],
                'excess_W': exp['excess_W'],
                'COP': exp['COP'],
                'duration_days': exp['duration_days'],
                'deuterium_loading': exp['DPd'] if exp['DPd'] is not None else 0.5,
                'temperature_K': exp['temperature_K'],
                'pressure_Pa': exp['pressure_Pa'],
                'reproducibility': exp['reproducibility'],
                'screening_energy_eV': Us,
                'reaction_occurred': 1,
            })
        return pd.DataFrame(records)

    def generate_combined_dataset(
        self,
        n_synthetic: int = 5000,
        noise_level: float = 0.05,
        include_mizuno: bool = True,
        include_neutron: bool = True,
    ) -> pd.DataFrame:
        """Generate combined real + synthetic dataset.

        Args:
            include_mizuno: Include 53 real Mizuno R19 data points (recommended).
            include_neutron: Include 10 Mizuno SUS304 neutron experiments.
        """
        synthetic = self.generate_parameter_sweep(n_synthetic, noise_level)
        experimental = self.generate_experimental_dataframe()

        # Mark data source
        synthetic['data_source'] = 'synthetic'
        experimental['data_source'] = 'experimental'

        frames = [synthetic, experimental]

        if include_mizuno:
            mizuno = self.generate_mizuno_dataframe()
            frames.append(mizuno)

        if include_neutron:
            neutron = self.generate_neutron_dataframe()
            frames.append(neutron)

        # Find common columns across all frames
        common_cols = set(frames[0].columns)
        for f in frames[1:]:
            common_cols &= set(f.columns)
        common_cols = list(common_cols)

        combined = pd.concat([f[common_cols] for f in frames], ignore_index=True)
        return combined


def get_feature_columns() -> list[str]:
    """Return feature column names for ML training."""
    return [
        'lattice_constant_A', 'debye_temperature_K', 'electron_density_A3',
        'screening_energy_eV', 'beam_energy_keV', 'temperature_K',
        'deuterium_loading', 'pressure_Pa', 'diffusion_coefficient',
        'enhancement_factor', 'log_cross_section',
        'barrier_reduction_maxwell', 'barrier_reduction_coulomb', 'barrier_reduction_cherepanov',
        'log_rate_maxwell', 'log_rate_coulomb', 'log_rate_cherepanov',
        'above_loading_threshold', 'above_storms_threshold',
    ]


if __name__ == '__main__':
    gen = LENRDataGenerator(seed=42)

    print("Generating synthetic dataset...")
    df = gen.generate_parameter_sweep(n_samples=1000)
    print(f"Shape: {df.shape}")
    print(f"Reaction rate: {df['reaction_occurred'].mean():.1%}")
    print(f"Mean excess heat (when > 0): {df[df['excess_heat_W'] > 0]['excess_heat_W'].mean():.2f} W")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample:\n{df.head()}")

    print("\n\nExperimental data:")
    exp_df = gen.generate_experimental_dataframe()
    print(exp_df[['lab', 'material', 'excess_W', 'COP', 'deuterium_loading']].to_string())
