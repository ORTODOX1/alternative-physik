"""
Tests for CherepanovEngine — photon mass, no Coulomb barrier.
"""
import pytest


class TestCherepanovEngine:
    """Cherepanov physics: photon mass density, medium resistance."""

    def test_calculate_pd(self, cherepanov_engine):
        """Basic Pd calculation works."""
        r = cherepanov_engine.calculate('Pd', 2.5, 340, 0.9, 0.0, 0.1)
        assert r is not None
        assert hasattr(r, 'medium_resistance')
        assert hasattr(r, 'photon_mass_density')
        assert hasattr(r, 'reaction_probability')

    def test_medium_resistance_positive(self, cherepanov_engine):
        """Medium resistance is always positive."""
        for mat in ['Pd', 'Ni', 'Fe', 'Ti']:
            r = cherepanov_engine.calculate(mat, 2.5, 300, 0.5, 0.0, 0.05)
            assert r.medium_resistance > 0, f"Negative resistance for {mat}"

    def test_defects_lower_resistance(self, cherepanov_engine):
        """Key Cherepanov prediction: defects lower medium resistance."""
        r_annealed = cherepanov_engine.calculate('Pd', 2.5, 300, 0.9, 0.0, 0.005)
        r_cold_rolled = cherepanov_engine.calculate('Pd', 2.5, 300, 0.9, 0.0, 0.5)
        assert r_cold_rolled.medium_resistance < r_annealed.medium_resistance, \
            "Cold-rolled should have LOWER resistance than annealed"

    def test_defects_increase_reaction_prob(self, cherepanov_engine):
        """More defects → higher reaction probability (more channels)."""
        r_low = cherepanov_engine.calculate('Pd', 2.5, 300, 0.9, 0.0, 0.01)
        r_high = cherepanov_engine.calculate('Pd', 2.5, 300, 0.9, 0.0, 0.5)
        assert r_high.reaction_probability >= r_low.reaction_probability, \
            "Higher defects should give >= reaction probability"

    def test_photon_mass_density_positive(self, cherepanov_engine):
        """Photon mass density is always positive."""
        r = cherepanov_engine.calculate('Ni', 2.5, 350, 0.5, 0.0, 0.1)
        assert r.photon_mass_density > 0

    def test_magnetic_flux(self, cherepanov_engine):
        """Magnetic flux B [kg/s] is returned."""
        r = cherepanov_engine.calculate('Pd', 2.5, 300, 0.9, 0.0, 0.1)
        assert hasattr(r, 'magnetic_flux_B_kg_s')

    def test_loading_effect(self, cherepanov_engine):
        """Higher D loading should increase photon mass density."""
        r_low = cherepanov_engine.calculate('Pd', 2.5, 300, 0.3, 0.0, 0.1)
        r_high = cherepanov_engine.calculate('Pd', 2.5, 300, 0.95, 0.0, 0.1)
        assert r_high.photon_mass_density >= r_low.photon_mass_density

    def test_multiple_materials(self, cherepanov_engine):
        """Engine handles various materials."""
        materials = ['Pd', 'Ni', 'Ti', 'Fe', 'Au', 'Pt', 'Cu', 'Zr', 'Ta']
        for mat in materials:
            r = cherepanov_engine.calculate(mat, 2.5, 300, 0.5, 0.0, 0.05)
            assert r is not None, f"Failed for {mat}"

    def test_barrier_result_wrapper(self, cherepanov_engine):
        """cherepanov_to_barrier_result provides V2 compatibility."""
        cr = cherepanov_engine.calculate('Pd', 2.5, 300, 0.9, 0.0, 0.1)
        br = cherepanov_engine.cherepanov_to_barrier_result(cr)
        assert hasattr(br, 'barrier_keV')
        assert hasattr(br, 'screening_eV')
