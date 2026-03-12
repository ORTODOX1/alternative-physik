"""
Tests for PhysicsEngine — 3 physics modes.
PhysicsEngine.calculate_barrier(material, E_cm_keV, T_K, D_loading, defect_concentration, B_field_T)
No 'mode' parameter — mode is set on the engine's internal engines dict.
"""
import pytest


class TestPhysicsEngine:
    """Core physics engine tests."""

    def test_default_barrier(self, physics_engine):
        """Default call returns reasonable barrier."""
        result = physics_engine.calculate_barrier('Pd', 2.5, 300, 0.9)
        assert result is not None
        assert result.barrier_keV > 0
        assert result.barrier_keV < 1000

    def test_barrier_result_fields(self, physics_engine):
        """BarrierResult has all expected fields."""
        r = physics_engine.calculate_barrier('Pd', 2.5, 300, 0.9)
        assert hasattr(r, 'barrier_keV')
        assert hasattr(r, 'effective_barrier_keV')
        assert hasattr(r, 'screening_eV')
        assert hasattr(r, 'penetration_probability')
        assert hasattr(r, 'reaction_rate_relative')
        assert hasattr(r, 'mode')
        assert hasattr(r, 'notes')

    def test_different_materials(self, physics_engine):
        """Engine works for multiple materials."""
        materials = ['Pd', 'Ni', 'Ti', 'Fe', 'Au']
        for mat in materials:
            r = physics_engine.calculate_barrier(mat, 2.5, 300, 0.5)
            assert r is not None, f"Failed for {mat}"
            assert r.barrier_keV > 0, f"Zero barrier for {mat}"

    def test_temperature_effect(self, physics_engine):
        """Different temperatures produce results."""
        r_cold = physics_engine.calculate_barrier('Pd', 2.5, 100, 0.9)
        r_hot = physics_engine.calculate_barrier('Pd', 2.5, 1000, 0.9)
        assert r_cold is not None
        assert r_hot is not None

    def test_loading_effect(self, physics_engine):
        """Higher loading should give higher screening."""
        r_low = physics_engine.calculate_barrier('Pd', 2.5, 300, 0.3)
        r_high = physics_engine.calculate_barrier('Pd', 2.5, 300, 0.95)
        assert r_low is not None
        assert r_high is not None
        assert r_high.screening_eV >= r_low.screening_eV

    def test_defect_parameter(self, physics_engine):
        """Defect concentration parameter accepted."""
        r = physics_engine.calculate_barrier('Pd', 2.5, 300, 0.9,
                                             defect_concentration=0.5)
        assert r is not None

    def test_b_field_parameter(self, physics_engine):
        """B field parameter accepted."""
        r = physics_engine.calculate_barrier('Pd', 2.5, 300, 0.9,
                                             B_field_T=0.5)
        assert r is not None

    def test_energy_range(self, physics_engine):
        """Works across wide energy range."""
        for E in [0.1, 1.0, 5.0, 25.0, 100.0]:
            r = physics_engine.calculate_barrier('Pd', E, 300, 0.9)
            assert r is not None, f"Failed at E={E} keV"


class TestPhysicsConstants:
    """Test that physics constants are accessible and sane."""

    def test_lenr_constants_import(self):
        from lenr_constants import NUCLEAR, SCREENING_EXPERIMENTAL
        assert len(NUCLEAR) > 0
        assert len(SCREENING_EXPERIMENTAL) >= 5

    def test_gamow_penetration(self):
        from lenr_constants import gamow_penetration
        p = gamow_penetration(10.0)
        assert 0 < p < 1

    def test_cross_section(self):
        from lenr_constants import cross_section_DD
        sigma = cross_section_DD(100.0)
        assert sigma > 0

    def test_enhancement_factor(self):
        from lenr_constants import enhancement_factor
        ef = enhancement_factor(2.5, 300)
        assert ef >= 1.0  # Screening always enhances
