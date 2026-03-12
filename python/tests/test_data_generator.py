"""
Tests for LENRDataGeneratorV2 — synthetic + real data pipeline.
"""
import pytest
import pandas as pd


class TestDataGeneratorV2:
    """Data generation pipeline tests."""

    def test_generate_synthetic(self, data_generator):
        """Synthetic generation produces valid DataFrame."""
        df = data_generator.generate_synthetic(n_samples=50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 50
        assert df.shape[1] > 10

    def test_generate_real_excess_heat(self, data_generator):
        """Real excess heat data generation."""
        df = data_generator.generate_real_excess_heat()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_generate_combined(self, data_generator):
        """Combined dataset merges synthetic + real."""
        df = data_generator.generate_combined(n_synthetic=30)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 30

    def test_v3_features(self, data_generator):
        """V3 mode produces 72+ features."""
        df = data_generator.generate_combined(
            n_synthetic=20, use_v3_features=True
        )
        assert df.shape[1] >= 72, f"Only {df.shape[1]} columns, expected 72+"

    def test_no_nans_in_key_columns(self, data_generator):
        """Key columns should not have NaN values."""
        df = data_generator.generate_synthetic(n_samples=30)
        key_cols = ['material', 'temperature_K', 'deuterium_loading']
        for col in key_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                assert nan_count == 0, f"{col} has {nan_count} NaNs"

    def test_temperature_range(self, data_generator):
        """Generated temperatures should be physically reasonable."""
        df = data_generator.generate_synthetic(n_samples=100)
        if 'temperature_K' in df.columns:
            assert df['temperature_K'].min() > 0, "Temperature below 0 K"
            assert df['temperature_K'].max() < 5000, "Temperature above 5000 K"

    def test_loading_range(self, data_generator):
        """D/M loading ratio should be in reasonable range."""
        df = data_generator.generate_synthetic(n_samples=100)
        if 'deuterium_loading' in df.columns:
            assert df['deuterium_loading'].min() >= 0
            assert df['deuterium_loading'].max() <= 1.5

    def test_alloy_data_generation(self, data_generator):
        """Alloy data generation works."""
        df = data_generator.generate_alloy_data(
            n_binary=10, n_ternary=5, n_composites=5, defect_variations=1
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_exfor_data(self, data_generator):
        """EXFOR data generation works (may use fallback)."""
        try:
            df = data_generator.generate_exfor_data()
            assert isinstance(df, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"EXFOR data unavailable: {e}")


class TestMaterialNormalization:
    """Test _normalize_material() helper — prevents .replace('O','') bug."""

    def test_simple_elements(self, data_generator):
        """Pure element names stay unchanged."""
        for elem in ('Pd', 'Ni', 'Fe', 'Ti', 'Au', 'Cu', 'W', 'Al'):
            assert data_generator._normalize_material(elem) == elem

    def test_co_not_stripped(self, data_generator):
        """Co must NOT become 'C' (the old .replace('O','') bug)."""
        assert data_generator._normalize_material('Co') == 'Co'

    def test_mo_not_stripped(self, data_generator):
        """Mo must NOT become 'M' (the old .replace('O','') bug)."""
        assert data_generator._normalize_material('Mo') == 'Mo'

    def test_pdo_oxide(self, data_generator):
        """PdO → Pd (oxide stripping)."""
        assert data_generator._normalize_material('PdO') == 'Pd'

    def test_beo_oxide(self, data_generator):
        """BeO → Be (oxide stripping)."""
        assert data_generator._normalize_material('BeO') == 'Be'

    def test_suffix_raiola(self, data_generator):
        """Experiment suffix stripped correctly."""
        assert data_generator._normalize_material('Pd_Raiola') == 'Pd'
        assert data_generator._normalize_material('Fe_Huke') == 'Fe'
        assert data_generator._normalize_material('Ti_Kasagi') == 'Ti'

    def test_nano_prefix(self, data_generator):
        """nano_ prefix stripped correctly."""
        assert data_generator._normalize_material('nano_Pd') == 'Pd'

    def test_compound_alloys(self, data_generator):
        """Compound alloy names resolve to primary element."""
        assert data_generator._normalize_material('NiCu') == 'Ni'
        assert data_generator._normalize_material('NiPd') == 'Ni'
        assert data_generator._normalize_material('PdNi') == 'Pd'

    def test_special_names(self, data_generator):
        """SUS304 → Fe, Constantan → Cu."""
        assert data_generator._normalize_material('SUS304') == 'Fe'
        assert data_generator._normalize_material('Constantan') == 'Cu'


class TestCherepanovFeaturesFunc:
    """Test cherepanov_features() helper function."""

    def test_material_fills_chi_m(self, cherepanov_engine):
        """Passing material fills magnetic_susceptibility_abs."""
        from cherepanov_engine import cherepanov_features
        cr = cherepanov_engine.calculate('Pd', 2.5, 300, 0.5)
        feats = cherepanov_features(cr, material='Pd')
        assert feats['magnetic_susceptibility_abs'] > 0, \
            "chi_m_abs should be > 0 for Pd"

    def test_no_material_zero_chi(self, cherepanov_engine):
        """Without material, chi_m_abs defaults to 0."""
        from cherepanov_engine import cherepanov_features
        cr = cherepanov_engine.calculate('Pd', 2.5, 300, 0.5)
        feats = cherepanov_features(cr)
        assert feats['magnetic_susceptibility_abs'] == 0.0

    def test_all_8_features_present(self, cherepanov_engine):
        """All 8 Cherepanov features are in the dict."""
        from cherepanov_engine import cherepanov_features
        cr = cherepanov_engine.calculate('Ni', 2.5, 300, 0.1)
        feats = cherepanov_features(cr, material='Ni')
        expected_keys = [
            'magnetic_susceptibility_abs', 'photon_mass_density',
            'photon_mass_density_critical', 'medium_resistance',
            'lattice_focusing_factor', 'defect_concentration',
            'magnetic_flux_B_kg_s', 'photon_phonon_coupling',
        ]
        for k in expected_keys:
            assert k in feats, f"Missing key: {k}"


class TestFeatureColumns:
    """Test feature column definitions."""

    def test_v2_features_count(self):
        from lenr_comprehensive_data import get_feature_columns_v2
        cols = get_feature_columns_v2()
        assert len(cols) == 64, f"V2 should have 64 features, got {len(cols)}"

    def test_v3_features_count(self):
        from lenr_comprehensive_data import get_feature_columns_v3
        cols = get_feature_columns_v3()
        assert len(cols) == 72, f"V3 should have 72 features, got {len(cols)}"

    def test_v3_includes_cherepanov(self):
        from lenr_comprehensive_data import get_feature_columns_v3
        cols = get_feature_columns_v3()
        cherepanov_features = [
            'magnetic_susceptibility_abs',
            'photon_mass_density',
            'medium_resistance',
            'lattice_focusing_factor',
        ]
        for f in cherepanov_features:
            assert f in cols, f"V3 missing Cherepanov feature: {f}"
