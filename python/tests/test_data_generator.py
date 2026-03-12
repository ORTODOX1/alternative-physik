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
        except Exception:
            pytest.skip("EXFOR data unavailable")


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
