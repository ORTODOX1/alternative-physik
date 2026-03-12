"""
Tests for EXFORLoader — IAEA cross-section data.
Tests for ML model imports.
"""
import pytest


class TestEXFORLoader:
    """EXFOR data loading and fallback."""

    def test_import(self):
        """Module imports successfully."""
        from exfor_loader import EXFORLoader
        loader = EXFORLoader()
        assert loader is not None

    def test_fallback_data(self):
        """Fallback data available when API unreachable."""
        from exfor_loader import EXFORLoader
        loader = EXFORLoader()
        df = loader.get_fallback_data()
        assert df is not None
        assert len(df) >= 50

    def test_fallback_columns(self):
        """Fallback data has required columns."""
        from exfor_loader import EXFORLoader
        loader = EXFORLoader()
        df = loader.get_fallback_data()
        required = ['energy_keV', 'cross_section_mb']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_energy_range(self):
        """Cross-section data covers relevant energy range."""
        from exfor_loader import EXFORLoader
        loader = EXFORLoader()
        df = loader.get_fallback_data()
        if len(df) > 0 and 'energy_keV' in df.columns:
            assert df['energy_keV'].min() > 0
            assert df['energy_keV'].max() > 10

    def test_screening_compilation(self):
        """Screening compilation returns data."""
        from exfor_loader import EXFORLoader
        loader = EXFORLoader()
        df = loader.get_screening_compilation()
        assert df is not None
        assert len(df) > 0

    def test_bosch_hale_grid(self):
        """Bosch-Hale parameterization grid generation."""
        from exfor_loader import EXFORLoader
        loader = EXFORLoader()
        df = loader.generate_bosch_hale_grid()
        assert df is not None
        assert len(df) > 0


class TestModels:
    """ML model imports."""

    def test_xgboost_classifier_import(self):
        from models.xgboost_classifier import LENRClassifier
        model = LENRClassifier()
        assert model is not None

    def test_dnn_regressor_import(self):
        """DNN regressor imports (requires torch)."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from models.dnn_regressor import LENRDNNRegressor
        model = LENRDNNRegressor()
        assert model is not None

    def test_anomaly_detector_import(self):
        """Anomaly detector — test independently."""
        try:
            from models.anomaly_detector import LENRAnomalyDetector
            model = LENRAnomalyDetector()
            assert model is not None
        except ImportError:
            pytest.skip("Dependency not available")
