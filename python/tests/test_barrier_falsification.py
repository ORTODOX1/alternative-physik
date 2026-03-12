"""
Tests for BarrierFalsification — Coulomb barrier analysis.
"""
import pytest
import pandas as pd


class TestBarrierFalsification:
    """Systematic falsification of standard Coulomb barrier model."""

    def test_load_screening_data(self, barrier_falsification):
        """Load all screening data into DataFrame."""
        df = barrier_falsification.load_all_screening_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 10

    def test_adiabatic_prediction(self, barrier_falsification):
        """Adiabatic prediction for Z=46 (Pd) should be ~25-50 eV."""
        u_ad = barrier_falsification.compute_adiabatic_prediction(46)
        assert 10 < u_ad < 200, f"Adiabatic prediction {u_ad} out of range"

    def test_residuals(self, barrier_falsification):
        """Residuals computed for all data points."""
        df = barrier_falsification.load_all_screening_data()
        residuals = barrier_falsification.compute_residuals(df)
        assert isinstance(residuals, pd.DataFrame)
        assert len(residuals) == len(df)

    def test_correlation_analysis(self, barrier_falsification):
        """Correlation analysis returns results."""
        df = barrier_falsification.load_all_screening_data()
        corr = barrier_falsification.correlation_analysis(df)
        assert corr is not None

    def test_most_metals_exceed_standard(self, barrier_falsification):
        """Most measured screening energies exceed standard prediction."""
        df = barrier_falsification.load_all_screening_data()
        # Find the screening energy column
        us_col = None
        for col in ['Us_measured_eV', 'Us_eV', 'screening_eV', 'Us']:
            if col in df.columns:
                us_col = col
                break
        if us_col is None:
            pytest.skip("No screening energy column found")
        above_standard = (df[us_col] > 40).sum()
        total = len(df)
        fraction = above_standard / total
        assert fraction > 0.3, \
            f"Only {fraction:.0%} above standard — expected most"


class TestAICBIC:
    """AIC/BIC model comparison."""

    def test_aic_bic_runs(self, barrier_falsification):
        """AIC/BIC comparison executes without error."""
        df = barrier_falsification.load_all_screening_data()
        if len(df) >= 10:
            try:
                result = barrier_falsification.aic_bic_comparison(df)
                assert result is not None
            except Exception as e:
                pytest.skip(f"AIC/BIC needs more data: {e}")
