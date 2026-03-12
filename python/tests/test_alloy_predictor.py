"""
Tests for AlloyPredictor — material discovery engine.
"""
import pytest
import pandas as pd


class TestAlloyPredictor:
    """Alloy prediction and material discovery."""

    def test_pure_metal_prediction(self, alloy_predictor):
        """Predict LENR potential for pure Pd."""
        pred = alloy_predictor.predict_lenr_potential(
            {'Pd': 1.0}, defect_concentration=0.05
        )
        assert pred is not None
        assert 0 < pred.lenr_score <= 100
        assert pred.predicted_Us_eV > 0
        assert pred.predicted_COP > 0

    def test_binary_alloy(self, alloy_predictor):
        """50/50 binary alloy prediction."""
        pred = alloy_predictor.predict_lenr_potential(
            {'Pd': 0.5, 'Ni': 0.5}, defect_concentration=0.1
        )
        assert pred is not None
        assert 0 < pred.lenr_score <= 100

    def test_ternary_alloy(self, alloy_predictor):
        """Ternary alloy prediction."""
        pred = alloy_predictor.predict_lenr_potential(
            {'Pd': 0.5, 'Ni': 0.3, 'Ti': 0.2}, defect_concentration=0.1
        )
        assert pred is not None
        assert 0 < pred.lenr_score <= 100

    def test_composite_with_oxide(self, alloy_predictor):
        """Metal/oxide composite prediction."""
        pred = alloy_predictor.predict_lenr_potential(
            {'Pd': 1.0}, oxide_matrix='PdO', oxide_fraction=0.1,
            defect_concentration=0.15
        )
        assert pred is not None
        assert pred.lenr_score > 0

    def test_defect_effect(self, alloy_predictor):
        """More defects → higher LENR score (Cherepanov)."""
        pred_low = alloy_predictor.predict_lenr_potential(
            {'Pd': 1.0}, defect_concentration=0.005
        )
        pred_high = alloy_predictor.predict_lenr_potential(
            {'Pd': 1.0}, defect_concentration=0.5
        )
        assert pred_high.lenr_score >= pred_low.lenr_score, \
            "Higher defects should give >= LENR score"

    def test_heatmap_generation(self, alloy_predictor):
        """Binary heatmap data generation."""
        elements = ['Pd', 'Ni', 'Fe']
        df = alloy_predictor.generate_heatmap_data(
            elements=elements, metric='lenr_score'
        )
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)

    def test_composition_optimization(self, alloy_predictor):
        """Find optimal composition for binary alloy."""
        df = alloy_predictor.find_optimal_composition(
            'Pd', 'Ni', n_steps=10
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert 'lenr_score' in df.columns

    def test_scan_ternary(self, alloy_predictor):
        """Ternary alloy scan produces ranked results."""
        df = alloy_predictor.scan_ternary_alloys(
            base_metals=['Pd', 'Ni', 'Fe'],
            defect_concentration=0.1
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'lenr_score' in df.columns

    def test_top_predictions(self, alloy_predictor):
        """Top predictions ranked list."""
        df = alloy_predictor.get_top_predictions(
            n_top=10, include_ternary=False,
            include_composites=False
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 10
        assert 'rank' in df.columns

    def test_score_components(self, alloy_predictor):
        """Prediction has all 5 score components."""
        pred = alloy_predictor.predict_lenr_potential(
            {'Ni': 1.0}, defect_concentration=0.1
        )
        assert hasattr(pred, 'screening_score')
        assert hasattr(pred, 'loading_score')
        assert hasattr(pred, 'magnetic_score')
        assert hasattr(pred, 'defect_score')
        assert hasattr(pred, 'thermal_score')
        # All scores 0-100
        for attr in ['screening_score', 'loading_score', 'magnetic_score',
                      'defect_score', 'thermal_score']:
            val = getattr(pred, attr)
            assert 0 <= val <= 100, f"{attr}={val} out of range"


class TestElementDatabase:
    """Test element and oxide databases."""

    def test_element_db_loaded(self):
        from alloy_predictor import ELEMENT_DB
        assert len(ELEMENT_DB) >= 15  # At least 15 elements

    def test_oxide_db_loaded(self):
        from alloy_predictor import OXIDE_DB
        assert len(OXIDE_DB) >= 5  # At least 5 oxides

    def test_pd_properties(self):
        from alloy_predictor import ELEMENT_DB
        pd_data = ELEMENT_DB.get('Pd')
        assert pd_data is not None
        assert 'Z' in pd_data
        assert pd_data['Z'] == 46
