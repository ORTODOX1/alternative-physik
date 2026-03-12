"""
Anomaly detection for LENR data filtering.
Uses Isolation Forest + statistical checks to identify
outliers and potentially erroneous measurements.
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib


@dataclass
class AnomalyResult:
    """Results from anomaly detection."""
    n_total: int
    n_anomalies: int
    anomaly_fraction: float
    anomaly_indices: np.ndarray
    anomaly_scores: np.ndarray
    feature_anomaly_counts: dict[str, int]


class LENRAnomalyDetector:
    """Isolation Forest anomaly detector for LENR data.

    Identifies outliers in experimental and synthetic data
    that may represent measurement errors or unusual physics.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False
        self.contamination = contamination

    def fit_detect(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> AnomalyResult:
        """Fit detector and identify anomalies."""
        self.feature_names = feature_cols

        X = df[feature_cols].values.astype(np.float32)
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e30, neginf=-1e30)
        X_scaled = self.scaler.fit_transform(X_clean)

        # Fit and predict
        labels = self.model.fit_predict(X_scaled)  # 1 = normal, -1 = anomaly
        scores = self.model.decision_function(X_scaled)  # lower = more anomalous

        self.is_fitted = True

        anomaly_mask = labels == -1
        anomaly_indices = np.where(anomaly_mask)[0]

        # Per-feature anomaly analysis: which features contribute most
        feature_anomaly_counts = {}
        if anomaly_mask.any():
            anomaly_data = X_scaled[anomaly_mask]
            normal_data = X_scaled[~anomaly_mask]
            normal_mean = normal_data.mean(axis=0)
            normal_std = normal_data.std(axis=0) + 1e-8

            for i, col in enumerate(feature_cols):
                # Count how many anomalies have this feature > 2 sigma
                deviations = np.abs(anomaly_data[:, i] - normal_mean[i]) / normal_std[i]
                feature_anomaly_counts[col] = int((deviations > 2).sum())

        feature_anomaly_counts = dict(sorted(
            feature_anomaly_counts.items(), key=lambda x: x[1], reverse=True,
        ))

        return AnomalyResult(
            n_total=len(X),
            n_anomalies=int(anomaly_mask.sum()),
            anomaly_fraction=float(anomaly_mask.mean()),
            anomaly_indices=anomaly_indices,
            anomaly_scores=scores,
            feature_anomaly_counts=feature_anomaly_counts,
        )

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove anomalies and return clean data."""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted yet. Call fit_detect() first.")

        X = df[self.feature_names].values.astype(np.float32)
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e30, neginf=-1e30)
        X_scaled = self.scaler.transform(X_clean)
        labels = self.model.predict(X_scaled)

        return df[labels == 1].reset_index(drop=True)

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted yet.")

        X = self.scaler.transform(
            np.nan_to_num(df[self.feature_names].values.astype(np.float32))
        )
        return self.model.decision_function(X)

    def physics_sanity_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply physics-based sanity checks.

        Returns DataFrame with added column 'physics_valid' (bool).
        Checks:
        - Cross-section should be reasonable range
        - Enhancement factor should be > 1 with screening
        - Barrier reduction should be in [0, 1]
        - Loading ratio should be in [0, 1.2]
        """
        result = df.copy()
        valid = pd.Series(True, index=df.index)

        if 'log_cross_section' in df.columns:
            valid &= df['log_cross_section'] > -50  # not absurdly small
            valid &= df['log_cross_section'] < 5    # not absurdly large

        if 'enhancement_factor' in df.columns:
            valid &= df['enhancement_factor'] >= 1.0
            valid &= df['enhancement_factor'] < 1e20

        for col in ['barrier_reduction_maxwell', 'barrier_reduction_coulomb', 'barrier_reduction_cherepanov']:
            if col in df.columns:
                valid &= df[col] >= 0
                valid &= df[col] <= 1.01  # allow tiny float error

        if 'deuterium_loading' in df.columns:
            valid &= df['deuterium_loading'] >= 0
            valid &= df['deuterium_loading'] <= 1.2

        if 'temperature_K' in df.columns:
            valid &= df['temperature_K'] > 0

        if 'excess_heat_W' in df.columns:
            valid &= df['excess_heat_W'] >= 0

        result['physics_valid'] = valid
        return result

    def save(self, path: str):
        """Save detector."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'LENRAnomalyDetector':
        """Load saved detector."""
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.feature_names = data['feature_names']
        obj.is_fitted = True
        return obj
