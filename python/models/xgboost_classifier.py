"""
XGBoost classifier for LENR reaction prediction.
Binary classification: will a nuclear reaction occur under given conditions?
Includes SHAP analysis for feature importance.
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import shap
except ImportError:
    shap = None

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class ClassifierResult:
    """Results from classifier training."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion_matrix: np.ndarray
    classification_report: str
    feature_importance: dict[str, float]
    cv_scores: Optional[np.ndarray] = None
    shap_values: Optional[np.ndarray] = None
    shap_expected: Optional[float] = None


class LENRClassifier:
    """XGBoost classifier for LENR reaction prediction.

    Predicts binary outcome: reaction_occurred (0/1).
    Uses SHAP for interpretability.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
    ):
        if xgb is None:
            raise ImportError("xgboost is required: pip install xgboost")

        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': random_state,
            'tree_method': 'hist',
        }
        if scale_pos_weight is not None:
            self.params['scale_pos_weight'] = scale_pos_weight

        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = 'reaction_occurred',
        test_size: float = 0.2,
        do_cv: bool = True,
        n_cv_folds: int = 5,
    ) -> ClassifierResult:
        """Train the classifier and return results."""
        self.feature_names = feature_cols

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.int32)

        # Handle class imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0 and 'scale_pos_weight' not in self.params:
            self.params['scale_pos_weight'] = n_neg / n_pos

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y,
        )

        # Train
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        self.is_fitted = True

        # Predict
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        # Feature importance (gain-based)
        importance = self.model.feature_importances_
        feat_imp = dict(zip(feature_cols, importance))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

        # Cross-validation
        cv_scores = None
        if do_cv:
            skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                xgb.XGBClassifier(**self.params),
                X_scaled, y, cv=skf, scoring='roc_auc',
            )

        # SHAP analysis
        shap_values = None
        shap_expected = None
        if shap is not None:
            try:
                explainer = shap.TreeExplainer(self.model)
                sv = explainer.shap_values(X_test)
                shap_values = sv
                shap_expected = float(explainer.expected_value)
            except Exception:
                pass

        return ClassifierResult(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            auc_roc=auc,
            confusion_matrix=cm,
            classification_report=report,
            feature_importance=feat_imp,
            cv_scores=cv_scores,
            shap_values=shap_values,
            shap_expected=shap_expected,
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict reaction probability for new data."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet. Call train() first.")
        X = self.scaler.transform(df[self.feature_names].values.astype(np.float32))
        return self.model.predict_proba(X)[:, 1]

    def predict_binary(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary reaction outcome."""
        probs = self.predict(df)
        return (probs >= threshold).astype(int)

    def save(self, path: str):
        """Save model and scaler."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'LENRClassifier':
        """Load saved model."""
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.feature_names = data['feature_names']
        obj.params = data['params']
        obj.is_fitted = True
        return obj

    def get_shap_summary(self, X_test: np.ndarray) -> dict[str, float]:
        """Get mean absolute SHAP values per feature."""
        if shap is None:
            raise ImportError("shap is required: pip install shap")
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")

        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(sv).mean(axis=0)
        return dict(sorted(
            zip(self.feature_names, mean_abs_shap),
            key=lambda x: x[1], reverse=True,
        ))
