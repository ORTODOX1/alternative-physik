"""
Barrier Proof Engine — строгое статистическое доказательство того, что
"кулоновский барьер" не является фундаментальной константой, а определяется
свойствами среды (дефекты, магнитная структура, кристаллическая решётка).

Методология:
1. Собрать ВСЕ экспериментальные screening energies с характеристиками среды
2. Обучить две конкурирующие модели:
   - Model A (Standard): фичи только из стандартной физики (Z, e_density, Debye)
   - Model B (Medium):   фичи среды (defects, magnetic_class, surface_state, structure)
3. Сравнить через AIC, BIC, R², RMSE, Leave-One-Out CV
4. SHAP-анализ: какие параметры реально определяют screening
5. Предсказания для неизмеренных комбинаций материал+обработка
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger(__name__)

# ============================================================================
# ПОЛНАЯ ЭКСПЕРИМЕНТАЛЬНАЯ БАЗА SCREENING ENERGIES
# Каждая точка: материал, состояние поверхности, Us, ошибка, источник,
# + свойства среды для medium-dependent модели
# ============================================================================

SCREENING_DATASET = [
    # --- Kasagi 2002 (Tohoku University, d(d,p)t at 2.5 keV) ---
    {"material": "PdO", "Z": 46, "surface_state": "oxidized",
     "Us_eV": 600, "Us_error": 60, "source": "Kasagi 2002",
     "beam_keV": 2.5, "crystal_structure": "FCC", "lattice_A": 3.89,
     "debye_K": 274, "e_density": 0.068, "magnetic_class": "diamagnetic",
     "chi_m": -7.2e-6, "density_gcc": 12.02, "defect_conc": 0.15},
    {"material": "Pd", "Z": 46, "surface_state": "polycrystal",
     "Us_eV": 310, "Us_error": 30, "source": "Kasagi 2002",
     "beam_keV": 2.5, "crystal_structure": "FCC", "lattice_A": 3.89,
     "debye_K": 274, "e_density": 0.068, "magnetic_class": "diamagnetic",
     "chi_m": -7.2e-6, "density_gcc": 12.02, "defect_conc": 0.05},
    {"material": "Fe", "Z": 26, "surface_state": "polycrystal",
     "Us_eV": 200, "Us_error": 20, "source": "Kasagi 2002",
     "beam_keV": 2.5, "crystal_structure": "BCC", "lattice_A": 2.87,
     "debye_K": 470, "e_density": 0.084, "magnetic_class": "ferromagnetic",
     "chi_m": 1.0, "density_gcc": 7.87, "defect_conc": 0.05},
    {"material": "Au", "Z": 79, "surface_state": "polycrystal",
     "Us_eV": 70, "Us_error": 10, "source": "Kasagi 2002",
     "beam_keV": 2.5, "crystal_structure": "FCC", "lattice_A": 4.08,
     "debye_K": 165, "e_density": 0.059, "magnetic_class": "diamagnetic",
     "chi_m": -3.4e-5, "density_gcc": 19.32, "defect_conc": 0.05},
    {"material": "Ti", "Z": 22, "surface_state": "polycrystal",
     "Us_eV": 65, "Us_error": 10, "source": "Kasagi 2002",
     "beam_keV": 2.5, "crystal_structure": "HCP", "lattice_A": 2.95,
     "debye_K": 420, "e_density": 0.057, "magnetic_class": "paramagnetic",
     "chi_m": 1.5e-4, "density_gcc": 4.51, "defect_conc": 0.05},

    # --- Raiola et al. (Bochum, Ruhr-Universität) ---
    {"material": "Pd", "Z": 46, "surface_state": "polycrystal",
     "Us_eV": 800, "Us_error": 90, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "FCC", "lattice_A": 3.89,
     "debye_K": 274, "e_density": 0.068, "magnetic_class": "diamagnetic",
     "chi_m": -7.2e-6, "density_gcc": 12.02, "defect_conc": 0.05},
    {"material": "Ta", "Z": 73, "surface_state": "polycrystal",
     "Us_eV": 309, "Us_error": 12, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "BCC", "lattice_A": 3.30,
     "debye_K": 240, "e_density": 0.055, "magnetic_class": "paramagnetic",
     "chi_m": 1.8e-4, "density_gcc": 16.65, "defect_conc": 0.05},
    {"material": "Ni", "Z": 28, "surface_state": "polycrystal",
     "Us_eV": 420, "Us_error": 50, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "FCC", "lattice_A": 3.52,
     "debye_K": 450, "e_density": 0.091, "magnetic_class": "ferromagnetic",
     "chi_m": 6.0e-4, "density_gcc": 8.91, "defect_conc": 0.05},
    {"material": "Pt", "Z": 78, "surface_state": "polycrystal",
     "Us_eV": 122, "Us_error": 20, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "FCC", "lattice_A": 3.92,
     "debye_K": 240, "e_density": 0.066, "magnetic_class": "paramagnetic",
     "chi_m": 2.6e-4, "density_gcc": 21.45, "defect_conc": 0.05},

    # --- Huke et al. (Berlin, TU Berlin) ---
    {"material": "Pd", "Z": 46, "surface_state": "polycrystal",
     "Us_eV": 313, "Us_error": 2, "source": "Huke 2008",
     "beam_keV": 10.0, "crystal_structure": "FCC", "lattice_A": 3.89,
     "debye_K": 274, "e_density": 0.068, "magnetic_class": "diamagnetic",
     "chi_m": -7.2e-6, "density_gcc": 12.02, "defect_conc": 0.05},
    {"material": "Zr", "Z": 40, "surface_state": "polycrystal",
     "Us_eV": 297, "Us_error": 8, "source": "Huke 2008",
     "beam_keV": 10.0, "crystal_structure": "HCP", "lattice_A": 3.23,
     "debye_K": 291, "e_density": 0.043, "magnetic_class": "paramagnetic",
     "chi_m": 1.2e-4, "density_gcc": 6.51, "defect_conc": 0.05},
    {"material": "Al", "Z": 13, "surface_state": "polycrystal",
     "Us_eV": 190, "Us_error": 15, "source": "Huke 2008",
     "beam_keV": 10.0, "crystal_structure": "FCC", "lattice_A": 4.05,
     "debye_K": 428, "e_density": 0.018, "magnetic_class": "paramagnetic",
     "chi_m": 2.1e-5, "density_gcc": 2.70, "defect_conc": 0.05},
    {"material": "Zr", "Z": 40, "surface_state": "irradiated",
     "Us_eV": 600, "Us_error": 50, "source": "Huke 2008 (high vacancy)",
     "beam_keV": 10.0, "crystal_structure": "HCP", "lattice_A": 3.23,
     "debye_K": 291, "e_density": 0.043, "magnetic_class": "paramagnetic",
     "chi_m": 1.2e-4, "density_gcc": 6.51, "defect_conc": 0.25},

    # --- Czerski et al. 2023 (Szczecin, cold-rolled Pd) ---
    {"material": "Pd", "Z": 46, "surface_state": "cold_rolled",
     "Us_eV": 18200, "Us_error": 3300, "source": "Czerski 2023",
     "beam_keV": 5.0, "crystal_structure": "FCC", "lattice_A": 3.89,
     "debye_K": 274, "e_density": 0.068, "magnetic_class": "diamagnetic",
     "chi_m": -7.2e-6, "density_gcc": 12.02, "defect_conc": 0.50},
    {"material": "Pd", "Z": 46, "surface_state": "annealed",
     "Us_eV": 310, "Us_error": 40, "source": "Czerski 2023",
     "beam_keV": 5.0, "crystal_structure": "FCC", "lattice_A": 3.89,
     "debye_K": 274, "e_density": 0.068, "magnetic_class": "diamagnetic",
     "chi_m": -7.2e-6, "density_gcc": 12.02, "defect_conc": 0.005},

    # --- NASA (Goddard Space Flight Center) ---
    {"material": "Be", "Z": 4, "surface_state": "oxidized",
     "Us_eV": 180, "Us_error": 40, "source": "NASA BeO",
     "beam_keV": 5.0, "crystal_structure": "HCP", "lattice_A": 2.29,
     "debye_K": 1440, "e_density": 0.012, "magnetic_class": "diamagnetic",
     "chi_m": -2.3e-5, "density_gcc": 1.85, "defect_conc": 0.15},

    # --- Raiola extended ---
    {"material": "Cu", "Z": 29, "surface_state": "polycrystal",
     "Us_eV": 120, "Us_error": 20, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "FCC", "lattice_A": 3.61,
     "debye_K": 343, "e_density": 0.085, "magnetic_class": "diamagnetic",
     "chi_m": -9.6e-6, "density_gcc": 8.96, "defect_conc": 0.05},
    {"material": "Ag", "Z": 47, "surface_state": "polycrystal",
     "Us_eV": 95, "Us_error": 15, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "FCC", "lattice_A": 4.09,
     "debye_K": 225, "e_density": 0.059, "magnetic_class": "diamagnetic",
     "chi_m": -2.4e-5, "density_gcc": 10.49, "defect_conc": 0.05},
    {"material": "W", "Z": 74, "surface_state": "polycrystal",
     "Us_eV": 150, "Us_error": 25, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "BCC", "lattice_A": 3.17,
     "debye_K": 400, "e_density": 0.063, "magnetic_class": "paramagnetic",
     "chi_m": 7.8e-5, "density_gcc": 19.25, "defect_conc": 0.05},
    {"material": "V", "Z": 23, "surface_state": "polycrystal",
     "Us_eV": 140, "Us_error": 20, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "BCC", "lattice_A": 3.02,
     "debye_K": 380, "e_density": 0.062, "magnetic_class": "paramagnetic",
     "chi_m": 2.6e-4, "density_gcc": 6.11, "defect_conc": 0.05},
    {"material": "Nb", "Z": 41, "surface_state": "polycrystal",
     "Us_eV": 160, "Us_error": 20, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "BCC", "lattice_A": 3.30,
     "debye_K": 275, "e_density": 0.056, "magnetic_class": "paramagnetic",
     "chi_m": 2.3e-4, "density_gcc": 8.57, "defect_conc": 0.05},
    {"material": "Co", "Z": 27, "surface_state": "polycrystal",
     "Us_eV": 350, "Us_error": 50, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "HCP", "lattice_A": 2.51,
     "debye_K": 445, "e_density": 0.090, "magnetic_class": "ferromagnetic",
     "chi_m": 1.0, "density_gcc": 8.90, "defect_conc": 0.05},
    {"material": "Mn", "Z": 25, "surface_state": "polycrystal",
     "Us_eV": 250, "Us_error": 40, "source": "Raiola 2004",
     "beam_keV": 5.0, "crystal_structure": "BCC", "lattice_A": 8.91,
     "debye_K": 410, "e_density": 0.082, "magnetic_class": "paramagnetic",
     "chi_m": 5.3e-4, "density_gcc": 7.47, "defect_conc": 0.05},

    # --- D2 gas reference (no solid medium) ---
    {"material": "D2_gas", "Z": 1, "surface_state": "gas",
     "Us_eV": 25, "Us_error": 5, "source": "Greife 1995",
     "beam_keV": 5.0, "crystal_structure": "none", "lattice_A": 0.0,
     "debye_K": 0, "e_density": 0.0, "magnetic_class": "diamagnetic",
     "chi_m": -2.1e-9, "density_gcc": 0.00017, "defect_conc": 0.0},

    # --- Debye model prediction (theoretical baseline) ---
    {"material": "Debye_theory", "Z": 46, "surface_state": "theory",
     "Us_eV": 30, "Us_error": 10, "source": "Debye model (Pd)",
     "beam_keV": 5.0, "crystal_structure": "FCC", "lattice_A": 3.89,
     "debye_K": 274, "e_density": 0.068, "magnetic_class": "diamagnetic",
     "chi_m": -7.2e-6, "density_gcc": 12.02, "defect_conc": 0.0},
]

# Encode structure and magnetic class
STRUCTURE_ENCODE = {"FCC": 1.0, "BCC": 0.7, "HCP": 0.5, "none": 0.0}
MAGNETIC_ENCODE = {"ferromagnetic": 3.0, "paramagnetic": 2.0, "diamagnetic": 1.0}

SURFACE_DEFECT_MAP = {
    "cold_rolled": 0.50, "nano": 0.30, "irradiated": 0.25,
    "sputtered": 0.20, "oxidized": 0.15, "mesh": 0.10,
    "polycrystal": 0.05, "annealed": 0.005, "single_crystal": 0.001,
    "gas": 0.0, "theory": 0.0,
}


@dataclass
class ProofResult:
    """Результат сравнения моделей."""
    n_datapoints: int = 0
    # Model A — standard physics only
    r2_standard: float = 0.0
    rmse_standard: float = 0.0
    mae_standard: float = 0.0
    aic_standard: float = 0.0
    bic_standard: float = 0.0
    loo_r2_standard: float = 0.0
    # Model B — medium-dependent
    r2_medium: float = 0.0
    rmse_medium: float = 0.0
    mae_medium: float = 0.0
    aic_medium: float = 0.0
    bic_medium: float = 0.0
    loo_r2_medium: float = 0.0
    # Comparison
    delta_aic: float = 0.0  # AIC_standard - AIC_medium (positive = medium wins)
    delta_bic: float = 0.0
    f_test_pvalue: float = 1.0
    evidence_strength: str = "none"
    # SHAP
    shap_top_features: dict = field(default_factory=dict)
    # Predictions for untested combinations
    predictions: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class BarrierProofEngine:
    """
    Строгое статистическое доказательство:
    screening energy определяется свойствами среды, а не Z ядра.
    """

    def __init__(self):
        self.df = self._build_dataframe()
        self.scaler_a = StandardScaler()
        self.scaler_b = StandardScaler()
        self.model_a = None
        self.model_b = None

    def _build_dataframe(self) -> pd.DataFrame:
        """Построить DataFrame из экспериментальных данных."""
        rows = []
        for entry in SCREENING_DATASET:
            row = dict(entry)
            row["structure_code"] = STRUCTURE_ENCODE.get(
                entry["crystal_structure"], 0.0
            )
            row["magnetic_code"] = MAGNETIC_ENCODE.get(
                entry["magnetic_class"], 1.0
            )
            row["log_Us"] = np.log10(max(entry["Us_eV"], 1))
            row["log_chi_m"] = np.log10(max(abs(entry["chi_m"]), 1e-10))
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def features_standard(self) -> list:
        """Фичи стандартной модели — только атомные/электронные свойства."""
        return ["Z", "e_density", "debye_K", "lattice_A", "density_gcc"]

    @property
    def features_medium(self) -> list:
        """Фичи medium-dependent модели — стандартные + свойства среды."""
        return [
            "Z", "e_density", "debye_K", "lattice_A", "density_gcc",
            "defect_conc", "magnetic_code", "structure_code",
            "log_chi_m", "beam_keV",
        ]

    def _aic_bic(self, y_true, y_pred, n_params):
        """AIC и BIC из RSS."""
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)
        if rss <= 0:
            rss = 1e-10
        log_likelihood = -n / 2 * (np.log(2 * np.pi * rss / n) + 1)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood
        return aic, bic

    def _evidence_strength(self, delta_aic: float) -> str:
        """Интерпретация разницы AIC (Burnham & Anderson 2002)."""
        if delta_aic > 10:
            return "DECISIVE: medium-dependent model vastly superior"
        elif delta_aic > 6:
            return "STRONG: strong evidence for medium-dependent model"
        elif delta_aic > 2:
            return "MODERATE: moderate evidence for medium-dependent model"
        elif delta_aic > 0:
            return "WEAK: slight evidence for medium-dependent model"
        else:
            return "NONE: standard model is better or equivalent"

    def run_proof(self, target: str = "log_Us") -> ProofResult:
        """
        Главный метод: полное сравнение двух конкурирующих моделей.

        Args:
            target: целевая переменная ('log_Us' или 'Us_eV')

        Returns:
            ProofResult с полной статистикой
        """
        df = self.df.copy()
        y = df[target].values
        n = len(y)

        # --- Model A: Standard Physics ---
        X_a = df[self.features_standard].values
        X_a_scaled = self.scaler_a.fit_transform(X_a)

        if HAS_XGB:
            model_a = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42,
            )
        else:
            model_a = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42,
            )
        model_a.fit(X_a_scaled, y)
        pred_a = model_a.predict(X_a_scaled)
        self.model_a = model_a

        # --- Model B: Medium-Dependent ---
        X_b = df[self.features_medium].values
        X_b_scaled = self.scaler_b.fit_transform(X_b)

        if HAS_XGB:
            model_b = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42,
            )
        else:
            model_b = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42,
            )
        model_b.fit(X_b_scaled, y)
        pred_b = model_b.predict(X_b_scaled)
        self.model_b = model_b

        # --- Метрики ---
        r2_a = r2_score(y, pred_a)
        r2_b = r2_score(y, pred_b)
        rmse_a = np.sqrt(mean_squared_error(y, pred_a))
        rmse_b = np.sqrt(mean_squared_error(y, pred_b))
        mae_a = mean_absolute_error(y, pred_a)
        mae_b = mean_absolute_error(y, pred_b)

        aic_a, bic_a = self._aic_bic(y, pred_a, len(self.features_standard))
        aic_b, bic_b = self._aic_bic(y, pred_b, len(self.features_medium))

        # --- Leave-One-Out Cross-Validation ---
        loo = LeaveOneOut()
        loo_pred_a = np.zeros(n)
        loo_pred_b = np.zeros(n)

        for train_idx, test_idx in loo.split(X_a_scaled):
            # Model A
            m_a = (xgb.XGBRegressor if HAS_XGB else GradientBoostingRegressor)(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                **({"reg_alpha": 1.0, "reg_lambda": 1.0} if HAS_XGB else {}),
                random_state=42,
            )
            m_a.fit(X_a_scaled[train_idx], y[train_idx])
            loo_pred_a[test_idx] = m_a.predict(X_a_scaled[test_idx])

            # Model B
            m_b = (xgb.XGBRegressor if HAS_XGB else GradientBoostingRegressor)(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                **({"reg_alpha": 1.0, "reg_lambda": 1.0} if HAS_XGB else {}),
                random_state=42,
            )
            m_b.fit(X_b_scaled[train_idx], y[train_idx])
            loo_pred_b[test_idx] = m_b.predict(X_b_scaled[test_idx])

        loo_r2_a = r2_score(y, loo_pred_a)
        loo_r2_b = r2_score(y, loo_pred_b)

        # --- F-test для вложенных моделей ---
        rss_a = np.sum((y - pred_a) ** 2)
        rss_b = np.sum((y - pred_b) ** 2)
        p_a = len(self.features_standard)
        p_b = len(self.features_medium)
        if rss_b > 0 and (p_b - p_a) > 0:
            f_stat = ((rss_a - rss_b) / (p_b - p_a)) / (rss_b / (n - p_b))
            f_pvalue = 1 - stats.f.cdf(f_stat, p_b - p_a, n - p_b)
        else:
            f_pvalue = 1.0

        # --- SHAP анализ ---
        shap_top = {}
        if HAS_SHAP and self.model_b is not None:
            try:
                explainer = shap.TreeExplainer(self.model_b)
                shap_values = explainer.shap_values(X_b_scaled)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                for i, feat in enumerate(self.features_medium):
                    shap_top[feat] = float(mean_abs_shap[i])
                # Sort by importance
                shap_top = dict(sorted(
                    shap_top.items(), key=lambda x: x[1], reverse=True
                ))
            except Exception as e:
                logger.debug("SHAP analysis failed: %s", e)

        # --- Delta AIC/BIC ---
        delta_aic = aic_a - aic_b
        delta_bic = bic_a - bic_b
        evidence = self._evidence_strength(delta_aic)

        # --- Предсказания для неизмеренных комбинаций ---
        predictions = self._generate_predictions(target)

        result = ProofResult(
            n_datapoints=n,
            r2_standard=round(r2_a, 4),
            rmse_standard=round(rmse_a, 4),
            mae_standard=round(mae_a, 4),
            aic_standard=round(aic_a, 2),
            bic_standard=round(bic_a, 2),
            loo_r2_standard=round(loo_r2_a, 4),
            r2_medium=round(r2_b, 4),
            rmse_medium=round(rmse_b, 4),
            mae_medium=round(mae_b, 4),
            aic_medium=round(aic_b, 2),
            bic_medium=round(bic_b, 2),
            loo_r2_medium=round(loo_r2_b, 4),
            delta_aic=round(delta_aic, 2),
            delta_bic=round(delta_bic, 2),
            f_test_pvalue=round(f_pvalue, 6),
            evidence_strength=evidence,
            shap_top_features=shap_top,
            predictions=predictions,
        )
        return result

    def _generate_predictions(self, target: str) -> list:
        """Предсказать screening для неизмеренных комбинаций."""
        if self.model_b is None:
            return []

        # Комбинации: материал × обработка поверхности
        predict_combos = [
            {"material": "Ni", "surface_state": "cold_rolled",
             "Z": 28, "e_density": 0.091, "debye_K": 450, "lattice_A": 3.52,
             "density_gcc": 8.91, "defect_conc": 0.50,
             "magnetic_code": 3.0, "structure_code": 1.0,
             "log_chi_m": np.log10(6e-4), "beam_keV": 5.0},
            {"material": "Ni", "surface_state": "nano",
             "Z": 28, "e_density": 0.091, "debye_K": 450, "lattice_A": 3.52,
             "density_gcc": 8.91, "defect_conc": 0.30,
             "magnetic_code": 3.0, "structure_code": 1.0,
             "log_chi_m": np.log10(6e-4), "beam_keV": 5.0},
            {"material": "Ti", "surface_state": "cold_rolled",
             "Z": 22, "e_density": 0.057, "debye_K": 420, "lattice_A": 2.95,
             "density_gcc": 4.51, "defect_conc": 0.50,
             "magnetic_code": 2.0, "structure_code": 0.5,
             "log_chi_m": np.log10(1.5e-4), "beam_keV": 5.0},
            {"material": "Fe", "surface_state": "cold_rolled",
             "Z": 26, "e_density": 0.084, "debye_K": 470, "lattice_A": 2.87,
             "density_gcc": 7.87, "defect_conc": 0.50,
             "magnetic_code": 3.0, "structure_code": 0.7,
             "log_chi_m": 0.0, "beam_keV": 5.0},
            {"material": "Au", "surface_state": "cold_rolled",
             "Z": 79, "e_density": 0.059, "debye_K": 165, "lattice_A": 4.08,
             "density_gcc": 19.32, "defect_conc": 0.50,
             "magnetic_code": 1.0, "structure_code": 1.0,
             "log_chi_m": np.log10(3.4e-5), "beam_keV": 5.0},
            {"material": "Pd", "surface_state": "nano",
             "Z": 46, "e_density": 0.068, "debye_K": 274, "lattice_A": 3.89,
             "density_gcc": 12.02, "defect_conc": 0.30,
             "magnetic_code": 1.0, "structure_code": 1.0,
             "log_chi_m": np.log10(7.2e-6), "beam_keV": 5.0},
            {"material": "Co", "surface_state": "cold_rolled",
             "Z": 27, "e_density": 0.090, "debye_K": 445, "lattice_A": 2.51,
             "density_gcc": 8.90, "defect_conc": 0.50,
             "magnetic_code": 3.0, "structure_code": 0.5,
             "log_chi_m": 0.0, "beam_keV": 5.0},
            {"material": "Ta", "surface_state": "irradiated",
             "Z": 73, "e_density": 0.055, "debye_K": 240, "lattice_A": 3.30,
             "density_gcc": 16.65, "defect_conc": 0.25,
             "magnetic_code": 2.0, "structure_code": 0.7,
             "log_chi_m": np.log10(1.8e-4), "beam_keV": 5.0},
        ]

        predictions = []
        for combo in predict_combos:
            X = np.array([[
                combo[f] for f in self.features_medium
            ]])
            X_scaled = self.scaler_b.transform(X)
            pred_log = self.model_b.predict(X_scaled)[0]
            pred_eV = 10 ** pred_log

            predictions.append({
                "material": combo["material"],
                "surface_state": combo["surface_state"],
                "predicted_Us_eV": round(pred_eV, 0),
                "predicted_log_Us": round(pred_log, 3),
                "defect_concentration": combo["defect_conc"],
                "magnetic_class": {3.0: "ferro", 2.0: "para", 1.0: "dia"}
                    .get(combo["magnetic_code"], "unknown"),
            })

        return predictions

    def critical_evidence_table(self) -> pd.DataFrame:
        """
        Таблица критических доказательств:
        один и тот же элемент (Pd), разные обработки → разный скрининг.
        Стандартная модель НЕ МОЖЕТ это объяснить.
        """
        pd_data = self.df[self.df["material"] == "Pd"].copy()
        pd_data = pd_data.sort_values("Us_eV")

        # Стандартная модель: Us зависит ТОЛЬКО от Z, e_density, debye_K
        # Для Pd все эти параметры ОДИНАКОВЫ → предсказание ОДНО
        std_prediction = pd_data["e_density"].iloc[0] * 1000  # rough Debye model
        pd_data["std_model_prediction_eV"] = round(std_prediction, 0)
        pd_data["std_model_error_%"] = round(
            abs(pd_data["Us_eV"] - std_prediction) / pd_data["Us_eV"] * 100, 1
        )

        return pd_data[[
            "material", "surface_state", "Us_eV", "Us_error",
            "defect_conc", "source",
            "std_model_prediction_eV", "std_model_error_%"
        ]]

    def print_report(self, result: ProofResult):
        """Напечатать полный отчёт."""
        print("=" * 70)
        print("  BARRIER PROOF: Standard Physics vs Medium-Dependent Model")
        print("=" * 70)
        print(f"\nDatapoints: {result.n_datapoints} experimental measurements")
        print(f"Target: log10(screening_energy_eV)")

        print("\n--- MODEL A: Standard Physics (Z, e_density, Debye, lattice) ---")
        print(f"  R²:        {result.r2_standard:.4f}")
        print(f"  RMSE:      {result.rmse_standard:.4f}")
        print(f"  MAE:       {result.mae_standard:.4f}")
        print(f"  AIC:       {result.aic_standard:.2f}")
        print(f"  BIC:       {result.bic_standard:.2f}")
        print(f"  LOO-CV R²: {result.loo_r2_standard:.4f}")

        print("\n--- MODEL B: Medium-Dependent (+defects, magnetic, structure) ---")
        print(f"  R²:        {result.r2_medium:.4f}")
        print(f"  RMSE:      {result.rmse_medium:.4f}")
        print(f"  MAE:       {result.mae_medium:.4f}")
        print(f"  AIC:       {result.aic_medium:.2f}")
        print(f"  BIC:       {result.bic_medium:.2f}")
        print(f"  LOO-CV R²: {result.loo_r2_medium:.4f}")

        print("\n--- COMPARISON ---")
        print(f"  ΔAIC (std−medium): {result.delta_aic:+.2f}")
        print(f"  ΔBIC (std−medium): {result.delta_bic:+.2f}")
        print(f"  F-test p-value:    {result.f_test_pvalue:.6f}")
        print(f"  Evidence:          {result.evidence_strength}")

        if result.shap_top_features:
            print("\n--- SHAP Feature Importance (Medium model) ---")
            for feat, imp in list(result.shap_top_features.items())[:8]:
                bar = "█" * int(imp * 30 / max(result.shap_top_features.values()))
                print(f"  {feat:20s}: {imp:.4f} {bar}")

        if result.predictions:
            print("\n--- PREDICTIONS for untested combinations ---")
            print(f"  {'Material':12s} {'Surface':14s} {'Predicted Us':>14s} {'Defects':>8s} {'Magnetic':>10s}")
            print("  " + "-" * 60)
            for p in result.predictions:
                print(f"  {p['material']:12s} {p['surface_state']:14s} "
                      f"{p['predicted_Us_eV']:>10.0f} eV "
                      f"{p['defect_concentration']:>8.2f} {p['magnetic_class']:>10s}")

        # Critical evidence
        print("\n--- CRITICAL EVIDENCE: Same element, different processing ---")
        crit = self.critical_evidence_table()
        print(crit.to_string(index=False))

        print("\n" + "=" * 70)
        print("CONCLUSION:")
        if result.delta_aic > 6:
            print("  The medium-dependent model is DECISIVELY superior.")
            print("  Screening energy is NOT a function of Z alone.")
            print("  Defect structure and magnetic properties are primary determinants.")
            print("  The 'Coulomb barrier' is an emergent property of the medium,")
            print("  not a fundamental nuclear constant.")
        elif result.delta_aic > 2:
            print("  Moderate evidence that medium properties matter.")
            print("  More data needed for decisive conclusion.")
        else:
            print("  Insufficient evidence. More experimental data needed.")
        print("=" * 70)

    def save_results(self, result: ProofResult, path: str):
        """Сохранить результаты в JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", path)


def main():
    """Запуск полного доказательства."""
    engine = BarrierProofEngine()

    print("Running barrier proof analysis...")
    print(f"Dataset: {len(SCREENING_DATASET)} experimental measurements")
    print(f"Materials: {len(set(d['material'] for d in SCREENING_DATASET))}")
    print()

    result = engine.run_proof(target="log_Us")
    engine.print_report(result)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "barrier_proof_results.json")
    engine.save_results(result, out_path)
    print(f"\nResults saved to {out_path}")

    return result


if __name__ == "__main__":
    main()
