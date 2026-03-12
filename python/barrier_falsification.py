"""
Coulomb Barrier Falsification Module
=====================================
Systematic analysis: WHERE does the standard model BREAK?

Compares two frameworks:
  - Maxwell (standard): U_s = f(Z, e_density, theta_D)
  - Cherepanov:         U_s = f(chi_m, defects, a/theta_D, loading)

Key experimental evidence:
  - Czerski 2023: cold-rolled Pd → U_e = 18,200 eV (728x theory!)
  - PdO > Pd (600 vs 310 eV) → oxide as photon mass lens
  - Ferromagnets (Ni, Fe, Co) → systematically higher screening
  - T-independence (Kasagi) → Planck-like flat region at RT

Uses:
  - OLS regression with AIC/BIC comparison
  - XGBoost feature importance (Maxwell vs Cherepanov features)
  - SHAP values for interpretability
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# MATERIAL PROPERTIES FOR ANALYSIS
# =============================================================================
# Unified property table for all materials with screening data.
# Sources: CRC Handbook, ASM International, Kittel, various papers.

MATERIAL_PROPERTIES = {
    'D2_gas':     {'Z': 1,   'e_density': 0.0,   'theta_D': 0,   'a': 0,      'chi_m': 0,        'fermi_E': 0,     'mag_class': 'none',         'structure': 'gas'},
    'Pd':         {'Z': 46,  'e_density': 0.34,   'theta_D': 274, 'a': 3.891,  'chi_m': -7.2e-6,  'fermi_E': 5.56,  'mag_class': 'diamagnetic',  'structure': 'FCC'},
    'PdO':        {'Z': 46,  'e_density': 0.25,   'theta_D': 300, 'a': 3.04,   'chi_m': 1.2e-4,   'fermi_E': 3.0,   'mag_class': 'paramagnetic', 'structure': 'FCC'},
    'Ni':         {'Z': 28,  'e_density': 0.16,   'theta_D': 450, 'a': 3.524,  'chi_m': 600e-6,   'fermi_E': 7.04,  'mag_class': 'ferromagnetic','structure': 'FCC'},
    'Fe':         {'Z': 26,  'e_density': 0.17,   'theta_D': 470, 'a': 2.867,  'chi_m': 1.0,      'fermi_E': 7.47,  'mag_class': 'ferromagnetic','structure': 'BCC'},
    'Co':         {'Z': 27,  'e_density': 0.16,   'theta_D': 445, 'a': 2.507,  'chi_m': 1.0,      'fermi_E': 7.30,  'mag_class': 'ferromagnetic','structure': 'HCP'},
    'Ti':         {'Z': 22,  'e_density': 0.051,  'theta_D': 420, 'a': 2.951,  'chi_m': 153e-6,   'fermi_E': 4.87,  'mag_class': 'paramagnetic', 'structure': 'HCP'},
    'Au':         {'Z': 79,  'e_density': 0.059,  'theta_D': 165, 'a': 4.078,  'chi_m': -2.8e-6,  'fermi_E': 5.53,  'mag_class': 'diamagnetic',  'structure': 'FCC'},
    'Pt':         {'Z': 78,  'e_density': 0.066,  'theta_D': 240, 'a': 3.924,  'chi_m': 193e-6,   'fermi_E': 5.74,  'mag_class': 'paramagnetic', 'structure': 'FCC'},
    'W':          {'Z': 74,  'e_density': 0.063,  'theta_D': 400, 'a': 3.165,  'chi_m': 59e-6,    'fermi_E': 5.77,  'mag_class': 'paramagnetic', 'structure': 'BCC'},
    'Cu':         {'Z': 29,  'e_density': 0.085,  'theta_D': 343, 'a': 3.615,  'chi_m': -5.5e-6,  'fermi_E': 7.00,  'mag_class': 'diamagnetic',  'structure': 'FCC'},
    'Zr':         {'Z': 40,  'e_density': 0.043,  'theta_D': 291, 'a': 3.232,  'chi_m': -13.8e-6, 'fermi_E': 5.36,  'mag_class': 'diamagnetic',  'structure': 'HCP'},
    'Ta':         {'Z': 73,  'e_density': 0.055,  'theta_D': 240, 'a': 3.301,  'chi_m': 154e-6,   'fermi_E': 5.59,  'mag_class': 'paramagnetic', 'structure': 'BCC'},
    'Al':         {'Z': 13,  'e_density': 0.018,  'theta_D': 428, 'a': 4.050,  'chi_m': 16.5e-6,  'fermi_E': 11.7,  'mag_class': 'paramagnetic', 'structure': 'FCC'},
    'Nb':         {'Z': 41,  'e_density': 0.056,  'theta_D': 275, 'a': 3.300,  'chi_m': 195e-6,   'fermi_E': 5.32,  'mag_class': 'paramagnetic', 'structure': 'BCC'},
    'V':          {'Z': 23,  'e_density': 0.058,  'theta_D': 380, 'a': 3.024,  'chi_m': 255e-6,   'fermi_E': 5.28,  'mag_class': 'paramagnetic', 'structure': 'BCC'},
    'Cr':         {'Z': 24,  'e_density': 0.065,  'theta_D': 630, 'a': 2.885,  'chi_m': 180e-6,   'fermi_E': 5.85,  'mag_class': 'paramagnetic', 'structure': 'BCC'},
    'Mn':         {'Z': 25,  'e_density': 0.06,   'theta_D': 410, 'a': 8.913,  'chi_m': 529e-6,   'fermi_E': 5.20,  'mag_class': 'paramagnetic', 'structure': 'BCC'},
    'Sn':         {'Z': 50,  'e_density': 0.037,  'theta_D': 200, 'a': 5.832,  'chi_m': -2.2e-5,  'fermi_E': 10.2,  'mag_class': 'diamagnetic',  'structure': 'BCT'},
    'In':         {'Z': 49,  'e_density': 0.039,  'theta_D': 108, 'a': 3.252,  'chi_m': -1.1e-5,  'fermi_E': 8.63,  'mag_class': 'diamagnetic',  'structure': 'BCT'},
    'Ag':         {'Z': 47,  'e_density': 0.059,  'theta_D': 225, 'a': 4.085,  'chi_m': -1.95e-5, 'fermi_E': 5.49,  'mag_class': 'diamagnetic',  'structure': 'FCC'},
    'Yb':         {'Z': 70,  'e_density': 0.020,  'theta_D': 118, 'a': 5.485,  'chi_m': 1.1e-6,   'fermi_E': 3.0,   'mag_class': 'paramagnetic', 'structure': 'FCC'},
    'Be_BeO':     {'Z': 4,   'e_density': 0.025,  'theta_D': 1000,'a': 2.70,   'chi_m': 5.0e-5,   'fermi_E': 14.3,  'mag_class': 'paramagnetic', 'structure': 'HCP'},
    'C':          {'Z': 6,   'e_density': 0.0,    'theta_D': 2230,'a': 3.567,  'chi_m': -6.2e-6,  'fermi_E': 0,     'mag_class': 'diamagnetic',  'structure': 'diamond'},
    'Si':         {'Z': 14,  'e_density': 0.0,    'theta_D': 645, 'a': 5.431,  'chi_m': -3.2e-6,  'fermi_E': 0,     'mag_class': 'diamagnetic',  'structure': 'diamond'},
    'Ge':         {'Z': 32,  'e_density': 0.0,    'theta_D': 374, 'a': 5.658,  'chi_m': -7.7e-6,  'fermi_E': 0,     'mag_class': 'diamagnetic',  'structure': 'diamond'},
    # === Raiola 2002 metals (newly added) ===
    'Mg':         {'Z': 12,  'e_density': 0.013,  'theta_D': 400, 'a': 3.209,  'chi_m': 13.1e-6,  'fermi_E': 7.08,  'mag_class': 'paramagnetic', 'structure': 'HCP'},
    'Zn':         {'Z': 30,  'e_density': 0.013,  'theta_D': 327, 'a': 2.665,  'chi_m': -11.4e-6, 'fermi_E': 9.47,  'mag_class': 'diamagnetic',  'structure': 'HCP'},
    'Y':          {'Z': 39,  'e_density': 0.038,  'theta_D': 280, 'a': 3.648,  'chi_m': 187e-6,   'fermi_E': 3.19,  'mag_class': 'paramagnetic', 'structure': 'HCP'},
    'Mo':         {'Z': 42,  'e_density': 0.064,  'theta_D': 450, 'a': 3.147,  'chi_m': 72e-6,    'fermi_E': 6.82,  'mag_class': 'paramagnetic', 'structure': 'BCC'},
    'Ru':         {'Z': 44,  'e_density': 0.074,  'theta_D': 600, 'a': 2.706,  'chi_m': 39e-6,    'fermi_E': 6.52,  'mag_class': 'paramagnetic', 'structure': 'HCP'},
    'Rh':         {'Z': 45,  'e_density': 0.076,  'theta_D': 480, 'a': 3.803,  'chi_m': 102e-6,   'fermi_E': 5.37,  'mag_class': 'paramagnetic', 'structure': 'FCC'},
    'Cd':         {'Z': 48,  'e_density': 0.031,  'theta_D': 209, 'a': 2.979,  'chi_m': -18e-6,   'fermi_E': 7.47,  'mag_class': 'diamagnetic',  'structure': 'HCP'},
    'Hf':         {'Z': 72,  'e_density': 0.044,  'theta_D': 252, 'a': 3.195,  'chi_m': 75e-6,    'fermi_E': 5.00,  'mag_class': 'paramagnetic', 'structure': 'HCP'},
    'Re':         {'Z': 75,  'e_density': 0.069,  'theta_D': 416, 'a': 2.761,  'chi_m': 67e-6,    'fermi_E': 6.10,  'mag_class': 'paramagnetic', 'structure': 'HCP'},
    'Ir':         {'Z': 77,  'e_density': 0.072,  'theta_D': 420, 'a': 3.839,  'chi_m': 25e-6,    'fermi_E': 5.60,  'mag_class': 'paramagnetic', 'structure': 'FCC'},
    'Pb':         {'Z': 82,  'e_density': 0.013,  'theta_D': 105, 'a': 4.951,  'chi_m': -15.5e-6, 'fermi_E': 9.47,  'mag_class': 'diamagnetic',  'structure': 'FCC'},
    'B':          {'Z': 5,   'e_density': 0.0,    'theta_D': 1480,'a': 8.73,   'chi_m': -6.7e-6,  'fermi_E': 0,     'mag_class': 'diamagnetic',  'structure': 'rhombohedral'},
    'BeO':        {'Z': 4,   'e_density': 0.0,    'theta_D': 1280,'a': 2.698,  'chi_m': -1.0e-5,  'fermi_E': 0,     'mag_class': 'diamagnetic',  'structure': 'HCP'},
}

# Defect state → concentration mapping
DEFECT_MAP = {
    'cold_rolled': 0.50,
    'nano': 0.30,
    'irradiated': 0.25,
    'sputtered': 0.20,
    'oxidized': 0.15,
    'mesh': 0.10,
    'polycrystal': 0.05,
    'annealed': 0.005,
    'single_crystal': 0.001,
    'insulator': 0.001,
    'semiconductor': 0.003,
    'N/A': 0.0,
}


class BarrierFalsification:
    """Systematic comparison: standard model vs Cherepanov."""

    def load_all_screening_data(self) -> pd.DataFrame:
        """Load complete screening energy compilation with material properties."""
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from exfor_loader import SCREENING_COMPILATION

        rows = []
        for (mat, Us, err, author, year, method, defect_state) in SCREENING_COMPILATION:
            props = MATERIAL_PROPERTIES.get(mat, MATERIAL_PROPERTIES.get('Pd'))

            # Defect concentration
            defects = DEFECT_MAP.get(defect_state, 0.05)

            # Lattice focusing (Cherepanov)
            structure_f = {'FCC': 1.0, 'BCC': 0.7, 'HCP': 0.5}.get(
                props.get('structure', 'FCC'), 0.5)

            # Magnetic class factor
            mag_factor = {
                'ferromagnetic': 10.0,
                'paramagnetic': 2.0,
                'diamagnetic': 0.5,
                'none': 0.0,
            }.get(props.get('mag_class', 'diamagnetic'), 0.5)

            # Cherepanov "focusing" = structure × magnetic × lattice_ratio
            a = props.get('a', 3.5)
            theta_D = max(props.get('theta_D', 300), 1)
            focusing = structure_f * mag_factor * (a / theta_D * 100)

            rows.append({
                'material': mat,
                'Us_measured_eV': Us,
                'Us_error_eV': err,
                'author': author,
                'year': year,
                'method': method,
                'defect_state': defect_state,
                # Maxwell features
                'Z': props['Z'],
                'e_density': props['e_density'],
                'theta_D': theta_D,
                'fermi_E': props.get('fermi_E', 5.0),
                # Cherepanov features
                'chi_m_abs': abs(props.get('chi_m', 1e-6)),
                'defect_concentration': defects,
                'focusing': focusing,
                'a_over_theta': a / theta_D,
                'mag_class': props.get('mag_class', 'diamagnetic'),
                'structure': props.get('structure', 'FCC'),
                # Derived
                'log_Us': np.log10(max(Us, 1)),
                'is_ferromagnetic': 1 if props.get('mag_class') == 'ferromagnetic' else 0,
                'is_metal': 1 if props.get('e_density', 0) > 0 else 0,
            })

        return pd.DataFrame(rows)

    def compute_adiabatic_prediction(self, Z: int, e_density: float = 0.0) -> float:
        """Standard theory prediction for D-D screening in metals.

        For D-D reactions (both Z=1), screening comes from:
          1. D2 molecular electrons: U_ad ~ 25 eV (Assenbaum 1987)
          2. Metal conduction electrons (Thomas-Fermi):
             U_TF ~ 30 + e_density * 50 eV

        Combined: U_predicted ~ 25 + small metal correction.
        Maximum theoretical prediction: ~50 eV for high-density metals.

        KEY: Any measured value >> 50 eV FALSIFIES the standard model.
        Czerski 2023: 18,200 eV = 364x this prediction.
        """
        if Z <= 1:
            return 25.0  # D2 gas: molecular electrons only

        # D-D in metals: molecular + conduction electron screening
        # Thomas-Fermi gives ~30-50 eV for metals
        U_molecular = 25.0
        U_conduction = e_density * 50.0  # scales with electron density
        return U_molecular + U_conduction

    def compute_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute residuals: measured - predicted (standard model)."""
        df = df.copy()

        # Adiabatic prediction (uses Z and electron density)
        df['Us_predicted_eV'] = df.apply(
            lambda r: self.compute_adiabatic_prediction(r['Z'], r.get('e_density', 0.0)),
            axis=1,
        )

        # Residuals
        df['residual_eV'] = df['Us_measured_eV'] - df['Us_predicted_eV']
        df['ratio'] = df['Us_measured_eV'] / df['Us_predicted_eV'].clip(lower=1)
        df['log_ratio'] = np.log10(df['ratio'].clip(lower=0.01))

        # Classification
        df['anomalous'] = df['ratio'] > 3.0  # > 3x theory = anomalous
        df['extreme_anomaly'] = df['ratio'] > 50.0  # > 50x = extreme

        return df

    def correlation_analysis(self, df: pd.DataFrame) -> dict:
        """Pearson & Spearman correlations with residuals.

        Expected (if Cherepanov is right):
          - residual vs chi_m_abs → STRONG positive
          - residual vs defects  → STRONG positive
          - residual vs focusing → STRONG positive
          - residual vs e_density → WEAK (Maxwell variable)
          - residual vs Z → WEAK (Maxwell variable)
        """
        from scipy import stats

        df = self.compute_residuals(df)

        # Filter out gas-phase and theory entries
        metals = df[df['is_metal'] == 1].copy()

        results = {}
        variables = {
            # Maxwell features (should be WEAK if barrier doesn't exist)
            'Z': 'Maxwell: atomic number',
            'e_density': 'Maxwell: electron density',
            'theta_D': 'Maxwell: Debye temperature',
            'fermi_E': 'Maxwell: Fermi energy',
            # Cherepanov features (should be STRONG)
            'chi_m_abs': 'Cherepanov: |magnetic susceptibility|',
            'defect_concentration': 'Cherepanov: defect concentration',
            'focusing': 'Cherepanov: lattice focusing',
            'a_over_theta': 'Cherepanov: a/theta_D ratio',
            'is_ferromagnetic': 'Cherepanov: ferromagnetic flag',
        }

        for var, description in variables.items():
            x = metals[var].values
            y = metals['log_ratio'].values

            # Skip if constant
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                results[var] = {
                    'description': description,
                    'pearson_r': 0.0, 'pearson_p': 1.0,
                    'spearman_r': 0.0, 'spearman_p': 1.0,
                }
                continue

            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            results[var] = {
                'description': description,
                'pearson_r': float(pr),
                'pearson_p': float(pp),
                'spearman_r': float(sr),
                'spearman_p': float(sp),
            }

        return results

    def aic_bic_comparison(self, df: pd.DataFrame) -> dict:
        """Compare two OLS models using AIC and BIC.

        Model A (Maxwell):    log(Us) = a0 + a1*Z + a2*e_density + a3*theta_D
        Model B (Cherepanov):  log(Us) = b0 + b1*chi_m_abs + b2*defects + b3*(a/theta_D) + b4*focusing

        LOWER AIC/BIC = BETTER model.
        Expected: Cherepanov wins.
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            logger.warning('statsmodels not installed; AIC/BIC comparison skipped')
            return {'error': 'pip install statsmodels'}

        df = self.compute_residuals(df)
        metals = df[df['is_metal'] == 1].copy()

        if len(metals) < 10:
            return {'error': 'Not enough metal data points'}

        y = metals['log_Us'].values

        # Model A: Maxwell features
        X_maxwell = metals[['Z', 'e_density', 'theta_D', 'fermi_E']].values
        X_maxwell = sm.add_constant(X_maxwell)

        # Model B: Cherepanov features
        X_cherepanov = metals[['chi_m_abs', 'defect_concentration', 'a_over_theta', 'focusing']].values
        X_cherepanov = sm.add_constant(X_cherepanov)

        try:
            model_A = sm.OLS(y, X_maxwell).fit()
            model_B = sm.OLS(y, X_cherepanov).fit()
        except Exception as e:
            return {'error': str(e)}

        return {
            'maxwell': {
                'aic': model_A.aic,
                'bic': model_A.bic,
                'r_squared': model_A.rsquared,
                'r_squared_adj': model_A.rsquared_adj,
                'n_params': model_A.df_model + 1,
                'features': ['Z', 'e_density', 'theta_D', 'fermi_E'],
            },
            'cherepanov': {
                'aic': model_B.aic,
                'bic': model_B.bic,
                'r_squared': model_B.rsquared,
                'r_squared_adj': model_B.rsquared_adj,
                'n_params': model_B.df_model + 1,
                'features': ['chi_m_abs', 'defects', 'a/theta_D', 'focusing'],
            },
            'winner': 'cherepanov' if model_B.aic < model_A.aic else 'maxwell',
            'delta_aic': model_A.aic - model_B.aic,
            'delta_bic': model_A.bic - model_B.bic,
        }

    def ml_comparison(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> dict:
        """XGBoost comparison: Maxwell features vs Cherepanov features.

        Two separate models trained to predict log(Us).
        Compare R2 and RMSE via cross-validation.
        """
        try:
            from xgboost import XGBRegressor
            from sklearn.model_selection import cross_val_score
        except ImportError:
            logger.warning('xgboost/sklearn not installed')
            return {'error': 'pip install xgboost scikit-learn'}

        df = self.compute_residuals(df)
        metals = df[df['is_metal'] == 1].copy()

        if len(metals) < 10:
            return {'error': 'Not enough data'}

        y = metals['log_Us'].values

        # Feature sets
        maxwell_cols = ['Z', 'e_density', 'theta_D', 'fermi_E']
        cherepanov_cols = ['chi_m_abs', 'defect_concentration', 'a_over_theta', 'focusing', 'is_ferromagnetic']

        X_maxwell = metals[maxwell_cols].values
        X_cherepanov = metals[cherepanov_cols].values

        # Models
        params = dict(n_estimators=100, max_depth=4, learning_rate=0.1,
                      random_state=42, verbosity=0)

        n_cv = min(n_splits, len(metals) // 2)
        if n_cv < 2:
            n_cv = 2

        model_A = XGBRegressor(**params)
        model_B = XGBRegressor(**params)

        scores_A = cross_val_score(model_A, X_maxwell, y, cv=n_cv, scoring='r2')
        scores_B = cross_val_score(model_B, X_cherepanov, y, cv=n_cv, scoring='r2')

        # Fit on full data for feature importance
        model_A.fit(X_maxwell, y)
        model_B.fit(X_cherepanov, y)

        importance_A = dict(zip(maxwell_cols, model_A.feature_importances_))
        importance_B = dict(zip(cherepanov_cols, model_B.feature_importances_))

        return {
            'maxwell': {
                'r2_mean': float(scores_A.mean()),
                'r2_std': float(scores_A.std()),
                'features': maxwell_cols,
                'importance': importance_A,
            },
            'cherepanov': {
                'r2_mean': float(scores_B.mean()),
                'r2_std': float(scores_B.std()),
                'features': cherepanov_cols,
                'importance': importance_B,
            },
            'winner': 'cherepanov' if scores_B.mean() > scores_A.mean() else 'maxwell',
            'delta_r2': float(scores_B.mean() - scores_A.mean()),
        }

    def full_report(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Run complete falsification analysis and return report."""
        if df is None:
            df = self.load_all_screening_data()

        df = self.compute_residuals(df)

        # Summary statistics
        metals = df[df['is_metal'] == 1]
        n_anomalous = int(metals['anomalous'].sum())
        n_extreme = int(metals['extreme_anomaly'].sum())
        max_ratio = float(metals['ratio'].max())
        max_material = metals.loc[metals['ratio'].idxmax(), 'material'] if len(metals) > 0 else 'N/A'
        max_author = metals.loc[metals['ratio'].idxmax(), 'author'] if len(metals) > 0 else 'N/A'

        report = {
            'summary': {
                'total_measurements': len(df),
                'metal_measurements': len(metals),
                'n_anomalous_gt_3x': n_anomalous,
                'n_extreme_gt_50x': n_extreme,
                'max_ratio': max_ratio,
                'max_material': max_material,
                'max_author': max_author,
                'fraction_anomalous': n_anomalous / max(len(metals), 1),
            },
        }

        # Correlations
        try:
            report['correlations'] = self.correlation_analysis(df)
        except Exception as e:
            report['correlations'] = {'error': str(e)}

        # AIC/BIC
        try:
            report['aic_bic'] = self.aic_bic_comparison(df)
        except Exception as e:
            report['aic_bic'] = {'error': str(e)}

        # ML comparison
        try:
            report['ml'] = self.ml_comparison(df)
        except Exception as e:
            report['ml'] = {'error': str(e)}

        return report


# =============================================================================
# CLI
# =============================================================================
if __name__ == '__main__':
    import json
    logging.basicConfig(level=logging.INFO)

    bf = BarrierFalsification()

    # Load data
    df = bf.load_all_screening_data()
    print(f'Loaded {len(df)} screening measurements')

    # Residuals
    df = bf.compute_residuals(df)
    metals = df[df['is_metal'] == 1]

    print(f'\n=== STANDARD MODEL FAILURES ===')
    print(f'Metal measurements: {len(metals)}')
    print(f'Anomalous (>3x theory): {metals["anomalous"].sum()}/{len(metals)}')
    print(f'Extreme (>50x theory): {metals["extreme_anomaly"].sum()}/{len(metals)}')
    print(f'Max ratio: {metals["ratio"].max():.0f}x ({metals.loc[metals["ratio"].idxmax(), "material"]})')

    print(f'\nTop 10 anomalies:')
    top = metals.nlargest(10, 'ratio')
    for _, r in top.iterrows():
        print(f'  {r["material"]:>12s}  Us={r["Us_measured_eV"]:>8.0f} eV  '
              f'predicted={r["Us_predicted_eV"]:>6.0f} eV  '
              f'ratio={r["ratio"]:>7.0f}x  '
              f'{r["defect_state"]:>14s}  ({r["author"]} {r["year"]})')

    # Correlations
    print(f'\n=== CORRELATION ANALYSIS ===')
    corrs = bf.correlation_analysis(df)
    print(f'{"Variable":>25s}  {"Spearman r":>10s}  {"p-value":>10s}  Framework')
    print('-' * 70)
    for var, c in sorted(corrs.items(), key=lambda x: -abs(x[1]['spearman_r'])):
        framework = 'Cherepanov' if 'Cherepanov' in c['description'] else 'Maxwell'
        sig = '***' if c['spearman_p'] < 0.001 else '**' if c['spearman_p'] < 0.01 else '*' if c['spearman_p'] < 0.05 else ''
        print(f'{var:>25s}  {c["spearman_r"]:>+10.3f}  {c["spearman_p"]:>10.4f}{sig:>3s}  {framework}')

    # AIC/BIC
    print(f'\n=== AIC/BIC MODEL COMPARISON ===')
    aic = bf.aic_bic_comparison(df)
    if 'error' not in aic:
        for model in ['maxwell', 'cherepanov']:
            m = aic[model]
            print(f'  {model:>12s}: AIC={m["aic"]:.1f}, BIC={m["bic"]:.1f}, R2={m["r_squared"]:.3f}')
        print(f'  Winner: {aic["winner"]} (delta_AIC={aic["delta_aic"]:.1f})')
    else:
        print(f'  Error: {aic["error"]}')

    # ML comparison
    print(f'\n=== ML (XGBoost) COMPARISON ===')
    ml = bf.ml_comparison(df)
    if 'error' not in ml:
        for model in ['maxwell', 'cherepanov']:
            m = ml[model]
            print(f'  {model:>12s}: R2={m["r2_mean"]:.3f} +/- {m["r2_std"]:.3f}')
            top_feat = sorted(m['importance'].items(), key=lambda x: -x[1])[:3]
            print(f'    Top features: {", ".join(f"{k}={v:.3f}" for k, v in top_feat)}')
        print(f'  Winner: {ml["winner"]} (delta_R2={ml["delta_r2"]:.3f})')
    else:
        print(f'  Error: {ml["error"]}')
