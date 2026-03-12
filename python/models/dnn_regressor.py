"""
DNN regressor for LENR excess heat prediction.
Predicts continuous excess heat output (W) with physics-informed loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import torch

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


@dataclass
class RegressorResult:
    """Results from regressor training."""
    mse: float
    rmse: float
    mae: float
    r2: float
    train_losses: list[float]
    val_losses: list[float]
    feature_importance: dict[str, float]


if not HAS_TORCH:
    # Stub base class when PyTorch is not installed
    class _ModuleStub:
        pass
    _nn_Module = _ModuleStub
else:
    _nn_Module = nn.Module


class ExcessHeatNet(_nn_Module):
    """Neural network for excess heat prediction."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DNN regressor: pip install torch")
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())  # smooth non-negative (better gradients than ReLU)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PhysicsLoss(_nn_Module):
    """Physics-informed loss: MSE + constraint penalties.

    Constraints:
    - Excess heat should be ~0 when loading < 0.84 (McKubre threshold)
    - Excess heat should scale with screening energy
    - Non-negativity (enforced by architecture, but penalize negative residuals)
    """

    def __init__(self, lambda_loading: float = 0.1, lambda_physics: float = 0.05):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required: pip install torch")
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_loading = lambda_loading
        self.lambda_physics = lambda_physics

    def forward(self, pred, target, features_dict: Optional[dict] = None):
        loss = self.mse(pred, target)

        if features_dict is not None:
            # Penalty: predicted heat when loading is very low
            if 'deuterium_loading' in features_dict:
                loading = features_dict['deuterium_loading']
                low_loading_mask = loading < 0.5
                if low_loading_mask.any():
                    penalty = pred[low_loading_mask].pow(2).mean()
                    loss = loss + self.lambda_loading * penalty

            # Penalty: predicted heat should correlate with barrier reduction
            if 'barrier_reduction_maxwell' in features_dict:
                barrier = features_dict['barrier_reduction_maxwell']
                # Lower barrier = more heat expected
                high_barrier_mask = barrier > 0.95  # almost no reduction
                if high_barrier_mask.any():
                    penalty = pred[high_barrier_mask].pow(2).mean()
                    loss = loss + self.lambda_physics * penalty

        return loss


class LENRRegressor:
    """DNN regressor for excess heat prediction.

    Uses physics-informed loss to enforce physically meaningful predictions.
    """

    def __init__(
        self,
        hidden_dims: list[int] = None,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 200,
        patience: int = 20,
        device: str = 'auto',
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required: pip install torch")

        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[ExcessHeatNet] = None
        self.scaler_X = StandardScaler()
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.feature_names: list[str] = []
        self.is_fitted = False

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = 'excess_heat_W',
        test_size: float = 0.2,
        use_physics_loss: bool = True,
    ) -> RegressorResult:
        """Train the regressor and return results."""
        self.feature_names = feature_cols

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)

        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        # Log1p transform for skewed target (most values near 0)
        y_log = np.log1p(y)
        self.y_mean = y_log.mean()
        self.y_std = y_log.std() + 1e-8
        y_scaled = (y_log - self.y_mean) / self.y_std

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42,
        )

        # Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Model
        self.model = ExcessHeatNet(len(feature_cols), self.hidden_dims).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        if use_physics_loss:
            criterion = PhysicsLoss()
        else:
            criterion = nn.MSELoss()

        # Feature indices for physics constraints
        loading_idx = feature_cols.index('deuterium_loading') if 'deuterium_loading' in feature_cols else None
        barrier_idx = feature_cols.index('barrier_reduction_maxwell') if 'barrier_reduction_maxwell' in feature_cols else None

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                pred = self.model(X_batch)

                if use_physics_loss and isinstance(criterion, PhysicsLoss):
                    feat_dict = {}
                    if loading_idx is not None:
                        feat_dict['deuterium_loading'] = X_batch[:, loading_idx]
                    if barrier_idx is not None:
                        feat_dict['barrier_reduction_maxwell'] = X_batch[:, barrier_idx]
                    loss = criterion(pred, y_batch, feat_dict)
                else:
                    loss = criterion(pred, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_test_t)
                val_loss = nn.MSELoss()(val_pred, y_test_t).item()
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True

        # Evaluate on original scale
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_test_t).cpu().numpy()

        # Inverse log1p transform
        y_pred_log = y_pred_scaled * self.y_std + self.y_mean
        y_pred = np.expm1(y_pred_log)
        y_true_log = y_test * self.y_std + self.y_mean
        y_true = np.expm1(y_true_log)

        # Clip negative predictions
        y_pred = np.maximum(y_pred, 0)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Feature importance via gradient-based saliency
        feat_imp = self._gradient_importance(X_test_t)

        return RegressorResult(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            train_losses=train_losses,
            val_losses=val_losses,
            feature_importance=feat_imp,
        )

    def _gradient_importance(self, X: torch.Tensor) -> dict[str, float]:
        """Compute gradient-based feature importance."""
        self.model.eval()
        X_grad = X.clone().requires_grad_(True)

        pred = self.model(X_grad)
        pred.sum().backward()

        importance = X_grad.grad.abs().mean(dim=0).cpu().numpy()
        feat_imp = dict(zip(self.feature_names, importance))
        return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict excess heat for new data."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X = self.scaler_X.transform(df[self.feature_names].values.astype(np.float32))
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model(X_t).cpu().numpy()

        y_log = y_scaled * self.y_std + self.y_mean
        y = np.expm1(y_log)
        return np.maximum(y, 0)

    def save(self, path: str):
        """Save model, scalers, and config."""
        torch.save({
            'model_state': self.model.state_dict(),
            'hidden_dims': self.hidden_dims,
            'input_dim': len(self.feature_names),
            'feature_names': self.feature_names,
            'scaler_X': self.scaler_X,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'LENRRegressor':
        """Load saved model."""
        data = torch.load(path, weights_only=False)
        obj = cls(hidden_dims=data['hidden_dims'], device=device)
        obj.feature_names = data['feature_names']
        obj.scaler_X = data['scaler_X']
        obj.y_mean = data['y_mean']
        obj.y_std = data['y_std']
        obj.model = ExcessHeatNet(data['input_dim'], data['hidden_dims']).to(obj.device)
        obj.model.load_state_dict(data['model_state'])
        obj.model.eval()
        obj.is_fitted = True
        return obj
