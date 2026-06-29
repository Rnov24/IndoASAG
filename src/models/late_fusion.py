"""Late Fusion models for short answer grading.

Implements:
    - LateFusionRidge: Global Ridge regression prediction stacking.
    - LateFusionCARG: A novel, lightweight Consensus-Aware Regression Gating
      mechanism that dynamically weights predictions per sample based on agreement.
"""

import numpy as np
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.optim as optim


class LateFusionRidge:
    """Prediction-level fusion using Ridge Regression stacking.

    Finds global weights to combine predictions from PLM and morphological models:
        Final_Score = w_deep * PLM_pred + w_shallow * Morph_pred + Intercept
    """

    def __init__(self, alpha: float = 1.0):
        self.ridge = Ridge(alpha=alpha, positive=True) # positive=True to ensure weights are non-negative
        self._fitted = False

    def fit(self, pred_deep: np.ndarray, pred_shallow: np.ndarray, y_true: np.ndarray) -> "LateFusionRidge":
        X = np.column_stack([pred_deep, pred_shallow])
        self.ridge.fit(X, y_true)
        self._fitted = True
        return self

    def predict(self, pred_deep: np.ndarray, pred_shallow: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X = np.column_stack([pred_deep, pred_shallow])
        return np.clip(self.ridge.predict(X), 0.0, 1.0)

    def get_weights(self) -> dict:
        if not self._fitted:
            raise RuntimeError("Model must be fitted to retrieve weights.")
        coefs = self.ridge.coef_
        total = abs(coefs[0]) + abs(coefs[1])
        return {
            "w_deep": float(coefs[0]),
            "w_shallow": float(coefs[1]),
            "bias": float(self.ridge.intercept_),
            "pct_deep": float(abs(coefs[0]) / total) if total > 0.0 else 0.5,
            "pct_shallow": float(abs(coefs[1]) / total) if total > 0.0 else 0.5,
        }


class CARGGateModule(nn.Module):
    """Mini PyTorch Module to learn regression-based consensus gating."""

    def __init__(self):
        super().__init__()
        # Input features: [pred_deep, pred_shallow, consensus]
        self.gate = nn.Linear(3, 1)

    def forward(self, pred_deep, pred_shallow, consensus):
        # Stack inputs into a single tensor
        x = torch.stack([pred_deep, pred_shallow, consensus], dim=-1)
        # Gating weight g in range [0, 1]
        g = torch.sigmoid(self.gate(x)).squeeze(-1)
        # Combined prediction
        pred_fused = g * pred_deep + (1.0 - g) * pred_shallow
        return pred_fused, g


class LateFusionCARG:
    """Consensus-Aware Regression Gating (CARG) Late Fusion.

    Learns to dynamically weight predictions per sample:
        Consensus = 1.0 - |PLM_pred - Morph_pred|
        g = Sigmoid(w1 * PLM_pred + w2 * Morph_pred + w3 * Consensus + bias)
        Final_Score = g * PLM_pred + (1.0 - g) * Morph_pred
    """

    def __init__(self, lr: float = 0.01, epochs: int = 200, seed: int = 42):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.model = None
        self._fitted = False

    def fit(self, pred_deep: np.ndarray, pred_shallow: np.ndarray, y_true: np.ndarray) -> "LateFusionCARG":
        # Ensure reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Convert inputs to PyTorch tensors
        t_deep = torch.tensor(pred_deep, dtype=torch.float)
        t_shallow = torch.tensor(pred_shallow, dtype=torch.float)
        t_true = torch.tensor(y_true, dtype=torch.float)
        
        # Calculate consensus (prediction agreement)
        t_consensus = 1.0 - torch.abs(t_deep - t_shallow)
        
        # Initialize gate model
        self.model = CARGGateModule()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        # Optimization loop
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred_fused, _ = self.model(t_deep, t_shallow, t_consensus)
            loss = loss_fn(pred_fused, t_true)
            loss.backward()
            optimizer.step()
            
        self._fitted = True
        return self

    def predict(self, pred_deep: np.ndarray, pred_shallow: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
            
        self.model.eval()
        t_deep = torch.tensor(pred_deep, dtype=torch.float)
        t_shallow = torch.tensor(pred_shallow, dtype=torch.float)
        t_consensus = 1.0 - torch.abs(t_deep - t_shallow)
        
        with torch.no_grad():
            pred_fused, _ = self.model(t_deep, t_shallow, t_consensus)
            
        # Convert back to numpy and clip to [0, 1]
        preds = pred_fused.cpu().numpy()
        return np.clip(preds, 0.0, 1.0)

    def predict_with_gates(self, pred_deep: np.ndarray, pred_shallow: np.ndarray) -> tuple:
        """Generate predictions and return the calculated gating weight (g) per sample."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
            
        self.model.eval()
        t_deep = torch.tensor(pred_deep, dtype=torch.float)
        t_shallow = torch.tensor(pred_shallow, dtype=torch.float)
        t_consensus = 1.0 - torch.abs(t_deep - t_shallow)
        
        with torch.no_grad():
            pred_fused, g = self.model(t_deep, t_shallow, t_consensus)
            
        preds = np.clip(pred_fused.cpu().numpy(), 0.0, 1.0)
        gates = g.cpu().numpy()
        return preds, gates
