"""Handcrafted morphological features regression model (SVR and Ridge).

Trains SVR or Ridge models on extracted Sastrawi morphological features
using 5 seeds x 5-fold cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def train_hc_fold(
    train_features: np.ndarray,
    train_scores: np.ndarray,
    val_features: np.ndarray,
    model_type: str = "svr",
    seed: int = 42,
    fold: int = 0
) -> np.ndarray:
    """Train a handcrafted features regressor on a single fold.

    Scales features using StandardScaler before training.

    Args:
        train_features: Feature matrix for training fold.
        train_scores: Target scores for training fold.
        val_features: Feature matrix for validation fold.
        model_type: "svr" or "ridge".
        seed: Random state seed.
        fold: Fold index.

    Returns:
        Validation fold prediction array.
    """
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_val = scaler.transform(val_features)
    
    # Initialize model
    if model_type.lower() == "svr":
        # Standard robust RBF kernel SVR
        model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    elif model_type.lower() == "ridge":
        model = Ridge(alpha=1.0, random_state=seed)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'svr' or 'ridge'.")
        
    # Fit model
    model.fit(X_train, train_scores)
    
    # Predict
    preds = model.predict(X_val)
    
    # Clip predictions to valid [0, 1] normalized score range
    preds = np.clip(preds, 0.0, 1.0)
    
    return preds
