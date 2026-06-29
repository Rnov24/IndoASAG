"""Early Fusion baseline model (Feature Concatenation).

Concatenates the 768-dimensional PLM contextual embeddings and the 11-dimensional
handcrafted Sastrawi features to form a combined 779-dimensional vector, then
trains SVR or Ridge regressor across 5 seeds x 5-fold cross-validation.
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def train_early_fusion_fold(
    bert_embeddings: np.ndarray,
    hc_features: np.ndarray,
    train_scores: np.ndarray,
    val_bert_embeddings: np.ndarray,
    val_hc_features: np.ndarray,
    model_type: str = "svr",
    seed: int = 42,
    fold: int = 0
) -> np.ndarray:
    """Train an early fusion regressor on a single fold.

    Concatenates BERT embeddings and Sastrawi features, scales them, and trains.

    Args:
        bert_embeddings: Train BERT embeddings (shape: n_samples, 768).
        hc_features: Train morphological features (shape: n_samples, 11).
        train_scores: Train target scores (shape: n_samples).
        val_bert_embeddings: Val BERT embeddings (shape: n_samples, 768).
        val_hc_features: Val morphological features (shape: n_samples, 11).
        model_type: "svr" or "ridge".
        seed: Random state seed.
        fold: Fold index.

    Returns:
        Validation predictions.
    """
    # Naive concatenation of features (768 + 11 = 779 dimensions)
    X_train = np.column_stack([bert_embeddings, hc_features])
    X_val = np.column_stack([val_bert_embeddings, val_hc_features])
    
    # Scale concatenated features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    # Initialize model
    if model_type.lower() == "svr":
        model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    elif model_type.lower() == "ridge":
        model = Ridge(alpha=1.0, random_state=seed)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'svr' or 'ridge'.")
        
    # Fit model
    model.fit(X_train_s, train_scores)
    
    # Predict
    preds = model.predict(X_val_s)
    
    # Clip predictions to [0, 1] range
    preds = np.clip(preds, 0.0, 1.0)
    
    return preds
