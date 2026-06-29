"""Evaluation, statistical validation, LOQO, and ablation modules.

Implements:
    - calculate_metrics: QWK, Pearson, Spearman, MAE, RMSE.
    - perform_paired_ttest: Paired t-tests on instance absolute errors.
    - run_loqo_validation: Leave-One-Question-Out cross-validation.
    - run_ablation_study: Handcrafted feature group ablation analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard ASAG/AES performance metrics.

    Args:
        y_true: Normalized ground truth scores (range [0, 1]).
        y_pred: Normalized predicted scores (range [0, 1]).

    Returns:
        Dictionary containing QWK, Pearson, Spearman, RMSE, and MAE.
    """
    # Pearson correlation
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    
    # Spearman rank correlation
    spearman_rs, _ = stats.spearmanr(y_true, y_pred)
    
    # Absolute errors
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Quadratic Weighted Kappa (QWK)
    # Convert [0, 1] normalized scores to A-E ordinal categories (mapped to 0-4 integers)
    # E: [0, 0.2), D: [0.2, 0.4), C: [0.4, 0.6), B: [0.6, 0.8), A: [0.8, 1.0]
    bins = [0.2, 0.4, 0.6, 0.8]
    y_true_bins = np.digitize(y_true, bins)
    y_pred_bins = np.digitize(np.clip(y_pred, 0.0, 1.0), bins)
    qwk = cohen_kappa_score(y_true_bins, y_pred_bins, weights="quadratic")
    
    return {
        "QWK": float(qwk),
        "Pearson": float(pearson_r),
        "Spearman": float(spearman_rs),
        "RMSE": float(rmse),
        "MAE": float(mae)
    }


def perform_paired_ttest(err_a: np.ndarray, err_b: np.ndarray) -> Tuple[float, float]:
    """Perform a paired t-test between absolute error vectors of two models.

    Used to verify if the performance difference is statistically significant.

    Args:
        err_a: Absolute error vector of Model A.
        err_b: Absolute error vector of Model B.

    Returns:
        Tuple of (t_statistic, p_value).
    """
    t_stat, p_val = stats.ttest_rel(err_a, err_b)
    return float(t_stat), float(p_val)


def run_loqo_validation(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    model_type: str = "svr",
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Execute Leave-One-Question-Out (LOQO) cross-validation.

    Trains the model iteratively leaving out one unique question (topic_question_id)
    to test the model's capability to generalize to unseen questions.

    Args:
        df: Master dataset DataFrame.
        features_df: Handcrafted Sastrawi features DataFrame.
        model_type: "svr" or "ridge".
        seed: Random seed.

    Returns:
        Tuple of (loqo_predictions, loqo_metrics).
    """
    # Create unique question keys (e.g., "Lifestyle_1")
    df["q_key"] = df["topic"].astype(str) + "_" + df["question_id"].astype(str)
    unique_q_keys = df["q_key"].unique()
    
    loqo_preds = np.zeros(len(df))
    y_true = df["score_norm"].values
    X_features = features_df.values
    
    print(f"[LOQO] Starting LOQO validation across {len(unique_q_keys)} questions...")
    
    for q_key in unique_q_keys:
        val_mask = df["q_key"] == q_key
        train_mask = ~val_mask
        
        X_train, y_train = X_features[train_mask], y_true[train_mask]
        X_val = X_features[val_mask]
        
        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        # Train model
        if model_type.lower() == "svr":
            model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
        else:
            model = Ridge(alpha=1.0, random_state=seed)
            
        model.fit(X_train_s, y_train)
        preds = model.predict(X_val_s)
        
        loqo_preds[val_mask] = np.clip(preds, 0.0, 1.0)
        
    metrics = calculate_metrics(y_true, loqo_preds)
    return loqo_preds, metrics


def run_ablation_study(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    seeds: List[int],
    folds: int = 5
) -> Dict[str, Dict[str, float]]:
    """Perform a feature group ablation study.

    Systematically trains morphological SVR models leaving out one group of features
    at a time, evaluating the impact across all seeds and folds.

    Args:
        df: Master training dataset.
        features_df: Handcrafted Sastrawi features DataFrame.
        seeds: List of random seeds.
        folds: Number of folds.

    Returns:
        Dictionary mapping ablation name -> mean metrics.
    """
    from morphological import FEATURE_GROUPS, get_feature_names
    
    y_true = df["score_norm"].values
    results = {}
    
    # 1. Full Features Baseline
    full_preds_list = []
    for seed in seeds:
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        oof = np.zeros(len(df))
        for train_idx, val_idx in kf.split(df):
            X_train, y_train = features_df.values[train_idx], y_true[train_idx]
            X_val = features_df.values[val_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_va_s = scaler.transform(X_val)
            
            model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
            model.fit(X_tr_s, y_train)
            oof[val_idx] = np.clip(model.predict(X_va_s), 0.0, 1.0)
        full_preds_list.append(oof)
        
    full_metrics = pd.DataFrame([calculate_metrics(y_true, p) for p in full_preds_list]).mean().to_dict()
    results["Full Features"] = full_metrics
    
    # 2. Ablate each group
    for group_name in FEATURE_GROUPS.keys():
        print(f"[Ablation] Evaluating performance leaving out group: {group_name}...")
        
        # Determine remaining features
        remaining_groups = [g for g in FEATURE_GROUPS.keys() if g != group_name]
        remaining_cols = get_feature_names(remaining_groups)
        X_ablation = features_df[remaining_cols].values
        
        ablation_preds_list = []
        for seed in seeds:
            kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
            oof = np.zeros(len(df))
            for train_idx, val_idx in kf.split(df):
                X_train, y_train = X_ablation[train_idx], y_true[train_idx]
                X_val = X_ablation[val_idx]
                
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train)
                X_va_s = scaler.transform(X_val)
                
                model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
                model.fit(X_tr_s, y_train)
                oof[val_idx] = np.clip(model.predict(X_va_s), 0.0, 1.0)
            ablation_preds_list.append(oof)
            
        abl_metrics = pd.DataFrame([calculate_metrics(y_true, p) for p in ablation_preds_list]).mean().to_dict()
        results[f"W/O {group_name.capitalize()}"] = abl_metrics
        
    return results
