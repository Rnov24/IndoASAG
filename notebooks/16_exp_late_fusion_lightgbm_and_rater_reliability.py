# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/Rnov24/IndoASAG/blob/main/notebooks/16_exp_late_fusion_lightgbm_and_rater_reliability.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="title_header"
# # Experiment 16: Inter-Rater Reliability Analysis and Gradient-Boosted Base Learners in Late-Fusion Stacking
#
# In this experiment, we investigate two fundamental components of our automated short answer grading framework:
# 1. **Human Annotation Reliability Assessment**: We formally evaluate the reliability and agreement among the three expert human raters using the Intraclass Correlation Coefficient (ICC under both two-way random and mixed-effects models) and Krippendorff's alpha (interval distance metric), alongside pairwise Quadratic Weighted Kappa (QWK) and Pearson correlation coefficients.
# 2. **Gradient-Boosted Decision Trees as Late-Fusion Base Learners**: We evaluate LightGBM on the 11 handcrafted linguistic features within the late-fusion stacking architecture across five random seeds (42, 100, 2026, 777, 999). We benchmark two-way fusion (IndoBERT + LightGBM) and three-way fusion (IndoBERT + SVR + LightGBM) against our baseline late-fusion stacking model (IndoBERT + SVR) on the held-out test partition.

# %% [markdown] id="sec_env_setup"
# ## 0. Environment Setup

# %% id="env_setup_code"
import os
import sys

# Windows UTF-8 stdout configuration
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Detect execution runtime
try:
    import google.colab
    IN_COLAB = True
    print("[Info] Running in Google Colab environment.")
    if not os.path.exists("/content/IndoASAG"):
        try:
            from google.colab import userdata
            GH_TOKEN = userdata.get("GITHUB_TOKEN")
        except Exception:
            GH_TOKEN = None

        if GH_TOKEN:
            repo_url = f"https://{GH_TOKEN}@github.com/Rnov24/IndoASAG.git"
        else:
            repo_url = "https://github.com/Rnov24/IndoASAG.git"

        os.system(f"git clone {repo_url} /content/IndoASAG")
    os.chdir("/content/IndoASAG")
    os.system("pip install -q -r requirements.txt lightgbm")
    REPO_ROOT = "/content/IndoASAG"
except ImportError:
    IN_COLAB = False
    try:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        cwd = os.path.abspath(os.getcwd())
        if os.path.basename(cwd) == "notebooks":
            REPO_ROOT = os.path.abspath(os.path.join(cwd, ".."))
        else:
            REPO_ROOT = cwd
    print(f"[Info] Running in local environment: {REPO_ROOT}")

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f as f_dist
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Directory definitions
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PREDS_DIR   = os.path.join(RESULTS_DIR, "predictions")
FIGS_DIR    = os.path.join(RESULTS_DIR, "figures")
REVISION_DIR = os.path.join(RESULTS_DIR, "revision")

RATER_DIR       = os.path.join(REVISION_DIR, "r2-009_rater_reliability")
LGBM_FUSION_DIR = os.path.join(REVISION_DIR, "r2-008_lgbm_late_fusion")

os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(RATER_DIR, exist_ok=True)
os.makedirs(LGBM_FUSION_DIR, exist_ok=True)

# Also create output directory in latex if local workspace
LATEX_FIGS_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "latex", "results", "figures"))
if os.path.exists(os.path.dirname(LATEX_FIGS_DIR)):
    os.makedirs(LATEX_FIGS_DIR, exist_ok=True)

SEEDS = [42, 100, 2026, 777, 999]
BINS = [0.2, 0.4, 0.6, 0.8]

print(f"[Config] Seeds: {SEEDS}")
print(f"[Config] Discretization bins: {BINS}")

# %% [markdown] id="sec_data_loading"
# ## 1. Dataset & Annotation Loading

# %% id="data_loading_code"
master_df = pd.read_parquet(os.path.join(DATASET_DIR, "dataset_master.parquet"))
train_df  = pd.read_parquet(os.path.join(DATASET_DIR, "train.parquet"))
test_df   = pd.read_parquet(os.path.join(DATASET_DIR, "test.parquet"))

train_feat_df = pd.read_parquet(os.path.join(DATASET_DIR, "train_features.parquet"))
test_feat_df  = pd.read_parquet(os.path.join(DATASET_DIR, "test_features.parquet"))

print(f"[Info] Master dataset samples: {len(master_df)}")
print(f"[Info] Training split samples: {len(train_df)}")
print(f"[Info] Testing split samples:  {len(test_df)}")
print(f"[Info] Linguistic features:    {train_feat_df.columns.tolist()}")

# Verify rater columns
rater_cols = ["score_m1", "score_m2", "score_m3"]
print("\n[Summary] Human Rater Score Statistics (Scale 0-100):")
print(master_df[rater_cols].describe().round(3))

# %% [markdown] id="sec_rater_reliability"
# ## 2. Formal Inter-Rater Reliability Analysis
#
# We evaluate the inter-rater reliability of the three human raters using:
# - Two-Way Random-Effects ICC: $\text{ICC}(2,1)$ for single rater agreement and $\text{ICC}(2,k)$ for the composite average of $k=3$ raters.
# - Two-Way Mixed-Effects ICC: $\text{ICC}(3,1)$ and $\text{ICC}(3,k)$ for consistency.
# - Krippendorff's alpha ($\alpha$) under an interval distance metric.
# - Pairwise Pearson correlations and Quadratic Weighted Kappa (discretized into 5 ordinal categories using $[0.2, 0.4, 0.6, 0.8]$ thresholds).

# %% id="rater_reliability_calc"
def calculate_two_way_icc(ratings: np.ndarray, alpha_level: float = 0.05):
    """
    Computes Two-Way ANOVA Intraclass Correlation Coefficients (ICC)
    along with 95% Confidence Intervals.
    
    ratings: array of shape (n_targets, k_raters)
    """
    n, k = ratings.shape
    row_mean = ratings.mean(axis=1)
    col_mean = ratings.mean(axis=0)
    grand_mean = ratings.mean()

    # Sum of Squares
    SSR = k * np.sum((row_mean - grand_mean) ** 2)
    SSC = n * np.sum((col_mean - grand_mean) ** 2)
    SST = np.sum((ratings - grand_mean) ** 2)
    SSE = SST - SSR - SSC

    # Mean Squares
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    # Point estimates
    # ICC(2,1): Two-way random effects, single rater, absolute agreement
    icc21 = (MSR - MSE) / (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE))
    # ICC(2,k): Two-way random effects, average of k raters, absolute agreement
    icc2k = (MSR - MSE) / (MSR + (MSC - MSE) / n)

    # ICC(3,1): Two-way mixed effects, single rater, consistency
    icc31 = (MSR - MSE) / (MSR + (k - 1) * MSE)
    # ICC(3,k): Two-way mixed effects, average of k raters, consistency
    icc3k = (MSR - MSE) / MSR

    # F-statistics and Confidence Intervals for ICC(2,1)
    # Following McGraw and Wong (1996) approximation
    F_obs = MSR / MSE
    df1 = n - 1
    df2 = (n - 1) * (k - 1)

    F_low = F_obs / f_dist.ppf(1 - alpha_level / 2, df1, df2)
    F_high = F_obs * f_dist.ppf(1 - alpha_level / 2, df2, df1)

    # Approximate 95% CI bounds for ICC(2,1)
    ci_low_21 = (F_low - 1) / (F_low + (k - 1))
    ci_high_21 = (F_high - 1) / (F_high + (k - 1))

    # Approximate 95% CI bounds for ICC(2,k)
    ci_low_2k = 1 - (1 / F_low)
    ci_high_2k = 1 - (1 / F_high)

    return {
        "MSR": float(MSR),
        "MSC": float(MSC),
        "MSE": float(MSE),
        "ICC(2,1)": float(icc21),
        "ICC(2,1)_95CI": (float(max(0.0, ci_low_21)), float(min(1.0, ci_high_21))),
        "ICC(2,k)": float(icc2k),
        "ICC(2,k)_95CI": (float(max(0.0, ci_low_2k)), float(min(1.0, ci_high_2k))),
        "ICC(3,1)": float(icc31),
        "ICC(3,k)": float(icc3k),
    }

def calculate_krippendorff_alpha_interval(ratings: np.ndarray) -> float:
    """
    Computes Krippendorff's alpha for fully observed interval data across k raters.
    ratings: array of shape (n_targets, k_raters)
    """
    n, k = ratings.shape
    # Pairwise squared difference within each target
    diffs_sq = np.zeros(n)
    for i in range(k):
        for j in range(i + 1, k):
            diffs_sq += (ratings[:, i] - ratings[:, j]) ** 2
    
    n_pairs = k * (k - 1) / 2.0
    D_o = np.mean(diffs_sq) / n_pairs
    
    # Total variance
    var_total = np.var(ratings, ddof=1)
    D_e = 2.0 * var_total
    
    return float(1.0 - (D_o / D_e))

# Extract normalized ratings in [0, 1]
ratings_raw = master_df[rater_cols].values
ratings_norm = ratings_raw / 100.0

icc_results = calculate_two_way_icc(ratings_norm)
krip_alpha = calculate_krippendorff_alpha_interval(ratings_norm)

# Pairwise metrics
r12, _ = pearsonr(ratings_norm[:, 0], ratings_norm[:, 1])
r13, _ = pearsonr(ratings_norm[:, 0], ratings_norm[:, 2])
r23, _ = pearsonr(ratings_norm[:, 1], ratings_norm[:, 2])
mean_pearson = float(np.mean([r12, r13, r23]))

qwk12 = cohen_kappa_score(np.digitize(ratings_norm[:, 0], BINS), np.digitize(ratings_norm[:, 1], BINS), weights="quadratic")
qwk13 = cohen_kappa_score(np.digitize(ratings_norm[:, 0], BINS), np.digitize(ratings_norm[:, 2], BINS), weights="quadratic")
qwk23 = cohen_kappa_score(np.digitize(ratings_norm[:, 1], BINS), np.digitize(ratings_norm[:, 2], BINS), weights="quadratic")
mean_qwk = float(np.mean([qwk12, qwk13, qwk23]))

mean_inter_rater_std = float(master_df["score_manual_std"].mean())

print("\n" + "="*65)
print("FORMAL INTER-RATER RELIABILITY BENCHMARK (N=2,162)")
print("="*65)
print(f"  Mean Inter-Rater Standard Deviation:  {mean_inter_rater_std:.3f} points (scale 0-100)")
print(f"  Pairwise Pearson Correlation (r):     {mean_pearson:.4f} (r12={r12:.4f}, r13={r13:.4f}, r23={r23:.4f})")
print(f"  Pairwise Quadratic Weighted Kappa:    {mean_qwk:.4f} (q12={qwk12:.4f}, q13={qwk13:.4f}, q23={qwk23:.4f})")
print(f"  Krippendorff's Alpha (Interval):      {krip_alpha:.4f}")
print(f"  ICC(2,1) [Single Rater Agreement]:   {icc_results['ICC(2,1)']:.4f} (95% CI: [{icc_results['ICC(2,1)_95CI'][0]:.4f}, {icc_results['ICC(2,1)_95CI'][1]:.4f}])")
print(f"  ICC(2,k) [Average Rater Agreement]:  {icc_results['ICC(2,k)']:.4f} (95% CI: [{icc_results['ICC(2,k)_95CI'][0]:.4f}, {icc_results['ICC(2,k)_95CI'][1]:.4f}])")
print(f"  ICC(3,1) [Single Rater Consistency]: {icc_results['ICC(3,1)']:.4f}")
print(f"  ICC(3,k) [Average Rater Consistency]:{icc_results['ICC(3,k)']:.4f}")

# Save metrics summary table
reliability_df = pd.DataFrame([
    {"Metric": "Mean Inter-Rater Std", "Value": round(mean_inter_rater_std, 4), "Interpretation": "Score dispersion per answer (0-100)"},
    {"Metric": "Pairwise Pearson r", "Value": round(mean_pearson, 4), "Interpretation": "Mean pairwise linear correlation"},
    {"Metric": "Pairwise QWK", "Value": round(mean_qwk, 4), "Interpretation": "Mean pairwise quadratic weighted kappa"},
    {"Metric": "Krippendorff Alpha", "Value": round(krip_alpha, 4), "Interpretation": "Interval reliability coefficient (>0.80 highly reliable)"},
    {"Metric": "ICC(2,1)", "Value": round(icc_results["ICC(2,1)"], 4), "Interpretation": f"Two-way random single rater absolute agreement (95% CI: {icc_results['ICC(2,1)_95CI'][0]:.4f}-{icc_results['ICC(2,1)_95CI'][1]:.4f})"},
    {"Metric": "ICC(2,k)", "Value": round(icc_results["ICC(2,k)"], 4), "Interpretation": f"Two-way random k=3 raters absolute agreement (95% CI: {icc_results['ICC(2,k)_95CI'][0]:.4f}-{icc_results['ICC(2,k)_95CI'][1]:.4f})"},
    {"Metric": "ICC(3,1)", "Value": round(icc_results["ICC(3,1)"], 4), "Interpretation": "Two-way mixed single rater consistency"},
    {"Metric": "ICC(3,k)", "Value": round(icc_results["ICC(3,k)"], 4), "Interpretation": "Two-way mixed k=3 raters consistency"},
])
reliability_df.to_csv(os.path.join(RATER_DIR, "inter_rater_reliability_table.csv"), index=False)
print(f"[OK] Saved reliability metrics to {os.path.join(RATER_DIR, 'inter_rater_reliability_table.csv')}")

# %% [markdown] id="sec_lgbm_training"
# ## 3. LightGBM Base Learner Modeling across 5 Seeds
#
# We train LightGBM on the 11 handcrafted linguistic features across 5 folds and 5 random seeds to produce leak-free Out-of-Fold (OOF) train predictions ($N=1,729$) and held-out test predictions ($N=433$).

# %% id="lgbm_training_code"
X_train_hc = train_feat_df.values
y_train    = train_df["score_norm"].values

X_test_hc  = test_feat_df.values
y_test     = test_df["score_norm"].values

lgbm_seed_metrics = []

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(train_df))
    test_fold_preds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
        X_tr, y_tr = X_train_hc[tr_idx], y_train[tr_idx]
        X_va       = X_train_hc[va_idx]

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed + fold,
            n_jobs=1,
            verbose=-1
        )
        model.fit(X_tr, y_tr)
        oof_preds[va_idx] = np.clip(model.predict(X_va), 0.0, 1.0)
        test_fold_preds.append(np.clip(model.predict(X_test_hc), 0.0, 1.0))

    # Mean test prediction across folds for this seed
    seed_test_pred = np.mean(test_fold_preds, axis=0)

    # Save predictions
    np.save(os.path.join(PREDS_DIR, f"exp16_lgbm_oof_seed{seed}.npy"), oof_preds)
    np.save(os.path.join(PREDS_DIR, f"exp16_lgbm_test_seed{seed}.npy"), seed_test_pred)

    # Evaluate test performance
    qwk = cohen_kappa_score(np.digitize(y_test, BINS), np.digitize(seed_test_pred, BINS), weights="quadratic")
    r, _ = pearsonr(y_test, seed_test_pred)
    rs, _ = spearmanr(y_test, seed_test_pred)
    mae = mean_absolute_error(y_test, seed_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, seed_test_pred))

    lgbm_seed_metrics.append({
        "seed": seed,
        "QWK": qwk,
        "Pearson": r,
        "Spearman": rs,
        "RMSE": rmse,
        "MAE": mae
    })
    print(f"  [LightGBM HC] Seed {seed:<4} | QWK: {qwk:.4f} | Pearson: {r:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

lgbm_results_df = pd.DataFrame(lgbm_seed_metrics)
print("\n[Summary] LightGBM Handcrafted Model Test Performance (5 seeds):")
print(f"  QWK:     {lgbm_results_df['QWK'].mean():.4f} ± {lgbm_results_df['QWK'].std():.4f}")
print(f"  Pearson: {lgbm_results_df['Pearson'].mean():.4f} ± {lgbm_results_df['Pearson'].std():.4f}")
print(f"  MAE:     {lgbm_results_df['MAE'].mean():.4f} ± {lgbm_results_df['MAE'].std():.4f}")
print(f"  RMSE:    {lgbm_results_df['RMSE'].mean():.4f} ± {lgbm_results_df['RMSE'].std():.4f}")

# %% [markdown] id="sec_late_fusion_benchmark"
# ## 4. Multi-Architecture Late-Fusion Stacking Benchmark
#
# We evaluate three late-fusion configurations with non-negative Ridge stacking ($\alpha=1.0$):
# 1. **Baseline Late Fusion Ridge (IndoBERT + SVR)**: Stacking deep contextual semantics with SVR linguistic predictions.
# 2. **Alternative Late Fusion Ridge (IndoBERT + LightGBM)**: Stacking deep contextual semantics with gradient-boosted decision trees.
# 3. **Comprehensive 3-Way Late Fusion Ridge (IndoBERT + SVR + LightGBM)**: Joint stacking combining deep representations, margin-based shallow regression, and ensemble gradient-boosted trees.

# %% id="late_fusion_benchmark_code"
benchmark_records = []

for seed in SEEDS:
    # 1. Load predictions
    ib_oof  = np.load(os.path.join(PREDS_DIR, f"exp2_indobert_oof_seed{seed}.npy"))
    ib_test = np.load(os.path.join(PREDS_DIR, f"exp2_indobert_test_seed{seed}.npy"))

    svr_oof  = np.load(os.path.join(PREDS_DIR, f"exp3_hc_oof_seed{seed}.npy"))
    svr_test = np.load(os.path.join(PREDS_DIR, f"exp3_hc_test_seed{seed}.npy"))

    lgb_oof  = np.load(os.path.join(PREDS_DIR, f"exp16_lgbm_oof_seed{seed}.npy"))
    lgb_test = np.load(os.path.join(PREDS_DIR, f"exp16_lgbm_test_seed{seed}.npy"))

    # Config A: Baseline IndoBERT + SVR
    ridge_base = Ridge(alpha=1.0, positive=True)
    ridge_base.fit(np.column_stack([ib_oof, svr_oof]), y_train)
    pred_base = np.clip(ridge_base.predict(np.column_stack([ib_test, svr_test])), 0.0, 1.0)

    # Config B: 2-Way IndoBERT + LightGBM
    ridge_lgbm = Ridge(alpha=1.0, positive=True)
    ridge_lgbm.fit(np.column_stack([ib_oof, lgb_oof]), y_train)
    pred_lgbm = np.clip(ridge_lgbm.predict(np.column_stack([ib_test, lgb_test])), 0.0, 1.0)

    # Config C: 3-Way IndoBERT + SVR + LightGBM
    ridge_3way = Ridge(alpha=1.0, positive=True)
    ridge_3way.fit(np.column_stack([ib_oof, svr_oof, lgb_oof]), y_train)
    pred_3way = np.clip(ridge_3way.predict(np.column_stack([ib_test, svr_test, lgb_test])), 0.0, 1.0)

    # Helper evaluation
    def record_metrics(config_name, preds, weights_dict):
        qwk = cohen_kappa_score(np.digitize(y_test, BINS), np.digitize(preds, BINS), weights="quadratic")
        r, _ = pearsonr(y_test, preds)
        rs, _ = spearmanr(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        benchmark_records.append({
            "model": config_name,
            "seed": seed,
            "QWK": qwk,
            "Pearson": r,
            "Spearman": rs,
            "RMSE": rmse,
            "MAE": mae,
            **weights_dict
        })

    record_metrics("Late Fusion (IndoBERT + SVR)", pred_base, {
        "w_plm": float(ridge_base.coef_[0]),
        "w_svr": float(ridge_base.coef_[1]),
        "w_lgbm": 0.0,
        "intercept": float(ridge_base.intercept_)
    })

    record_metrics("Late Fusion (IndoBERT + LightGBM)", pred_lgbm, {
        "w_plm": float(ridge_lgbm.coef_[0]),
        "w_svr": 0.0,
        "w_lgbm": float(ridge_lgbm.coef_[1]),
        "intercept": float(ridge_lgbm.intercept_)
    })

    record_metrics("3-Way Late Fusion (IndoBERT + SVR + LightGBM)", pred_3way, {
        "w_plm": float(ridge_3way.coef_[0]),
        "w_svr": float(ridge_3way.coef_[1]),
        "w_lgbm": float(ridge_3way.coef_[2]),
        "intercept": float(ridge_3way.intercept_)
    })

benchmark_df = pd.DataFrame(benchmark_records)
benchmark_df.to_csv(os.path.join(LGBM_FUSION_DIR, "late_fusion_per_seed_runs.csv"), index=False)

# Aggregated Table
agg_summary = benchmark_df.groupby("model").agg({
    "QWK": ["mean", "std"],
    "Pearson": ["mean", "std"],
    "Spearman": ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE": ["mean", "std"]
}).round(4)

print("\n" + "="*75)
print("LATE-FUSION ARCHITECTURE COMPARISON (Held-Out Test Set, 5 Seeds)")
print("="*75)
for model_name, grp in benchmark_df.groupby("model"):
    print(f"\n{model_name}:")
    print(f"  QWK:      {grp['QWK'].mean():.4f} ± {grp['QWK'].std():.4f}")
    print(f"  Pearson:  {grp['Pearson'].mean():.4f} ± {grp['Pearson'].std():.4f}")
    print(f"  Spearman: {grp['Spearman'].mean():.4f} ± {grp['Spearman'].std():.4f}")
    print(f"  MAE:      {grp['MAE'].mean():.4f} ± {grp['MAE'].std():.4f}")
    print(f"  RMSE:     {grp['RMSE'].mean():.4f} ± {grp['RMSE'].std():.4f}")

# %% [markdown] id="sec_bootstrap_significance"
# ## 5. Statistical Significance & Paired Bootstrap Analysis
#
# To strictly evaluate whether integrating LightGBM into late fusion provides a statistically reliable advantage over baseline stacking, we execute a paired bootstrap test ($B = 1,000$ resamples) on the test predictions.

# %% id="bootstrap_calc_code"
B = 1000
rng = np.random.default_rng(2026)
n_test = len(y_test)

# Average ensemble test predictions across seeds
base_ensemble_preds = np.zeros(n_test)
lgbm_ensemble_preds = np.zeros(n_test)
threeway_ensemble_preds = np.zeros(n_test)

for seed in SEEDS:
    ib_oof  = np.load(os.path.join(PREDS_DIR, f"exp2_indobert_oof_seed{seed}.npy"))
    ib_test = np.load(os.path.join(PREDS_DIR, f"exp2_indobert_test_seed{seed}.npy"))
    svr_oof  = np.load(os.path.join(PREDS_DIR, f"exp3_hc_oof_seed{seed}.npy"))
    svr_test = np.load(os.path.join(PREDS_DIR, f"exp3_hc_test_seed{seed}.npy"))
    lgb_oof  = np.load(os.path.join(PREDS_DIR, f"exp16_lgbm_oof_seed{seed}.npy"))
    lgb_test = np.load(os.path.join(PREDS_DIR, f"exp16_lgbm_test_seed{seed}.npy"))

    r_base = Ridge(alpha=1.0, positive=True).fit(np.column_stack([ib_oof, svr_oof]), y_train)
    r_lgbm = Ridge(alpha=1.0, positive=True).fit(np.column_stack([ib_oof, lgb_oof]), y_train)
    r_3w   = Ridge(alpha=1.0, positive=True).fit(np.column_stack([ib_oof, svr_oof, lgb_oof]), y_train)

    base_ensemble_preds += np.clip(r_base.predict(np.column_stack([ib_test, svr_test])), 0.0, 1.0) / len(SEEDS)
    lgbm_ensemble_preds += np.clip(r_lgbm.predict(np.column_stack([ib_test, lgb_test])), 0.0, 1.0) / len(SEEDS)
    threeway_ensemble_preds += np.clip(r_3w.predict(np.column_stack([ib_test, svr_test, lgb_test])), 0.0, 1.0) / len(SEEDS)

# Paired Bootstrap
boot_delta_qwk = []
boot_delta_mae = []

for b in range(B):
    idx = rng.choice(n_test, size=n_test, replace=True)
    y_b = y_test[idx]
    y_base_b = base_ensemble_preds[idx]
    y_lgb_b  = lgbm_ensemble_preds[idx]

    qwk_base = cohen_kappa_score(np.digitize(y_b, BINS), np.digitize(y_base_b, BINS), weights="quadratic")
    qwk_lgb  = cohen_kappa_score(np.digitize(y_b, BINS), np.digitize(y_lgb_b, BINS), weights="quadratic")

    mae_base = mean_absolute_error(y_b, y_base_b)
    mae_lgb  = mean_absolute_error(y_b, y_lgb_b)

    boot_delta_qwk.append(qwk_lgb - qwk_base)
    boot_delta_mae.append(mae_lgb - mae_base)

boot_delta_qwk = np.array(boot_delta_qwk)
boot_delta_mae = np.array(boot_delta_mae)

ci_qwk = np.percentile(boot_delta_qwk, [2.5, 97.5])
ci_mae = np.percentile(boot_delta_mae, [2.5, 97.5])
p_val_qwk = float(np.mean(boot_delta_qwk <= 0.0))
p_val_mae = float(np.mean(boot_delta_mae >= 0.0))

print("\n" + "="*70)
print("PAIRED BOOTSTRAP SIGNIFICANCE (Late Fusion LightGBM vs Late Fusion SVR)")
print("="*75)
print(f"  ΔQWK Mean:  {np.mean(boot_delta_qwk):+.4f} (95% CI: [{ci_qwk[0]:+.4f}, {ci_qwk[1]:+.4f}], p={p_val_qwk:.4f})")
print(f"  ΔMAE Mean:  {np.mean(boot_delta_mae):+.4f} (95% CI: [{ci_mae[0]:+.4f}, {ci_mae[1]:+.4f}], p={p_val_mae:.4f})")

bootstrap_res_df = pd.DataFrame([{
    "Comparison": "Late Fusion (IB+LightGBM) vs Late Fusion (IB+SVR)",
    "Delta_QWK_Mean": round(float(np.mean(boot_delta_qwk)), 4),
    "Delta_QWK_95CI_Low": round(float(ci_qwk[0]), 4),
    "Delta_QWK_95CI_High": round(float(ci_qwk[1]), 4),
    "p_value_QWK": round(p_val_qwk, 4),
    "Delta_MAE_Mean": round(float(np.mean(boot_delta_mae)), 4),
    "Delta_MAE_95CI_Low": round(float(ci_mae[0]), 4),
    "Delta_MAE_95CI_High": round(float(ci_mae[1]), 4),
    "p_value_MAE": round(p_val_mae, 4),
}])
bootstrap_res_df.to_csv(os.path.join(LGBM_FUSION_DIR, "bootstrap_significance_lgbm_vs_svr.csv"), index=False)
print(f"[OK] Saved bootstrap significance table.")

# %% [markdown] id="sec_diagnostic_visualizations"
# ## 6. Diagnostic Visualizations

# %% id="visualizations_code"
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = sns.color_palette("deep")

# 1. Figure: Inter-Rater Agreement Matrix
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

corr_matrix = master_df[rater_cols].corr()
corr_matrix.columns = ["Rater 1", "Rater 2", "Rater 3"]
corr_matrix.index   = ["Rater 1", "Rater 2", "Rater 3"]

sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, ax=axes[0], vmin=0.85, vmax=1.0)
axes[0].set_title("Pairwise Pearson Correlation (r)")

qwk_matrix = np.array([
    [1.0, qwk12, qwk13],
    [qwk12, 1.0, qwk23],
    [qwk13, qwk23, 1.0]
])
sns.heatmap(qwk_matrix, annot=True, fmt=".3f", cmap="YlOrRd", cbar=True, ax=axes[1],
            xticklabels=["Rater 1", "Rater 2", "Rater 3"],
            yticklabels=["Rater 1", "Rater 2", "Rater 3"], vmin=0.85, vmax=1.0)
axes[1].set_title("Pairwise Quadratic Weighted Kappa (QWK)")

plt.tight_layout()
fig_rater_path = os.path.join(FIGS_DIR, "inter_rater_agreement_matrix.png")
fig.savefig(fig_rater_path, bbox_inches="tight")
if os.path.exists(LATEX_FIGS_DIR):
    fig.savefig(os.path.join(LATEX_FIGS_DIR, "inter_rater_agreement_matrix.png"), bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved inter-rater agreement matrix figure: {fig_rater_path}")

# 2. Figure: Comparison of Late-Fusion Architectures
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

models_order = [
    "Late Fusion (IndoBERT + SVR)",
    "Late Fusion (IndoBERT + LightGBM)",
    "3-Way Late Fusion (IndoBERT + SVR + LightGBM)"
]

summary_plot_df = benchmark_df.groupby("model").agg({"QWK": ["mean", "std"]}).loc[models_order]
means = summary_plot_df[("QWK", "mean")].values
stds  = summary_plot_df[("QWK", "std")].values

x = np.arange(len(models_order))
bars = ax.bar(x, means, yerr=stds, capsize=5, color=[palette[0], palette[2], palette[3]], alpha=0.85, width=0.55)

ax.set_ylabel("Quadratic Weighted Kappa (QWK)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(["IndoBERT + SVR\n(Baseline)", "IndoBERT + LightGBM\n(2-Way)", "IndoBERT + SVR + LightGBM\n(3-Way)"], fontsize=10)
ax.set_ylim(0.85, 0.92)

for bar, mean_val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, mean_val + 0.003, f"{mean_val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
fig_arch_path = os.path.join(FIGS_DIR, "late_fusion_architectures_comparison.png")
fig.savefig(fig_arch_path, bbox_inches="tight")
if os.path.exists(LATEX_FIGS_DIR):
    fig.savefig(os.path.join(LATEX_FIGS_DIR, "late_fusion_architectures_comparison.png"), bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved late-fusion architectures comparison figure: {fig_arch_path}")

# %% [markdown] id="sec_colab_git_sync"
# ## 7. Colab Git Sync & Teardown

# %% id="colab_sync_code"
import subprocess

def _run_git(*args):
    """Execute git command safely via subprocess."""
    r = subprocess.run(["git"] + list(args), cwd=REPO_ROOT, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print(r.stderr.strip())
    return r.returncode

if IN_COLAB:
    from google.colab import userdata
    try:
        GH_TOKEN = userdata.get("GITHUB_TOKEN")
    except userdata.SecretNotFoundError:
        print("[Warning] Secret 'GITHUB_TOKEN' not found in Colab Secrets.")
        GH_TOKEN = None
    except Exception as e:
        print(f"[Warning] Authentication error ({type(e).__name__}). Proceeding without git push.")
        GH_TOKEN = None

    if GH_TOKEN:
        print("\n" + "="*50)
        print("PUSHING EXPERIMENT RESULTS TO GITHUB")
        print("="*50)

        try:
            GH_USER = "Rnov24"
            GH_REPO = "IndoASAG"
            GH_EMAIL = "rrrijal24@gmail.com"
            repo_url = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{GH_REPO}.git"

            _run_git("config", "--global", "user.email", GH_EMAIL)
            _run_git("config", "--global", "user.name", GH_USER)

            _run_git("add", "notebooks/16_exp_late_fusion_lightgbm_and_rater_reliability.ipynb")
            _run_git("add", "results/revision/r2-008_lgbm_late_fusion/")
            _run_git("add", "results/revision/r2-009_rater_reliability/")
            _run_git("add", "results/figures/inter_rater_agreement_matrix.png")
            _run_git("add", "results/figures/late_fusion_architectures_comparison.png")

            _run_git("commit", "-m", "feat(exp16): add inter-rater reliability and LightGBM late fusion benchmark")
            _run_git("pull", repo_url, "main", "--rebase")
            rc = _run_git("push", repo_url, "main")

            if rc == 0:
                print("[Success] Results successfully pushed to GitHub.")
                print("[Info] Disconnecting Colab runtime...")
                from google.colab import runtime
                runtime.unassign()
            else:
                print("[Failed] Failed to push data to GitHub.")
        except Exception as e:
            print(f"[Error] Failed to interact with git: {e}")
