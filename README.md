# IndoASAG: Integrating Pretrained Language Models and Handcrafted Linguistic Features via Late-Fusion Stacking for Indonesian Automated Short Answer Grading

Official research repository containing datasets, extraction pipelines, model checkpoints, experimental notebooks, and publication figures for Indonesian Automated Short Answer Grading (ASAG).

---

## 📌 Overview

Automated Short Answer Grading (ASAG) in morphologically rich, agglutinative languages such as Indonesian presents distinct challenges due to complex affixation (prefixes, infixes, suffixes, confixes) and contextual synonymy. This repository provides the complete experimental framework and diagnostic suite for comparing:
1. **Semantic Deep Representations**: Fine-tuned IndoBERT (`indobenchmark/indobert-base-p2`) with Layer-wise Learning Rate Decay (LLRD), gradual unfreezing, Multi-Sample Dropout (MSD), and R-Drop regularized loss.
2. **Handcrafted Linguistic Features**: 11 surface lexical, syntactic, and morphological features including Sastrawi root stemming, keyword coverage, bigram overlap, and text length ratios.
3. **Fusion Architectures**: Representation-level Early Fusion vs. Decision-level Late-Fusion Stacking (Ridge, LightGBM, and Consensus-Aware Regression Gating - CARG).

---

## 📊 Benchmark Highlights

Evaluated on the held-out test set ($N=433$) across 5 random seeds (42, 100, 2026, 777, 999):

| Model Architecture | QWK | Pearson ($r$) | Spearman ($\rho$) | RMSE | MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baselines** | | | | | |
| TF-IDF Baseline | $0.811$ | $0.869$ | $0.805$ | $0.1330$ | $0.1035$ |
| SBERT Baseline (Frozen) | $0.679$ | $0.741$ | $0.719$ | $0.1805$ | $0.1370$ |
| IndoBERT (Frozen) + Ridge | $0.823$ | $0.859$ | $0.831$ | $0.1292$ | $0.1007$ |
| **Standalone Modules** | | | | | |
| HC Linguistic SVR | $0.838$ | $0.891$ | $0.856$ | $0.1223$ | $0.0913$ |
| HC Linguistic LightGBM | $0.868$ | $0.923$ | $0.901$ | $0.1082$ | $0.0768$ |
| IndoBERT (Fine-Tuned) | $0.878 \pm 0.008$ | $0.930 \pm 0.002$ | $0.908 \pm 0.002$ | $0.1044 \pm 0.0051$ | $0.0797 \pm 0.0049$ |
| **Early Fusion Configurations** | | | | | |
| Early Fusion (Raw Concatenation SVR) | $0.433 \pm 0.037$ | $0.921 \pm 0.005$ | $0.899 \pm 0.008$ | $0.2015 \pm 0.0109$ | $0.1572 \pm 0.0085$ |
| Early Fusion (PCA-50 SVR) | $0.555 \pm 0.184$ | $0.877 \pm 0.021$ | $0.857 \pm 0.027$ | $0.1910 \pm 0.0269$ | $0.1561 \pm 0.0253$ |
| Early Fusion (Ridge, $\alpha=1.0$) | $0.208 \pm 0.272$ | $0.602 \pm 0.293$ | $0.600 \pm 0.273$ | $0.4028 \pm 0.1085$ | $0.3416 \pm 0.0907$ |
| **Late Fusion Stacking** | | | | | |
| Late Fusion (CARG) | $0.884 \pm 0.006$ | $0.931 \pm 0.002$ | $0.909 \pm 0.002$ | $0.1010 \pm 0.0018$ | $0.0776 \pm 0.0019$ |
| **Late Fusion (Ridge)** | $\mathbf{0.890 \pm 0.006}$ | $\mathbf{0.931 \pm 0.002}$ | $\mathbf{0.911 \pm 0.002}$ | $\mathbf{0.0984 \pm 0.0015}$ | $\mathbf{0.0737 \pm 0.0016}$ |
| **Late Fusion (LightGBM Stack)** | $\mathbf{0.899 \pm 0.004}$ | $\mathbf{0.933 \pm 0.001}$ | $\mathbf{0.912 \pm 0.002}$ | $\mathbf{0.0945 \pm 0.0012}$ | $\mathbf{0.0683 \pm 0.0012}$ |

---

## 📁 Repository Structure

```
IndoASAG/
├── dataset/                    # Dataset corpus and fold definitions
│   ├── raw/                    # Raw university exam responses
│   └── processed/              # Preprocessed splits and extracted feature sets
├── notebooks/                  # Reproducible experimental pipelines (.ipynb)
│   ├── 00_lexical_sbert_baselines.ipynb
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_exp_deep_learning.ipynb
│   ├── 03_exp_handcrafted.ipynb
│   ├── 04_exp_early_fusion.ipynb
│   ├── 05_exp_late_fusion.ipynb
│   ├── 06_evaluation_ablation.ipynb
│   ├── 07_loqo_full_model.ipynb
│   ├── 08_early_fusion_sensitivity.ipynb
│   ├── 09_frozen_indobert_baseline.ipynb
│   ├── 10_lgbm_hc_comparison.ipynb
│   ├── 11_dimensional_dominance_analysis.ipynb
│   ├── 12_cohens_d_effect_size.ipynb
│   ├── 13_dataset_stats_and_qwk_thresholds.ipynb
│   ├── 14_fix_figures_late_fusion_ridge.ipynb
│   ├── 15_fix_heatmap_and_scatter_ensemble.ipynb
│   └── 16_exp_late_fusion_lightgbm_and_rater_reliability.ipynb
├── results/                    # Exported figures and numerical summaries
│   ├── figures/                # Publication-grade PNG charts (Figures 1-6)
│   └── tables/                 # Benchmark metrics and ablation outputs
├── src/                        # Modular source code
│   ├── feature_extraction.py   # Sastrawi stemmer and linguistic extraction
│   ├── models.py               # IndoBERT regressor and stacker wrappers
│   └── evaluation.py           # Quadratic Weighted Kappa and IRR metrics
├── requirements.txt            # Python dependencies
└── README.md                   # Repository documentation
```

---

## 🔬 Experimental Notebook Overview

| Notebook | Focus & Objective |
| :--- | :--- |
| `00_lexical_sbert_baselines` | Implementation of unigram TF-IDF and frozen multilingual SBERT baselines. |
| `01_dataset_analysis` | Descriptive statistics, word count distributions, and score category frequencies. |
| `02_exp_deep_learning` | IndoBERT fine-tuning across 25 folds (5 seeds $\times$ 5 folds) with LLRD and R-Drop. |
| `03_exp_handcrafted` | Extraction of 11 handcrafted linguistic features and SVR modeling. |
| `04_exp_early_fusion` | Raw and PCA-compressed feature-concatenation early-fusion benchmarks. |
| `05_exp_late_fusion` | Ridge stacking and Consensus-Aware Regression Gating (CARG) meta-learners. |
| `06_evaluation_ablation` | Systematic feature group ablation on the linguistic feature set. |
| `07_loqo_full_model` | 40-fold Leave-One-Question-Out (LOQO) unseen prompt evaluation protocol. |
| `08_early_fusion_sensitivity` | Sensitivity analysis of PCA dimensionality ($10 \to 50$) and Ridge $\alpha$. |
| `09_frozen_indobert_baseline` | Frozen IndoBERT representation baseline with linear Ridge regression. |
| `10_lgbm_hc_comparison` | Non-linear tabular comparison (LightGBM, XGBoost, Random Forest) on linguistic features. |
| `11_dimensional_dominance_analysis` | Mathematical analysis of L2 norm imbalance and RBF kernel distance distortion. |
| `12_cohens_d_effect_size` | Non-parametric bootstrap resampling ($B=1,000$) and paired test-set significance. |
| `13_dataset_stats_and_qwk_thresholds` | Threshold discretization sensitivity analysis across alternative interval schemes. |
| `14_fix_figures_late_fusion_ridge` | Initial generation of Late Fusion Ridge scatter and residual plots. |
| `15_fix_heatmap_and_scatter_ensemble` | High-resolution 3-panel scatter (Figure 2) and 3-panel residual histogram (Figure 4). |
| `16_exp_late_fusion_lightgbm_and_rater_reliability` | Inter-Rater Reliability ($\text{ICC}$, Krippendorff's $\alpha$) and LightGBM late fusion. |

---

## 🛠️ Installation & Setup

### Environment Setup
```bash
git clone https://github.com/Rnov24/IndoASAG.git
cd IndoASAG

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### Reproducing Main Experiments
All notebooks are self-contained and run sequentially from `00` through `16`:
```bash
jupyter notebook notebooks/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Dataset is provided under CC-BY-SA 4.0.
