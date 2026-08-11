# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Eksperimen 7: Leave-One-Question-Out (LOQO) — Full Late Fusion Ridge Pipeline
#
# **Konteks Revisi:** Merespons komentar Reviewer B (#3) dan Reviewer D (#1) pada Round 1 review.
#
# Eksperimen ini mengevaluasi kemampuan generalisasi **seluruh pipeline Late Fusion Ridge**
# terhadap pertanyaan yang tidak terlihat saat training (unseen questions), menggunakan
# Leave-One-Question-Out (LOQO) cross-validation pada dataset training.
#
# ## Desain LOQO
#
# Dataset memiliki **40 pertanyaan unik** (4 topik × 10 soal), masing-masing dengan ~50–57 respons.
# Setiap fold LOQO:
# 1. **Satu pertanyaan** (±55 sampel) → validation set (unseen question)
# 2. **39 pertanyaan sisanya** (~1674 sampel) → training set
#
# Pipeline per fold (fully leak-free):
# - **HC Morph SVR**: dilatih ulang pada 39 pertanyaan, prediksi pada 1 pertanyaan heldout
# - **IndoBERT Head**: Ridge regression di atas embeddings [CLS] yang sudah tersimpan (exp2),
#   dilatih pada 39 pertanyaan, prediksi pada 1 pertanyaan heldout
# - **Late Fusion Ridge Stacker**: dilatih pada OOF predictions (dari exp2 + exp3) untuk
#   39 pertanyaan training, prediksi pada output IndoBERT Head + HC dari heldout question
#
# **Catatan Komputasi:** LOQO dijalankan **1x (seed 42)** tanpa replikasi 5-seed
# mengingat biaya komputasi. Ini sesuai saran Reviewer B: "run at least once (without
# repeating seeds) to demonstrate that performance does not collapse."

# %% [markdown]
# ## 0. Persiapan Lingkungan & Konfigurasi

# %%
import sys
import os

# Deteksi apakah kode berjalan di Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("[Info] Menjalankan notebook di lingkungan Google Colab.")
    if not os.path.exists("/content/IndoASAG"):
        try:
            from google.colab import userdata
            GH_TOKEN = userdata.get('GITHUB_TOKEN')
        except Exception:
            GH_TOKEN = None

        if GH_TOKEN:
            repo_url = f"https://{GH_TOKEN}@github.com/Rnov24/IndoASAG.git"
        else:
            repo_url = "https://github.com/Rnov24/IndoASAG.git"

        os.system(f"git clone {repo_url} /content/IndoASAG")
    os.chdir("/content/IndoASAG")
    os.system("pip install -q -r requirements.txt")
    REPO_ROOT = "/content/IndoASAG"
except ImportError:
    IN_COLAB = False
    try:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        REPO_ROOT = os.path.abspath(os.getcwd())
    print(f"[Info] Menjalankan notebook di lingkungan lokal: {REPO_ROOT}")

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr

from models.late_fusion import LateFusionRidge
from models.hc_regressor import train_hc_fold

# Konfigurasi Direktori
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PREDS_DIR   = os.path.join(RESULTS_DIR, "predictions")
LOQO_DIR    = os.path.join(RESULTS_DIR, "revision", "r1-001_loqo")

os.makedirs(LOQO_DIR, exist_ok=True)

# Seed yang digunakan untuk LOQO (sesuai keputusan: 1x tanpa replikasi)
LOQO_SEED = 42

print(f"[Config] LOQO Seed   : {LOQO_SEED}")
print(f"[Config] Results Dir : {LOQO_DIR}")

# %% [markdown]
# ## 1. Pemuatan Dataset & Fitur

# %%
# Load training data (memiliki kolom 'topic' dan 'question_id')
train_df = pd.read_parquet(os.path.join(DATASET_DIR, "train.parquet"))
train_features_df = pd.read_parquet(os.path.join(DATASET_DIR, "train_features.parquet"))

# Buat composite question group key: "Topic_QID" sebagai fold identifier
train_df = train_df.reset_index(drop=True)
train_df["q_group"] = train_df["topic"].astype(str) + "_Q" + train_df["question_id"].astype(str)

y_true = train_df["score_norm"].values
X_features = train_features_df.values

print(f"[Info] Training samples      : {len(train_df)}")
print(f"[Info] Morphological features: {X_features.shape[1]}")
print(f"[Info] Unique question groups: {train_df['q_group'].nunique()}")
print()
print("Distribusi sampel per q_group:")
group_counts = train_df.groupby("q_group").size()
print(f"  min={group_counts.min()}, max={group_counts.max()}, mean={group_counts.mean():.1f}")

# %%
# Load IndoBERT [CLS] embeddings untuk training set (dihasilkan Eksperimen 02)
emb_path = os.path.join(PREDS_DIR, f"exp2_indobert_embeddings_seed{LOQO_SEED}.npy")

if not os.path.exists(emb_path):
    try:
        from huggingface_hub import hf_hub_download
        print(f"[HuggingFace] Mengunduh exp2_indobert_embeddings_seed{LOQO_SEED}.npy...")
        hf_hub_download(
            repo_id="Rnov24/IndoASAG-model",
            filename=f"exp2_indobert_embeddings_seed{LOQO_SEED}.npy",
            local_dir=PREDS_DIR
        )
    except Exception as e:
        print(f"[Warning] Gagal mengunduh dari HuggingFace: {e}")

indobert_embeddings = np.load(emb_path)
print(f"[Info] IndoBERT embeddings shape: {indobert_embeddings.shape}")  # (1729, 768)

# %%
# Load OOF predictions dari Eksperimen 02 (IndoBERT) dan 03 (HC Morph SVR)
# — digunakan untuk melatih stacker Ridge dalam setiap fold LOQO
oof_plm = np.load(os.path.join(PREDS_DIR, f"exp2_indobert_oof_seed{LOQO_SEED}.npy"))
oof_hc  = np.load(os.path.join(PREDS_DIR, f"exp3_hc_oof_seed{LOQO_SEED}.npy"))

print(f"[Info] OOF IndoBERT shape: {oof_plm.shape}")
print(f"[Info] OOF HC Morph shape: {oof_hc.shape}")

# %% [markdown]
# ## 2. Helper: Evaluasi Metrik

# %%
def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Hitung QWK, Pearson, Spearman, RMSE, MAE."""
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)

    r, _  = pearsonr(y_true, y_pred_clipped)
    rs, _ = spearmanr(y_true, y_pred_clipped)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    mae   = mean_absolute_error(y_true, y_pred_clipped)

    bins = [0.2, 0.4, 0.6, 0.8]
    y_true_bins = np.digitize(y_true, bins)
    y_pred_bins = np.digitize(y_pred_clipped, bins)
    qwk = cohen_kappa_score(y_true_bins, y_pred_bins, weights="quadratic")

    return {"QWK": qwk, "Pearson": r, "Spearman": rs, "RMSE": rmse, "MAE": mae}

# %% [markdown]
# ## 3. Pipeline LOQO — Full Late Fusion Ridge
#
# Untuk setiap fold (1 pertanyaan heldout):
# 1. **HC Morph SVR**: retrain StandardScaler + SVR pada 39 pertanyaan, prediksi heldout
# 2. **IndoBERT Head**: fit Ridge pada embeddings 39 pertanyaan, prediksi heldout
# 3. **Late Fusion Ridge Stacker**: fit pada OOF preds (exp2+exp3) dari 39 pertanyaan,
#    prediksi pada [IndoBERT_head_pred, HC_pred] untuk heldout question

# %%
q_groups = sorted(train_df["q_group"].unique())
n_folds  = len(q_groups)

loqo_predictions = np.zeros(len(train_df))  # LOQO predictions untuk seluruh train set
fold_results = []

print(f"Memulai LOQO ({n_folds} folds)...\n")

for fold_idx, held_out_q in enumerate(q_groups):
    # --- Split indices ---
    val_mask   = train_df["q_group"] == held_out_q
    train_mask = ~val_mask

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]

    y_train = y_true[train_idx]
    y_val   = y_true[val_idx]

    # --- 1. HC Morph SVR ---
    X_hc_train = X_features[train_idx]
    X_hc_val   = X_features[val_idx]

    scaler_hc = StandardScaler()
    X_hc_train_s = scaler_hc.fit_transform(X_hc_train)
    X_hc_val_s   = scaler_hc.transform(X_hc_val)

    svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    svr.fit(X_hc_train_s, y_train)
    hc_val_preds = np.clip(svr.predict(X_hc_val_s), 0.0, 1.0)

    # --- 2. IndoBERT Head (Ridge on frozen embeddings) ---
    emb_train = indobert_embeddings[train_idx]
    emb_val   = indobert_embeddings[val_idx]

    scaler_emb = StandardScaler()
    emb_train_s = scaler_emb.fit_transform(emb_train)
    emb_val_s   = scaler_emb.transform(emb_val)

    ridge_head = Ridge(alpha=1.0)
    ridge_head.fit(emb_train_s, y_train)
    plm_val_preds = np.clip(ridge_head.predict(emb_val_s), 0.0, 1.0)

    # --- 3. Late Fusion Ridge Stacker ---
    # Fit stacker pada OOF predictions (exp2 + exp3) dari 39 training questions
    oof_plm_train = oof_plm[train_idx]
    oof_hc_train  = oof_hc[train_idx]

    stacker = LateFusionRidge(alpha=1.0)
    stacker.fit(oof_plm_train, oof_hc_train, y_train)

    # Prediksi pada held-out question menggunakan output step 1 & 2
    loqo_val_preds = stacker.predict(plm_val_preds, hc_val_preds)
    loqo_predictions[val_idx] = loqo_val_preds

    # --- Evaluasi per fold ---
    m = evaluate_metrics(y_val, loqo_val_preds)
    fold_results.append({
        "fold"      : fold_idx + 1,
        "q_group"   : held_out_q,
        "n_val"     : len(val_idx),
        **m
    })

    print(f"  Fold {fold_idx+1:02d}/{n_folds} | {held_out_q:<22} | "
          f"n={len(val_idx):3d} | QWK={m['QWK']:.4f} | MAE={m['MAE']:.4f}")

print("\n[OK] LOQO selesai.")

# %% [markdown]
# ## 4. Agregasi Hasil LOQO

# %%
# Evaluasi agregat: gunakan seluruh LOQO predictions vs y_true
loqo_overall = evaluate_metrics(y_true, loqo_predictions)

print("=" * 60)
print("HASIL LOQO — FULL LATE FUSION RIDGE (Seed 42, 40 Folds)")
print("=" * 60)
print(f"  QWK      : {loqo_overall['QWK']:.4f}")
print(f"  Pearson  : {loqo_overall['Pearson']:.4f}")
print(f"  Spearman : {loqo_overall['Spearman']:.4f}")
print(f"  RMSE     : {loqo_overall['RMSE']:.4f}")
print(f"  MAE      : {loqo_overall['MAE']:.4f}")
print("=" * 60)

# Perbandingan dengan skor held-out test set (dari Eksperimen 05, seed 42)
ridge_test_preds = np.load(
    os.path.join(PREDS_DIR, f"exp5_late_fusion_ridge_test_seed{LOQO_SEED}.npy")
)
test_df = pd.read_parquet(os.path.join(DATASET_DIR, "test.parquet"))
held_out_overall = evaluate_metrics(test_df["score_norm"].values, ridge_test_preds)

print()
print("PERBANDINGAN:")
print(f"  {'Metrik':<12} {'Held-Out Test':>15} {'LOQO':>10} {'Delta':>10}")
print(f"  {'-'*47}")
for k in ["QWK", "MAE", "Pearson"]:
    delta = loqo_overall[k] - held_out_overall[k]
    print(f"  {k:<12} {held_out_overall[k]:>15.4f} {loqo_overall[k]:>10.4f} {delta:>+10.4f}")

# %%
# Simpan fold-level results sebagai CSV
fold_df = pd.DataFrame(fold_results)
fold_df.to_csv(os.path.join(LOQO_DIR, "loqo_fold_results.csv"), index=False)
print(f"\n[OK] Fold results disimpan ke: {LOQO_DIR}/loqo_fold_results.csv")

# Simpan LOQO predictions
np.save(os.path.join(LOQO_DIR, "loqo_late_fusion_ridge_preds.npy"), loqo_predictions)
print(f"[OK] LOQO predictions disimpan ke: {LOQO_DIR}/loqo_late_fusion_ridge_preds.npy")

# %% [markdown]
# ## 5. Visualisasi Distribusi QWK per Fold

# %%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Plot 1: QWK per fold (bar chart) ---
ax = axes[0]
colors = ["#4C72B0" if q.startswith("Lifestyle") else
          "#DD8452" if q.startswith("Olahraga") else
          "#55A868" if q.startswith("Politik")  else
          "#C44E52"
          for q in fold_df["q_group"]]

bars = ax.bar(range(1, n_folds + 1), fold_df["QWK"], color=colors, alpha=0.85, edgecolor="white")
ax.axhline(loqo_overall["QWK"], color="black", linestyle="--", linewidth=1.5,
           label=f"Overall LOQO QWK = {loqo_overall['QWK']:.3f}")
ax.axhline(held_out_overall["QWK"], color="gray", linestyle=":", linewidth=1.5,
           label=f"Held-Out Test QWK = {held_out_overall['QWK']:.3f}")
ax.set_xlabel("Fold (Question)")
ax.set_ylabel("QWK")
ax.set_title("QWK per LOQO Fold\n(warna = topik)")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.0)

# Legend topik (warna)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4C72B0", label="Lifestyle"),
    Patch(facecolor="#DD8452", label="Olahraga"),
    Patch(facecolor="#55A868", label="Politik"),
    Patch(facecolor="#C44E52", label="Teknologi"),
]
ax.legend(handles=legend_elements + [
    plt.Line2D([0], [0], color="black", linestyle="--", label=f"LOQO Overall ({loqo_overall['QWK']:.3f})"),
    plt.Line2D([0], [0], color="gray", linestyle=":", label=f"Held-Out Test ({held_out_overall['QWK']:.3f})"),
], fontsize=8, loc="lower right")

# --- Plot 2: QWK distribusi per topik (boxplot) ---
ax2 = axes[1]
topics = ["Lifestyle", "Olahraga", "Politik", "Teknologi"]
topic_qwks = [
    fold_df[fold_df["q_group"].str.startswith(t)]["QWK"].values
    for t in topics
]
bp = ax2.boxplot(topic_qwks, labels=topics, patch_artist=True,
                 medianprops=dict(color="black", linewidth=2))

topic_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for patch, color in zip(bp["boxes"], topic_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax2.axhline(loqo_overall["QWK"], color="black", linestyle="--", linewidth=1.5,
            label=f"Overall LOQO = {loqo_overall['QWK']:.3f}")
ax2.set_ylabel("QWK")
ax2.set_title("Distribusi QWK per Topik\n(LOQO 40 Folds)")
ax2.legend(fontsize=9)

plt.tight_layout()
fig_path = os.path.join(LOQO_DIR, "loqo_qwk_distribution.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"[OK] Figur disimpan ke: {fig_path}")

# %% [markdown]
# ## 6. Rangkuman untuk Revisi Paper
#
# Cell ini mencetak teks ringkasan yang siap disalin ke dalam manuscript.

# %%
print("=" * 65)
print("TEKS RANGKUMAN UNTUK MANUSCRIPT (Section 4.4 / Table)")
print("=" * 65)
print()
print(f"To evaluate generalization to unseen questions, we conducted")
print(f"Leave-One-Question-Out (LOQO) cross-validation across all")
print(f"40 unique questions in the training set. In each of the 40")
print(f"folds, one question (~55 responses) was held out entirely,")
print(f"and all three pipeline components (HC Morph SVR, IndoBERT")
print(f"regression head, and Late Fusion Ridge stacker) were retrained")
print(f"on the remaining 39 questions.")
print()
print(f"The full Late Fusion Ridge pipeline achieves a LOQO QWK of")
print(f"{loqo_overall['QWK']:.3f} (MAE = {loqo_overall['MAE']:.4f}),")
print(f"compared to {held_out_overall['QWK']:.3f} on the random held-out")
print(f"test set (seed {LOQO_SEED}). The ΔQWK of")
print(f"{loqo_overall['QWK'] - held_out_overall['QWK']:+.3f} indicates that")
if loqo_overall["QWK"] >= 0.80:
    print(f"the system maintains strong performance on unseen questions,")
    print(f"demonstrating its generalization capability beyond known questions.")
else:
    print(f"there is a performance drop under the question-disjoint setting,")
    print(f"suggesting that generalization to fully unseen questions remains")
    print(f"an open challenge and a direction for future work.")

# %% [markdown]
# ## 7. Publikasi Otomatis ke GitHub & Pemutusan Runtime

# %%
import subprocess

def _run_git(*args):
    """Menjalankan command git menggunakan subprocess."""
    r = subprocess.run(["git"] + list(args), cwd=REPO_ROOT, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print(r.stderr.strip())
    return r.returncode

if IN_COLAB:
    from google.colab import userdata
    try:
        GH_TOKEN = userdata.get('GITHUB_TOKEN')
    except userdata.SecretNotFoundError:
        print("[Peringatan] Kunci rahasia 'GITHUB_TOKEN' tidak ditemukan di Colab Secrets.")
        GH_TOKEN = None
    except Exception as e:
        print(f"[Peringatan] Kendala autentikasi ({type(e).__name__}). Melanjutkan tanpa git push.")
        GH_TOKEN = None

    if GH_TOKEN:
        print("\n" + "="*50)
        print("MENGIRIMKAN UPDATE HASIL LOQO KE GITHUB")
        print("="*50)

        try:
            GH_USER  = "Rnov24"
            GH_REPO  = "IndoASAG"
            GH_EMAIL = "rrrijal24@gmail.com"
            repo_url = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{GH_REPO}.git"

            _run_git("config", "--global", "user.email", GH_EMAIL)
            _run_git("config", "--global", "user.name", GH_USER)

            # Stage notebook dan semua output LOQO
            _run_git("add", "notebooks/07_loqo_full_model.ipynb")
            _run_git("add", "results/revision/r1-001_loqo/")

            _run_git("commit", "-m",
                     "[R1-001] run LOQO on full Late Fusion Ridge pipeline (40 folds, seed 42)")
            _run_git("pull", repo_url, "revision/r1-001-loqo-full-model", "--rebase")
            rc = _run_git("push", repo_url, "revision/r1-001-loqo-full-model")

            if rc == 0:
                print("[Sukses] Hasil LOQO berhasil dikirim ke GitHub (branch: revision/r1-001-loqo-full-model).")
                print("[Info] Mengeksekusi penghentian otomatis runtime Colab...")
                from google.colab import runtime
                runtime.unassign()
            else:
                print("[Gagal] Gagal mengirim data ke GitHub.")
        except Exception as e:
            print(f"[Kesalahan] Gagal berinteraksi dengan git: {e}")
