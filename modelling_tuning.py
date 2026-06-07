"""
modelling_tuning.py
===================
Script pelatihan model dengan Hyperparameter Tuning + Manual Logging MLflow.
Digunakan untuk Kriteria 2 Advance:
  - Manual logging (bukan autolog)
  - Hyperparameter tuning dengan GridSearchCV
  - Minimal 2 artefak tambahan di luar autolog:
      1. Confusion Matrix (gambar)
      2. ROC-AUC Curve (gambar)
      3. Feature Importance (gambar)
      4. Classification Report (teks)

Menggunakan DagsHub sebagai MLflow Tracking Server online.

Author  : Yusfitasari
Dataset : Heart Disease Dataset
Task    : Binary Classification
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')


# ==============================================================
# KONFIGURASI DAGSHUB + MLFLOW
# ==============================================================

# Ganti dengan username dan nama repo DagsHub kamu
DAGSHUB_USERNAME  = "<username_dagshub>"
DAGSHUB_REPO_NAME = "<nama_repo_dagshub>"

dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPO_NAME,
    mlflow=True
)

MLFLOW_EXPERIMENT_NAME = "Heart-Disease-Tuning"

# --- Path Dataset ---
TRAIN_PATH = "heart_preprocessing/heart_train.csv"
TEST_PATH  = "heart_preprocessing/heart_test.csv"
TARGET_COL = "target"

# --- Hyperparameter Grid ---
PARAM_GRID = {
    "n_estimators" : [50, 100, 200],
    "max_depth"    : [5, 10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf" : [1, 2],
}


# ==============================================================
# FUNGSI UTILITAS
# ==============================================================

def load_data():
    """Memuat dataset train dan test."""
    print("[INFO] Memuat dataset...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(TARGET_COL, axis=1)
    y_train = train_df[TARGET_COL]
    X_test  = test_df.drop(TARGET_COL, axis=1)
    y_test  = test_df[TARGET_COL]

    print(f"       Train : {X_train.shape}, Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


def hyperparameter_tuning(X_train, y_train):
    """Melakukan GridSearchCV untuk mencari parameter terbaik."""
    print("[INFO] Melakukan hyperparameter tuning (GridSearchCV)...")
    print(f"       Total kombinasi: {len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_samples_split']) * len(PARAM_GRID['min_samples_leaf'])}")

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"       Best params : {grid_search.best_params_}")
    print(f"       Best CV F1  : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X_test, y_test):
    """Menghitung semua metrik evaluasi."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall"   : float(recall_score(y_test, y_pred)),
        "f1_score" : float(f1_score(y_test, y_pred)),
        "roc_auc"  : float(roc_auc_score(y_test, y_pred_prob)),
    }
    return metrics, y_pred, y_pred_prob


# ==============================================================
# ARTEFAK TAMBAHAN
# ==============================================================

def save_confusion_matrix(y_test, y_pred, path="confusion_matrix.png"):
    """Artefak 1: Confusion Matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sehat", "Sakit"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Random Forest (Tuned)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"       Confusion matrix disimpan: {path}")
    return path


def save_roc_curve(y_test, y_pred_prob, path="roc_curve.png"):
    """Artefak 2: ROC-AUC Curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score   = roc_auc_score(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC-AUC Curve - Random Forest (Tuned)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"       ROC curve disimpan: {path}")
    return path


def save_feature_importance(model, feature_names, path="feature_importance.png"):
    """Artefak 3: Feature Importance."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_n       = min(15, len(feature_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(top_n),
        importances[indices[:top_n]][::-1],
        color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[:top_n]][::-1])
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Top Feature Importances - Random Forest (Tuned)", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"       Feature importance disimpan: {path}")
    return path


def save_classification_report(y_test, y_pred, path="classification_report.txt"):
    """Artefak 4: Classification Report teks."""
    report = classification_report(y_test, y_pred, target_names=["Sehat", "Sakit"])
    with open(path, "w") as f:
        f.write("Classification Report - Random Forest (Tuned)\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    print(f"       Classification report disimpan: {path}")
    return path


# ==============================================================
# PIPELINE UTAMA
# ==============================================================

def run():
    print("=" * 60)
    print("  MODELLING TUNING - Heart Disease Classification")
    print("  Author: Yusfitasari | MLflow + DagsHub")
    print("=" * 60)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="RandomForest_GridSearch_ManualLog"):

        # 1. Load data
        X_train, X_test, y_train, y_test = load_data()

        # 2. Hyperparameter tuning
        best_model, best_params, best_cv_f1 = hyperparameter_tuning(X_train, y_train)

        # 3. Evaluasi
        metrics, y_pred, y_pred_prob = evaluate_model(best_model, X_test, y_test)

        print("\n[INFO] Hasil Evaluasi:")
        for k, v in metrics.items():
            print(f"       {k:10s}: {v:.4f}")

        # ----------------------------------------------------------
        # 4. MANUAL LOGGING — Parameter
        # ----------------------------------------------------------
        print("\n[INFO] Manual logging: parameters...")
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # ----------------------------------------------------------
        # 5. MANUAL LOGGING — Metrik
        # ----------------------------------------------------------
        print("[INFO] Manual logging: metrics...")
        mlflow.log_metric("accuracy",       metrics["accuracy"])
        mlflow.log_metric("precision",      metrics["precision"])
        mlflow.log_metric("recall",         metrics["recall"])
        mlflow.log_metric("f1_score",       metrics["f1_score"])
        mlflow.log_metric("roc_auc",        metrics["roc_auc"])
        mlflow.log_metric("best_cv_f1",     best_cv_f1)

        # ----------------------------------------------------------
        # 6. MANUAL LOGGING — Model
        # ----------------------------------------------------------
        print("[INFO] Manual logging: model...")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="random_forest_model",
            registered_model_name="HeartDisease-RandomForest"
        )

        # ----------------------------------------------------------
        # 7. ARTEFAK TAMBAHAN (Advance: minimal 2)
        # ----------------------------------------------------------
        print("[INFO] Menyimpan artefak tambahan...")
        cm_path   = save_confusion_matrix(y_test, y_pred)
        roc_path  = save_roc_curve(y_test, y_pred_prob)
        fi_path   = save_feature_importance(best_model, list(X_train.columns))
        rep_path  = save_classification_report(y_test, y_pred)

        mlflow.log_artifact(cm_path,  artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")
        mlflow.log_artifact(fi_path,  artifact_path="plots")
        mlflow.log_artifact(rep_path, artifact_path="reports")

        # ----------------------------------------------------------
        # 8. Tag run
        # ----------------------------------------------------------
        mlflow.set_tag("author",    "Yusfitasari")
        mlflow.set_tag("model",     "RandomForestClassifier")
        mlflow.set_tag("tuning",    "GridSearchCV")
        mlflow.set_tag("dataset",   "Heart Disease")
        mlflow.set_tag("level",     "advance")

        print("\n✅ modelling_tuning.py selesai!")
        print(f"   Cek hasil di DagsHub MLflow: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")


if __name__ == "__main__":
    run()
