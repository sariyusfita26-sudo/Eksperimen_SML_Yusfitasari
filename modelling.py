"""
modelling.py
============
Script pelatihan model Machine Learning untuk Heart Disease Classification.
Menggunakan MLflow autolog untuk tracking eksperimen.

Untuk Kriteria 2 Basic:
- Model: Random Forest Classifier
- Tracking: MLflow autolog (lokal atau DagsHub)
- Dataset: heart_preprocessing/heart_train.csv

Author  : Yusfitasari
Dataset : Heart Disease Dataset
Task    : Binary Classification
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


# ==============================================================
# KONFIGURASI
# ==============================================================

# --- DagsHub / MLflow Tracking ---
# Uncomment baris di bawah ini jika menggunakan DagsHub (Advance)
# import dagshub
# dagshub.init(repo_owner='<username_dagshub>', repo_name='<nama_repo>', mlflow=True)

# Untuk lokal (Basic/Skilled), biarkan default (mlruns/ di direktori ini)
MLFLOW_EXPERIMENT_NAME = "Heart-Disease-Classification"

# --- Path Dataset ---
TRAIN_PATH = "heart_preprocessing/heart_train.csv"
TEST_PATH  = "heart_preprocessing/heart_test.csv"
TARGET_COL = "target"

# --- Parameter Model ---
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1
}


# ==============================================================
# FUNGSI UTAMA
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


def train_model(X_train, y_train):
    """Melatih model Random Forest."""
    print("[INFO] Melatih model Random Forest...")
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    print("       Training selesai!")
    return model


def evaluate_model(model, X_test, y_test):
    """Menghitung metrik evaluasi model."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1_score" : f1_score(y_test, y_pred),
        "roc_auc"  : roc_auc_score(y_test, y_pred_prob),
    }
    return metrics, y_pred


def run():
    """Pipeline utama: setup MLflow → train → evaluate → log."""
    print("=" * 60)
    print("  MODELLING - Heart Disease Classification")
    print("  Author: Yusfitasari")
    print("=" * 60)

    # Setup MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # Aktifkan autolog
        mlflow.sklearn.autolog()

        # Load data
        X_train, X_test, y_train, y_test = load_data()

        # Train
        model = train_model(X_train, y_train)

        # Evaluate
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        print("\n[INFO] Hasil Evaluasi:")
        for k, v in metrics.items():
            print(f"       {k:10s}: {v:.4f}")

        print("\n[INFO] Run selesai. Cek MLflow UI dengan:")
        print("       mlflow ui")

    print("\n✅ modelling.py selesai!")


if __name__ == "__main__":
    run()
