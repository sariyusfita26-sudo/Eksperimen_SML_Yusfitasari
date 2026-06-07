"""
automate_Yusfitasari.py
=======================
Script otomatisasi preprocessing dataset Heart Disease.
Script ini mengkonversi seluruh tahapan eksperimen di notebook
menjadi fungsi-fungsi yang dapat dijalankan secara otomatis.

Author  : Yusfitasari
Dataset : Heart Disease Dataset
Task    : Binary Classification (0=Sehat, 1=Penyakit Jantung)
"""

import pandas as pd
import numpy as np
import os
import argparse
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# ==============================================================
# KONFIGURASI
# ==============================================================
NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_MULTI = ['cp', 'restecg', 'slope', 'ca', 'thal']
TARGET_COL = 'target'
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = 'heart_preprocessing'


# ==============================================================
# FUNGSI-FUNGSI PREPROCESSING
# ==============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Parameters
    ----------
    filepath : str
        Path ke file CSV dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame berisi dataset yang dimuat.
    """
    print(f'[1/7] Memuat dataset dari: {filepath}')
    df = pd.read_csv(filepath)
    print(f'      Shape: {df.shape}')
    print(f'      Kolom: {list(df.columns)}')
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa duplikat.
    """
    print(f'[2/7] Menghapus data duplikat...')
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    print(f'      Dihapus : {before - after} baris duplikat')
    print(f'      Sisa    : {after} baris')
    return df_clean


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values dengan imputasi median (numerik)
    dan modus (kategorikal).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa missing values.
    """
    print(f'[3/7] Menangani missing values...')
    df_clean = df.copy()
    total_missing = df_clean.isnull().sum().sum()

    if total_missing == 0:
        print('      Tidak ada missing values ditemukan.')
        return df_clean

    # Imputasi median untuk numerik
    for col in NUMERICAL_COLS:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f'      {col}: diisi median ({median_val:.2f})')

    # Imputasi modus untuk kategorikal
    for col in CATEGORICAL_MULTI:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f'      {col}: diisi modus ({mode_val})')

    print(f'      Missing values tersisa: {df_clean.isnull().sum().sum()}')
    return df_clean


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR Capping
    (nilai ekstrem di-clip ke batas bawah/atas IQR).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan outlier sudah ditangani.
    """
    print(f'[4/7] Menangani outlier (IQR Capping)...')
    df_clean = df.copy()

    for col in NUMERICAL_COLS:
        if col not in df_clean.columns:
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)].shape[0]
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        print(f'      {col}: {outlier_count} outlier di-cap ke [{lower:.2f}, {upper:.2f}]')

    return df_clean


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan One-Hot Encoding pada fitur kategorikal
    dengan lebih dari 2 nilai unik.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan fitur kategorikal sudah di-encode.
    """
    print(f'[5/7] Encoding fitur kategorikal (One-Hot)...')
    cols_to_encode = [c for c in CATEGORICAL_MULTI if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
    print(f'      Shape sebelum encoding : {df.shape}')
    print(f'      Shape setelah encoding  : {df_encoded.shape}')
    return df_encoded


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan normalisasi fitur numerik menggunakan StandardScaler
    (mean=0, std=1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan fitur numerik sudah dinormalisasi.
    """
    print(f'[6/7] Normalisasi fitur numerik (StandardScaler)...')
    df_scaled = df.copy()
    scaler = StandardScaler()
    cols_to_scale = [c for c in NUMERICAL_COLS if c in df_scaled.columns]
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    print(f'      Fitur yang dinormalisasi: {cols_to_scale}')
    return df_scaled


def split_and_save(df: pd.DataFrame, output_dir: str) -> dict:
    """
    Membagi dataset menjadi train dan test set, lalu menyimpan
    hasilnya ke folder output.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang sudah dipreprocess.
    output_dir : str
        Path folder untuk menyimpan output.

    Returns
    -------
    dict
        Dictionary berisi path file output.
    """
    print(f'[7/7] Split data dan simpan ke {output_dir}/')
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Simpan file
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    paths = {
        'train': os.path.join(output_dir, 'heart_train.csv'),
        'test': os.path.join(output_dir, 'heart_test.csv'),
        'full': os.path.join(output_dir, 'heart_preprocessed.csv'),
    }

    train_df.to_csv(paths['train'], index=False)
    test_df.to_csv(paths['test'], index=False)
    df.to_csv(paths['full'], index=False)

    print(f'      Train : {len(train_df)} baris → {paths["train"]}')
    print(f'      Test  : {len(test_df)} baris → {paths["test"]}')
    print(f'      Full  : {len(df)} baris → {paths["full"]}')

    return paths


# ==============================================================
# PIPELINE UTAMA
# ==============================================================

def run_preprocessing(input_path: str, output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara otomatis.

    Parameters
    ----------
    input_path : str
        Path ke file CSV dataset raw.
    output_dir : str
        Folder output untuk dataset hasil preprocessing.

    Returns
    -------
    pd.DataFrame
        Dataset final yang sudah siap untuk pelatihan model.
    """
    print('=' * 60)
    print('  PIPELINE PREPROCESSING - Heart Disease Dataset')
    print('  Author: Yusfitasari')
    print('=' * 60)

    # Jalankan setiap langkah secara berurutan
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = encode_categorical(df)
    df = normalize_features(df)
    paths = split_and_save(df, output_dir)

    print('\n' + '=' * 60)
    print('  ✅ Preprocessing selesai!')
    print(f'  Output disimpan di folder: {output_dir}/')
    print('=' * 60)

    return df


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automate preprocessing untuk Heart Disease Dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='heart_raw.csv',
        help='Path ke file CSV dataset raw (default: heart_raw.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_DIR,
        help=f'Folder output hasil preprocessing (default: {OUTPUT_DIR})'
    )

    args = parser.parse_args()
    result_df = run_preprocessing(
        input_path=args.input,
        output_dir=args.output
    )
    print(f'\nShape dataset final: {result_df.shape}')
