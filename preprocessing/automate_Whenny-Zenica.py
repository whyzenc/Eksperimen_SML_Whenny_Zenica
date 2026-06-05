#!/usr/bin/env python3
"""
Skrip Otomatisasi Preprocessing - Proyek Akhir
Nama: Whenny Zenica
Dataset: Melbourne Housing Snapshot (melb_data.csv)
"""

print("="*50)
print("MEMULAI OTOMATISASI PREPROCESSING")
print("="*50)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Memuat file dataset mentah dengan proteksi FileNotFoundError."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Kritis: File tidak ditemukan di: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset berhasil dimuat. Dimensi awal: {df.shape[0]} baris x {df.shape[1]} kolom.")
    return df

def pipeline_preprocessing(df):
    """Pipeline pemrosesan data: Drop fitur, Imputasi, Label Encoding, Winsorization, dan Scaling."""
    df_clean = df.copy()
    
    # 1. Menghapus Kolom Tidak Relevan
    unnecessary_cols = ['Address', 'SellerG', 'Date']
    df_clean = df_clean.drop(columns=unnecessary_cols, errors='ignore')
    print("-> Kolom tidak relevan berhasil dihapus.")
    
    # 2. Pemisahan Fitur (X) dan Target (y)
    X = df_clean.drop(columns=['Price'])
    y = df_clean['Price']
    
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # 3. Penanganan Missing Values Secara Otomatis
    if num_cols:
        imputer_num = SimpleImputer(strategy='median')
        X[num_cols] = imputer_num.fit_transform(X[num_cols])
    if cat_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
    print("-> Imputasi nilai kosong (missing values) selesai.")
    
    # 4. Encoding Data Kategorikal (Label Encoding)
    if cat_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in cat_cols:
            X[col] = le.fit_transform(X[col].astype(str))
        print("-> Encoding data kategorikal selesai.")
        
    # 5. Penanganan Outlier menggunakan Metode Winsorization (IQR)
    outlier_cols = ['Rooms', 'Distance', 'Landsize', 'BuildingArea']
    for col in outlier_cols:
        if col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
            X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
    print("-> Penanganan pencilan (outliers) dengan Winsorization selesai.")
    
    # 6. Normalisasi/Standarisasi Fitur Numerik
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    print("-> Standarisasi seluruh skala fitur numerik selesai.")
    
    # 7. Gabungkan Kembali Fitur Berpola dengan Target Asli
    df_preprocessed = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    return df_preprocessed

def main():
    # Menentukan jalur input dan output yang aman untuk eksekusi lokal dari dalam folder preprocessing
    input_path = os.path.join("..", "melb_data.csv")
    output_dir = os.path.join("..", "melb_data_preprocessing")
    output_file = os.path.join(output_dir, "melb_preprocessed.csv")
    
    # Membuat folder output utama di luar jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Menjalankan alur pipeline
    df_raw = load_data(input_path)
    print("\nMenjalankan tahapan preprocessing data...")
    df_final = pipeline_preprocessing(df_raw)
    
    # Menyimpan ke file fisik CSV
    df_final.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print("PREPROCESSING SUKSES DIJALANKAN SECARA OTOMATIS!")
    print(f"Hasil akhir disimpan di: {output_file}")
    print(f"Shape akhir dataset: {df_final.shape[0]} baris x {df_final.shape[1]} kolom.")
    print("="*50)

if __name__ == "__main__":
    main()