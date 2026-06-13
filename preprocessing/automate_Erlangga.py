import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing():
    print("Memulai proses otomatisasi data preprocessing...")
    
    # 1. Memuat Dataset
    df = sns.load_dataset('titanic')
    print(f"Data awal termuat dengan ukuran: {df.shape}")
    
    # 2. Membersihkan Data (Drop & Fill)
    df_clean = df.drop(columns=['deck', 'alive', 'class', 'who', 'adult_male', 'embark_town'])
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])
    
    # 3. Encoding Kategorikal
    label_cols = ['sex', 'embarked', 'alone']
    le = LabelEncoder()
    for col in label_cols:
        df_clean[col] = le.fit_transform(df_clean[col])
        
    # 4. Memisahkan Fitur (X) dan Target (y)
    X = df_clean.drop(columns=['survived'])
    y = df_clean['survived']
    
    # 5. Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_final = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 6. Menggabungkan kembali menjadi dataset siap latih dan menyimpannya
    dataset_ready = pd.concat([X_final, y], axis=1)
    dataset_ready.to_csv("titanic_clean.csv", index=False)
    
    print("Preprocessing selesai!")
    print(f"Data bersih disimpan sebagai 'titanic_clean.csv' dengan ukuran: {dataset_ready.shape}")

# Eksekusi fungsi utama jika file ini dijalankan
if __name__ == "__main__":
    run_preprocessing()