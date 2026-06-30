import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Setup MLflow Lokal
# Kriteria Basic: Disimpan secara lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Latihan Credit Scoring")

# 2. AKTIFKAN AUTOLOG
# INI WAJIB ADA untuk memenuhi syarat Basic
mlflow.sklearn.autolog()

# 3. Load Data Bersih
# Pastikan path ini sesuai dengan folder di dalam folder Membangun_model
df = pd.read_csv('namadataset_preprocessing/data_bersih.csv')
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# 4. Training Sederhana
with mlflow.start_run():
    # Tanpa hyperparameter tuning, pakai default saja
    model = RandomForestClassifier()
    
    # MLflow akan otomatis mencatat parameter dan metrik lewat autolog()
    model.fit(X, y)
    
    # Print sekilas saja
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    print(f"Model trained with accuracy: {acc}")