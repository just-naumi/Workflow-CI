import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dagshub # Tambahkan ini supaya DagsHub-nya terinisialisasi dengan benar

# Inisialisasi DagsHub agar koneksi lebih stabil di GitHub Actions
dagshub.init(repo_owner="just-naumi", repo_name="Eksperimen_SML_naumi", mlflow=True)

# Ganti ke URI DagsHub
mlflow.set_tracking_uri("https://dagshub.com/just-naumi/Eksperimen_SML_naumi.mlflow")
mlflow.set_experiment("Latihan Credit Scoring")

# AKTIFKAN AUTOLOG
mlflow.sklearn.autolog()

# Load Data Bersih
df = pd.read_csv('namadataset_preprocessing/data_bersih.csv')
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Training Sederhana
with mlflow.start_run() as run:
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Simpan dengan nama 'model' agar bisa dibaca oleh build-docker
    mlflow.sklearn.log_model(model, artifact_path="model")
    
    print(f"Model logged with run_id: {run.info.run_id}")