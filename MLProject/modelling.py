import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dagshub

# Inisialisasi DagsHub agar koneksi lebih stabil di GitHub Actions
dagshub.init(repo_owner="just-naumi", repo_name="Eksperimen_SML_naumi", mlflow=True)

# Set tracking URI ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/just-naumi/Eksperimen_SML_naumi.mlflow")
mlflow.set_experiment("Latihan Credit Scoring")

# Nonaktifkan autolog agar artifact path bisa dikontrol manual
# (autolog menyimpan ke path "model" juga, tapi kita pastikan eksplisit)
mlflow.sklearn.autolog(log_models=False)

# Load Data Bersih
df = pd.read_csv('namadataset_preprocessing/data_bersih.csv')
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Training
with mlflow.start_run() as run:
    params = {"n_estimators": 100, "max_depth": 5}
    model = RandomForestClassifier(**params)
    model.fit(X, y)

    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    print(f"Model trained with accuracy: {acc}")
    mlflow.log_metric("accuracy", acc)

    # Simpan model dengan artifact_path="model" agar bisa dibaca oleh mlflow build-docker
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model logged with run_id: {run.info.run_id}")