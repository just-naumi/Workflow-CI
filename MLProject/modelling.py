import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Gunakan env vars yang sudah diset oleh GitHub Actions workflow
# (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD)
# Tidak perlu dagshub.init() agar tidak ada konflik dengan run context

mlflow.set_experiment("Latihan Credit Scoring")
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
    mlflow.log_param("n_estimators", params["n_estimators"])
    mlflow.log_param("max_depth", params["max_depth"])
    mlflow.log_metric("accuracy", acc)

    # 1. Log ke DagsHub untuk tracking (opsional, tidak diandalkan untuk build-docker)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # 2. Simpan model ke lokal agar bisa digunakan untuk build-docker
    #    tanpa bergantung pada download dari DagsHub
    local_model_path = "saved_model"
    mlflow.sklearn.save_model(model, local_model_path)
    print(f"Model disimpan lokal di: {local_model_path}")

    run_id = run.info.run_id
    print(f"Model logged dengan run_id: {run_id}")

    # Simpan run_id ke file untuk digunakan step berikutnya di CI
    with open('run_id.txt', 'w') as f:
        f.write(run_id)
    print(f"run_id disimpan ke run_id.txt: {run_id}")