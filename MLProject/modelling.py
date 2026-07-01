import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Tracking URI dan credentials dibaca otomatis dari env vars:
# MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
# yang sudah diset oleh GitHub Actions workflow.
# Ketika dipanggil via 'mlflow run .', MLFLOW_RUN_ID juga diset otomatis
# sehingga mlflow.start_run() akan JOIN ke run yang ada (bukan buat nested run baru).

mlflow.set_experiment("Latihan Credit Scoring")
mlflow.sklearn.autolog(log_models=False)

# Load Data Bersih
df = pd.read_csv('namadataset_preprocessing/data_bersih.csv')
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Training — jika dipanggil via 'mlflow run .', start_run() join ke run MLflow project
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

    # Log model ke DagsHub/MLflow untuk tracking
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Simpan model ke lokal — digunakan untuk build-docker
    # agar tidak bergantung pada download artifact dari DagsHub
    local_model_path = "saved_model"
    mlflow.sklearn.save_model(model, local_model_path)
    print(f"Model disimpan lokal di: {local_model_path}")

    run_id = run.info.run_id
    print(f"Model logged dengan run_id: {run_id}")

    # Tulis run_id ke file — dibaca oleh step CI berikutnya
    with open('run_id.txt', 'w') as f:
        f.write(run_id)
    print(f"run_id disimpan ke run_id.txt: {run_id}")