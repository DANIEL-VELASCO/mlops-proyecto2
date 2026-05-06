import os
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Contrato estricto de columnas (Sección 3.3)
FEATURE_COLUMNS = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "race_encoded",
    "gender_encoded", "admission_type_encoded", "discharge_encoded",
    "admission_source_encoded", "a1c_result_encoded",
    "metformin_encoded", "insulin_encoded"
]
TARGET_COLUMN = "readmitted"

def run_training(db_conn, mlflow_uri, experiment_name, model_name, batch_id) -> dict:
    # 1. Configurar MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # 2. Conectar a PostgreSQL y extraer datos limpios
    engine = create_engine(f"postgresql://mlops_user:mlops_pass@{db_conn}:5432/mlops_db")
    df = pd.read_sql(
        f"SELECT {', '.join(FEATURE_COLUMNS + [TARGET_COLUMN])} FROM clean.diabetes_clean",
        engine
    )

    X = df[FEATURE_COLUMNS].astype("float32")
    y = df[TARGET_COLUMN]

    # 3. Split de datos
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Iniciar el run de MLflow y Entrenar
    with mlflow.start_run() as run:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=200,
                random_state=42,
                solver="lbfgs",
                n_jobs=1,
            ))
        ])
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_val, proba),
            "f1": f1_score(y_val, preds),
            "recall": recall_score(y_val, preds),
        }

        mlflow.log_params({"model_type": "LogisticRegression", "batch_id": batch_id, "max_iter": 200})
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions(model_name)
        version = latest_versions[0].version if latest_versions else "1"

        return {
            "run_id": run.info.run_id,
            "model_version": version,
            "metrics": metrics,
            "promoted": False
        }
