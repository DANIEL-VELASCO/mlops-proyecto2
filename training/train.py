import os
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

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
    df = pd.read_sql("SELECT * FROM clean.diabetes_clean", engine)
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 3. Split de datos
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Iniciar el run de MLflow y Entrenar
    with mlflow.start_run() as run:
        # Usamos RandomForest con class_weight="balanced" por el desbalanceo clínico
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        model.fit(X_tr, y_tr)
        
        # Predicciones
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]
        
        # Métricas
        metrics = {
            "roc_auc": roc_auc_score(y_val, proba),
            "f1": f1_score(y_val, preds),
            "recall": recall_score(y_val, preds),
        }
        
        # Registrar en MLflow
        mlflow.log_params({"n_estimators": 100, "batch_id": batch_id, "model_type": "RandomForest"})
        mlflow.log_metrics(metrics)
        
        # Registrar modelo en el Model Registry
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        
        # Obtener la versión recién registrada
        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions(model_name)
        version = latest_versions[0].version if latest_versions else "1"
        
        return {
            "run_id": run.info.run_id,
            "model_version": version,
            "metrics": metrics,
            "promoted": False
        }