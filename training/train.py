import os
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "race_encoded",
    "gender_encoded", "admission_type_encoded", "discharge_encoded",
    "admission_source_encoded", "a1c_result_encoded",
    "metformin_encoded", "insulin_encoded"
]
TARGET_COLUMN = "readmitted"

# Candidatos a entrenar — se registran todos, gana el mejor por roc_auc
MODEL_CANDIDATES = [
    {
        "model_type": "LogisticRegression",
        "params": {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
        "build": lambda p: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", n_jobs=1, **p))
        ])
    },
    {
        "model_type": "RandomForest",
        "params": {"n_estimators": 50, "max_depth": 10},
        "build": lambda p: RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, **p
        )
    },
    {
        "model_type": "RandomForest",
        "params": {"n_estimators": 100, "max_depth": 15},
        "build": lambda p: RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, **p
        )
    },
]


def run_training(db_conn, mlflow_uri, experiment_name, model_name, batch_id) -> dict:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    engine = create_engine(f"postgresql://mlops_user:mlops_pass@{db_conn}:5432/mlops_db")
    df = pd.read_sql(
        f"SELECT {', '.join(FEATURE_COLUMNS + [TARGET_COLUMN])} FROM clean.diabetes_clean",
        engine
    )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = []

    for candidate in MODEL_CANDIDATES:
        model_type = candidate["model_type"]
        params     = candidate["params"]
        model      = candidate["build"](params)

        with mlflow.start_run(run_name=f"{model_type}_{batch_id}") as run:
            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            proba = model.predict_proba(X_val)[:, 1]

            metrics = {
                "roc_auc": roc_auc_score(y_val, proba),
                "f1":      f1_score(y_val, preds),
                "recall":  recall_score(y_val, preds),
            }

            log_params = {"model_type": model_type, "batch_id": batch_id}
            log_params.update(params)
            mlflow.log_params(log_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(model_name)
            version = versions[0].version if versions else "1"

            results.append({
                "run_id":        run.info.run_id,
                "model_version": version,
                "model_type":    model_type,
                "params":        params,
                "metrics":       metrics,
            })

    # Seleccionar el mejor candidato por roc_auc
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])

    import logging
    logging.getLogger(__name__).info(
        f"Modelos entrenados: {len(results)} | "
        f"Ganador: {best['model_type']} {best['params']} | "
        f"ROC-AUC: {best['metrics']['roc_auc']:.4f}"
    )

    return {
        "run_id":        best["run_id"],
        "model_version": best["model_version"],
        "metrics":       best["metrics"],
        "promoted":      False,
    }
