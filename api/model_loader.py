import os
import mlflow.pyfunc

MODEL_NAME  = os.getenv("MLFLOW_MODEL_NAME", "diabetes_model")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

_model = None
_model_info = {}

def load_model():
    global _model, _model_info
    
    # 1. Configurar URI de tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # 2. Cargar el modelo usando el alias (ej: models:/diabetes_model@champion)
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    _model = mlflow.pyfunc.load_model(uri)
    
    # 3. Obtener metadatos de esa versión específica
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    
    _model_info = {
        "model_name": MODEL_NAME,
        "model_version": mv.version,
        "model_alias": MODEL_ALIAS,
    }

def get_model():
    if _model is None:
        load_model()
    return _model

def get_model_info() -> dict:
    return _model_info