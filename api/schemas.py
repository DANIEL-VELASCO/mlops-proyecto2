from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    race_encoded: int
    gender_encoded: int
    admission_type_encoded: int
    discharge_encoded: int
    admission_source_encoded: int
    a1c_result_encoded: int
    metformin_encoded: int
    insulin_encoded: int

class PredictResponse(BaseModel):
    prediction: int           # 0 o 1
    probability: float        # score del modelo
    model_name: str           # MLFLOW_MODEL_NAME
    model_version: str        # versión en MLflow
    model_alias: str          # "champion"
    response_time_ms: float
    request_id: str