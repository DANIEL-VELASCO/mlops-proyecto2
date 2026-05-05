import streamlit as st
import requests
import os

# La API estará en http://fastapi-service:8000 dentro de Kubernetes
API_URL = os.getenv("API_URL", "http://fastapi-service:8000/predict")

st.set_page_config(page_title="Diabetes Inferencia API", page_icon="🏥", layout="centered")

st.title("🏥 Predicción de Readmisión (Diabetes)")
st.write("Ingrese los datos del paciente para predecir si será readmitido en el hospital. Los valores por defecto son un ejemplo base.")

with st.form("predict_form"):
    st.subheader("Datos Clínicos y Demográficos")
    
    col1, col2, col3 = st.columns(3)
    
    # Se usan los valores de ejemplo exactos definidos en el contrato de Locust (Sección 4.3)
    with col1:
        age = st.number_input("Edad (age)", value=3)
        time_in_hospital = st.number_input("Días en hospital", value=5)
        num_lab_procedures = st.number_input("Pruebas de lab", value=40)
        num_procedures = st.number_input("Procedimientos", value=1)
        num_medications = st.number_input("Medicamentos", value=10)
        number_outpatient = st.number_input("Consultas ext.", value=0)
    
    with col2:
        number_emergency = st.number_input("Emergencias", value=0)
        number_inpatient = st.number_input("Hosp. previas", value=1)
        number_diagnoses = st.number_input("Diagnósticos", value=7)
        race_encoded = st.number_input("Raza (encoded)", value=1)
        gender_encoded = st.number_input("Género (encoded)", value=0)
        admission_type_encoded = st.number_input("Tipo Admisión", value=1)
        
    with col3:
        discharge_encoded = st.number_input("Tipo Alta", value=1)
        admission_source_encoded = st.number_input("Origen Admisión", value=7)
        a1c_result_encoded = st.number_input("Resultado A1C", value=0)
        metformin_encoded = st.number_input("Metformina", value=1)
        insulin_encoded = st.number_input("Insulina", value=2)
        
    submitted = st.form_submit_button("Realizar Predicción", type="primary")

if submitted:
    payload = {
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "race_encoded": race_encoded,
        "gender_encoded": gender_encoded,
        "admission_type_encoded": admission_type_encoded,
        "discharge_encoded": discharge_encoded,
        "admission_source_encoded": admission_source_encoded,
        "a1c_result_encoded": a1c_result_encoded,
        "metformin_encoded": metformin_encoded,
        "insulin_encoded": insulin_encoded
    }
    
    try:
        with st.spinner('Consultando modelo en MLflow a través de la API...'):
            response = requests.post(API_URL, json=payload, timeout=10)
            
        if response.status_code == 200:
            data = response.json()
            st.success("✅ ¡Inferencia exitosa!")
            
            # Formato condicional del resultado
            if data["prediction"] == 1:
                st.error("### 🚨 Resultado: PACIENTE READMITIDO (1)")
            else:
                st.info("### ✅ Resultado: NO READMITIDO (0)")
                
            st.write(f"**Probabilidad estimada:** `{data['probability']:.2%}`")
            
            # Criterio obligatorio: Mostrar versión del modelo
            st.divider()
            st.caption(f"🤖 **Modelo en uso:** `{data['model_name']}` | 📌 **Versión:** `{data['model_version']}` | 🏷️ **Alias:** `{data['model_alias']}`")
            st.caption(f"⏱️ **Latencia API:** `{data['response_time_ms']:.2f} ms` | 🆔 **Req ID:** `{data['request_id']}`")
            
        else:
            st.error(f"❌ Error en la API (Status {response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ No se pudo conectar con la API en `{API_URL}`. Revisa que el pod de FastAPI esté corriendo. Detalles: {e}")