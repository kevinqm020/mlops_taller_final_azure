import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pycaret.classification import *

# Cargar de datos
df = pd.read_csv("mlops_taller_final_azure\data_prestamos.csv")

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Servidor MLflow local
mlflow.set_experiment("mlops_example")

exp = setup(df, target="loan_status", log_experiment=False, session_id=123, data_split_shuffle=True)

# Iniciar manualmente un run en MLflow
with mlflow.start_run(run_name="prestamo_experiment"):
    try:
        print("Comparando modelos")
        best_model = compare_models(n_select=1)
        tuned_model = tune_model(best_model)
        final_model = finalize_model(tuned_model)

        print("Guardando modelo")
        model_path = "prestamo_modelo"
        save_model(final_model, model_path)

        print("Registrando modelo en MLflow")
        mlflow.sklearn.log_model(final_model, "best_model")

        print("Registrando parámetros en MLflow")
        mlflow.log_param("session_id", 123)
        mlflow.log_param("data_split_shuffle", True)

        print("Entrenamiento finalizado con éxito")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")