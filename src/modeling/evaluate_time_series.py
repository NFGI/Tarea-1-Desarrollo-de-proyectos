import os
import sys

# Permite importar desde src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.metrics import f1_score

from data.dataset import cargar_parquet, preprocess
from features.build_features import features
from modeling.predict import load_model


def evaluate_months(months,
                    model_path="models/rf_taxi_model.pkl",
                    target_col="high_tip"):
    """
    Para cada mes en 'months' (formato "MM", ej. ["06","07","08"]):
      1. Descarga y preprocesa el Parquet de ese mes.
      2. Genera X, y.
      3. Calcula F1-score.
    Devuelve un DataFrame con columnas: mes, n_ejemplos, f1_score.
    """
    # Carga el modelo una sola vez
    model = load_model(model_path)
    records = []

    for m in months:
        # Construye la URL del Parquet
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-{m}.parquet"
        
        # 1) Carga y limpieza
        df = cargar_parquet(url)
        df = preprocess(df, target_col)

        # 2) Separar X e y
        X = df[features]
        y = df[target_col]

        # 3) Calcular F1-score
        f1 = f1_score(y, model.predict(X))

        records.append({
            "mes": f"2020-{m}",
            "n_ejemplos": len(df),
            "f1_score": f1
        })

    return pd.DataFrame(records)