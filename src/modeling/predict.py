import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.metrics import f1_score
from features.build_features import features

TARGET_COL = "high_tip"

def load_model(model_path="models/rf_taxi_model.pkl"):
    """
    Carga el modelo serializado desde disco.
    """
    return joblib.load(model_path)

def evaluate(model, df):
    """
    Calcula y muestra el F1-score en df (debe incluir TARGET_COL y features).
    """
    y_true = df[TARGET_COL]
    y_pred = model.predict(df[features])
    f1 = f1_score(y_true, y_pred)
    print(f"🔍 F1-score: {f1:.4f}")
    return f1