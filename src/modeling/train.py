import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import cargar_parquet, preprocess
from features.build_features import features
import joblib
from sklearn.ensemble import RandomForestClassifier

TARGET_COL = "high_tip"

def train(path_or_url, model_path="models/rf_taxi_model.pkl"):
    """
    1. Carga y preprocesa el DataFrame.
    2. Entrena un RandomForestClassifier (n_estimators=100, max_depth=10).
    3. Serializa el modelo en model_path.
    """
    # 1. Cargar y procesar
    df = cargar_parquet(path_or_url)
    df = preprocess(df, target_col=TARGET_COL)

    # 2. Entrenar
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(df[features], df[TARGET_COL])

    # 3. Guardar
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Modelo entrenado y guardado en {model_path}")
    return clf

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python train.py <ruta_o_url_parquet>")
        sys.exit(1)
    train(sys.argv[1])