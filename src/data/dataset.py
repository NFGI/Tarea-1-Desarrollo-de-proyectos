# src/data/dataset.py

import pandas as pd

def cargar_parquet(url):
    """
    Carga un archivo Parquet desde una URL o ruta local.
    """
    df = pd.read_parquet(url)
    return df
