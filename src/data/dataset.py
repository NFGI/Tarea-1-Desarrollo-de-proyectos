import pandas as pd

def cargar_parquet(url):
    df = pd.read_parquet(url)
    return df
