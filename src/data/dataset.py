import pandas as pd
from features.build_features import features

EPS = 1e-7  # Para evitar división por cero

def cargar_parquet(path_or_url):
    """
    Carga un archivo Parquet desde una URL o ruta local.
    Requiere pyarrow y fsspec para HTTP/S.
    """
    df = pd.read_parquet(path_or_url, engine='pyarrow')
    return df

def preprocess(df, target_col):
    """
    - Filtra tarifas positivas.
    - Calcula tip_fraction y crea la variable binaria target_col (high_tip).
    - Agrega features de tiempo y velocidad.
    - Selecciona sólo las columnas de features + target.
    - Rellena NaNs y define tipos optimizados.
    """
    # Basic cleaning
    df = df[df['fare_amount'] > 0].reset_index(drop=True)

    # add target
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[target_col] = df['tip_fraction'] > 0.2

    # add features
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour']    = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute']  = df['tpep_pickup_datetime'].dt.minute
    df['work_hours']     = (
        (df['pickup_weekday'] >= 0) &
        (df['pickup_weekday'] <= 4) &
        (df['pickup_hour'] >= 8) &
        (df['pickup_hour'] <= 18)
    )
    df['trip_time']      = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.seconds
    df['trip_speed']     = df['trip_distance'] / (df['trip_time'] + EPS)

    # seleccionamos sólo features + target
    df = df[['tpep_dropoff_datetime'] + features + [target_col]]
    df[features + [target_col]] = df[features + [target_col]] \
        .astype("float32") \
        .fillna(-1.0)
    df[target_col] = df[target_col].astype("int32")

    return df.reset_index(drop=True)