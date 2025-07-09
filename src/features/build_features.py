"""
Define las listas de columnas usadas como features.
"""
numeric_feat = [
    "pickup_weekday",
    "pickup_hour",
    "work_hours",
    "pickup_minute",
    "passenger_count",
    "trip_distance",
    "trip_time",
    "trip_speed"
]
categorical_feat = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID"
]

# Lista definitiva de features
features = numeric_feat + categorical_feat