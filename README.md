# Tarea-1-Desarrollo-de-proyectos

# NYC Taxi Modelo de Predicción de Propinas

Este repositorio contiene el código para entrenar, evaluar y analizar un modelo de predicción de propinas (high\_tip) en viajes de taxi en la ciudad de Nueva York.

## Estructura del proyecto

```
TAREA-1-DESARROLLO-DE-PROYECTOS/
├── data/                           # Carpeta para datos raw y procesados
│   ├── raw/                        # Parquets descargados
│   └── processed/                  # Parquets limpios y con features
├── models/                         # Modelos serializados (.pkl)
├── notebooks/                      # Jupyter notebooks de exploración y análisis
│   └── 00_nyc_taxi_model.ipynb     # Notebook principal
├── src/                            # Código fuente modular
│   ├── __init__.py                 # Inicializador de paquete
│   ├── config.py                   # Variables y rutas globales
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py              # Carga y limpieza de datos
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py       # Definición de features
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── train.py                # Entrenamiento y serialización
│   │   ├── predict.py              # Carga modelo y evaluación
│   │   └── evaluate_time_series.py # Evaluación mes a mes
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                # Funciones de graficado
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

## Requisitos previos

* Python 3.8+ instalado en el sistema.
* Git (opcional, para clonar el repositorio).

## Instalación y configuración

1. Clona este repositorio:

   ```bash
   git clone https://github.com/TU_USUARIO/nyc-taxi-model.git
   cd nyc-taxi-model
   ```
2. Crea y activa un entorno virtual:

   ```bash
   python -m venv .venv
   # PowerShell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   # o CMD
   .\.venv\Scripts\activate.bat
   ```
3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

### 1. Entrenar el modelo

Entrena el RandomForest en datos de enero 2020 y guarda el modelo:

```bash
python src/modeling/train.py https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet
```

* Salida: `models/rf_taxi_model.pkl` y reporte de clasificación en consola.

### 2. Evaluación mes a mes

Desde el notebook ejecuta:

```python
# Dentro de notebooks/00_nyc_taxi_model.ipynb
import sys, os
os.chdir('..')
sys.path.insert(0, 'src')
from modeling.evaluate_time_series import evaluate_months
meses = ["06","07","08"]
df_res = evaluate_months(meses)
display(df_res)
```




*Autor: Nicolás González*
