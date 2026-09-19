# Predicción de ventas mediante regresión lineal

**Curso:** CADI - Introducción a Machine Learning

**Autor:** Ferley Antonio Moreno Cruz
**Institución:** Universidad de Cundinamarca

## Objetivo

Este proyecto desarrolla y evalúa un modelo de regresión lineal para estimar las ventas de un producto a partir de la inversión publicitaria en televisión, radio y periódicos. El análisis documenta todo el flujo de trabajo: descarga de datos, controles de calidad, análisis exploratorio, entrenamiento, evaluación y validación de supuestos.

## Fuente y descarga de los datos

Los datos proceden de un recurso abierto en formato CSV: [Advertising.csv](https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv). El script realiza una solicitud HTTP con `requests`, por lo que el proceso corresponde a una descarga reproducible desde la web. No se presenta como una API en sentido estricto.

La ejecución conserva una copia del archivo descargado en `data/raw/advertising.csv` y genera la versión procesada en `data/processed/advertising_limpio.csv`. Esto permite verificar el origen, la transformación y los resultados del análisis.

El conjunto contiene 200 observaciones y las siguientes variables:

| Variable | Tipo | Descripción |
|---|---|---|
| `Sales` | Numérica continua | Ventas del producto, en miles de unidades. |
| `TV` | Numérica continua | Inversión publicitaria en televisión, en miles de dólares. |
| `Radio` | Numérica continua | Inversión publicitaria en radio, en miles de dólares. |
| `Newspaper` | Numérica continua | Inversión publicitaria en periódicos, en miles de dólares. |

## Estructura del repositorio

```text
.
├── data/
│   ├── raw/                 # Archivo descargado sin modificaciones
│   └── processed/           # Datos validados y preparados
├── figures/                 # Gráficos para el informe
├── outputs/                 # Tablas, métricas y diagnósticos exportados
├── src/
│   ├── analisis_regresion.py
│   └── TallerRegresiónLineal.ipynb
├── requirements.txt
└── README.md
```

El archivo `src/analisis_regresion.py` es el flujo reproducible principal. El notebook se conserva como apoyo exploratorio y visual.

## Requisitos e instalación

Se requiere Python 3.10 o superior.

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instale las dependencias y ejecute el análisis:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/analisis_regresion.py
```

## Flujo de análisis

El script ejecuta los siguientes pasos:

1. Descarga el CSV desde la fuente abierta y conserva una copia bruta.
2. Verifica columnas esperadas y convierte las variables a formato numérico.
3. Identifica valores nulos, elimina registros duplicados y reporta valores atípicos mediante el criterio IQR.
4. Genera estadísticos descriptivos, correlaciones y un gráfico de dispersión entre televisión y ventas.
5. Ajusta una regresión lineal múltiple con división 70/30 y `random_state=42`.
6. Calcula MAE, MSE, RMSE y R² en el conjunto de prueba.
7. Evalúa normalidad de residuos, homocedasticidad, linealidad y multicolinealidad.
8. Compara el modelo base con un modelo reducido y una alternativa lineal que incorpora la interacción entre televisión y radio.

## Resultados principales

El modelo base utiliza `TV`, `Radio` y `Newspaper` como predictores. Con la partición 70/30 y semilla 42, obtuvo los siguientes resultados en prueba:

| Métrica | Modelo base |
|---|---:|
| MAE | 1,5117 |
| MSE | 3,7968 |
| RMSE | 1,9485 |
| R² | 0,8609 |

La comparación exploratoria muestra que incluir la interacción entre televisión y radio mejora el desempeño predictivo. Esta alternativa sigue siendo un modelo lineal respecto de sus coeficientes, aunque incluye una variable derivada.

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Modelo base | 1,5117 | 1,9485 | 0,8609 |
| Modelo reducido TV y Radio | 1,4759 | 1,9155 | 0,8656 |
| Modelo con interacción TV y Radio | 0,7460 | 0,9616 | 0,9661 |

## Controles de calidad y validación

En la fuente descargada se identificaron cero valores nulos y cero registros duplicados. El análisis detectó dos posibles valores atípicos en `Newspaper`; se conservan porque no existe evidencia de que sean errores de medición.

Para el modelo base, los diagnósticos deben interpretarse de la siguiente manera:

| Prueba o indicador | Resultado | Lectura |
|---|---:|---|
| Shapiro-Wilk de residuos | p < 0,001 | Se cuestiona la normalidad de los residuos. |
| Breusch-Pagan | p = 0,1284 | No hay evidencia suficiente de heterocedasticidad. |
| RESET de Ramsey | p < 0,001 | La especificación lineal simple puede ser insuficiente. |
| VIF de predictores | Entre 1,00 y 1,14 | No hay multicolinealidad problemática. |

Por ello, el modelo base es útil como referencia para la actividad, pero no se deben presentar sus supuestos como plenamente cumplidos. El informe debe explicar estas limitaciones y presentar la comparación de modelos con prudencia.

## Archivos generados

Al ejecutar el script se crean los siguientes insumos para el informe:

- `figures/eda_tv_ventas.png`
- `figures/validacion_modelo.png`
- `outputs/calidad_datos.json`
- `outputs/estadisticos_descriptivos.csv`
- `outputs/matriz_correlacion.csv`
- `outputs/coeficientes_modelo.csv`
- `outputs/comparacion_modelos.csv`
- `outputs/vif.csv`
- `outputs/resultados_modelo.json`

## Limitaciones y trabajo futuro

Los datos tienen un tamaño reducido y corresponden a un conjunto de acceso abierto, por lo que los resultados no deben extrapolarse automáticamente a un contexto comercial real. Como trabajo futuro se recomienda revisar modelos alternativos, validar con particiones adicionales o validación cruzada y justificar cualquier transformación de variables a partir de los diagnósticos obtenidos.
