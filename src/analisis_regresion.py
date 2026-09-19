"""Descarga, limpieza, análisis y validación de una regresión lineal.

El script conserva el archivo bruto, genera un archivo procesado, exporta las
tablas y figuras necesarias para el informe y deja resultados reproducibles.
"""

from __future__ import annotations

import io
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor


URL_DATOS = (
    "https://raw.githubusercontent.com/justmarkham/"
    "scikit-learn-videos/master/data/Advertising.csv"
)
PREDICTORES = ["TV", "Radio", "Newspaper"]
OBJETIVO = "Sales"
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_BRUTA = RAIZ_PROYECTO / "data" / "raw" / "advertising.csv"
RUTA_PROCESADA = RAIZ_PROYECTO / "data" / "processed" / "advertising_limpio.csv"
CARPETA_FIGURAS = RAIZ_PROYECTO / "figures"
CARPETA_SALIDAS = RAIZ_PROYECTO / "outputs"


def preparar_carpetas() -> None:
    """Crea la estructura de salida sin depender de rutas locales absolutas."""
    for carpeta in [RUTA_BRUTA.parent, RUTA_PROCESADA.parent, CARPETA_FIGURAS, CARPETA_SALIDAS]:
        carpeta.mkdir(parents=True, exist_ok=True)


def descargar_datos() -> tuple[pd.DataFrame, dict[str, str]]:
    """Descarga el CSV público y conserva una copia bruta para trazabilidad."""
    respuesta = requests.get(URL_DATOS, timeout=30)
    respuesta.raise_for_status()
    RUTA_BRUTA.write_bytes(respuesta.content)
    trazabilidad = {
        "fuente_url": URL_DATOS,
        "metodo_descarga": "HTTP GET mediante requests",
        "formato": "CSV",
        "fecha_descarga_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_archivo_bruto": hashlib.sha256(respuesta.content).hexdigest(),
        "archivo_bruto": str(RUTA_BRUTA.relative_to(RAIZ_PROYECTO)),
    }
    return pd.read_csv(io.BytesIO(respuesta.content), index_col=0), trazabilidad


def validar_y_limpiar(datos: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Valida tipos, trata nulos, elimina duplicados y reporta posibles atípicos.

    Los atípicos se reportan, pero no se eliminan automáticamente porque pueden
    corresponder a observaciones reales y relevantes para el análisis.
    """
    columnas_esperadas = PREDICTORES + [OBJETIVO]
    columnas_faltantes = set(columnas_esperadas) - set(datos.columns)
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(columnas_faltantes)}")

    datos_limpios = datos[columnas_esperadas].copy()
    for columna in columnas_esperadas:
        datos_limpios[columna] = pd.to_numeric(datos_limpios[columna], errors="coerce")

    nulos_antes = datos_limpios.isna().sum().to_dict()
    filas_sin_objetivo = int(datos_limpios[OBJETIVO].isna().sum())
    datos_limpios = datos_limpios.dropna(subset=[OBJETIVO]).copy()

    for columna in PREDICTORES:
        if datos_limpios[columna].isna().any():
            datos_limpios[columna] = datos_limpios[columna].fillna(datos_limpios[columna].median())

    duplicados_antes = int(datos_limpios.duplicated().sum())
    datos_limpios = datos_limpios.drop_duplicates().reset_index(drop=True)

    atipicos_iqr: dict[str, int] = {}
    for columna in columnas_esperadas:
        q1, q3 = datos_limpios[columna].quantile([0.25, 0.75])
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        atipicos_iqr[columna] = int(
            ((datos_limpios[columna] < limite_inferior) | (datos_limpios[columna] > limite_superior)).sum()
        )

    calidad = {
        "registros_recibidos": int(len(datos)),
        "registros_despues_limpieza": int(len(datos_limpios)),
        "nulos_antes": {clave: int(valor) for clave, valor in nulos_antes.items()},
        "nulos_despues": {clave: int(valor) for clave, valor in datos_limpios.isna().sum().to_dict().items()},
        "filas_eliminadas_por_objetivo_nulo": filas_sin_objetivo,
        "duplicados_eliminados": duplicados_antes,
        "atipicos_detectados_iqr": atipicos_iqr,
        "tratamiento_de_atipicos": "Se conservan y se interpretan; no se eliminan sin justificar un error de medición.",
    }
    return datos_limpios, calidad


def guardar_eda(datos: pd.DataFrame) -> None:
    """Exporta tablas y una figura de análisis exploratorio."""
    datos.describe().T.round(4).to_csv(CARPETA_SALIDAS / "estadisticos_descriptivos.csv")
    datos.corr(numeric_only=True).round(4).to_csv(CARPETA_SALIDAS / "matriz_correlacion.csv")

    pendiente, intercepto = np.polyfit(datos["TV"], datos[OBJETIVO], 1)
    eje_x = np.linspace(datos["TV"].min(), datos["TV"].max(), 200)

    figura, eje = plt.subplots(figsize=(8, 5))
    eje.scatter(datos["TV"], datos[OBJETIVO], color="#276fbf", alpha=0.7, edgecolors="black")
    eje.plot(eje_x, pendiente * eje_x + intercepto, "--", color="#c1121f", linewidth=2)
    eje.set_title("Relación entre inversión en TV y ventas")
    eje.set_xlabel("Inversión en TV miles de dólares")
    eje.set_ylabel("Ventas miles de unidades")
    eje.grid(True, linestyle="--", alpha=0.5)
    figura.tight_layout()
    figura.savefig(CARPETA_FIGURAS / "eda_tv_ventas.png", dpi=300)
    plt.close(figura)


def interpretar_prueba(p_valor: float, hipotesis_nula: str) -> str:
    """Devuelve una interpretación prudente y uniforme para las pruebas."""
    if p_valor >= 0.05:
        return f"No hay evidencia estadística suficiente para rechazar que {hipotesis_nula}."
    return f"Se encuentra evidencia estadística para cuestionar que {hipotesis_nula}."


def comparar_modelos(
    x_entrenamiento: pd.DataFrame,
    x_prueba: pd.DataFrame,
    y_entrenamiento: pd.Series,
    y_prueba: pd.Series,
) -> pd.DataFrame:
    """Compara el modelo base con alternativas lineales interpretables.

    La interacción TV Radio no cambia la naturaleza lineal del modelo respecto
    de sus coeficientes; se incluye para explorar una posible combinación entre
    ambos medios publicitarios.
    """
    alternativas = [
        ("Modelo base", x_entrenamiento, x_prueba),
        (
            "Modelo reducido TV y Radio",
            x_entrenamiento[["TV", "Radio"]],
            x_prueba[["TV", "Radio"]],
        ),
        (
            "Modelo con interacción TV y Radio",
            x_entrenamiento.assign(TV_Radio=x_entrenamiento["TV"] * x_entrenamiento["Radio"]),
            x_prueba.assign(TV_Radio=x_prueba["TV"] * x_prueba["Radio"]),
        ),
    ]
    filas = []
    for nombre, x_train, x_test in alternativas:
        modelo = LinearRegression().fit(x_train, y_entrenamiento)
        predicciones = modelo.predict(x_test)
        filas.append(
            {
                "modelo": nombre,
                "mae": float(mean_absolute_error(y_prueba, predicciones)),
                "mse": float(mean_squared_error(y_prueba, predicciones)),
                "rmse": float(np.sqrt(mean_squared_error(y_prueba, predicciones))),
                "r2": float(r2_score(y_prueba, predicciones)),
            }
        )
    return pd.DataFrame(filas).sort_values("rmse").reset_index(drop=True)


def modelar_y_validar(
    datos: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Entrena el modelo, calcula métricas y revisa los supuestos principales."""
    x = datos[PREDICTORES]
    y = datos[OBJETIVO]
    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
        x, y, test_size=0.30, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(x_entrenamiento, y_entrenamiento)
    predicciones_prueba = modelo.predict(x_prueba)

    metricas = {
        "mae": float(mean_absolute_error(y_prueba, predicciones_prueba)),
        "mse": float(mean_squared_error(y_prueba, predicciones_prueba)),
        "rmse": float(np.sqrt(mean_squared_error(y_prueba, predicciones_prueba))),
        "r2": float(r2_score(y_prueba, predicciones_prueba)),
        "observaciones_entrenamiento": int(len(x_entrenamiento)),
        "observaciones_prueba": int(len(x_prueba)),
        "random_state": 42,
    }
    comparacion = comparar_modelos(x_entrenamiento, x_prueba, y_entrenamiento, y_prueba)

    coeficientes = pd.DataFrame(
        {
            "variable": PREDICTORES,
            "coeficiente": modelo.coef_,
        }
    )
    intercepto = float(modelo.intercept_)

    x_entrenamiento_constante = sm.add_constant(x_entrenamiento, has_constant="add")
    modelo_ols = sm.OLS(y_entrenamiento, x_entrenamiento_constante).fit()
    residuos = modelo_ols.resid
    ajustados = modelo_ols.fittedvalues

    shapiro_estadistico, shapiro_p = shapiro(residuos)
    bp_lm, bp_lm_p, _, _ = het_breuschpagan(residuos, x_entrenamiento_constante)
    reset = linear_reset(modelo_ols, power=2, use_f=True)

    vif = pd.DataFrame(
        {
            "variable": PREDICTORES,
            "vif": [
                variance_inflation_factor(x_entrenamiento.to_numpy(), indice)
                for indice in range(len(PREDICTORES))
            ],
        }
    )

    diagnosticos = {
        "normalidad_shapiro": {
            "estadistico": float(shapiro_estadistico),
            "p_valor": float(shapiro_p),
            "interpretacion": interpretar_prueba(float(shapiro_p), "los residuos sigan una distribución normal"),
        },
        "homocedasticidad_breusch_pagan": {
            "estadistico_lm": float(bp_lm),
            "p_valor": float(bp_lm_p),
            "interpretacion": interpretar_prueba(float(bp_lm_p), "la varianza de los residuos sea constante"),
        },
        "linealidad_reset_ramsey": {
            "estadistico_f": float(reset.statistic),
            "p_valor": float(reset.pvalue),
            "interpretacion": interpretar_prueba(float(reset.pvalue), "la especificación lineal sea suficiente"),
        },
        "multicolinealidad_vif": {
            "valores": {fila.variable: float(fila.vif) for fila in vif.itertuples(index=False)},
            "interpretacion": "Valores de VIF menores de 5 no sugieren multicolinealidad problemática.",
        },
    }

    crear_graficos_validacion(y_prueba, predicciones_prueba, ajustados, residuos)
    resultados = {
        "intercepto": intercepto,
        "metricas_prueba": metricas,
        "diagnosticos": diagnosticos,
        "comparacion_exploratoria_de_modelos": comparacion.to_dict(orient="records"),
    }
    return resultados, coeficientes, vif, comparacion


def crear_graficos_validacion(
    y_prueba: pd.Series,
    predicciones_prueba: np.ndarray,
    valores_ajustados: pd.Series,
    residuos: pd.Series,
) -> None:
    """Crea gráficos para evaluación predictiva y diagnóstico de supuestos."""
    figura, ejes = plt.subplots(1, 3, figsize=(17, 5))

    minimo = min(y_prueba.min(), predicciones_prueba.min())
    maximo = max(y_prueba.max(), predicciones_prueba.max())
    ejes[0].scatter(y_prueba, predicciones_prueba, color="#168aad", edgecolors="black", alpha=0.75)
    ejes[0].plot([minimo, maximo], [minimo, maximo], "--", color="#c1121f", linewidth=2)
    ejes[0].set_title("Ventas reales y predichas")
    ejes[0].set_xlabel("Ventas reales")
    ejes[0].set_ylabel("Ventas predichas")
    ejes[0].grid(True, linestyle="--", alpha=0.5)

    ejes[1].scatter(valores_ajustados, residuos, color="#7b2cbf", edgecolors="black", alpha=0.75)
    ejes[1].axhline(0, color="#c1121f", linestyle="--", linewidth=2)
    ejes[1].set_title("Residuos y valores ajustados")
    ejes[1].set_xlabel("Valores ajustados")
    ejes[1].set_ylabel("Residuos")
    ejes[1].grid(True, linestyle="--", alpha=0.5)

    sm.qqplot(residuos, line="45", ax=ejes[2])
    ejes[2].set_title("Gráfico Q Q de residuos")
    ejes[2].grid(True, linestyle="--", alpha=0.5)

    figura.tight_layout()
    figura.savefig(CARPETA_FIGURAS / "validacion_modelo.png", dpi=300)
    plt.close(figura)


def guardar_json(ruta: Path, contenido: dict[str, Any]) -> None:
    """Guarda resultados con codificación UTF-8 para reutilizarlos en el informe."""
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2)


def main() -> None:
    preparar_carpetas()
    datos_brutos, trazabilidad = descargar_datos()
    datos_limpios, calidad = validar_y_limpiar(datos_brutos)
    calidad["trazabilidad_fuente"] = trazabilidad
    datos_limpios.to_csv(RUTA_PROCESADA, index=False)
    guardar_eda(datos_limpios)

    resultados, coeficientes, vif, comparacion = modelar_y_validar(datos_limpios)
    coeficientes.round(6).to_csv(CARPETA_SALIDAS / "coeficientes_modelo.csv", index=False)
    vif.round(4).to_csv(CARPETA_SALIDAS / "vif.csv", index=False)
    comparacion.round(6).to_csv(CARPETA_SALIDAS / "comparacion_modelos.csv", index=False)
    guardar_json(CARPETA_SALIDAS / "calidad_datos.json", calidad)
    guardar_json(CARPETA_SALIDAS / "resultados_modelo.json", resultados)

    metricas = resultados["metricas_prueba"]
    print("Análisis finalizado.")
    print(f"Registros analizados: {calidad['registros_despues_limpieza']}")
    print(f"MAE: {metricas['mae']:.4f}")
    print(f"MSE: {metricas['mse']:.4f}")
    print(f"RMSE: {metricas['rmse']:.4f}")
    print(f"R²: {metricas['r2']:.4f}")


if __name__ == "__main__":
    main()
