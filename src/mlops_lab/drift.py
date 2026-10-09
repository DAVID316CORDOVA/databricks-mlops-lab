"""Data drift con PySpark: PSI y KS. Spark cuenta (distribuido) y Python hace la matematica,
porque el PSI son ~10 numeros por variable."""
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from scipy.stats import ks_2samp

EPS = 1e-6   # evita log(0) cuando un rango queda vacio


def _bin_counts(df: DataFrame, column: str, inner_edges: list) -> list:
    """Cuenta cuantas filas caen en cada rango. Solo usa SQL de Spark (sin Spark ML), asi
    funciona igual en serverless. Rango i = [borde_i, borde_i+1); los extremos son -inf e +inf."""
    # indice del rango = cuantos bordes son <= al valor
    bin_idx = sum((F.col(column) >= F.lit(float(e))).cast("int") for e in inner_edges)
    rows = df.where(F.col(column).isNotNull()).groupBy(bin_idx.alias("_bin")).count().collect()
    counts = [0] * (len(inner_edges) + 1)
    for r in rows:
        counts[int(r["_bin"])] = r["count"]
    return counts


def _sample_values(df: DataFrame, column: str, total: int, sample_size: int = 2000, seed: int = 42):
    """Muestra pequena de valores (la necesita el test KS)."""
    fraction = min(1.0, sample_size * 1.5 / max(total, 1))
    rows = df.select(column).dropna().sample(False, fraction, seed).limit(sample_size).collect()
    return [float(r[0]) for r in rows]


def compute_baseline(df: DataFrame, features: list, n_bins: int = 10) -> dict:
    """Foto de los datos de entrenamiento: rangos, % de datos por rango y una muestra."""
    total = df.count()
    probs = [i / n_bins for i in range(1, n_bins)]
    base = {}
    for c in features:
        inner = sorted(set(df.approxQuantile(c, probs, 0.001)))   # bordes por cuantiles
        if not inner:                                             # columna constante
            inner = [0.0]
        counts = _bin_counts(df, c, inner)
        n = sum(counts)
        base[c] = {
            "inner_edges": inner,
            "expected_pct": [x / n for x in counts],
            "sample": _sample_values(df, c, total),
        }
    return {"features": base}


def psi(expected_pct, actual_pct) -> float:
    """PSI = suma de (actual - esperado) * ln(actual / esperado), rango por rango."""
    e = np.clip(np.asarray(expected_pct, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual_pct, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status_from_psi(value: float) -> str:
    if value < 0.1:
        return "ok"          # estable
    if value < 0.25:
        return "warning"     # cambio moderado: vigilar
    return "alert"           # cambio fuerte: investigar


def drift_report(baseline: dict, new_df: DataFrame) -> pd.DataFrame:
    """Compara los datos nuevos contra la base. Una fila por feature."""
    total = new_df.count()
    rows = []
    for c, ref in baseline["features"].items():
        counts = _bin_counts(new_df, c, ref["inner_edges"])    # en los MISMOS rangos del train
        n = sum(counts)
        actual = [x / n for x in counts]
        p = psi(ref["expected_pct"], actual)
        ks = ks_2samp(ref["sample"], _sample_values(new_df, c, total)).statistic
        rows.append({"feature": c, "psi": round(p, 4), "ks": round(float(ks), 4),
                     "status": status_from_psi(p)})
    return pd.DataFrame(rows)


def simulate_drift(df: DataFrame, factor: float) -> DataFrame:
    """Para demos: multiplica distancia y duracion, como si los viajes se hubieran alargado."""
    return (df.withColumn("trip_distance", F.col("trip_distance") * factor)
              .withColumn("duration_min", F.col("duration_min") * factor))