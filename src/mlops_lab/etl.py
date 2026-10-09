"""Transformaciones PySpark (bronze -> silver). Son funciones puras DataFrame -> DataFrame:
se pueden probar con un Spark local, sin Databricks."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from mlops_lab.config import REFERENCE_FRACTION

REQUIRED_SOURCE_COLUMNS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount",
]


def build_bronze_df(source: DataFrame) -> DataFrame:
    """Bronze = copia fiel del origen + la marca de cuando se ingirio."""
    return source.withColumn("_ingested_at", F.current_timestamp())


def assign_windows(df: DataFrame, reference_fraction: float = REFERENCE_FRACTION) -> DataFrame:
    """Etiqueta cada fila como 'reference' (mas antigua) o 'new' (mas reciente), por fecha."""
    ts = F.col("pickup_ts").cast("long")
    cutoff = df.select(ts.alias("ts")).approxQuantile("ts", [reference_fraction], 0.001)[0]
    return df.withColumn("window", F.when(ts <= F.lit(cutoff), "reference").otherwise("new"))


def build_silver_df(bronze: DataFrame, reference_fraction: float = REFERENCE_FRACTION) -> DataFrame:
    """Silver = datos tipados, limpios y con las features que usa el modelo."""
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in bronze.columns]
    if missing:   # error claro si el origen no tiene lo esperado
        raise ValueError(f"Faltan columnas en el origen: {missing}. Disponibles: {bronze.columns}")

    pickup = F.col("tpep_pickup_datetime").cast("timestamp")
    dropoff = F.col("tpep_dropoff_datetime").cast("timestamp")
    duration_min = (F.unix_timestamp(dropoff) - F.unix_timestamp(pickup)) / 60.0

    silver = (
        bronze.select(
            pickup.alias("pickup_ts"),
            F.col("trip_distance").cast("double").alias("trip_distance"),
            duration_min.cast("double").alias("duration_min"),
            F.hour(pickup).cast("double").alias("pickup_hour"),
            F.dayofweek(pickup).cast("double").alias("pickup_dow"),
            F.col("fare_amount").cast("double").alias("fare_amount"),
        )
        .dropna()
        # Reglas de calidad: descartamos viajes imposibles o corruptos
        .filter(F.col("trip_distance").between(0.1, 100))
        .filter(F.col("duration_min").between(1, 240))
        .filter((F.col("fare_amount") > 0) & (F.col("fare_amount") <= 300))
        .dropDuplicates()
    )
    return assign_windows(silver, reference_fraction)


