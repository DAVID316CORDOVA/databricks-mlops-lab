"""Fixtures compartidas: un Spark local y datos de taxi sinteticos (con filas malas a proposito)."""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest
from pyspark.sql import SparkSession

from mlops_lab.etl import build_bronze_df, build_silver_df

os.environ["PYSPARK_PYTHON"] = sys.executable          # que los workers usen este mismo Python
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

SOURCE_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount"]


@pytest.fixture(scope="session")
def spark():
    session = (SparkSession.builder.master("local[2]").appName("tests")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false")
               .config("spark.driver.bindAddress", "127.0.0.1")
               .config("spark.driver.host", "127.0.0.1")
               .getOrCreate())
    yield session
    session.stop()


def make_source_df(spark, n=1000, bad_rows=True, seed=0):
    """Imita la tabla de origen. La tarifa depende de distancia y duracion (se puede aprender)."""
    rng = np.random.default_rng(seed)
    start = datetime(2016, 1, 1)
    rows = []
    for i in range(n):
        pickup = start + timedelta(minutes=int(i * 30 * 24 * 60 / n))   # repartido en 30 dias
        distance = float(rng.uniform(0.3, 15))
        duration = float(2 + distance * 3 + rng.uniform(0, 5))
        fare = float(max(3.0, 2.5 + 2.2 * distance + 0.35 * duration + rng.normal(0, 0.5)))
        rows.append((pickup, pickup + timedelta(minutes=duration), distance, fare))
    if bad_rows:                                         # filas que el silver debe descartar
        t = start + timedelta(days=1)
        rows += [(t, t + timedelta(minutes=10), 0.0, 10.0),            # distancia cero
                 (t, t + timedelta(minutes=10), 3.0, -5.0),            # tarifa negativa
                 (t, t - timedelta(minutes=10), 3.0, 12.0)]            # llega antes de salir
    return spark.createDataFrame(rows, SOURCE_COLUMNS)


@pytest.fixture(scope="session")
def silver_df(spark):
    return build_silver_df(build_bronze_df(make_source_df(spark))).cache()