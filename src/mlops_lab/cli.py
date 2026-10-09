"""Puntos de entrada de las dos primeras tareas del job (bronze y silver)."""
import argparse

from pyspark.sql import SparkSession

from mlops_lab.config import SOURCE_TABLE, LabConfig
from mlops_lab.etl import build_bronze_df, build_silver_df


def _parse(argv=None) -> LabConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    args = parser.parse_args(argv)
    return LabConfig(args.catalog, args.schema)


def _write(df, table: str) -> None:
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)


def ingest_bronze(argv=None):
    cfg = _parse(argv)
    spark = SparkSession.builder.getOrCreate()
    # Idempotente: si el esquema ya existe no pasa nada (asi dev y staging se crean solos)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.schema}")
    bronze = build_bronze_df(spark.table(SOURCE_TABLE))
    _write(bronze, cfg.bronze)
    print(f"bronze: {spark.table(cfg.bronze).count()} filas en {cfg.bronze}")


def build_silver(argv=None):
    cfg = _parse(argv)
    spark = SparkSession.builder.getOrCreate()
    silver = build_silver_df(spark.table(cfg.bronze))
    _write(silver, cfg.silver)
    print(f"silver: {spark.table(cfg.silver).count()} filas en {cfg.silver}")