import pytest
from conftest import make_source_df
from pyspark.sql import functions as F

from mlops_lab.etl import build_bronze_df, build_silver_df


def test_silver_drops_bad_rows(spark):
    silver = build_silver_df(build_bronze_df(make_source_df(spark, n=500, bad_rows=True)))
    assert silver.count() == 500            # las 3 filas malas desaparecen


def test_silver_has_expected_columns(silver_df):
    assert set(silver_df.columns) == {
        "pickup_ts", "trip_distance", "duration_min", "pickup_hour", "pickup_dow",
        "fare_amount", "window"}


def test_silver_rejects_source_without_required_columns(spark):
    source = make_source_df(spark, n=50, bad_rows=False).drop("fare_amount")
    with pytest.raises(ValueError, match="Faltan columnas"):
        build_silver_df(build_bronze_df(source))


def test_windows_split_roughly_70_30(silver_df):
    share_new = silver_df.filter(F.col("window") == "new").count() / silver_df.count()
    assert 0.25 < share_new < 0.35


def test_windows_are_chronological(silver_df):
    ref_max = silver_df.filter(F.col("window") == "reference").agg(F.max("pickup_ts")).first()[0]
    new_min = silver_df.filter(F.col("window") == "new").agg(F.min("pickup_ts")).first()[0]
    assert ref_max <= new_min               # lo "nuevo" es siempre posterior a lo de entrenamiento