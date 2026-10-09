"""Scoring batch distribuido: el modelo se aplica como UDF de Spark sobre los datos 'new'."""
import argparse

import mlflow
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from mlops_lab.config import FEATURES, TARGET, LabConfig


def score(spark, df: DataFrame, model_uri: str) -> DataFrame:
    """Agrega la prediccion y el error absoluto. Funciona con modelos de Spark ML y de sklearn."""
    predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")
    return (df.withColumn("prediction", predict_udf(*[F.col(c) for c in FEATURES]))
              .withColumn("abs_error", F.abs(F.col(TARGET) - F.col("prediction"))))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args(argv)
    cfg = LabConfig(args.catalog, args.schema)

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")

    new_df = spark.table(cfg.silver).filter(F.col("window") == "new")
    scored = score(spark, new_df, f"models:/{cfg.model_name}@{args.alias}")   # carga por alias

    (scored.write.mode("overwrite").option("overwriteSchema", "true")
           .saveAsTable(cfg.predictions))
    print("predicciones guardadas en", cfg.predictions)


if __name__ == "__main__":
    main()