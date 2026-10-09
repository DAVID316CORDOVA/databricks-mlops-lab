"""Monitoreo de drift sobre el trafico REAL del endpoint de Model Serving.
Lee la tabla de inferencia (cada peticion guardada), extrae las entradas del modelo y las compara
con la foto de entrenamiento (PSI y KS), igual que monitor.py pero con datos de produccion."""
import argparse
from datetime import datetime, timezone

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from mlops_lab.config import FEATURES, LabConfig
from mlops_lab.drift import drift_report


class DriftAlert(Exception):
    """Hace fallar el job a proposito: Databricks envia el correo de 'job fallido'."""


# Formato de la columna `request`: {"dataframe_records": [{"trip_distance": 2.2, ...}, ...]}
REQUEST_SCHEMA = "dataframe_records ARRAY<STRUCT<{}>>".format(
    ", ".join(f"{c}: DOUBLE" for c in FEATURES))


def parse_requests(payload: DataFrame) -> DataFrame:
    """De la tabla de inferencia saca una fila por viaje, con las columnas del modelo."""
    return (payload.filter(F.col("status_code") == 200)           # solo peticiones correctas
            .select("request_time", F.from_json("request", REQUEST_SCHEMA).alias("r"))
            .select("request_time", F.explode("r.dataframe_records").alias("rec"))
            .select("request_time", *[F.col(f"rec.{c}").alias(c) for c in FEATURES]))


def check_endpoint_drift(payload: DataFrame, baseline: dict, minutes: int, min_rows: int):
    """Devuelve (informe, n_filas). Si hay pocas filas devuelve informe=None: el PSI con
    muy pocos datos es ruido, mejor no alertar."""
    recent = payload.filter(
        F.col("request_time") >= F.expr(f"current_timestamp() - INTERVAL {int(minutes)} MINUTES"))
    inputs = parse_requests(recent)          # sin .cache(): serverless no lo soporta
    n = inputs.count()
    if n < min_rows:
        return None, n
    return drift_report(baseline, inputs), n


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--minutes", type=int, default=60, help="ventana de tiempo a revisar")
    parser.add_argument("--min-rows", type=int, default=30, help="minimo de filas para evaluar")
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args(argv)
    cfg = LabConfig(args.catalog, args.schema)

    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")

    # La foto de entrenamiento del modelo que esta en produccion (alias champion)
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(cfg.model_name, args.alias)
    baseline = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/drift_baseline.json")

    report, n = check_endpoint_drift(spark.table(cfg.payload), baseline, args.minutes, args.min_rows)
    if report is None:
        print(f"Solo {n} peticiones en los ultimos {args.minutes} min (minimo {args.min_rows}). "
              "No se evalua drift.")
        return

    report["model_version"] = str(version.version)
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    report["n_rows"] = n
    report["window_minutes"] = args.minutes
    spark.createDataFrame(report).write.mode("append").saveAsTable(cfg.endpoint_drift_metrics)
    print(report.to_string(index=False))

    alerts = report[report["status"] == "alert"]["feature"].tolist()
    if alerts and args.fail_on_alert:
        raise DriftAlert(f"Drift fuerte en trafico real: {alerts} ({n} peticiones)")


if __name__ == "__main__":
    main()