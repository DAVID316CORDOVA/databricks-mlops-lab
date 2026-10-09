"""Monitoreo: data drift (PSI y KS) sobre las entradas + error real del modelo."""
import argparse
from datetime import datetime, timezone

import mlflow
import pandas as pd
from pyspark.sql import functions as F

from mlops_lab.config import LabConfig
from mlops_lab.drift import drift_report, simulate_drift


class DriftAlert(Exception):
    """Se lanza (solo con --fail-on-alert) para que el job falle y llegue la notificacion."""


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--simulate-factor", type=float, default=1.0)   # 1.0 = datos reales
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args(argv)
    cfg = LabConfig(args.catalog, args.schema)

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")

    # 1) Que modelo esta en produccion y con que datos se entreno
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(cfg.model_name, args.alias)
    baseline = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/drift_baseline.json")
    train_mae = client.get_run(version.run_id).data.metrics.get("mae")

    # 2) Datos nuevos (opcionalmente alterados para simular drift)
    new_df = spark.table(cfg.silver).filter(F.col("window") == "new")
    simulated = args.simulate_factor != 1.0
    if simulated:
        new_df = simulate_drift(new_df, args.simulate_factor)

    # 3) Data drift por feature
    report = drift_report(baseline, new_df)
    checked_at = datetime.now(timezone.utc).isoformat()
    report["model_version"] = str(version.version)
    report["checked_at"] = checked_at
    report["simulated"] = simulated
    spark.createDataFrame(report).write.mode("append").saveAsTable(cfg.drift_metrics)
    print(report.to_string(index=False))

    # 4) Error real: aqui SI tenemos la tarifa real, asi que medimos si el modelo se degrada
    window_mae = spark.table(cfg.predictions).agg(F.avg("abs_error")).first()[0]
    perf = pd.DataFrame([{
        "model_version": str(version.version), "checked_at": checked_at,
        "train_mae": float(train_mae), "window_mae": float(window_mae),
        "mae_ratio": float(window_mae / train_mae),
    }])
    spark.createDataFrame(perf).write.mode("append").saveAsTable(cfg.performance_metrics)
    print(perf.to_string(index=False))

    alerts = report[report["status"] == "alert"]["feature"].tolist()
    if alerts and args.fail_on_alert:
        raise DriftAlert(f"Drift fuerte en: {alerts}")


if __name__ == "__main__":
    main()