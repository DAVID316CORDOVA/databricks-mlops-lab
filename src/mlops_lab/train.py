"""Entrenamiento con dos motores intercambiables: Spark ML (distribuido) o scikit-learn.
La logica (train_models) no depende de MLflow ni de Unity Catalog, asi se prueba en local."""
import argparse

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor as SkRandomForest
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from mlops_lab.config import FEATURES, TARGET, LabConfig
from mlops_lab.drift import compute_baseline


# ---------------------------------------------------------------- motor Spark ML
def _train_spark(df: DataFrame, seed: int):
    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features", handleInvalid="skip")
    estimators = {
        "linear_regression": LinearRegression(labelCol=TARGET),
        "random_forest": RandomForestRegressor(labelCol=TARGET, numTrees=50, maxDepth=8, seed=seed),
        "gbt": GBTRegressor(labelCol=TARGET, maxIter=40, maxDepth=5, seed=seed),
    }
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
    results = {}
    for name, est in estimators.items():
        model = Pipeline(stages=[assembler, est]).fit(train_df)      # el pipeline incluye el assembler
        preds = model.transform(test_df)
        metrics = {m: RegressionEvaluator(labelCol=TARGET, predictionCol="prediction",
                                          metricName=m).evaluate(preds) for m in ("rmse", "mae")}
        results[name] = {"model": model, **metrics}
    return results


# ---------------------------------------------------------------- motor scikit-learn
def _train_sklearn(df: DataFrame, seed: int):
    pdf = df.select(*FEATURES, TARGET).toPandas()                     # la ventana reference es chica
    X_train, X_test, y_train, y_test = train_test_split(
        pdf[FEATURES], pdf[TARGET], test_size=0.2, random_state=seed)
    estimators = {
        "linear_regression": SkLinearRegression(),
        "random_forest": SkRandomForest(n_estimators=100, max_depth=10, n_jobs=-1, random_state=seed),
        "gbt": GradientBoostingRegressor(random_state=seed),
    }
    results = {}
    for name, est in estimators.items():
        est.fit(X_train, y_train)
        preds = est.predict(X_test)
        results[name] = {"model": est,
                         "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                         "mae": float(mean_absolute_error(y_test, preds))}
    return results


def train_models(df: DataFrame, engine: str = "spark", seed: int = 42):
    """Entrena 3 modelos y devuelve (resultados, nombre del mejor segun RMSE)."""
    if engine not in ("spark", "sklearn"):
        raise ValueError(f"engine debe ser 'spark' o 'sklearn', no '{engine}'")
    results = _train_spark(df, seed) if engine == "spark" else _train_sklearn(df, seed)
    best = min(results, key=lambda name: results[name]["rmse"])
    return results, best


# ---------------------------------------------------------------- gobierno del modelo
def decide_alias(client, model_name: str, new_rmse: float) -> str:
    """El primer modelo es 'champion'. Despues, solo reemplaza al champion si su RMSE es mejor;
    si no, queda como 'challenger'."""
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return "champion"
    champion_rmse = client.get_run(champion.run_id).data.metrics.get("rmse")
    if champion_rmse is None or new_rmse < champion_rmse:
        return "champion"
    return "challenger"


def _sample_inputs_and_preds(engine, model, df: DataFrame):
    """Ejemplo de entradas y salidas para que MLflow registre la 'signature' del modelo."""
    sample = df.select(*FEATURES).limit(200)
    pdf = sample.toPandas()
    if engine == "spark":
        preds = model.transform(sample).select("prediction").toPandas()["prediction"]
    else:
        preds = model.predict(pdf)
    return pdf, preds


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--engine", default="spark", choices=["spark", "sklearn"])
    args = parser.parse_args(argv)
    cfg = LabConfig(args.catalog, args.schema)

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    silver = spark.table(cfg.silver)
    reference = silver.filter(F.col("window") == "reference")

    results, best = train_models(reference, args.engine)
    print({n: {"rmse": round(r["rmse"], 3), "mae": round(r["mae"], 3)} for n, r in results.items()})

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(cfg.experiment)

    with mlflow.start_run(run_name=f"train_{args.engine}") as run:
        for name, r in results.items():                                  # una sub-run por modelo
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({"model": name, "engine": args.engine})
                mlflow.log_metrics({"rmse": r["rmse"], "mae": r["mae"]})

        mlflow.log_params({"best_model": best, "engine": args.engine, "silver_table": cfg.silver})
        mlflow.log_metrics({"rmse": results[best]["rmse"], "mae": results[best]["mae"]})

        # La foto de los datos de entrenamiento queda atada a ESTA run (y por tanto al modelo)
        mlflow.log_dict(compute_baseline(reference, FEATURES), "drift_baseline.json")

        model = results[best]["model"]
        inputs, outputs = _sample_inputs_and_preds(args.engine, model, reference)
        signature = mlflow.models.infer_signature(inputs, outputs)
        log_model = mlflow.spark.log_model if args.engine == "spark" else mlflow.sklearn.log_model
        extra = {} if args.engine == "spark" else {"serialization_format": "cloudpickle"}
        log_model(model, artifact_path="model", signature=signature,
                  registered_model_name=cfg.model_name, **extra)

    # Versiones registradas por ESTA run -> la mas reciente es la que acabamos de crear
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name = '{cfg.model_name}'")
    mine = [v for v in versions if v.run_id == run.info.run_id] or versions
    version = max(int(v.version) for v in mine)

    alias = decide_alias(client, cfg.model_name, results[best]["rmse"])
    client.set_registered_model_alias(cfg.model_name, alias, version)
    print(f"modelo {cfg.model_name} v{version} ({best}) -> alias '{alias}'")


if __name__ == "__main__":
    main()