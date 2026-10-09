import mlflow
import pytest

from mlops_lab.config import FEATURES, LabConfig
from mlops_lab.predict import score
from mlops_lab.train import _sample_inputs_and_preds, train_models


@pytest.mark.parametrize("engine", ["spark", "sklearn"])
def test_score_with_a_logged_model(spark, silver_df, tmp_path, engine):
    """Entrena, registra en un MLflow local y puntua con spark_udf (lo mismo que hace predict)."""
    # MLflow local con SQLite (las versiones nuevas ya no aceptan el almacenamiento por archivos)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    experiment_id = mlflow.create_experiment(
        f"test_{engine}", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment(experiment_id=experiment_id)
    results, best = train_models(silver_df, engine)
    model = results[best]["model"]
    inputs, outputs = _sample_inputs_and_preds(engine, model, silver_df)
    signature = mlflow.models.infer_signature(inputs, outputs)

    with mlflow.start_run():
        if engine == "spark":
            info = mlflow.spark.log_model(model, artifact_path="model", signature=signature)
        else:
            info = mlflow.sklearn.log_model(model, artifact_path="model", signature=signature,
                                            serialization_format="cloudpickle")

    scored = score(spark, silver_df, info.model_uri)
    assert {"prediction", "abs_error"} <= set(scored.columns)
    assert scored.count() == silver_df.count()
    assert scored.agg({"abs_error": "avg"}).first()[0] < 5


def test_config_builds_unity_catalog_names():
    cfg = LabConfig("workspace", "mlops_dev")
    assert cfg.silver == "workspace.mlops_dev.silver_trips"
    assert cfg.model_name == "workspace.mlops_dev.taxi_fare_model"
    assert FEATURES            # las features estan definidas