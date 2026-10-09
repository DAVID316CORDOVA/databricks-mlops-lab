import math

import pytest
from mlflow.exceptions import MlflowException

from mlops_lab.train import decide_alias, train_models


@pytest.mark.parametrize("engine", ["spark", "sklearn"])
def test_train_models_returns_three_models_and_picks_best(silver_df, engine):
    results, best = train_models(silver_df, engine)
    assert set(results) == {"linear_regression", "random_forest", "gbt"}
    for r in results.values():
        assert math.isfinite(r["rmse"]) and r["mae"] >= 0
        assert r["rmse"] < 5                                  # la tarifa si es aprendible
    assert results[best]["rmse"] == min(r["rmse"] for r in results.values())


def test_unknown_engine_raises(silver_df):
    with pytest.raises(ValueError):
        train_models(silver_df, "otro")


# --- gobierno del modelo: quien es champion (con un cliente falso, sin MLflow real) ---
class FakeClient:
    def __init__(self, champion_rmse=None):
        self.champion_rmse = champion_rmse

    def get_model_version_by_alias(self, name, alias):
        if self.champion_rmse is None:
            raise MlflowException("no existe")
        return type("V", (), {"run_id": "r1"})()

    def get_run(self, run_id):
        data = type("D", (), {"metrics": {"rmse": self.champion_rmse}})()
        return type("R", (), {"data": data})()


def test_first_model_becomes_champion():
    assert decide_alias(FakeClient(None), "m", 2.0) == "champion"


def test_better_model_replaces_champion():
    assert decide_alias(FakeClient(3.0), "m", 2.0) == "champion"


def test_worse_model_stays_challenger():
    assert decide_alias(FakeClient(1.5), "m", 2.0) == "challenger"