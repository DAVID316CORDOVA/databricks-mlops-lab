"""API de prediccion en tiempo real (FastAPI). Carga el modelo 'champion' desde el registro de
MLflow al arrancar y lo mantiene en memoria. Cada peticion solo hace model.predict."""
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Que modelo servir. Se cambia con una variable de entorno, sin tocar codigo.
MODEL_URI = os.getenv("MODEL_URI", "models:/workspace.mlops_dev.taxi_fare_model@champion")
FEATURES = ["trip_distance", "duration_min", "pickup_hour", "pickup_dow"]
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta UNA vez al arrancar: cargar el modelo es lento, predecir es rapido.
    if MODEL_URI.startswith("models:/"):
        mlflow.set_registry_uri("databricks-uc")      # registro de Unity Catalog
    state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
    yield
    state.clear()


app = FastAPI(title="Taxi fare API", lifespan=lifespan)


class Trip(BaseModel):
    """Validacion de entrada: si falta un campo o el tipo es malo, FastAPI responde 422."""
    trip_distance: float = Field(gt=0, le=100, description="millas")
    duration_min: float = Field(gt=0, le=240)
    pickup_hour: float = Field(ge=0, le=23)
    pickup_dow: float = Field(ge=1, le=7, description="1=domingo ... 7=sabado")


@app.get("/health")
def health():
    """Para el balanceador / orquestador: esta viva y con el modelo cargado?"""
    return {"status": "ok" if "model" in state else "loading", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(trips: list[Trip]):
    if "model" not in state:
        raise HTTPException(503, "modelo no cargado")
    df = pd.DataFrame([t.model_dump() for t in trips], columns=FEATURES)
    preds = state["model"].predict(df)
    return {"predictions": [round(float(p), 2) for p in preds]}