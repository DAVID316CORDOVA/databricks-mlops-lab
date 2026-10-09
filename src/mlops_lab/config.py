"""Configuracion central: nombres de tablas, features y parametros. Nada hardcodeado en la logica."""
from dataclasses import dataclass

SOURCE_TABLE = "samples.nyctaxi.trips"   # tabla de ejemplo que ya existe en el workspace

FEATURES = ["trip_distance", "duration_min", "pickup_hour", "pickup_dow"]   # entradas del modelo
TARGET = "fare_amount"                                                      # lo que se predice

# El 70% mas antiguo (por fecha) es la ventana "reference" (entrenamiento);
# el 30% mas reciente es la ventana "new" (lo que llega despues y se monitorea).
REFERENCE_FRACTION = 0.7


@dataclass(frozen=True)
class LabConfig:
    catalog: str
    schema: str

    def _table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"      # formato Unity Catalog

    @property
    def bronze(self) -> str:
        return self._table("bronze_trips")

    @property
    def silver(self) -> str:
        return self._table("silver_trips")

    @property
    def predictions(self) -> str:
        return self._table("predictions")

    @property
    def drift_metrics(self) -> str:
        return self._table("drift_metrics")

    @property
    def performance_metrics(self) -> str:
        return self._table("performance_metrics")

    @property
    def model_name(self) -> str:
        return self._table("taxi_fare_model")

    @property
    def experiment(self) -> str:
        return f"/Shared/mlops_lab_{self.schema}"           # un experimento por entorno
    
    
    @property
    def payload(self) -> str:
        return self._table("taxi_fare_payload")             # tabla de inferencia del endpoint

    @property
    def endpoint_drift_metrics(self) -> str:
        return self._table("endpoint_drift_metrics")
    
    @property
    def payload_for(self, prefix: str) -> str:
        """Tabla de inferencia de un endpoint: AI Gateway la llama <prefijo>_payload."""
        return self._table(f"{prefix}_payload")