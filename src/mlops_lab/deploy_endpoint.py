"""Crea o actualiza el endpoint de Model Serving para que sirva el modelo champion.
Si el endpoint no existe lo crea; si existe, lo actualiza a la version champion actual.
La tabla de inferencia se activa por AI Gateway (la forma antigua esta obsoleta).
Con --delete lo borra. Pensado para correr como paso del pipeline / del CD."""
import argparse

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)

from mlops_lab.config import LabConfig


def champion_version(model_name: str, alias: str) -> str:
    """Que version del modelo tiene el alias (champion)."""
    mlflow.set_registry_uri("databricks-uc")
    return str(mlflow.MlflowClient().get_model_version_by_alias(model_name, alias).version)


def deploy(w: WorkspaceClient, name: str, cfg: LabConfig, alias: str, prefix: str) -> str:
    """Crea o actualiza el endpoint. Devuelve la version desplegada."""
    version = champion_version(cfg.model_name, alias)
    entities = [ServedEntityInput(
        entity_name=cfg.model_name,
        entity_version=version,
        workload_size="Small",           # el tamano mas chico (barato)
        scale_to_zero_enabled=True,      # se apaga solo si no hay trafico
    )]

    try:
        w.serving_endpoints.get(name)
        existe = True
    except NotFound:
        existe = False

    if existe:
        print(f"El endpoint {name} ya existe: actualizando a la version {version}...")
        w.serving_endpoints.update_config_and_wait(name=name, served_entities=entities)
    else:
        print(f"Creando el endpoint {name} (puede tardar varios minutos)...")
        w.serving_endpoints.create_and_wait(
            name=name,
            config=EndpointCoreConfigInput(name=name, served_entities=entities))

    # Activa la tabla de inferencia (guarda cada peticion) con AI Gateway
    w.serving_endpoints.put_ai_gateway(
        name=name,
        inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=cfg.catalog,
            schema_name=cfg.schema,
            table_name_prefix=prefix,
            enabled=True,
        ),
    )
    return version


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--name", required=True, help="nombre del endpoint")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--prefix", default="taxi_fare", help="prefijo de la tabla de inferencia")
    parser.add_argument("--profile", default=None, help="perfil del CLI (solo desde tu PC)")
    parser.add_argument("--delete", action="store_true", help="borra el endpoint y termina")
    args = parser.parse_args(argv)
    cfg = LabConfig(args.catalog, args.schema)

    # En Databricks no hace falta perfil (usa la identidad del job); en tu PC si.
    w = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()

    if args.delete:
        w.serving_endpoints.delete(args.name)
        print(f"Endpoint {args.name} borrado.")
        return

    version = deploy(w, args.name, cfg, args.alias, args.prefix)
    print(f"Listo: endpoint {args.name} sirviendo la version {version}.")


if __name__ == "__main__":
    main()