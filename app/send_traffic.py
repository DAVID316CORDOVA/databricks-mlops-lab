"""Manda peticiones al endpoint de Model Serving. Con --shift 1 simula trafico normal;
con --shift 3 simula viajes 3 veces mas largos (para provocar drift)."""
import argparse
import json
import random
import subprocess
import time

import requests

HOST = "https://adb-7405611969133544.4.azuredatabricks.net"


def get_token(profile: str) -> str:
    """Pide un token temporal al CLI de Databricks (usa tu login, sin guardar secretos)."""
    out = subprocess.run(["databricks", "auth", "token", "--profile", profile],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["access_token"]


def make_trip(shift: float) -> dict:
    distance = random.lognormvariate(0.9, 0.7)          # mayoria de viajes cortos, algunos largos
    duration = 3 + distance * 4 + random.uniform(0, 6)
    return {"trip_distance": round(min(distance, 60) * shift, 2),
            "duration_min": round(min(duration, 200) * shift, 1),
            "pickup_hour": random.randint(0, 23),
            "pickup_dow": random.randint(1, 7)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="taxi-fare-dev")
    p.add_argument("--profile", default="practica")
    p.add_argument("--n", type=int, default=50, help="numero de peticiones")
    p.add_argument("--shift", type=float, default=1.0, help="1=normal, 3=drift fuerte")
    args = p.parse_args()

    url = f"{HOST}/serving-endpoints/{args.endpoint}/invocations"
    headers = {"Authorization": f"Bearer {get_token(args.profile)}"}

    for i in range(args.n):
        trip = make_trip(args.shift)
        # timeout largo: si el endpoint estaba en cero, la primera peticion tarda en arrancar
        r = requests.post(url, headers=headers, json={"dataframe_records": [trip]}, timeout=300)
        print(i + 1, r.status_code, trip, r.json())
        time.sleep(0.2)


if __name__ == "__main__":
    main()
