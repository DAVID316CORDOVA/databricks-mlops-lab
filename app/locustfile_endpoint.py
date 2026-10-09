"""Prueba de carga contra el endpoint de Model Serving."""
import os
import random

from locust import HttpUser, between, task

ENDPOINT = os.getenv("ENDPOINT_NAME", "taxi-fare-dev")


class ServingUser(HttpUser):
    host = "https://adb-7405611969133544.4.azuredatabricks.net"
    wait_time = between(0.1, 0.5)          # pausa entre peticiones; constant(0) = sin pausa

    def on_start(self):
        self.headers = {"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"}

    @task
    def predict(self):
        distance = round(random.lognormvariate(0.9, 0.7), 2)
        trip = {"trip_distance": min(distance, 60),
                "duration_min": round(3 + distance * 4 + random.uniform(0, 6), 1),
                "pickup_hour": random.randint(0, 23),
                "pickup_dow": random.randint(1, 7)}
        self.client.post(f"/serving-endpoints/{ENDPOINT}/invocations",
                         json={"dataframe_records": [trip]}, headers=self.headers,
                         name="invocations")
