import atexit
import json
import os
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MQTT_USER", "collier-test")
os.environ.setdefault("MQTT_PASSWORD", "collier-test")
os.environ.setdefault("MONGO_URI", "mongodb://127.0.0.1:27017")
os.environ.setdefault(
    "JWT_SECRET",
    "collier-ingestion-tests-secret-with-at-least-32-bytes",
)

import main
atexit.register(main.mongo_client.close)


class CollectionMemoire:
    def __init__(self, documents=()):
        self.documents = [deepcopy(document) for document in documents]

    @staticmethod
    def _correspond(document, filtre):
        for cle, valeur in filtre.items():
            courant = document.get(cle)
            if isinstance(valeur, dict):
                if "$ne" in valeur and courant == valeur["$ne"]:
                    return False
            elif courant != valeur:
                return False
        return True

    def find_one(self, filtre, projection=None):
        return next(
            (
                deepcopy(document)
                for document in self.documents
                if self._correspond(document, filtre)
            ),
            None,
        )

    def update_one(self, filtre, modification, upsert=False):
        document = next(
            (
                item
                for item in self.documents
                if self._correspond(item, filtre)
            ),
            None,
        )
        if document is None and upsert:
            document = deepcopy(modification.get("$setOnInsert", {}))
            for cle, valeur in filtre.items():
                if not isinstance(valeur, dict):
                    document.setdefault(cle, valeur)
            self.documents.append(document)
        if document is None:
            return SimpleNamespace(matched_count=0)
        document.update(deepcopy(modification.get("$set", {})))
        return SimpleNamespace(matched_count=1)

    def insert_one(self, document):
        self.documents.append(deepcopy(document))


class DiffuseurMemoire:
    def publier(self, evenement, client_id):
        pass


class IngestionCollierTests(unittest.TestCase):
    def test_device_id_et_horodatage_conservent_l_animal_historique(self):
        maintenant = datetime.now(timezone.utc)
        debut_ancien = maintenant - timedelta(days=3)
        fin_ancien = maintenant - timedelta(days=1)
        documents_colliers = [{
            "client_id": "client-a",
            "collier_id": "imei-123",
            "animal_id": "animal-actuel",
            "revision": 1,
            "affectations": [
                {
                    "animal_id": "animal-vendu",
                    "debut": debut_ancien,
                    "fin": fin_ancien,
                    "affecte_par": "owner-a",
                },
                {
                    "animal_id": "animal-actuel",
                    "debut": fin_ancien,
                    "fin": None,
                    "affecte_par": "owner-a",
                },
            ],
        }]
        positions = CollectionMemoire()
        clients = CollectionMemoire([{
            "_id": "client-a",
            "numero_client": "00000001",
        }])
        animaux = CollectionMemoire([{
            "client_id": "client-a",
            "animal_id": "animal-vendu",
            "ferme": "Ferme principale",
        }])
        timestamp_ancien = debut_ancien + timedelta(hours=1)
        message = SimpleNamespace(
            topic="farm/00000001/data",
            payload=json.dumps({
                "device_id": "imei-123",
                "lat": 36.7,
                "lon": 3.1,
                "vitesse": 0.4,
                "activite": "marche",
                "timestamp": timestamp_ancien.isoformat(),
            }).encode("utf-8"),
        )

        with (
            patch.object(main, "colliers", CollectionMemoire(documents_colliers)),
            patch.object(main, "positions", positions),
            patch.object(main, "clients_col", clients),
            patch.object(main, "animaux", animaux),
            patch.object(main, "diffuseur", DiffuseurMemoire()),
            patch.object(main, "gerer_alerte"),
            patch.object(main, "verifier_troupeau"),
            patch.object(main, "verifier_pature"),
        ):
            main.on_message(None, None, message)

        self.assertEqual(len(positions.documents), 1)
        self.assertEqual(positions.documents[0]["animal_id"], "animal-vendu")
        self.assertEqual(positions.documents[0]["collier_id"], "imei-123")
        self.assertEqual(positions.documents[0]["timestamp"], timestamp_ancien)


if __name__ == "__main__":
    unittest.main()
