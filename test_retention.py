import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("JWT_SECRET", "retention-test-secret-with-at-least-32-bytes")

from bson import ObjectId

from retention import nettoyer_donnees


class CollectionMemoire:
    def __init__(self):
        self.documents = []

    def _correspond(self, document, filtre):
        for champ, valeur in filtre.items():
            courant = document.get(champ)
            if isinstance(valeur, dict):
                if "$lt" in valeur and not courant < valeur["$lt"]:
                    return False
            elif courant != valeur:
                return False
        return True

    def find(self, filtre, projection=None):
        return [
            document for document in self.documents
            if self._correspond(document, filtre)
        ]

    def find_one_and_delete(self, filtre):
        document = next(
            (doc for doc in self.documents if self._correspond(doc, filtre)),
            None,
        )
        if document is not None:
            self.documents.remove(document)
        return document

    def delete_many(self, filtre):
        supprimés = [
            document for document in self.documents
            if self._correspond(document, filtre)
        ]
        self.documents = [
            document for document in self.documents
            if not self._correspond(document, filtre)
        ]
        return SimpleNamespace(deleted_count=len(supprimés))


class BaseMemoire:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, nom):
        return self.collections.setdefault(nom, CollectionMemoire())


class RetentionTests(unittest.TestCase):
    def test_applique_les_seuils_et_preserve_les_autres_tenants(self):
        maintenant = datetime(2026, 5, 1, tzinfo=timezone.utc)
        client_id = ObjectId()
        autre_client = ObjectId()
        base = BaseMemoire()

        def ajouter(collection, document):
            base[collection].documents.append(document)

        ajouter("positions", {
            "client_id": client_id,
            "timestamp": maintenant - timedelta(days=9, seconds=1),
        })
        ajouter("positions", {
            "client_id": client_id,
            "timestamp": maintenant - timedelta(days=9),
        })
        ajouter("positions", {
            "client_id": autre_client,
            "timestamp": maintenant - timedelta(days=20),
        })
        ajouter("mesures_activite", {
            "client_id": client_id,
            "timestamp": maintenant - timedelta(days=9, seconds=1),
        })
        ajouter("alerts", {
            "client_id": client_id,
            "timestamp": maintenant - timedelta(hours=26, seconds=1),
            "resolue": False,
            "animal_id": "a1",
            "type": "test",
        })
        ajouter("alerts", {
            "client_id": client_id,
            "timestamp": maintenant - timedelta(hours=26),
            "resolue": True,
        })
        ajouter("lstm_alertes_silencieuses", {
            "client_id": client_id,
            "cree_le": maintenant - timedelta(hours=26, seconds=1),
        })
        ajouter("temps_travail_evenements", {
            "client_id": client_id,
            "horodatage_utc": maintenant - timedelta(days=30, seconds=1),
        })
        ajouter("temps_travail_evenements", {
            "client_id": client_id,
            "horodatage_utc": maintenant - timedelta(days=30),
        })
        ajouter("pastures", {
            "client_id": client_id,
            "active": False,
            "cree_le": maintenant - timedelta(hours=171),
        })
        ajouter("conversations", {
            "_id": ObjectId(),
            "client_id": client_id,
            "updated_at": maintenant - timedelta(days=3, seconds=1),
        })
        conversation_active = ObjectId()
        ajouter("conversations", {
            "_id": conversation_active,
            "client_id": client_id,
            "updated_at": maintenant - timedelta(days=3),
        })
        ajouter("messages", {
            "conversation_id": base["conversations"].documents[-2]["_id"],
        })
        ajouter("messages", {"conversation_id": conversation_active})
        notifications = []

        bilan = nettoyer_donnees(
            base,
            client_id=client_id,
            maintenant=maintenant,
            notifier_alerte=lambda alerte, instant: notifications.append(
                (alerte, instant)
            ),
        )

        self.assertEqual(bilan, {
            "alertes": 1,
            "positions": 1,
            "mesures_activite": 1,
            "evenements_lstm": 1,
            "anciennes_zones": 0,
            "evenements_travail": 1,
            "conversations": 1,
            "messages": 1,
        })
        self.assertEqual(len(notifications), 1)
        self.assertEqual(len(base["positions"].documents), 2)
        self.assertEqual(len(base["alerts"].documents), 1)
        self.assertEqual(len(base["pastures"].documents), 1)
        self.assertEqual(len(base["messages"].documents), 1)
