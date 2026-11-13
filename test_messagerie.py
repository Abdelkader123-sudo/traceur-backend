import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "messagerie-test-secret-with-at-least-32-bytes")

import jwt
from bson import ObjectId
from fastapi import BackgroundTasks, HTTPException
from pymongo.errors import DuplicateKeyError

from auth import SECRET, creer_dependance_courant
from messagerie import (
    ConversationDirecte,
    NouveauGroupe,
    NouveauMessage,
    NouveauxMembres,
    InstallationFCM,
    creer_routeur_messagerie,
)


class Curseur(list):
    def sort(self, key, direction=None):
        if isinstance(key, list):
            for champ, sens in reversed(key):
                super().sort(key=lambda document: document.get(champ), reverse=sens < 0)
        else:
            super().sort(key=lambda document: document.get(key), reverse=direction == -1)
        return self

    def limit(self, limite):
        return Curseur(self[:limite])


class CollectionMemoire:
    def __init__(self):
        self.documents = []
        self.index_uniques = set()

    def create_index(self, *args, **kwargs):
        if kwargs.get("unique"):
            self.index_uniques.add(kwargs.get("name", "direct_key"))

    def _correspond(self, document, filtre):
        for cle, valeur in filtre.items():
            courant = document.get(cle)
            if isinstance(valeur, dict):
                if "$in" in valeur:
                    options = valeur["$in"]
                    if isinstance(courant, list):
                        if not any(element in courant for element in options):
                            return False
                    elif courant not in options:
                        return False
                elif "$exists" in valeur and (cle in document) != valeur["$exists"]:
                    return False
                elif "$ne" in valeur and courant == valeur["$ne"]:
                    return False
            elif isinstance(courant, list):
                if valeur not in courant:
                    return False
            elif courant != valeur:
                return False
        return True

    def find_one(self, filtre, projection=None):
        return next(
            (document for document in self.documents
             if self._correspond(document, filtre)),
            None,
        )

    def find(self, filtre, projection=None):
        return Curseur(
            document for document in self.documents
            if self._correspond(document, filtre)
        )

    def insert_one(self, document):
        if document.get("direct_key") and any(
            actuel.get("direct_key") == document["direct_key"]
            for actuel in self.documents
        ):
            raise DuplicateKeyError("direct key exists")
        document["_id"] = document.get("_id", ObjectId())
        self.documents.append(document)
        return SimpleNamespace(inserted_id=document["_id"])

    def update_one(self, filtre, changements):
        document = self.find_one(filtre)
        if document is None:
            return SimpleNamespace(modified_count=0)
        for cle, valeur in changements.get("$set", {}).items():
            document[cle] = valeur
        for cle, valeur in changements.get("$addToSet", {}).items():
            if isinstance(valeur, dict) and "$each" in valeur:
                ajouts = valeur["$each"]
            else:
                ajouts = [valeur]
            document.setdefault(cle, [])
            document[cle] = list(dict.fromkeys(document[cle] + ajouts))
        for cle, valeur in changements.get("$pull", {}).items():
            document[cle] = [element for element in document.get(cle, []) if element != valeur]
        return SimpleNamespace(modified_count=1)

    def update_many(self, filtre, changements):
        modifies = 0
        for document in self.documents:
            if not self._correspond(document, filtre):
                continue
            for cle, valeur in changements.get("$pull", {}).items():
                document[cle] = [
                    element for element in document.get(cle, []) if element != valeur
                ]
            modifies += 1
        return SimpleNamespace(modified_count=modifies)


class BaseMemoire:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, nom):
        return self.collections.setdefault(nom, CollectionMemoire())


def _route(routeur, chemin, methode):
    return next(
        route.endpoint
        for route in routeur.routes
        if route.path == chemin and methode in route.methods
    )


def _utilisateur(nom):
    return {
        "_id": ObjectId(),
        "nom": nom,
        "prenom": "",
        "role": "utilisateur",
        "client_id": "client-test",
        "numero_client": "00001",
    }


class MessagerieTests(unittest.TestCase):
    def test_routes_reutilisent_la_session_jwt_existante(self):
        users = CollectionMemoire()
        alice = _utilisateur("Alice")
        users.insert_one(alice)
        courant = creer_dependance_courant(users)
        jeton = jwt.encode(
            {
                "sub": str(alice["_id"]),
                "role": alice["role"],
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            SECRET,
            algorithm="HS256",
        )

        self.assertIs(courant(f"Bearer {jeton}"), alice)
        with self.assertRaises(HTTPException) as erreur:
            courant(None)
        self.assertEqual(erreur.exception.status_code, 401)

    def test_discussion_directe_unique_et_messages_prives(self):
        base = BaseMemoire()
        routeur = creer_routeur_messagerie(base)
        alice, bob, charlie = map(_utilisateur, ("Alice", "Bob", "Charlie"))
        users = base["utilisateurs"]
        for utilisateur in (alice, bob, charlie):
            users.insert_one(utilisateur)

        ouvrir = _route(routeur, "/messagerie/conversations/direct", "POST")
        premiere = ouvrir(
            ConversationDirecte(destinataire_id=str(bob["_id"])),
            utilisateur=alice,
        )
        seconde = ouvrir(
            ConversationDirecte(destinataire_id=str(alice["_id"])),
            utilisateur=bob,
        )
        self.assertEqual(premiere["id"], seconde["id"])
        self.assertEqual(premiere["titre"], "Bob")
        self.assertEqual(seconde["titre"], "Alice")

        envoyer = _route(
            routeur,
            "/messagerie/conversations/{conversation_id}/messages",
            "POST",
        )
        message = envoyer(
            premiere["id"],
            NouveauMessage(texte="Bonjour"),
            background_tasks=BackgroundTasks(),
            utilisateur=alice,
        )
        lire = _route(
            routeur,
            "/messagerie/conversations/{conversation_id}/messages",
            "GET",
        )
        self.assertEqual(
            lire(premiere["id"], limite=100, utilisateur=bob),
            [message],
        )
        with self.assertRaises(HTTPException) as erreur:
            lire(premiere["id"], limite=100, utilisateur=charlie)
        self.assertEqual(erreur.exception.status_code, 404)

    def test_groupe_et_refus_message_vide(self):
        base = BaseMemoire()
        routeur = creer_routeur_messagerie(base)
        alice, bob, charlie = map(_utilisateur, ("Alice", "Bob", "Charlie"))
        users = base["utilisateurs"]
        for utilisateur in (alice, bob, charlie):
            users.insert_one(utilisateur)

        creer = _route(routeur, "/messagerie/conversations/groupe", "POST")
        groupe = creer(
            NouveauGroupe(
                titre="Équipe",
                membres=[str(bob["_id"])],
            ),
            utilisateur=alice,
        )
        self.assertEqual(groupe["type"], "groupe")
        self.assertEqual(groupe["titre"], "Équipe")
        self.assertEqual(
            {membre["id"] for membre in groupe["membres"]},
            {str(alice["_id"]), str(bob["_id"])},
        )

        envoyer = _route(
            routeur,
            "/messagerie/conversations/{conversation_id}/messages",
            "POST",
        )
        with self.assertRaises(HTTPException) as erreur:
            envoyer(
                groupe["id"],
                NouveauMessage(texte="   "),
                background_tasks=BackgroundTasks(),
                utilisateur=alice,
            )
        self.assertEqual(erreur.exception.status_code, 422)

        ajouter = _route(
            routeur,
            "/messagerie/conversations/{conversation_id}/membres",
            "POST",
        )
        actualise = ajouter(
            groupe["id"],
            NouveauxMembres(membres=[str(charlie["_id"])]),
            utilisateur=alice,
        )
        self.assertEqual(len(actualise["membres"]), 3)

    def test_identifiant_fcm_enregistre_et_est_supprime_a_la_deconnexion(self):
        base = BaseMemoire()
        routeur = creer_routeur_messagerie(base)
        utilisateur = _utilisateur("Alice")
        autre = _utilisateur("Bob")
        autre["fcm_installation_ids"] = ["installation-de-test"]
        base["utilisateurs"].insert_one(utilisateur)
        base["utilisateurs"].insert_one(autre)
        enregistrer = _route(routeur, "/messagerie/installation-fcm", "POST")
        supprimer = _route(routeur, "/messagerie/installation-fcm", "DELETE")
        jeton = InstallationFCM(identifiant="installation-de-test")

        with patch("messagerie.fcm_prete", return_value=True):
            enregistrer(jeton, utilisateur=utilisateur)
            enregistrer(jeton, utilisateur=utilisateur)
        self.assertEqual(
            base["utilisateurs"].find_one({"_id": utilisateur["_id"]})[
                "fcm_installation_ids"
            ],
            ["installation-de-test"],
        )
        self.assertEqual(autre["fcm_installation_ids"], [])

        supprimer(jeton, utilisateur=utilisateur)
        self.assertEqual(
            base["utilisateurs"].find_one({"_id": utilisateur["_id"]})[
                "fcm_installation_ids"
            ],
            [],
        )

    def test_refuse_l_activation_fcm_si_le_serveur_n_est_pas_configure(self):
        routeur = creer_routeur_messagerie(BaseMemoire())
        enregistrer = _route(routeur, "/messagerie/installation-fcm", "POST")
        with patch("messagerie.fcm_prete", return_value=False):
            with self.assertRaises(HTTPException) as erreur:
                enregistrer(
                    InstallationFCM(identifiant="installation-de-test"),
                    utilisateur=_utilisateur("Alice"),
                )
        self.assertEqual(erreur.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
