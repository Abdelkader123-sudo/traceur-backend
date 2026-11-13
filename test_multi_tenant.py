import os
import unittest
import base64
import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("JWT_SECRET", "multitenant-test-secret-with-at-least-32-bytes")

import jwt
from bson import ObjectId
from fastapi import HTTPException
from pydantic import ValidationError

from auth import (
    Connexion,
    NouveauClient,
    NouveauCodeActivation,
    SECRET,
    _hacher,
    _migrer_client_existant,
    creer_dependance_courant,
    creer_routeur,
)
from messagerie import ConversationDirecte, creer_routeur_messagerie


class CollectionMemoire:
    def __init__(self):
        self.documents = []

    def create_index(self, *args, **kwargs):
        return None

    def index_information(self):
        return {}

    def _correspond(self, document, filtre):
        for cle, valeur in filtre.items():
            courant = document.get(cle)
            if isinstance(valeur, dict):
                if "$in" in valeur:
                    possibles = valeur["$in"]
                    if isinstance(courant, list):
                        if not any(item in courant for item in possibles):
                            return False
                    elif courant not in possibles:
                        return False
                elif "$ne" in valeur and courant == valeur["$ne"]:
                    return False
                elif "$exists" in valeur and (cle in document) != valeur["$exists"]:
                    return False
            elif isinstance(courant, list):
                if valeur not in courant:
                    return False
            elif courant != valeur:
                return False
        return True

    def find_one(self, filtre, projection=None):
        return next(
            (doc for doc in self.documents if self._correspond(doc, filtre)),
            None,
        )

    def find(self, filtre, projection=None):
        return [
            doc for doc in self.documents if self._correspond(doc, filtre)
        ]

    def insert_one(self, document):
        document["_id"] = document.get("_id", ObjectId())
        self.documents.append(document)
        return SimpleNamespace(inserted_id=document["_id"])

    def update_many(self, filtre, modification):
        documents = [
            document
            for document in self.documents
            if self._correspond(document, filtre)
        ]
        for document in documents:
            document.update(modification["$set"])
        return SimpleNamespace(modified_count=len(documents))


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


class IsolationClientTests(unittest.TestCase):
    def test_numero_client_facultatif_mais_valide_sil_est_fourni(self):
        donnees = {
            "nom": "Alice",
            "mot_de_passe": "mot-de-passe",
        }
        self.assertIsNone(Connexion(**donnees).numero_client)
        with self.assertRaises(ValidationError):
            Connexion(**donnees, numero_client="1234")
        self.assertEqual(
            Connexion(**donnees, numero_client="01234567").numero_client,
            "01234567",
        )

    def test_code_activation_est_unique_et_renvoie_un_qr_sans_secret_stocke(self):
        base = BaseMemoire()
        routeur = creer_routeur(base)
        creer_code = _route(routeur, "/auth/codes-activation", "POST")
        page = _route(routeur, "/admin/codes-activation", "GET")()
        self.assertEqual(page.headers["cache-control"], "no-store")
        precedent = os.environ.get("CLIENT_ENROLLMENT_ADMIN_KEY")
        os.environ["CLIENT_ENROLLMENT_ADMIN_KEY"] = "cle-admin-test"
        try:
            with self.assertRaises(HTTPException) as erreur:
                creer_code(
                    NouveauCodeActivation(duree_jours=1),
                    cle="mauvaise-cle",
                )
            self.assertEqual(erreur.exception.status_code, 403)
            resultat = creer_code(
                NouveauCodeActivation(duree_jours=1),
                cle="cle-admin-test",
            )
        finally:
            if precedent is None:
                os.environ.pop("CLIENT_ENROLLMENT_ADMIN_KEY", None)
            else:
                os.environ["CLIENT_ENROLLMENT_ADMIN_KEY"] = precedent

        self.assertEqual(resultat["expire_dans_jours"], 1)
        self.assertEqual(len(base["codes_activation"].documents), 1)
        code_stocke = base["codes_activation"].documents[0]
        self.assertNotIn("code_activation", code_stocke)
        self.assertEqual(
            code_stocke["code_hash"],
            hashlib.sha256(resultat["code_activation"].encode()).hexdigest(),
        )
        qr_svg = base64.b64decode(resultat["qr_svg_base64"])
        self.assertIn(b"<svg", qr_svg)

    def test_nouveau_client_ne_demande_plus_de_donnee_biometrique(self):
        client = NouveauClient(
            code_activation="code-de-test",
            nom="Admin",
            mot_de_passe="mot-de-passe",
            ferme="Ferme principale",
        )
        self.assertNotIn("appareil_biometrique", client.model_dump())

    def test_migration_rattache_les_donnees_au_client_historique(self):
        base = BaseMemoire()
        users = base["utilisateurs"]
        admin = {"nom": "Ferme A", "role": "superadmin"}
        utilisateur = {"nom": "Employé", "role": "utilisateur"}
        animal = {"animal_id": "collier-1"}
        position = {"animal_id": "collier-1"}
        users.insert_one(admin)
        users.insert_one(utilisateur)
        base["animaux"].insert_one(animal)
        base["positions"].insert_one(position)

        _migrer_client_existant(base, users, base["clients"])

        self.assertEqual(admin["client_id"], utilisateur["client_id"])
        self.assertEqual(admin["numero_client"], animal["numero_client"])
        self.assertEqual(animal["ferme"], "Ferme principale")
        self.assertEqual(position["client_id"], admin["client_id"])

    def test_connexion_demande_empreinte_sur_un_nouvel_appareil(self):
        base = BaseMemoire()
        utilisateur = {
            "_id": ObjectId(),
            "nom": "Alice",
            "nom_cle": "alice",
            "role": "superadmin",
            "client_id": ObjectId(),
            "numero_client": "00000001",
            "appareils_biometriques": ["appareil-inscription"],
            "mot_de_passe": _hacher("mot-de-passe"),
        }
        base["utilisateurs"].insert_one(utilisateur)
        routeur = creer_routeur(base)
        connexion = _route(routeur, "/auth/connexion", "POST")

        session = connexion(Connexion(
            numero_client="00000001",
            nom="Alice",
            mot_de_passe="mot-de-passe",
        ))

        self.assertTrue(session["verification_biometrique_requise"])

    def test_jwt_ne_peut_pas_changer_de_client(self):
        client_id = ObjectId()
        utilisateur = {
            "_id": ObjectId(),
            "role": "superadmin",
            "client_id": client_id,
            "numero_client": "00001234",
        }
        users = CollectionMemoire()
        users.insert_one(utilisateur)
        courant = creer_dependance_courant(users)
        token = jwt.encode(
            {
                "sub": str(utilisateur["_id"]),
                "client_id": str(client_id),
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            SECRET,
            algorithm="HS256",
        )
        self.assertIs(courant(f"Bearer {token}"), utilisateur)

        mauvais_token = jwt.encode(
            {
                "sub": str(utilisateur["_id"]),
                "client_id": str(ObjectId()),
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            SECRET,
            algorithm="HS256",
        )
        with self.assertRaises(HTTPException) as erreur:
            courant(f"Bearer {mauvais_token}")
        self.assertEqual(erreur.exception.status_code, 401)

    def test_discussion_interclient_refusee(self):
        base = BaseMemoire()
        routeur = creer_routeur_messagerie(base)
        client_a, client_b = ObjectId(), ObjectId()
        alice = {
            "_id": ObjectId(),
            "nom": "Alice",
            "prenom": "",
            "role": "superadmin",
            "client_id": client_a,
            "numero_client": "00000001",
        }
        bob = {
            "_id": ObjectId(),
            "nom": "Bob",
            "prenom": "",
            "role": "superadmin",
            "client_id": client_b,
            "numero_client": "00000002",
        }
        base["utilisateurs"].insert_one(alice)
        base["utilisateurs"].insert_one(bob)
        ouvrir = _route(routeur, "/messagerie/conversations/direct", "POST")

        with self.assertRaises(HTTPException) as erreur:
            ouvrir(
                ConversationDirecte(destinataire_id=str(bob["_id"])),
                utilisateur=alice,
            )
        self.assertEqual(erreur.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
