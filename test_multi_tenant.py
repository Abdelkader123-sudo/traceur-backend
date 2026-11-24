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
from pymongo import ReturnDocument

from auth import (
    Connexion,
    JetonSession,
    ModifierProfilClient,
    NouveauClient,
    NouveauCodeActivation,
    NouvelUtilisateur,
    PermissionsUtilisateur,
    SECRET,
    _hachage_cin,
    _hacher,
    _migrer_client_existant,
    _verifier,
    creer_dependance_courant,
    creer_routeur,
)
from messagerie import ConversationDirecte, creer_routeur_messagerie


class CollectionMemoire:
    def __init__(self):
        self.documents = []
        self.indexes = {}
        self.indexes_crees = []
        self.indexes_supprimes = []

    def create_index(self, *args, **kwargs):
        self.indexes_crees.append((args, kwargs))
        return kwargs.get("name", "index_test")

    def index_information(self):
        return dict(self.indexes)

    def drop_index(self, nom):
        self.indexes_supprimes.append(nom)
        self.indexes.pop(nom, None)

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

    def find_one_and_update(self, filtre, modification, return_document=None):
        document = self.find_one(filtre)
        if not document:
            return None
        avant = dict(document)
        document.update(modification["$set"])
        return document if return_document == ReturnDocument.AFTER else avant

    def delete_one(self, filtre):
        document = self.find_one(filtre)
        if not document:
            return SimpleNamespace(deleted_count=0)
        self.documents.remove(document)
        return SimpleNamespace(deleted_count=1)

    def update_one(self, filtre, modification):
        document = self.find_one(filtre)
        if not document:
            return SimpleNamespace(modified_count=0)
        document.update(modification["$set"])
        return SimpleNamespace(modified_count=1)

    def delete_many(self, filtre):
        documents = [
            document
            for document in self.documents
            if self._correspond(document, filtre)
        ]
        for document in documents:
            self.documents.remove(document)
        return SimpleNamespace(deleted_count=len(documents))


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
        self.assertEqual(session["utilisateur"]["role"], "client")

    def test_connexion_par_mot_de_passe_dans_la_ferme_associee(self):
        base = BaseMemoire()
        utilisateur = {
            "_id": ObjectId(),
            "nom": "Alice",
            "nom_cle": "alice",
            "role": "utilisateur",
            "client_id": ObjectId(),
            "numero_client": "00000001",
            "mot_de_passe": _hacher("mot-de-passe"),
        }
        base["utilisateurs"].insert_one(utilisateur)
        connexion = _route(creer_routeur(base), "/auth/connexion", "POST")

        session = connexion(Connexion(
            numero_client="00000001",
            mot_de_passe="mot-de-passe",
        ))

        self.assertEqual(session["utilisateur"]["id"], str(utilisateur["_id"]))

    def test_connexion_sans_code_ferme_retrouve_le_compte_par_cin(self):
        base = BaseMemoire()
        utilisateur = {
            "_id": ObjectId(),
            "nom": "Alice",
            "nom_cle": "alice",
            "role": "utilisateur",
            "client_id": ObjectId(),
            "numero_client": "00000001",
            "mot_de_passe": _hacher("12345678"),
            "cin_hash": _hachage_cin("12345678"),
        }
        base["utilisateurs"].insert_one(utilisateur)
        connexion = _route(creer_routeur(base), "/auth/connexion", "POST")

        session = connexion(Connexion(mot_de_passe="12345678"))

        self.assertEqual(session["utilisateur"]["id"], str(utilisateur["_id"]))
        self.assertEqual(session["utilisateur"]["numero_client"], "00000001")

    def test_connexion_sans_code_migre_le_cin_d_un_ancien_compte(self):
        base = BaseMemoire()
        utilisateur = {
            "_id": ObjectId(),
            "nom": "Alice",
            "nom_cle": "alice",
            "role": "utilisateur",
            "client_id": ObjectId(),
            "numero_client": "00000001",
            "mot_de_passe": _hacher("12345678"),
        }
        base["utilisateurs"].insert_one(utilisateur)
        connexion = _route(creer_routeur(base), "/auth/connexion", "POST")

        session = connexion(Connexion(mot_de_passe="12345678"))

        self.assertEqual(session["utilisateur"]["id"], str(utilisateur["_id"]))
        self.assertEqual(
            utilisateur["cin_hash"],
            _hachage_cin("12345678"),
        )

    def test_connexion_sans_code_refuse_un_cin_ambigu(self):
        base = BaseMemoire()
        for numero, nom in (("00000001", "Alice"), ("00000002", "Bob")):
            base["utilisateurs"].insert_one({
                "_id": ObjectId(),
                "nom": nom,
                "nom_cle": nom.lower(),
                "role": "utilisateur",
                "client_id": ObjectId(),
                "numero_client": numero,
                "mot_de_passe": _hacher("12345678"),
                "cin_hash": _hachage_cin("12345678"),
            })
        connexion = _route(creer_routeur(base), "/auth/connexion", "POST")

        with self.assertRaises(HTTPException) as erreur:
            connexion(Connexion(mot_de_passe="12345678"))

        self.assertEqual(erreur.exception.status_code, 409)

    def test_meme_mot_de_passe_ouvre_le_compte_de_chaque_ferme(self):
        base = BaseMemoire()
        utilisateur_a = {
            "_id": ObjectId(),
            "nom": "Alice",
            "prenom": "A",
            "role": "utilisateur",
            "client_id": ObjectId(),
            "numero_client": "00000001",
            "mot_de_passe": _hacher("mot-partage"),
        }
        utilisateur_b = {
            "_id": ObjectId(),
            "nom": "Bob",
            "prenom": "B",
            "role": "utilisateur",
            "client_id": ObjectId(),
            "numero_client": "00000002",
            "mot_de_passe": _hacher("mot-partage"),
        }
        base["utilisateurs"].insert_one(utilisateur_a)
        base["utilisateurs"].insert_one(utilisateur_b)
        connexion = _route(creer_routeur(base), "/auth/connexion", "POST")

        session_a = connexion(
            Connexion(numero_client="00000001", mot_de_passe="mot-partage")
        )
        session_b = connexion(
            Connexion(numero_client="00000002", mot_de_passe="mot-partage")
        )

        self.assertEqual(session_a["utilisateur"]["id"], str(utilisateur_a["_id"]))
        self.assertEqual(session_b["utilisateur"]["id"], str(utilisateur_b["_id"]))

    def test_connexion_sans_cin_correspondant_est_refusee(self):
        connexion = _route(
            creer_routeur(BaseMemoire()),
            "/auth/connexion",
            "POST",
        )
        with self.assertRaises(HTTPException) as erreur:
            connexion(Connexion(mot_de_passe="mot-de-passe"))
        self.assertEqual(erreur.exception.status_code, 401)

    def test_creation_et_modification_des_pages_autorisees(self):
        base = BaseMemoire()
        routeur = creer_routeur(base)
        client = {
            "_id": ObjectId(),
            "role": "superadmin",
            "client_id": ObjectId(),
            "numero_client": "00000001",
        }
        creer = _route(routeur, "/utilisateurs", "POST")
        utilisateur = creer(
            NouvelUtilisateur(
                nom="Alice",
                prenom="Test",
                residence="Rabat",
                telephone="0612345678",
                cin="12345678",
                pages_autorisees=[
                    "map",
                    "notifications",
                    "health",
                    "worktime",
                    "animal_sales",
                ],
            ),
            u=client,
        )
        self.assertEqual(utilisateur["role"], "utilisateur")
        self.assertEqual(
            utilisateur["pages_autorisees"],
            ["map", "notifications", "health", "worktime", "animal_sales"],
        )
        self.assertNotIn("age", utilisateur)

        modifier = _route(
            routeur,
            "/utilisateurs/{user_id}/permissions",
            "PATCH",
        )
        resultat = modifier(
            utilisateur["id"],
            PermissionsUtilisateur(
                pages_autorisees=[
                    "map",
                    "alerts",
                    "notifications",
                    "animal_sales",
                    "worktime",
                ]
            ),
            u=client,
        )
        self.assertEqual(
            resultat["pages_autorisees"],
            ["map", "alerts", "notifications", "animal_sales", "worktime"],
        )

    def test_mot_de_passe_reutilisable_entre_fermes_mais_unique_dans_la_ferme(self):
        base = BaseMemoire()
        creer = _route(creer_routeur(base), "/utilisateurs", "POST")
        client_a_id = ObjectId()
        client_b_id = ObjectId()
        client_a = {
            "_id": ObjectId(),
            "role": "superadmin",
            "client_id": client_a_id,
            "numero_client": "00000001",
        }
        client_b = {
            "_id": ObjectId(),
            "role": "superadmin",
            "client_id": client_b_id,
            "numero_client": "00000002",
        }
        premier = creer(
            NouvelUtilisateur(
                nom="Alice",
                prenom="Test",
                residence="Rabat",
                telephone="0612345678",
                cin="12345678",
                pages_autorisees=["map"],
            ),
            u=client_a,
        )
        enregistre = base["utilisateurs"].find_one({"_id": ObjectId(premier["id"])})
        self.assertEqual(enregistre["cin_hash"], _hachage_cin("12345678"))
        self.assertNotIn("cin", enregistre)

        second = creer(
            NouvelUtilisateur(
                nom="Bob",
                prenom="Test",
                residence="Tunis",
                telephone="0612345679",
                cin="12345678",
                pages_autorisees=["map"],
            ),
            u=client_b,
        )
        self.assertNotEqual(premier["id"], second["id"])
        self.assertEqual(
            base["utilisateurs"].find_one(
                {"_id": ObjectId(second["id"])}
            )["cin_hash"],
            _hachage_cin("12345678"),
        )

        with self.assertRaises(HTTPException) as erreur:
            creer(
                NouvelUtilisateur(
                    nom="Charlie",
                    prenom="Test",
                    residence="Rabat",
                    telephone="0612345680",
                    cin="12345678",
                    pages_autorisees=["map"],
                ),
                u=client_a,
            )
        self.assertEqual(erreur.exception.status_code, 409)
        self.assertIn("cette ferme", erreur.exception.detail)

    def test_remplace_lindex_global_des_mots_de_passe_par_un_index_par_ferme(self):
        base = BaseMemoire()
        users = base["utilisateurs"]
        users.indexes["cin_hash_1"] = {
            "key": [("cin_hash", 1)],
            "unique": True,
        }

        creer_routeur(base)

        self.assertIn("cin_hash_1", users.indexes_supprimes)
        index_scoped = next(
            kwargs
            for args, kwargs in users.indexes_crees
            if args[0] == [("client_id", 1), ("cin_hash", 1)]
        )
        self.assertTrue(index_scoped["unique"])

    def test_cin_deja_present_dans_un_compte_historique_est_reconnu(self):
        base = BaseMemoire()
        users = base["utilisateurs"]
        ancien_id = ObjectId()
        client_id = ObjectId()
        users.insert_one({
            "_id": ancien_id,
            "role": "utilisateur",
            "client_id": client_id,
            "mot_de_passe": _hacher("87654321"),
        })
        creer = _route(creer_routeur(base), "/utilisateurs", "POST")

        with self.assertRaises(HTTPException) as erreur:
            creer(
                NouvelUtilisateur(
                    nom="Nouveau",
                    prenom="Compte",
                    residence="Rabat",
                    telephone="0612345678",
                    cin="87654321",
                    pages_autorisees=["map"],
                ),
                u={
                    "_id": ObjectId(),
                    "role": "superadmin",
                    "client_id": client_id,
                    "numero_client": "00000001",
                },
            )

        self.assertEqual(erreur.exception.status_code, 409)
        self.assertEqual(
            users.find_one({"_id": ancien_id})["cin_hash"],
            _hachage_cin("87654321"),
        )

    def test_le_client_modifie_son_profil_et_le_changement_de_mot_de_passe(self):
        base = BaseMemoire()
        routeur = creer_routeur(base)
        client_id = ObjectId()
        utilisateur = {
            "_id": ObjectId(),
            "role": "superadmin",
            "est_superadmin": True,
            "client_id": client_id,
            "numero_client": "00000001",
            "nom": "Ancien nom",
            "prenom": "",
            "nom_cle": "ancien nom",
            "prenom_cle": "",
            "telephone": "",
            "residence": "",
            "mot_de_passe": _hacher("ancien-mot-de-passe"),
        }
        base["utilisateurs"].insert_one(utilisateur)
        base["clients"].insert_one({
            "_id": client_id,
            "numero_client": "00000001",
            "nom": "Ancien nom",
        })
        session_courante = ObjectId()
        session_autre = ObjectId()
        base["sessions_authentification"].insert_one({
            "_id": session_courante,
            "utilisateur_id": utilisateur["_id"],
        })
        base["sessions_authentification"].insert_one({
            "_id": session_autre,
            "utilisateur_id": utilisateur["_id"],
        })
        jeton = jwt.encode(
            {
                "sub": str(utilisateur["_id"]),
                "sid": str(session_courante),
                "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            },
            SECRET,
            algorithm="HS256",
        )

        modifier = _route(routeur, "/auth/profil", "PATCH")
        resultat = modifier(
            ModifierProfilClient(
                nom="Nouveau nom",
                telephone="0612345678",
                residence="Rabat",
                mot_de_passe_actuel="ancien-mot-de-passe",
                nouveau_mot_de_passe="nouveau-mot-de-passe",
            ),
            u=utilisateur,
            authorization=f"Bearer {jeton}",
        )

        self.assertEqual(resultat["nom"], "Nouveau nom")
        self.assertEqual(resultat["telephone"], "0612345678")
        self.assertEqual(resultat["residence"], "Rabat")
        self.assertTrue(
            _verifier(
                "nouveau-mot-de-passe",
                base["utilisateurs"].find_one({"_id": utilisateur["_id"]})[
                    "mot_de_passe"
                ],
            )
        )
        self.assertEqual(
            base["clients"].find_one({"_id": client_id})["nom"],
            "Nouveau nom",
        )
        self.assertIsNotNone(
            base["sessions_authentification"].find_one({
                "_id": session_courante,
            })
        )
        self.assertIsNone(
            base["sessions_authentification"].find_one({
                "_id": session_autre,
            })
        )

    def test_un_utilisateur_ne_peut_pas_modifier_le_profil_du_client(self):
        routeur = creer_routeur(BaseMemoire())
        modifier = _route(routeur, "/auth/profil", "PATCH")
        with self.assertRaises(HTTPException) as erreur:
            modifier(
                ModifierProfilClient(nom="Nouveau nom"),
                u={"role": "utilisateur"},
                authorization=None,
            )
        self.assertEqual(erreur.exception.status_code, 403)

    def test_renouvellement_fait_pivoter_jeton_et_deconnexion_le_revoque(self):
        base = BaseMemoire()
        utilisateur = {
            "_id": ObjectId(),
            "nom": "Alice",
            "nom_cle": "alice",
            "role": "superadmin",
            "client_id": ObjectId(),
            "numero_client": "00000001",
            "mot_de_passe": _hacher("mot-de-passe"),
        }
        base["utilisateurs"].insert_one(utilisateur)
        routeur = creer_routeur(base)
        authentifier = _route(routeur, "/auth/connexion", "POST")
        resultat = authentifier(Connexion(
            numero_client="00000001",
            nom="Alice",
            mot_de_passe="mot-de-passe",
        ))
        renouveler = _route(routeur, "/auth/renouveler", "POST")
        deconnecter = _route(routeur, "/auth/deconnexion", "POST")

        renouvele = renouveler(JetonSession(
            refresh_token=resultat["refresh_token"],
        ))
        self.assertNotEqual(renouvele["refresh_token"], resultat["refresh_token"])
        with self.assertRaises(HTTPException):
            renouveler(JetonSession(refresh_token=resultat["refresh_token"]))

        courant = creer_dependance_courant(
            base["utilisateurs"],
            base["sessions_authentification"],
        )
        self.assertIs(
            courant(f"Bearer {renouvele['token']}"),
            utilisateur,
        )
        deconnecter(JetonSession(refresh_token=renouvele["refresh_token"]))
        with self.assertRaises(HTTPException) as erreur:
            courant(f"Bearer {renouvele['token']}")
        self.assertEqual(erreur.exception.status_code, 401)

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
