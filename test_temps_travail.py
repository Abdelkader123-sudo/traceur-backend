import os
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("JWT_SECRET", "temps-travail-test-secret-with-at-least-32-bytes")

from bson import ObjectId
from fastapi import HTTPException

from temps_travail import (
    EvenementTravail,
    RapportPresence,
    ZoneTravail,
    creer_routeur_temps_travail,
)


class CollectionMemoire:
    def __init__(self):
        self.documents = []
        self.indexes = []

    def create_index(self, *args, **kwargs):
        self.indexes.append((args, kwargs))

    def _correspond(self, document, filtre):
        for cle, valeur in filtre.items():
            courant = document.get(cle)
            if isinstance(valeur, dict):
                if "$ne" in valeur and courant == valeur["$ne"]:
                    return False
                if "$gte" in valeur and courant < valeur["$gte"]:
                    return False
                if "$lte" in valeur and courant > valeur["$lte"]:
                    return False
            elif courant != valeur:
                return False
        return True

    def find_one(self, filtre, projection=None):
        return next(
            (document for document in self.documents if self._correspond(document, filtre)),
            None,
        )

    def find(self, filtre, projection=None):
        return [
            document
            for document in self.documents
            if self._correspond(document, filtre)
        ]

    def insert_one(self, document):
        document["_id"] = ObjectId()
        self.documents.append(document)
        return SimpleNamespace(inserted_id=document["_id"])

    def update_one(self, filtre, modification, upsert=False):
        document = self.find_one(filtre)
        if document is None:
            if not upsert:
                return
            document = dict(filtre)
            self.documents.append(document)
        document.update(modification["$set"])

    def delete_one(self, filtre):
        document = self.find_one(filtre)
        if document is not None:
            self.documents.remove(document)


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


def _evenement(
    evenement_id,
    type_evenement,
    heure,
    *,
    journee_id="journee-1",
    tache_id=None,
    nom_tache=None,
):
    return EvenementTravail(
        evenement_id=evenement_id,
        type=type_evenement,
        journee_id=journee_id,
        tache_id=tache_id,
        nom_tache=nom_tache,
        horodatage_utc=f"2026-11-20T{heure}:00Z",
        date_locale="2026-11-20",
        decalage_minutes=60,
    )


class TempsTravailTests(unittest.TestCase):
    def setUp(self):
        self.base = BaseMemoire()
        self.client_id = ObjectId()
        self.utilisateur = {
            "_id": ObjectId(),
            "client_id": self.client_id,
            "numero_client": "00001",
            "role": "admin",
            "nom": "Employe",
            "prenom": "Ali",
        }
        self.proprietaire = {
            "_id": ObjectId(),
            "client_id": self.client_id,
            "numero_client": "00001",
            "role": "superadmin",
            "nom": "Client",
            "prenom": "",
        }
        self.base["utilisateurs"].documents.extend(
            [self.utilisateur, self.proprietaire]
        )
        self.routeur = creer_routeur_temps_travail(self.base)
        self.enregistrer = _route(
            self.routeur,
            "/temps-travail/evenements",
            "POST",
        )
        self.lire_etat = _route(
            self.routeur,
            "/temps-travail/etat",
            "GET",
        )
        self.statistiques = _route(
            self.routeur,
            "/admin/temps-travail",
            "GET",
        )
        self.configuration_presence = _route(
            self.routeur,
            "/temps-travail/presence/configuration",
            "GET",
        )
        self.definir_zone = _route(
            self.routeur,
            "/admin/temps-travail/presence/zone",
            "PUT",
        )
        self.enregistrer_presence = _route(
            self.routeur,
            "/temps-travail/presence",
            "POST",
        )
        self.presences_admin = _route(
            self.routeur,
            "/admin/temps-travail/presence",
            "GET",
        )

    def test_enregistre_les_evenements_de_maniere_idempotente_et_renvoie_etat(self):
        evenements = [
            _evenement("journee-debut", "journee_debut", "08:00"),
            _evenement(
                "tache-debut",
                "tache_debut",
                "08:15",
                tache_id="tache-1",
                nom_tache="Soins",
            ),
            _evenement(
                "tache-fin",
                "tache_fin",
                "08:45",
                tache_id="tache-1",
                nom_tache="Soins",
            ),
            _evenement("journee-fin", "journee_fin", "09:00"),
        ]
        for evenement in evenements:
            self.assertEqual(
                self.enregistrer(evenement, self.utilisateur),
                {"evenement_id": evenement.evenement_id},
            )
        self.enregistrer(evenements[0], self.utilisateur)

        self.assertEqual(len(self.base["temps_travail_evenements"].documents), 4)
        etat = self.lire_etat(self.utilisateur)
        self.assertIsNone(etat["journee_active"])
        self.assertEqual(etat["journees_du_jour"][0]["duree_secondes"], 3600)
        self.assertEqual(etat["taches_du_jour"][0]["duree_secondes"], 1800)

    def test_refuse_les_transitions_invalides_sans_les_enregistrer(self):
        with self.assertRaises(HTTPException) as erreur:
            self.enregistrer(
                _evenement(
                    "tache-fin",
                    "tache_fin",
                    "08:45",
                    tache_id="tache-1",
                ),
                self.utilisateur,
            )
        self.assertEqual(erreur.exception.status_code, 409)
        self.assertEqual(self.base["temps_travail_evenements"].documents, [])

    def test_repare_une_journee_active_lorsqu_une_nouvelle_commence(self):
        evenements = [
            _evenement("journee-debut-1", "journee_debut", "08:00"),
            _evenement(
                "tache-debut-1",
                "tache_debut",
                "08:15",
                tache_id="tache-1",
                nom_tache="Soins",
            ),
            _evenement("journee-debut-2", "journee_debut", "09:00", journee_id="journee-2"),
            _evenement(
                "tache-fin-retardee",
                "tache_fin",
                "09:05",
                tache_id="tache-1",
            ),
            _evenement(
                "journee-fin-retardee",
                "journee_fin",
                "09:10",
                journee_id="journee-1",
            ),
            _evenement(
                "journee-fin-2",
                "journee_fin",
                "10:00",
                journee_id="journee-2",
            ),
        ]
        for evenement in evenements:
            self.enregistrer(evenement, self.utilisateur)

        etat = self.lire_etat(self.utilisateur)
        self.assertIsNone(etat["journee_active"])
        self.assertEqual(
            [jour["duree_secondes"] for jour in etat["journees_du_jour"]],
            [3600, 3600],
        )
        self.assertEqual(etat["taches_du_jour"][0]["duree_secondes"], 2700)

    def test_statistiques_admin_incluent_les_jours_sans_activite(self):
        for evenement in [
            _evenement("journee-debut", "journee_debut", "08:00"),
            _evenement("journee-fin", "journee_fin", "09:00"),
        ]:
            self.enregistrer(evenement, self.utilisateur)

        resultat = self.statistiques(
            debut=date(2026, 11, 19),
            fin=date(2026, 11, 20),
            utilisateur=self.proprietaire,
        )

        utilisateur = resultat["utilisateurs"][0]
        self.assertEqual(utilisateur["nom"], "Ali Employe")
        self.assertEqual(
            [jour["duree_secondes"] for jour in utilisateur["journees"]],
            [0, 3600],
        )

    def test_statistiques_ne_comptent_pas_une_journee_active_apres_minuit(self):
        self.enregistrer(
            _evenement("journee-debut", "journee_debut", "08:00"),
            self.utilisateur,
        )
        self.enregistrer(
            _evenement(
                "tache-debut",
                "tache_debut",
                "08:15",
                tache_id="tache-1",
                nom_tache="Soins",
            ),
            self.utilisateur,
        )

        resultat = self.statistiques(
            debut=date(2026, 11, 20),
            fin=date(2026, 11, 21),
            utilisateur=self.proprietaire,
        )

        journees = resultat["utilisateurs"][0]["journees"]
        self.assertEqual(journees[0]["duree_secondes"], 15 * 60 * 60)
        self.assertEqual(
            journees[0]["taches"][0]["duree_secondes"],
            14 * 60 * 60 + 45 * 60,
        )
        self.assertEqual(journees[1]["duree_secondes"], 0)

    def test_statistiques_bornent_une_journee_terminee_au_minuit_local(self):
        evenements = [
            _evenement("journee-debut", "journee_debut", "08:00"),
            _evenement(
                "tache-debut",
                "tache_debut",
                "08:15",
                tache_id="tache-1",
                nom_tache="Soins",
            ),
            EvenementTravail(
                evenement_id="journee-fin",
                type="journee_fin",
                journee_id="journee-1",
                horodatage_utc=datetime(2026, 11, 22, 10, tzinfo=timezone.utc),
                date_locale="2026-11-20",
                decalage_minutes=60,
            ),
        ]
        for evenement in evenements:
            self.enregistrer(evenement, self.utilisateur)

        resultat = self.statistiques(
            debut=date(2026, 11, 20),
            fin=date(2026, 11, 20),
            utilisateur=self.proprietaire,
        )

        journee = resultat["utilisateurs"][0]["journees"][0]
        self.assertEqual(journee["duree_secondes"], 15 * 60 * 60)
        self.assertEqual(
            journee["taches"][0]["duree_secondes"],
            14 * 60 * 60 + 45 * 60,
        )

    def test_statistiques_sont_reservees_au_proprietaire(self):
        with self.assertRaises(HTTPException) as erreur:
            self.statistiques(
                debut=date(2026, 11, 19),
                fin=date(2026, 11, 20),
                utilisateur=self.utilisateur,
            )
        self.assertEqual(erreur.exception.status_code, 403)

    def test_presence_admin_est_verifiee_dans_la_zone_pour_une_journee_active(self):
        maintenant = datetime.now(timezone.utc)
        date_locale = maintenant.date().isoformat()
        self.enregistrer(
            EvenementTravail(
                evenement_id="journee-presence",
                type="journee_debut",
                journee_id="journee-presence",
                horodatage_utc=maintenant,
                date_locale=date_locale,
                decalage_minutes=0,
            ),
            self.utilisateur,
        )
        self.definir_zone(
            ZoneTravail(latitude=36.8, longitude=10.1, rayon_m=100),
            self.proprietaire,
        )

        rapport = self.enregistrer_presence(
            RapportPresence(
                journee_id="journee-presence",
                latitude=36.8,
                longitude=10.1,
                precision_m=5,
            ),
            self.utilisateur,
        )
        resultat = self.presences_admin(self.proprietaire)

        self.assertTrue(rapport["presence_verifiable"])
        self.assertTrue(rapport["dans_ferme"])
        utilisateur = resultat["utilisateurs"][0]
        self.assertEqual(utilisateur["statut"], "dans_ferme")
        self.assertTrue(utilisateur["presence_verifiee"])
        self.assertEqual(utilisateur["latitude"], 36.8)
        self.assertEqual(utilisateur["longitude"], 10.1)

    def test_presence_hors_zone_est_signalee_avec_sa_position(self):
        maintenant = datetime.now(timezone.utc)
        self.enregistrer(
            EvenementTravail(
                evenement_id="journee-presence",
                type="journee_debut",
                journee_id="journee-presence",
                horodatage_utc=maintenant,
                date_locale=maintenant.date().isoformat(),
                decalage_minutes=0,
            ),
            self.utilisateur,
        )
        self.definir_zone(
            ZoneTravail(latitude=36.8, longitude=10.1, rayon_m=100),
            self.proprietaire,
        )
        self.enregistrer_presence(
            RapportPresence(
                journee_id="journee-presence",
                latitude=36.803,
                longitude=10.1,
                precision_m=5,
            ),
            self.utilisateur,
        )

        utilisateur = self.presences_admin(self.proprietaire)["utilisateurs"][0]
        self.assertEqual(utilisateur["statut"], "hors_ferme")
        self.assertTrue(utilisateur["presence_verifiee"])

    def test_position_a_la_limite_ne_confirme_pas_injustement_la_zone(self):
        maintenant = datetime.now(timezone.utc)
        self.enregistrer(
            EvenementTravail(
                evenement_id="journee-presence",
                type="journee_debut",
                journee_id="journee-presence",
                horodatage_utc=maintenant,
                date_locale=maintenant.date().isoformat(),
                decalage_minutes=0,
            ),
            self.utilisateur,
        )
        self.definir_zone(
            ZoneTravail(latitude=36.8, longitude=10.1, rayon_m=100),
            self.proprietaire,
        )
        self.enregistrer_presence(
            RapportPresence(
                journee_id="journee-presence",
                latitude=36.8009,
                longitude=10.1,
                precision_m=10,
            ),
            self.utilisateur,
        )

        utilisateur = self.presences_admin(self.proprietaire)["utilisateurs"][0]
        self.assertEqual(utilisateur["statut"], "precision_insuffisante")
        self.assertFalse(utilisateur["presence_verifiee"])

    def test_presence_refusee_sans_journee_active_et_zone_reservee_au_proprietaire(self):
        with self.assertRaises(HTTPException) as erreur:
            self.enregistrer_presence(
                RapportPresence(
                    journee_id="journee-inexistante",
                    latitude=36.8,
                    longitude=10.1,
                    precision_m=5,
                ),
                self.utilisateur,
            )
        self.assertEqual(erreur.exception.status_code, 409)

        with self.assertRaises(HTTPException) as erreur:
            self.definir_zone(
                ZoneTravail(latitude=36.8, longitude=10.1, rayon_m=100),
                self.utilisateur,
            )
        self.assertEqual(erreur.exception.status_code, 403)
        with self.assertRaises(HTTPException) as erreur:
            self.presences_admin(self.utilisateur)
        self.assertEqual(erreur.exception.status_code, 403)

    def test_position_simulee_ne_peut_pas_confirmer_la_presence(self):
        maintenant = datetime.now(timezone.utc)
        self.enregistrer(
            EvenementTravail(
                evenement_id="journee-presence",
                type="journee_debut",
                journee_id="journee-presence",
                horodatage_utc=maintenant,
                date_locale=maintenant.date().isoformat(),
                decalage_minutes=0,
            ),
            self.utilisateur,
        )
        self.definir_zone(
            ZoneTravail(latitude=36.8, longitude=10.1, rayon_m=100),
            self.proprietaire,
        )
        self.enregistrer_presence(
            RapportPresence(
                journee_id="journee-presence",
                latitude=36.8,
                longitude=10.1,
                precision_m=5,
                est_simulee=True,
            ),
            self.utilisateur,
        )

        utilisateur = self.presences_admin(self.proprietaire)["utilisateurs"][0]
        self.assertEqual(utilisateur["statut"], "non_verifiee")
        self.assertFalse(utilisateur["presence_verifiee"])
        self.assertEqual(utilisateur["latitude"], 36.8)
        self.assertEqual(utilisateur["longitude"], 10.1)

    def test_presence_perimee_ou_imprecise_n_est_pas_confirmee(self):
        maintenant = datetime.now(timezone.utc)
        self.enregistrer(
            EvenementTravail(
                evenement_id="journee-presence",
                type="journee_debut",
                journee_id="journee-presence",
                horodatage_utc=maintenant,
                date_locale=maintenant.date().isoformat(),
                decalage_minutes=0,
            ),
            self.utilisateur,
        )
        self.definir_zone(
            ZoneTravail(latitude=36.8, longitude=10.1, rayon_m=100),
            self.proprietaire,
        )
        self.enregistrer_presence(
            RapportPresence(
                journee_id="journee-presence",
                latitude=36.8,
                longitude=10.1,
                precision_m=150,
            ),
            self.utilisateur,
        )

        utilisateur = self.presences_admin(self.proprietaire)["utilisateurs"][0]
        self.assertEqual(utilisateur["statut"], "precision_insuffisante")
        self.assertFalse(utilisateur["presence_verifiee"])

        presence = self.base["temps_travail_presence"].documents[0]
        presence["precision_fiable"] = True
        presence["dans_ferme"] = True
        presence["recu_le_utc"] = datetime.now(timezone.utc) - timedelta(
            minutes=21,
        )
        utilisateur = self.presences_admin(self.proprietaire)["utilisateurs"][0]
        self.assertEqual(utilisateur["statut"], "non_verifiee")
        self.assertFalse(utilisateur["presence_verifiee"])
        self.assertIsNone(utilisateur["latitude"])
        self.assertIsNone(utilisateur["longitude"])

    def test_supervision_ne_retourne_que_les_utilisateurs_du_meme_client(self):
        self.base["utilisateurs"].documents.append(
            {
                "_id": ObjectId(),
                "client_id": ObjectId(),
                "numero_client": "00002",
                "role": "admin",
                "nom": "Autre",
                "prenom": "Client",
            }
        )

        resultat = self.presences_admin(self.proprietaire)

        self.assertEqual(
            [utilisateur["nom"] for utilisateur in resultat["utilisateurs"]],
            ["Ali Employe"],
        )

    def test_fin_de_journee_efface_le_dernier_rapport_de_presence(self):
        maintenant = datetime.now(timezone.utc)
        date_locale = maintenant.date().isoformat()
        self.enregistrer(
            EvenementTravail(
                evenement_id="journee-debut",
                type="journee_debut",
                journee_id="journee-presence",
                horodatage_utc=maintenant - timedelta(seconds=2),
                date_locale=date_locale,
                decalage_minutes=0,
            ),
            self.utilisateur,
        )
        self.definir_zone(
            ZoneTravail(latitude=36.8, longitude=10.1, rayon_m=100),
            self.proprietaire,
        )
        self.enregistrer_presence(
            RapportPresence(
                journee_id="journee-presence",
                latitude=36.8,
                longitude=10.1,
                precision_m=5,
            ),
            self.utilisateur,
        )

        self.enregistrer(
            EvenementTravail(
                evenement_id="journee-fin",
                type="journee_fin",
                journee_id="journee-presence",
                horodatage_utc=maintenant - timedelta(seconds=1),
                date_locale=date_locale,
                decalage_minutes=0,
            ),
            self.utilisateur,
        )

        self.assertEqual(self.base["temps_travail_presence"].documents, [])


if __name__ == "__main__":
    unittest.main()
