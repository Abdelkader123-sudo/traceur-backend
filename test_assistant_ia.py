import json
import urllib.request

import pytest
from bson import ObjectId
from fastapi import FastAPI
from fastapi.testclient import TestClient

from assistant_ia import (
    ErreurGemini,
    appeler_gemini,
    construire_statistiques,
    creer_routeur,
)


class FauxAnimaux:
    def __init__(self, fiches):
        self.fiches = fiches
        self.filtre = None
        self.projection = None

    def find(self, filtre, projection):
        self.filtre = filtre
        self.projection = projection
        return iter(self.fiches)


class FauxUtilisateurs:
    def __init__(self):
        self.filtres = []

    def count_documents(self, filtre):
        self.filtres.append(filtre)
        return 7 if "role" not in filtre else 6


def test_statistiques_limitees_a_la_ferme_et_sans_identifiants():
    client_id = ObjectId()
    animaux = FauxAnimaux(
        [
            {
                "_id": ObjectId(),
                "animal_id": "id-confidentiel",
                "client_id": client_id,
                "type": "Montbéliarde",
                "age_ans": 2,
            },
            {
                "_id": ObjectId(),
                "animal_id": "autre-id-confidentiel",
                "client_id": client_id,
                "type": "Barbarine (« Arbi »)",
                "age_ans": 0.5,
            },
            {
                "_id": ObjectId(),
                "animal_id": "id-type-libre",
                "client_id": client_id,
                "type": "texte arbitraire non classé",
                "age_ans": None,
            },
        ]
    )
    utilisateurs = FauxUtilisateurs()

    statistiques = construire_statistiques(animaux, utilisateurs, client_id)

    assert animaux.filtre == {"client_id": client_id}
    assert animaux.projection == {"_id": 0, "type": 1, "age_ans": 1}
    assert utilisateurs.filtres == [
        {"client_id": client_id},
        {"client_id": client_id, "role": {"$ne": "superadmin"}},
    ]
    assert statistiques["utilisateurs"] == {
        "total": 7,
        "employes_hors_proprietaire": 6,
    }
    assert statistiques["animaux"]["total"] == 3
    assert statistiques["animaux"]["par_espece"]["vaches"] == 1
    assert statistiques["animaux"]["par_espece"]["moutons"] == 1
    assert statistiques["animaux"]["par_espece"]["autres"] == 1
    assert statistiques["animaux"]["par_type"]["Montbéliarde"] == 1
    assert "id-confidentiel" not in json.dumps(statistiques)
    assert "texte arbitraire non classé" not in json.dumps(statistiques)


def test_route_repond_avec_statistiques_et_refuse_un_utilisateur_non_autorise(
    monkeypatch,
):
    client_id = ObjectId()
    utilisateurs = FauxUtilisateurs()
    animaux = FauxAnimaux([{"type": "Jersiaise", "age_ans": 4}])
    appels = []

    def generer(messages, cle_api, modele):
        appels.append((messages, cle_api, modele))
        return "La ferme compte une vache Jersiaise."

    utilisateur = {
        "client_id": client_id,
        "role": "utilisateur",
        "pages_autorisees": ["assistant"],
    }

    def courant():
        return utilisateur

    app = FastAPI()
    app.include_router(creer_routeur(animaux, utilisateurs, courant, generer))
    client = TestClient(app)
    monkeypatch.setenv("GEMINI_API_KEY", "cle-de-test")

    reponse = client.post(
        "/assistant/chat",
        json={"messages": [{"role": "user", "content": "Combien de vaches ?"}]},
    )

    assert reponse.status_code == 200
    assert reponse.json() == {"answer": "La ferme compte une vache Jersiaise."}
    assert appels[0][1:] == ("cle-de-test", "gemini-3.5-flash-lite")
    assert '"total": 1' in appels[0][0][1]["content"]

    utilisateur["pages_autorisees"] = ["animals"]
    refusee = client.post(
        "/assistant/chat",
        json={"messages": [{"role": "user", "content": "Combien ?"}]},
    )
    assert refusee.status_code == 403
    assert len(appels) == 1


def test_route_signale_l_absence_de_cle_gemini(monkeypatch):
    utilisateur = {"client_id": ObjectId(), "role": "superadmin"}
    app = FastAPI()
    app.include_router(
        creer_routeur(FauxAnimaux([]), FauxUtilisateurs(), lambda: utilisateur)
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    reponse = TestClient(app).post(
        "/assistant/chat",
        json={"messages": [{"role": "user", "content": "Combien d’animaux ?"}]},
    )

    assert reponse.status_code == 503


def test_appel_gemini_forme_la_requete_sans_appel_reseau(monkeypatch):
    class FauxReponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"candidates":[{"content":{"parts":[{"text":"Bonjour."}]}}]}'

    def urlopen(requete, timeout):
        assert timeout == 30
        assert requete.get_header("X-goog-api-key") == "cle-de-test"
        assert requete.full_url.endswith(
            "/models/gemini-3.5-flash-lite:generateContent"
        )
        corps = json.loads(requete.data)
        assert corps["systemInstruction"]["parts"][0]["text"] == "Consignes"
        assert corps["contents"] == [
            {"role": "user", "parts": [{"text": "Question précédente"}]},
            {"role": "model", "parts": [{"text": "Réponse précédente"}]},
            {"role": "user", "parts": [{"text": "Bonjour"}]}
        ]
        return FauxReponse()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert appeler_gemini(
        [
            {"role": "system", "content": "Consignes"},
            {"role": "assistant", "content": "Réponse précédente oubliée"},
            {"role": "user", "content": "Question précédente"},
            {"role": "assistant", "content": "Réponse précédente"},
            {"role": "user", "content": "Bonjour"},
        ],
        "cle-de-test",
        "gemini-3.5-flash-lite",
    ) == "Bonjour."


def test_route_convertit_les_erreurs_gemini_en_erreur_api(monkeypatch):
    def generer(*_):
        raise ErreurGemini(500)

    utilisateur = {"client_id": ObjectId(), "role": "superadmin"}
    app = FastAPI()
    app.include_router(
        creer_routeur(
            FauxAnimaux([]),
            FauxUtilisateurs(),
            lambda: utilisateur,
            generer,
        )
    )
    monkeypatch.setenv("GEMINI_API_KEY", "cle-de-test")

    reponse = TestClient(app).post(
        "/assistant/chat",
        json={"messages": [{"role": "user", "content": "Question"}]},
    )

    assert reponse.status_code == 502
    assert "HTTP 500" not in reponse.text


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (
            400,
            "La requête Gemini ou la clé configurée sur le serveur est invalide.",
        ),
        (401, "La clé Gemini configurée sur le serveur est invalide."),
        (429, "Le quota ou la limite de requêtes Gemini est atteint."),
        (404, "Le modèle Gemini configuré sur le serveur est introuvable."),
    ],
)
def test_route_explique_les_erreurs_de_configuration_gemini(
    monkeypatch,
    status_code,
    detail,
):
    def generer(*_):
        raise ErreurGemini(status_code)

    utilisateur = {"client_id": ObjectId(), "role": "superadmin"}
    app = FastAPI()
    app.include_router(
        creer_routeur(
            FauxAnimaux([]),
            FauxUtilisateurs(),
            lambda: utilisateur,
            generer,
        )
    )
    monkeypatch.setenv("GEMINI_API_KEY", "cle-de-test")

    reponse = TestClient(app).post(
        "/assistant/chat",
        json={"messages": [{"role": "user", "content": "Question"}]},
    )

    assert reponse.status_code == 503
    assert reponse.json()["detail"] == detail
