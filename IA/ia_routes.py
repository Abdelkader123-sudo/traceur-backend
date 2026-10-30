"""Routes FastAPI /ia : bilan quotidien et historique détaillé par animal (Isolation Forest).

Le modèle ne dépend plus de l'identifiant ni de l'échelle du collier : la référence de chaque animal
est calculée à chaque appel sur son propre historique (il en faut donc assez).

Collection de mesures d'activité, un document par message :
    {"animal_id": "cow01", "timestamp": datetime (UTC), "mi": 3, "steps": 1, "duree_s": 60}

Branchement dans main.py :
    app.include_router(creer_router(mesures_activite, appliquer_alerte_ia))

Routes :
    GET /ia/bilan                          -> bilan des 24 dernières heures (lecture seule)
    GET /ia/bilan?enregistrer=true         -> ouvre / résout aussi les alertes « comportement_inhabituel »
    GET /ia/bilan?heures=240               -> fenêtre d'historique plus longue (référence plus stable)
    GET /ia/historique/{animal_id}         -> courbe heure par heure : pas, routine attendue, score, anomalie
    GET /ia/historique/{animal_id}?heures=336
"""
import os
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

try:                                            # import en tant que package (IA.ia_routes)
    from .entrainer_modele_final import (scorer, bilan_quotidien, preparer,
                                         agreger_horaire, MIN_HEURES_REF)
except ImportError:                             # import direct (script dans le même dossier)
    from entrainer_modele_final import (scorer, bilan_quotidien, preparer,
                                        agreger_horaire, MIN_HEURES_REF)

DOSSIER = os.path.dirname(os.path.abspath(__file__))
CHEMIN_MODELE = os.getenv("IA_MODELE", os.path.join(DOSSIER, "modele_isolation_forest_final.joblib"))
FUSEAU = os.getenv("IA_FUSEAU", "Africa/Tunis")     # fuseau LOCAL de l'élevage (le rythme jour/nuit en dépend)
SEUIL_HEURES_ALERTE = 3      # nb d'heures en alerte sur 24 h pour classer « à surveiller » (valeur de départ, non validée)
HEURES_DEFAUT = 168          # 7 jours d'historique : référence personnelle plus stable
HEURES_MAX = 720             # 30 jours : limite pour éviter des calculs trop lourds
HEURES_MIN = MIN_HEURES_REF + 24   # 96 h : 72 h valides + ~18 h perdues par la moyenne glissante sur 24 h
_bundle = None


def charger_modele():
    global _bundle
    if _bundle is None:
        b = joblib.load(CHEMIN_MODELE)
        if "refs" in b or "vaches" in b:
            raise ValueError("Ancien modèle (liée aux identifiants) : relancer entrainer_modele_final.py")
        _bundle = b
    return _bundle


def lire_mesures(mesures, heures, animal_id=None):
    """Lit les mesures récentes dans MongoDB, au format attendu par scorer()."""
    debut = datetime.now(timezone.utc) - timedelta(hours=heures)
    filtre = {"timestamp": {"$gte": debut}}
    if animal_id:
        filtre["animal_id"] = animal_id
    docs = list(mesures.find(filtre,
                             {"_id": 0, "animal_id": 1, "timestamp": 1, "mi": 1, "steps": 1, "duree_s": 1}))
    if not docs:
        return pd.DataFrame(columns=["timestamp", "cow_name", "mi", "steps", "duree_s"])
    df = pd.DataFrame(docs).rename(columns={"animal_id": "cow_name"})
    t = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = t.dt.tz_convert(FUSEAU).dt.tz_localize(None)     # heure locale, comme à l'entraînement
    return df[[c for c in ["timestamp", "cow_name", "mi", "steps", "duree_s"] if c in df.columns]]


def calculer_bilan(mesures, heures=HEURES_DEFAUT):
    """Retourne (animaux, ignores) ; lève FileNotFoundError si le modèle est absent."""
    bundle = charger_modele()
    df = lire_mesures(mesures, heures)
    if df.empty:
        return [], []
    res = scorer(bundle, df)
    resume = bilan_quotidien(res) if not res.empty else pd.DataFrame(columns=["heures_alerte", "score_max"])

    animaux = [{"animal_id": str(a), "heures_alerte": int(r["heures_alerte"]),
                "score_max": float(r["score_max"]),
                "statut": "a_surveiller" if r["heures_alerte"] >= SEUIL_HEURES_ALERTE else "normal"}
               for a, r in resume.iterrows()]
    scores = {str(a) for a in resume.index}
    ignores = [{"animal_id": a,
                "raison": f"historique insuffisant (moins de {MIN_HEURES_REF} h valides) "
                          f"ou aucune mesure exploitable sur les dernières 24 h"}
               for a in sorted(set(df["cow_name"].astype(str))) if a not in scores]
    return animaux, ignores


def _nb(x, n=2):
    """Nombre arrondi, ou None si absent (JSON n'accepte pas NaN)."""
    return None if x is None or pd.isna(x) else round(float(x), n)


def calculer_historique(mesures, animal_id, heures=HEURES_DEFAUT):
    """Courbe heure par heure d'UN animal. Retourne None s'il n'a aucune mesure.

    Pour chaque heure : pas et mouvement mesurés, pas attendus (routine de l'animal à cette heure),
    score d'anomalie et indicateur d'anomalie. Les heures invalides (trop peu de messages) ont
    valide=False et aucune valeur : à afficher en gris, pas en zéro."""
    bundle = charger_modele()
    df = lire_mesures(mesures, heures, animal_id)
    if df.empty:
        return None

    brut = agreger_horaire(df)                                   # toutes les heures, NaN si invalide
    brut = brut[["timestamp", "steps", "mi"]]
    nb_valides = int(brut["steps"].notna().sum())

    X = pd.DataFrame()
    if nb_valides >= MIN_HEURES_REF:
        X = preparer(df)                                         # heures scorables uniquement
    if X.empty:
        points = [{"heure": t.isoformat(), "valide": bool(pd.notna(s)), "pas": _nb(s, 1), "mi": _nb(m, 1),
                   "pas_attendus": None, "score": None, "alerte": None}
                  for t, s, m in brut.itertuples(index=False)]
        return {"animal_id": animal_id, "statut": "historique_insuffisant",
                "raison": f"moins de {MIN_HEURES_REF} h valides : routine non calculable",
                "heures_valides": nb_valides, "points": points}

    X = X.copy()
    X["score"] = -bundle["modele"].decision_function(X[bundle["features"]])
    X["alerte"] = bundle["modele"].predict(X[bundle["features"]]) == -1

    # routine de l'animal : pas attendus pour chaque heure de la journée (médiane de ses propres heures)
    routine = X.groupby(X["timestamp"].dt.hour)["prof_1h"].first()
    scores = X.set_index("timestamp")[["score", "alerte"]]

    points = []
    for t, s, m in brut.itertuples(index=False):
        attendu = routine.get(t.hour)
        sc = scores.loc[t] if t in scores.index else None
        points.append({
            "heure": t.isoformat(),                              # heure LOCALE de l'élevage
            "valide": bool(pd.notna(s)),
            "pas": _nb(s, 1), "mi": _nb(m, 1),
            "pas_attendus": _nb(np.expm1(attendu), 1) if attendu is not None and pd.notna(attendu) else None,
            "score": _nb(sc["score"], 3) if sc is not None else None,
            "alerte": bool(sc["alerte"]) if sc is not None else None})

    dern = X[X["timestamp"] > X["timestamp"].max() - pd.Timedelta(hours=24)]
    h_alerte = int(dern["alerte"].sum())
    return {
        "animal_id": animal_id,
        "statut": "a_surveiller" if h_alerte >= SEUIL_HEURES_ALERTE else "normal",
        "heures_alerte_24h": h_alerte,
        "score_max_24h": _nb(dern["score"].max(), 3),
        "seuil_score": 0.0,                  # score > 0 = heure anormale
        "seuil_heures": SEUIL_HEURES_ALERTE,
        "heures_valides": nb_valides,
        "fuseau": FUSEAU,
        "points": points}


def creer_router(mesures, appliquer_alerte=None):
    """appliquer_alerte(animal_id, condition, message) : branche le bilan sur le système d'alertes."""
    router = APIRouter(prefix="/ia", tags=["ia"])

    @router.get("/bilan")
    def bilan(heures: int = HEURES_DEFAUT, enregistrer: bool = False):
        """Animaux dont le comportement est inhabituel sur les 24 dernières heures."""
        if heures < HEURES_MIN:
            raise HTTPException(400, f"Il faut au moins {HEURES_MIN} h d'historique (reçu : {heures}).")
        if enregistrer and appliquer_alerte is None:
            raise HTTPException(501, "Enregistrement des alertes non configuré.")
        try:
            animaux, ignores = calculer_bilan(mesures, heures)
        except FileNotFoundError:
            raise HTTPException(503, "Modèle introuvable : lancer entrainer_modele_final.py d'abord.")
        except ValueError as e:
            raise HTTPException(503, str(e))
        if not animaux and not ignores:
            return {"animaux": [], "ignores": [], "message": "Aucune mesure récente."}

        if enregistrer:
            for x in animaux:
                appliquer_alerte(
                    x["animal_id"], x["statut"] == "a_surveiller",
                    f"Comportement inhabituel : {x['heures_alerte']} h en alerte sur les dernières 24 h (à vérifier)")
        return {"animaux": animaux, "ignores": ignores, "seuil_heures": SEUIL_HEURES_ALERTE,
                "alertes_mises_a_jour": enregistrer}

    @router.get("/historique/{animal_id}")
    def historique(animal_id: str, heures: int = HEURES_DEFAUT):
        """Historique d'activité heure par heure d'un animal, avec les heures anormales."""
        if heures < HEURES_MIN or heures > HEURES_MAX:
            raise HTTPException(400, f"heures doit être compris entre {HEURES_MIN} et {HEURES_MAX} (reçu : {heures}).")
        try:
            res = calculer_historique(mesures, animal_id, heures)
        except FileNotFoundError:
            raise HTTPException(503, "Modèle introuvable : lancer entrainer_modele_final.py d'abord.")
        except ValueError as e:
            raise HTTPException(503, str(e))
        if res is None:
            raise HTTPException(404, f"Aucune mesure récente pour {animal_id}.")
        return res

    return router