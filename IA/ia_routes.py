"""Route FastAPI /ia/bilan : bilan quotidien « animaux à surveiller » (Isolation Forest).

Le modèle ne dépend plus de l'identifiant ni de l'échelle du collier : la référence de chaque animal
est calculée à chaque appel sur son propre historique (il en faut donc assez).

Collection de mesures d'activité, un document par message :
    {"animal_id": "cow01", "timestamp": datetime (UTC), "mi": 3, "steps": 1, "duree_s": 60}

Branchement dans main.py :
    app.include_router(creer_router(mesures_activite, appliquer_alerte_ia))

Routes :
    GET /ia/bilan                    -> bilan des 24 dernières heures (lecture seule)
    GET /ia/bilan?enregistrer=true   -> ouvre / résout aussi les alertes « comportement_inhabituel »
    GET /ia/bilan?heures=240         -> fenêtre d'historique plus longue (référence plus stable)
"""
import os
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

try:                                            # import en tant que package (IA.ia_routes)
    from .entrainer_modele_final import scorer, bilan_quotidien, MIN_HEURES_REF
except ImportError:                             # import direct (script dans le même dossier)
    from entrainer_modele_final import scorer, bilan_quotidien, MIN_HEURES_REF

DOSSIER = os.path.dirname(os.path.abspath(__file__))
CHEMIN_MODELE = os.getenv("IA_MODELE", os.path.join(DOSSIER, "modele_isolation_forest_final.joblib"))
FUSEAU = os.getenv("IA_FUSEAU", "Africa/Tunis")     # fuseau LOCAL de l'élevage (le rythme jour/nuit en dépend)
SEUIL_HEURES_ALERTE = 3      # nb d'heures en alerte sur 24 h pour classer « à surveiller » (valeur de départ, non validée)
HEURES_DEFAUT = 168          # 7 jours d'historique : référence personnelle plus stable
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


def lire_mesures(mesures, heures):
    """Lit les mesures récentes dans MongoDB, au format attendu par scorer()."""
    debut = datetime.now(timezone.utc) - timedelta(hours=heures)
    docs = list(mesures.find({"timestamp": {"$gte": debut}},
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

    return router