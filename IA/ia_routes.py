"""Routes FastAPI /ia : bilan quotidien et historique détaillé par animal (Isolation Forest).

Le modèle ne dépend plus de l'identifiant ni de l'échelle du collier : la référence de chaque animal
est calculée à chaque appel sur son propre historique (il en faut donc assez).

Collection de mesures d'activité, un document par message :
    {"animal_id": "cow01", "timestamp": datetime (UTC), "mi": 3, "steps": 1, "duree_s": 60}

Branchement dans main.py :
    app.include_router(creer_router(mesures_activite, appliquer_alerte_ia, lambda: P()["reference_jours"]))

Routes :
    GET /ia/bilan                          -> bilan des 24 dernières heures (lecture seule)
    GET /ia/bilan?enregistrer=true         -> ouvre / résout aussi les alertes « comportement_inhabituel »
    GET /ia/bilan?heures=240               -> fenêtre d'historique forcée (sinon : durée de référence du réglage)
    GET /ia/historique/{animal_id}         -> courbe heure par heure : pas, routine attendue, score, anomalie
    GET /ia/historique/{animal_id}?heures=336
    GET /ia/historique/{animal_id}?debut=2026-10-24&fin=2026-10-28
                                   -> plage de dates (heure locale ; fin EXCLUE, 30 jours max)

Référence : les N derniers jours du réglage « reference_jours », comptés jusqu'à la
DERNIÈRE MESURE disponible (et non jusqu'à l'heure actuelle). Avec des colliers qui envoient en continu,
c'est la même chose ; si les mesures s'arrêtent (simulation éteinte, collier hors service), la courbe
et la routine restent calculées sur les derniers jours réellement enregistrés. Même calcul pour
l'alerte automatique et pour la page. L'alerte « collier silencieux » signale l'arrêt des mesures.
"""
import os
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

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
HEURES_DEFAUT = 168          # 7 jours : durée de référence par défaut (sans réglage enregistré)
HEURES_MAX = 720             # 30 jours : limite pour éviter des calculs trop lourds
HEURES_MIN = MIN_HEURES_REF + 24   # 96 h : 72 h valides + ~18 h perdues par la moyenne glissante sur 24 h
_EPS = pd.Timedelta(seconds=1)     # la borne de fin des requêtes est exclusive
_COLONNES = ["timestamp", "cow_name", "mi", "steps", "duree_s"]
_bundle = None
_jours_reference = None      # callable sans argument -> nombre de jours (posé par creer_router)


def heures_reference_courantes(client_id=None):
    """Durée de la référence en heures : réglage utilisateur lu à l'instant, sinon HEURES_DEFAUT."""
    if _jours_reference is not None:
        try:
            heures = int(_jours_reference(client_id)) * 24
            if HEURES_MIN <= heures <= HEURES_MAX:
                return heures
        except Exception:
            pass
    return HEURES_DEFAUT


def charger_modele():
    global _bundle
    if _bundle is None:
        b = joblib.load(CHEMIN_MODELE)
        if "refs" in b or "vaches" in b:
            raise ValueError("Ancien modèle (liée aux identifiants) : relancer entrainer_modele_final.py")
        _bundle = b
    return _bundle


def _vers_utc(t_local):
    """Timestamp naïf en heure locale de l'élevage -> datetime UTC."""
    return t_local.tz_localize(FUSEAU).tz_convert("UTC").to_pydatetime()


def _dates(df):
    """Copie du DataFrame avec une colonne `timestamp` toujours de type datetime64 (jamais « object »).

    Indispensable avant agreger_horaire / preparer, qui utilisent l'accesseur .dt."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def fin_donnees(mesures, client_id=None):
    """Heure locale (naïve) de la dernière mesure d'activité enregistrée ; None s'il n'y en a aucune."""
    filtre = {"client_id": client_id} if client_id is not None else {}
    doc = mesures.find_one(filtre, {"_id": 0, "timestamp": 1}, sort=[("timestamp", -1)])
    if not doc:
        return None
    t = pd.Timestamp(doc["timestamp"])
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_convert(FUSEAU).tz_localize(None)


def lire_mesures(mesures, heures=None, animal_id=None, debut_local=None,
                 fin_local=None, client_id=None):
    """Lit les mesures dans MongoDB, au format attendu par scorer().

    Soit les `heures` dernières heures (avant maintenant), soit la plage [debut_local, fin_local)
    en heure locale. Le DataFrame renvoyé a toujours une colonne `timestamp` de type datetime,
    même quand il est vide."""
    if debut_local is not None:
        filtre = {"timestamp": {"$gte": _vers_utc(debut_local), "$lt": _vers_utc(fin_local)}}
    else:
        debut = datetime.now(timezone.utc) - timedelta(hours=heures)
        filtre = {"timestamp": {"$gte": debut}}
    if animal_id:
        filtre["animal_id"] = animal_id
    if client_id is not None:
        filtre["client_id"] = client_id
    docs = list(mesures.find(filtre,
                             {"_id": 0, "animal_id": 1, "timestamp": 1, "mi": 1, "steps": 1, "duree_s": 1}))
    if not docs:
        vide = pd.DataFrame(columns=_COLONNES)
        vide["timestamp"] = pd.to_datetime(vide["timestamp"])      # évite une colonne « object »
        return vide
    df = pd.DataFrame(docs).rename(columns={"animal_id": "cow_name"})
    t = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = t.dt.tz_convert(FUSEAU).dt.tz_localize(None)     # heure locale, comme à l'entraînement
    return df[[c for c in _COLONNES if c in df.columns]]


def calculer_bilan(mesures, heures=None, client_id=None):
    """Retourne (animaux, ignores) ; lève FileNotFoundError si le modèle est absent.

    heures=None : durée de référence courante (réglage « reference_jours »),
    comptée jusqu'à la dernière mesure disponible."""
    if heures is None:
        heures = heures_reference_courantes(client_id)
    bundle = charger_modele()
    fin = fin_donnees(mesures, client_id)
    if fin is None:
        return [], []
    df = lire_mesures(mesures, debut_local=fin - pd.Timedelta(hours=heures),
                      fin_local=fin + _EPS, client_id=client_id)
    if df.empty:
        return [], []
    df = _dates(df)
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
    """Nombre arrondi, ou None si absent ou non fini (JSON n'accepte pas NaN ni inf)."""
    if x is None or pd.isna(x):
        return None
    v = float(x)
    if not np.isfinite(v):
        return None
    return round(v, n)


def calculer_historique(mesures, animal_id, heures=HEURES_DEFAUT, debut=None, fin=None,
                        reference_heures=None, client_id=None):
    """Courbe heure par heure d'UN animal, jugée par rapport à une référence FIXE.

    Référence : les `reference_heures` qui précèdent la DERNIÈRE MESURE enregistrée (None = durée de
    référence courante). C'est exactement le calcul de l'alerte automatique (calculer_bilan), donc la
    page et l'alerte donnent le même verdict.

    La période affichée (`heures` dernières heures avant la dernière mesure, ou plage [debut, fin) en
    pd.Timestamp locaux, même d'un seul jour) ne change PAS la référence : elle ne sert qu'à choisir
    les heures affichées. Les heures plus anciennes que la référence (et les ~18 premières heures de
    celle-ci, perdues par les moyennes glissantes) sont affichées sans score.

    pas_attendus ne montre que le profil horaire (prof_1h) de la routine ; le modèle utilise aussi
    les profils sur 6 h et les médianes sur 24 h.
    Retourne None si l'animal n'a aucune mesure sur la période affichée."""
    if reference_heures is None:
        reference_heures = heures_reference_courantes(client_id)
    bundle = charger_modele()
    fin_ref = fin_donnees(mesures, client_id)
    if fin_ref is None:
        return None
    ref_debut = fin_ref - pd.Timedelta(hours=reference_heures)
    if debut is not None:
        aff_debut, aff_fin = debut, fin
    else:
        aff_debut, aff_fin = fin_ref - pd.Timedelta(hours=heures), fin_ref + _EPS

    # 1) données de la référence (N derniers jours enregistrés)
    df_ref = _dates(lire_mesures(
        mesures, animal_id=animal_id, debut_local=ref_debut,
        fin_local=fin_ref + _EPS, client_id=client_id))

    # 2) données à afficher : la référence + ce qui est plus ancien si la période le demande.
    #    On ne concatène que si des mesures plus anciennes existent : concaténer avec un
    #    DataFrame vide rendait la colonne « timestamp » de type object (erreur « .dt accessor »).
    df_aff = df_ref.copy()
    if aff_debut < ref_debut:
        df_old = lire_mesures(
            mesures, animal_id=animal_id, debut_local=aff_debut,
            fin_local=min(ref_debut, aff_fin), client_id=client_id)
        if not df_old.empty:
            df_aff = pd.concat([_dates(df_old), df_ref], ignore_index=True)
    if df_aff.empty:
        return None
    df_aff = _dates(df_aff)                                      # garde un vrai type date

    brut_tout = agreger_horaire(df_aff)[["timestamp", "steps", "mi"]]    # NaN = heure invalide
    nb_valides = int(brut_tout.loc[brut_tout["timestamp"] >= ref_debut, "steps"].notna().sum())
    brut = brut_tout[(brut_tout["timestamp"] >= aff_debut) & (brut_tout["timestamp"] < aff_fin)]
    if brut.empty:
        return None

    maintenant = pd.Timestamp.now(tz=FUSEAU).tz_localize(None)
    reference = {"debut": ref_debut.isoformat(), "fin": fin_ref.isoformat(), "heures": reference_heures,
                 "retard_h": round(max(0.0, (maintenant - fin_ref).total_seconds() / 3600), 1)}
    reference_jours = reference_heures // 24

    X = pd.DataFrame()
    if nb_valides >= MIN_HEURES_REF and not df_ref.empty:
        X = preparer(df_ref)                                     # heures scorables de la référence
    if X.empty:
        points = [{"heure": t.isoformat(), "valide": bool(pd.notna(s)), "pas": _nb(s, 1), "mi": _nb(m, 1),
                   "pas_attendus": None, "score": None, "alerte": None}
                  for t, s, m in brut.itertuples(index=False)]
        if nb_valides < MIN_HEURES_REF:
            raison = (f"{nb_valides} h valides sur les {reference_jours} derniers jours enregistrés "
                      f"(minimum {MIN_HEURES_REF} h) : routine non calculable")
        else:
            # Le décompte brut passe le minimum, mais les premières heures de la référence sont
            # perdues par les moyennes glissantes : il en reste trop peu pour calculer la routine.
            raison = (f"{nb_valides} h valides sur les {reference_jours} derniers jours enregistrés : "
                      f"insuffisant. Les premières heures de la référence sont perdues par les "
                      f"moyennes glissantes ; il faut environ {HEURES_MIN} h valides. "
                      f"Augmentez la durée de la référence.")
        return {"animal_id": animal_id, "statut": "historique_insuffisant",
                "raison": raison,
                "heures_valides": nb_valides, "reference": reference,
                "reference_jours": reference_jours, "points": points}

    X = X.copy()
    X["score"] = -bundle["modele"].decision_function(X[bundle["features"]])
    X["alerte"] = bundle["modele"].predict(X[bundle["features"]]) == -1

    # routine de l'animal : pas attendus pour chaque heure de la journée (médiane de ses propres heures)
    routine = X.groupby(X["timestamp"].dt.hour)["prof_1h"].first()
    # un seul score par timestamp (évite que .loc renvoie plusieurs lignes)
    scores = (X.drop_duplicates("timestamp")
                .set_index("timestamp")[["score", "alerte"]])

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
        "heures_valides": nb_valides,        # heures valides de la référence
        "reference": reference,
        "reference_jours": reference_jours,
        "fuseau": FUSEAU,
        "points": points}


def creer_router(mesures, appliquer_alerte=None, jours_reference=None, courant=None):
    """appliquer_alerte(animal_id, condition, message, client_id) : branche les alertes.

    jours_reference(client_id) : durée de référence propre au client
    (lue à chaque appel) ; None = 7 jours."""
    global _jours_reference
    _jours_reference = jours_reference
    router = APIRouter(prefix="/ia", tags=["ia"])

    @router.get("/bilan")
    def bilan(heures: Optional[int] = None, enregistrer: bool = False,
              utilisateur: dict = Depends(courant)):
        """Animaux dont le comportement est inhabituel sur les 24 dernières heures enregistrées.

        `heures` absent = durée de référence courante (réglage « reference_jours »)."""
        if heures is None:
            client_id = utilisateur["client_id"]
            heures = heures_reference_courantes(client_id)
        if heures < HEURES_MIN:
            raise HTTPException(400, f"Il faut au moins {HEURES_MIN} h d'historique (reçu : {heures}).")
        if enregistrer and appliquer_alerte is None:
            raise HTTPException(501, "Enregistrement des alertes non configuré.")
        try:
            animaux, ignores = calculer_bilan(mesures, heures, client_id)
        except FileNotFoundError:
            raise HTTPException(503, "Modèle introuvable : lancer entrainer_modele_final.py d'abord.")
        except ValueError as e:
            raise HTTPException(503, str(e))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        if not animaux and not ignores:
            return {"animaux": [], "ignores": [], "message": "Aucune mesure récente."}

        if enregistrer:
            for x in animaux:
                appliquer_alerte(
                    x["animal_id"], x["statut"] == "a_surveiller",
                    f"Comportement inhabituel : {x['heures_alerte']} h en alerte sur les dernières 24 h (à vérifier)",
                    utilisateur["client_id"])
        return {"animaux": animaux, "ignores": ignores, "seuil_heures": SEUIL_HEURES_ALERTE,
                "alertes_mises_a_jour": enregistrer}

    @router.get("/historique/{animal_id}")
    def historique(animal_id: str, heures: int = HEURES_DEFAUT,
                   debut: Optional[str] = None, fin: Optional[str] = None,
                   utilisateur: dict = Depends(courant)):
        """Historique d'activité heure par heure d'un animal, avec les heures anormales.

        Soit `heures` (dernières heures affichées), soit `debut` et `fin` (dates locales, fin EXCLUE).
        La référence est toujours celle du réglage « reference_jours », comptée jusqu'à la dernière
        mesure : la période choisie (même un seul jour) ne fait que sélectionner les heures affichées."""
        d = f = None
        if (debut is None) != (fin is None):
            raise HTTPException(400, "debut et fin doivent être fournis ensemble.")
        if debut is not None:
            try:
                d, f = pd.Timestamp(debut), pd.Timestamp(fin)
                d = d.tz_localize(None) if d.tzinfo else d
                f = f.tz_localize(None) if f.tzinfo else f
            except (ValueError, TypeError):
                raise HTTPException(400, "Dates invalides (format attendu : AAAA-MM-JJ).")
            if f <= d:
                raise HTTPException(400, "fin doit être postérieure à debut.")
            if f - d > pd.Timedelta(hours=HEURES_MAX):
                raise HTTPException(400, f"Période limitée à {HEURES_MAX // 24} jours.")
        elif heures < HEURES_MIN or heures > HEURES_MAX:
            raise HTTPException(400, f"heures doit être compris entre {HEURES_MIN} et {HEURES_MAX} (reçu : {heures}).")
        try:
            res = calculer_historique(
                mesures, animal_id, heures, d, f,
                client_id=utilisateur["client_id"])
        except FileNotFoundError:
            raise HTTPException(503, "Modèle introuvable : lancer entrainer_modele_final.py d'abord.")
        except ValueError as e:
            raise HTTPException(503, str(e))
        except Exception as e:
            # Erreur inattendue : traceback dans les logs Render, et réponse JSON
            # (avec en-têtes CORS) au lieu d'un 500 brut vu comme « Failed to fetch ».
            traceback.print_exc()
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        if res is None:
            raise HTTPException(404, f"Aucune mesure pour {animal_id} sur cette période.")
        return res

    return router