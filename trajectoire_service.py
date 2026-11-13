"""Alerte « mouvement agité » (LSTM de trajectoire) pour le backend FastAPI.

Principe : à chaque cycle (l'intervalle est fixé dans main.py, variable LSTM_INTERVALLE_S), pour chaque animal,
on évalue TOUTES les nouvelles positions, dans l'ordre. L'alerte s'ouvre quand l'erreur de prédiction dépasse
le seuil sur K positions CONSÉCUTIVES.
L'état (série en cours, dernière position évaluée) est gardé dans MongoDB entre deux exécutions.

MODE_SILENCIEUX = True : les alertes sont enregistrées à part, sans notifier l'utilisateur (à garder pendant la
phase de recalibrage avec les vraies données du collier).

Réglages par variables d'environnement (tous facultatifs) :
    LSTM_K            positions suspectes consécutives avant alerte      (défaut 1)
    LSTM_HISTORIQUE_H heures de positions relues à chaque cycle          (défaut 6)
    LSTM_ADAPTATIF    "true" = seuil propre à chaque animal              (défaut false)
    LSTM_PLANCHER_M   seuil minimal en mètres si adaptatif               (défaut 70)
    LSTM_FACTEUR      seuil = facteur × erreur typique de l'animal       (défaut 4)

Dépendances : numpy, onnxruntime, pymongo (déjà présent), tzdata (pip install tzdata) pour le fuseau Africa/Tunis.
Fichiers à déployer à côté de ce module : trajectoire.onnx (UN seul fichier) et trajectoire_meta.json.
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from itertools import groupby
from zoneinfo import ZoneInfo

import numpy as np

log = logging.getLogger("trajectoire")

# ----------------------------------------------------------------------------- réglages
DOSSIER = os.path.dirname(os.path.abspath(__file__))
FICHIER_ONNX = os.path.join(DOSSIER, "trajectoire.onnx")
FICHIER_META = os.path.join(DOSSIER, "trajectoire_meta.json")

N = 12                  # déplacements d'historique (identique à l'entraînement)
DT_MAX = 900            # s : au-delà, la série est coupée (identique à l'entraînement)
K = int(os.getenv("LSTM_K", "1"))                          # positions consécutives au-dessus du seuil
MODE_SILENCIEUX = True
HISTORIQUE_H = int(os.getenv("LSTM_HISTORIQUE_H", "6"))    # heures de positions relues à chaque cycle
RATTRAPAGE_MIN = 60     # au tout premier cycle d'un animal : on n'évalue que la dernière heure
TZ_LOCAL = ZoneInfo("Africa/Tunis")   # le modèle utilise l'heure LOCALE (sinus/cosinus de l'heure)

# Seuil adaptatif : seuil de l'animal = min(seuil du modèle, max(plancher, facteur × erreur typique de l'animal))
ADAPTATIF = os.getenv("LSTM_ADAPTATIF", "false").lower() == "true"   # éteint par défaut
PLANCHER_M = float(os.getenv("LSTM_PLANCHER_M", "70"))               # seuil minimal (m)
FACTEUR = float(os.getenv("LSTM_FACTEUR", "4"))                      # multiple de l'erreur typique

# ----------------------------------------------------------------------------- MongoDB (À ADAPTER)
COLL_POSITIONS = "positions"
CHAMP_ANIMAL, CHAMP_TEMPS, CHAMP_LAT, CHAMP_LON = "animal_id", "timestamp", "lat", "lon"
COLL_ETAT = "lstm_etat"
COLL_ALERTES_SILENCIEUSES = "lstm_alertes_silencieuses"
COLL_ALERTES = "alertes"          # utilisée seulement si MODE_SILENCIEUX = False (adapter le format du document)


# ----------------------------------------------------------------------------- modèle
class Modele:
    """Applique la normalisation puis le modèle ONNX : déplacements BRUTS (m) -> déplacement prévu (m)."""

    def __init__(self, fichier_onnx=FICHIER_ONNX, fichier_meta=FICHIER_META):
        import onnxruntime as ort
        with open(fichier_meta, encoding="utf-8") as f:
            d = json.load(f)
        meta = d.get("meta", d)
        if int(meta.get("N", N)) != N:
            raise ValueError("N du modèle différent de N du service")
        self.s_xy, self.s_dt, self.seuil = float(meta["s_xy"]), float(meta["s_dt"]), float(meta["seuil_m"])
        options = ort.SessionOptions()
        options.log_severity_level = 3          # n'affiche que les erreurs (pas les avertissements de forme)
        self.session = ort.InferenceSession(fichier_onnx, sess_options=options, providers=["CPUExecutionProvider"])

    def __call__(self, X):
        Xn = np.array(X, dtype=np.float32, copy=True)
        Xn[:, :, :2] /= self.s_xy
        Xn[:, :, 2] /= self.s_dt
        return self.session.run(None, {"x": Xn})[0] * self.s_xy


_modele = None


def charger_modele():
    global _modele
    if _modele is None:
        _modele = Modele()
    return _modele


# ----------------------------------------------------------------------------- prétraitement (miroir de lstm_final.py)
def _local_naif(t):
    """Instant -> heure locale Tunisie sans fuseau. Un instant SANS fuseau est supposé déjà local."""
    if t.tzinfo is not None:
        t = t.astimezone(TZ_LOCAL).replace(tzinfo=None)
    return t


def preparer_positions(positions):
    """[{'timestamp','lat','lon'}, ...] -> [(instant local, lat, lon)] triées, sans doublon ni valeur manquante."""
    pts = {}
    for p in positions:
        if p.get("lat") is None or p.get("lon") is None or p.get("timestamp") is None:
            continue
        pts[_local_naif(p["timestamp"])] = (float(p["lat"]), float(p["lon"]))
    return [(t, la, lo) for t, (la, lo) in sorted(pts.items())]


def deplacements(pts):
    """Positions -> déplacements (dx m, dy m, durée s, sin heure, cos heure) + instant de fin de chaque déplacement."""
    t0 = datetime(1970, 1, 1)
    ts = [p[0] for p in pts]
    lat = np.array([p[1] for p in pts])
    lon = np.array([p[2] for p in pts])
    m_lat, m_lon = 111320.0, 111320.0 * np.cos(np.radians(lat[0]))
    x, y = (lon - lon[0]) * m_lon, (lat - lat[0]) * m_lat
    t = np.array([(a - t0).total_seconds() for a in ts])
    h = np.array([a.hour + a.minute / 60 for a in ts])
    ang = 2 * np.pi * h[1:] / 24
    F = np.column_stack([np.diff(x), np.diff(y), np.diff(t), np.sin(ang), np.cos(ang)])
    return F, ts[1:]


def fenetres(F):
    """Fenêtres de N déplacements consécutifs + cible (le suivant). Retourne X, Y, idx (indice de la cible)."""
    vide = (np.empty((0, N, 5)), np.empty((0, 2)), np.empty(0, dtype=int))
    if len(F) <= N:
        return vide
    ok = (F[:, 2] > 0) & (F[:, 2] <= DT_MAX)
    c = np.concatenate([[0], np.cumsum(ok)])
    idx = np.arange(N, len(F))
    v = idx[(c[idx + 1] - c[idx - N]) == N + 1]
    if len(v) == 0:
        return vide
    return np.stack([F[i - N:i] for i in v]), F[v, :2], v


# ----------------------------------------------------------------------------- seuil adaptatif
def seuil_adaptatif(erreurs, seuil_max):
    """Seuil propre à l'animal : max(plancher, facteur × médiane de ses erreurs), plafonné au seuil du modèle.

    Si l'option est éteinte ou s'il y a moins de 5 erreurs, on garde le seuil du modèle."""
    if not ADAPTATIF or len(erreurs) < 5:
        return seuil_max
    typique = float(np.median(erreurs))
    return min(seuil_max, max(PLANCHER_M, FACTEUR * typique))


# ----------------------------------------------------------------------------- logique d'alerte (sans base de données)
def evaluer_animal(positions, etat, predire, seuil, k=K):
    """Évalue les nouvelles positions d'UN animal.

    positions : positions récentes (au moins ~14 avant la première nouvelle), dicts timestamp/lat/lon.
    etat      : {'derniere_cible': instant local|None, 'serie': int} ou None au tout premier cycle.
    Retourne (nouvel état, liste d'alertes).
    """
    pts = preparer_positions(positions)
    etat = dict(etat or {})
    if len(pts) < N + 2:
        return etat, []
    F, instants = deplacements(pts)
    X, Y, idx = fenetres(F)
    if len(idx) == 0:
        return etat, []
    derniere = etat.get("derniere_cible")
    if derniere is None:
        derniere = instants[-1] - timedelta(minutes=RATTRAPAGE_MIN)
    garder = [j for j, i in enumerate(idx) if instants[i] > derniere]
    if not garder:
        return etat, []
    previsions = predire(X)
    toutes = np.linalg.norm(previsions - Y, axis=1)     # erreurs de toutes les fenêtres (estime l'erreur typique)
    erreurs = toutes[garder]
    seuil = seuil_adaptatif(toutes, seuil)              # seuil propre à cet animal (= seuil du modèle si éteint)

    serie, prec, alertes = int(etat.get("serie", 0)), derniere, []
    for e, j in zip(erreurs, garder):
        i = int(idx[j])
        consecutif = i >= 1 and abs((instants[i - 1] - prec).total_seconds()) < 1
        if e > seuil:
            serie = serie + 1 if (consecutif and serie > 0) else 1
            if serie == k:
                lat_depart, lon_depart = pts[i][1], pts[i][2]
                lat_reelle, lon_reelle = pts[i + 1][1], pts[i + 1][2]
                dx_prevu, dy_prevu = map(float, previsions[j])
                m_lon = 111320.0 * np.cos(np.radians(pts[0][1]))
                alertes.append({"instant_local": instants[i], "erreur_m": float(e),
                                "seuil_m": float(seuil), "k": k,
                                "reel_lat": lat_reelle, "reel_lon": lon_reelle,
                                "prevu_lat": lat_depart + dy_prevu / 111320.0,
                                "prevu_lon": lon_depart + dx_prevu / m_lon})
        else:
            serie = 0
        prec = instants[i]
    return {"derniere_cible": prec, "serie": serie}, alertes


# ----------------------------------------------------------------------------- cycle complet (MongoDB, pymongo synchrone)
def executer_cycle(db, predire=None, seuil=None, client_id=None):
    """À appeler à intervalle régulier. `db` = base pymongo. Retourne un petit bilan."""
    modele = predire or charger_modele()
    seuil = seuil if seuil is not None else modele.seuil
    depuis = datetime.now(timezone.utc) - timedelta(hours=HISTORIQUE_H)
    filtre = {CHAMP_TEMPS: {"$gte": depuis}}
    if client_id is not None:
        filtre["client_id"] = client_id
    curseur = db[COLL_POSITIONS].find(
        filtre,
        {CHAMP_ANIMAL: 1, "client_id": 1, "numero_client": 1,
         CHAMP_TEMPS: 1, CHAMP_LAT: 1, CHAMP_LON: 1}
    ).sort([("client_id", 1), (CHAMP_ANIMAL, 1), (CHAMP_TEMPS, 1)])

    bilan = {"animaux": 0, "alertes": 0}
    for (tenant, animal), docs in groupby(
        curseur, key=lambda d: (d.get("client_id"), d[CHAMP_ANIMAL])
    ):
        identifiant_etat = f"{tenant}:{animal}"
        positions = []
        numero_client = None
        for d in docs:
            t = d[CHAMP_TEMPS]
            if t.tzinfo is None:                      # pymongo renvoie des dates naïves = UTC
                t = t.replace(tzinfo=timezone.utc)
            positions.append({"timestamp": t, "lat": d.get(CHAMP_LAT), "lon": d.get(CHAMP_LON)})
            numero_client = d.get("numero_client", numero_client)
        ancien = db[COLL_ETAT].find_one({"_id": identifiant_etat}) or {}
        dc = ancien.get("derniere_cible")
        if dc is not None and dc.tzinfo is not None:      # relue « aware » (client tz_aware) : on revient à l'heure locale naïve du service
            ancien["derniere_cible"] = dc.replace(tzinfo=None)
        etat, alertes = evaluer_animal(positions, ancien, modele, seuil)
        db[COLL_ETAT].replace_one(
            {"_id": identifiant_etat},
            {**etat, "client_id": tenant, "animal_id": animal,
             "maj": datetime.now(timezone.utc)},
            upsert=True)
        if alertes:
            enregistrer_alertes(db, animal, alertes, tenant, numero_client)
        bilan["animaux"] += 1
        bilan["alertes"] += len(alertes)
    log.info("cycle LSTM : %s", bilan)
    return bilan


def enregistrer_alertes(db, animal, alertes, client_id, numero_client):
    docs = [{"animal_id": animal, "type": "mouvement_agite_lstm",
             "client_id": client_id, "numero_client": numero_client,
             **a, "cree_le": datetime.now(timezone.utc)}
            for a in alertes]
    if MODE_SILENCIEUX:
        db[COLL_ALERTES_SILENCIEUSES].insert_many(docs)     # visibles seulement pour analyse, aucune notification
    else:
        db[COLL_ALERTES].insert_many(docs)                  # À ADAPTER au format de vos alertes existantes