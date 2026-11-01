"""Alerte « mouvement agité » (LSTM de trajectoire) pour le backend FastAPI.

Principe : toutes les 15 min, pour chaque animal, on évalue TOUTES les nouvelles positions (≈ 3, au pas de 5 min),
dans l'ordre. L'alerte s'ouvre quand l'erreur de prédiction dépasse le seuil sur K positions CONSÉCUTIVES.
L'état (série en cours, dernière position évaluée) est gardé dans MongoDB entre deux exécutions.

MODE_SILENCIEUX = True : les alertes sont enregistrées à part, sans notifier l'utilisateur (à garder pendant la
phase de recalibrage avec les vraies données du collier).

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
K = 4                   # positions consécutives au-dessus du seuil
MODE_SILENCIEUX = True
HISTORIQUE_H = 6        # heures de positions relues à chaque cycle
RATTRAPAGE_MIN = 60     # au tout premier cycle d'un animal : on n'évalue que la dernière heure
TZ_LOCAL = ZoneInfo("Africa/Tunis")   # le modèle utilise l'heure LOCALE (sinus/cosinus de l'heure)

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
    erreurs = np.linalg.norm(predire(X[garder]) - Y[garder], axis=1)

    serie, prec, alertes = int(etat.get("serie", 0)), derniere, []
    for e, j in zip(erreurs, garder):
        i = int(idx[j])
        consecutif = i >= 1 and abs((instants[i - 1] - prec).total_seconds()) < 1
        if e > seuil:
            serie = serie + 1 if (consecutif and serie > 0) else 1
            if serie == k:
                alertes.append({"instant_local": instants[i], "erreur_m": float(e),
                                "seuil_m": float(seuil), "k": k})
        else:
            serie = 0
        prec = instants[i]
    return {"derniere_cible": prec, "serie": serie}, alertes


# ----------------------------------------------------------------------------- cycle complet (MongoDB, pymongo synchrone)
def executer_cycle(db, predire=None, seuil=None):
    """À appeler toutes les 15 min. `db` = base pymongo. Retourne un petit bilan."""
    modele = predire or charger_modele()
    seuil = seuil if seuil is not None else modele.seuil
    depuis = datetime.now(timezone.utc) - timedelta(hours=HISTORIQUE_H)
    curseur = db[COLL_POSITIONS].find(
        {CHAMP_TEMPS: {"$gte": depuis}},
        {CHAMP_ANIMAL: 1, CHAMP_TEMPS: 1, CHAMP_LAT: 1, CHAMP_LON: 1}
    ).sort([(CHAMP_ANIMAL, 1), (CHAMP_TEMPS, 1)])

    bilan = {"animaux": 0, "alertes": 0}
    for animal, docs in groupby(curseur, key=lambda d: d[CHAMP_ANIMAL]):
        positions = []
        for d in docs:
            t = d[CHAMP_TEMPS]
            if t.tzinfo is None:                      # pymongo renvoie des dates naïves = UTC
                t = t.replace(tzinfo=timezone.utc)
            positions.append({"timestamp": t, "lat": d.get(CHAMP_LAT), "lon": d.get(CHAMP_LON)})
        ancien = db[COLL_ETAT].find_one({"_id": animal}) or {}
        etat, alertes = evaluer_animal(positions, ancien, modele, seuil)
        db[COLL_ETAT].replace_one({"_id": animal}, {**etat, "maj": datetime.now(timezone.utc)}, upsert=True)
        if alertes:
            enregistrer_alertes(db, animal, alertes)
        bilan["animaux"] += 1
        bilan["alertes"] += len(alertes)
    log.info("cycle LSTM : %s", bilan)
    return bilan


def enregistrer_alertes(db, animal, alertes):
    docs = [{"animal_id": animal, "type": "mouvement_agite_lstm", **a, "cree_le": datetime.now(timezone.utc)}
            for a in alertes]
    if MODE_SILENCIEUX:
        db[COLL_ALERTES_SILENCIEUSES].insert_many(docs)     # visibles seulement pour analyse, aucune notification
    else:
        db[COLL_ALERTES].insert_many(docs)                  # À ADAPTER au format de vos alertes existantes