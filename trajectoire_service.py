"""Alerte « mouvement agité » (LSTM de trajectoire) pour le backend FastAPI.

Principe : à chaque cycle (intervalle fixé dans main.py), le backend reconstruit les
centres du plus grand groupe actif de « Ferme principale », puis applique le modèle
ONNX existant à cette trajectoire, sans le réentraîner. L'état est unique par client
et conservé dans MongoDB.
L'alerte s'ouvre quand l'erreur dépasse le seuil sur K positions consécutives.

MODE_SILENCIEUX = True : les alertes sont enregistrées à part, sans notifier l'utilisateur.

Réglages par variables d'environnement (tous facultatifs) :
    LSTM_K            positions suspectes consécutives avant alerte      (défaut 1)
    LSTM_HISTORIQUE_H heures de positions relues à chaque cycle          (défaut 6)
    LSTM_ADAPTATIF    "true" = seuil propre à la trajectoire             (défaut false)
    LSTM_PLANCHER_M   seuil minimal en mètres si adaptatif               (défaut 70)
    LSTM_FACTEUR      seuil = facteur × erreur typique de la trajectoire (défaut 4)

Dépendances : numpy, onnxruntime, pymongo (déjà présent), tzdata (pip install tzdata) pour le fuseau Africa/Tunis.
Fichiers à déployer à côté de ce module : trajectoire.onnx (UN seul fichier) et trajectoire_meta.json.
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from DBSCAN import analyser_troupeaux

log = logging.getLogger("trajectoire")

# ----------------------------------------------------------------------------- réglages
DOSSIER = os.path.dirname(os.path.abspath(__file__))
FICHIER_ONNX = os.path.join(DOSSIER, "trajectoire.onnx")
FICHIER_META = os.path.join(DOSSIER, "trajectoire_meta.json")

N = 12                  # déplacements d'historique (identique à l'entraînement)
DT_MAX = 900            # s : au-delà, la série est coupée (identique à l'entraînement)
ID_TROUPEAU_PRINCIPAL = "troupeau_principal"
FERME_PRINCIPALE = "Ferme principale"
K = int(os.getenv("LSTM_K", "1"))                          # positions consécutives au-dessus du seuil
MODE_SILENCIEUX = True
HISTORIQUE_H = int(os.getenv("LSTM_HISTORIQUE_H", "6"))    # heures de positions relues à chaque cycle
RATTRAPAGE_MIN = 60     # au premier cycle : on n'évalue que la dernière heure
TZ_LOCAL = ZoneInfo("Africa/Tunis")   # le modèle utilise l'heure LOCALE (sinus/cosinus de l'heure)

# Seuil adaptatif : seuil de trajectoire = min(seuil du modèle, max(plancher, facteur × erreur typique))
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
        self.intervalle_s = int(meta.get("intervalle_s", 300))
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
    """Seuil propre à la trajectoire : max(plancher, facteur × médiane des erreurs), plafonné au modèle.

    Si l'option est éteinte ou s'il y a moins de 5 erreurs, on garde le seuil du modèle."""
    if not ADAPTATIF or len(erreurs) < 5:
        return seuil_max
    typique = float(np.median(erreurs))
    return min(seuil_max, max(PLANCHER_M, FACTEUR * typique))


# ----------------------------------------------------------------------------- logique d'alerte (sans base de données)
def evaluer_animal(positions, etat, predire, seuil, k=K):
    """Évalue les nouveaux points d'une trajectoire.

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
    seuil = seuil_adaptatif(toutes, seuil)              # seuil propre à la trajectoire (= seuil modèle si éteint)

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


def construire_centres_troupeau(
    positions,
    maintenant,
    parametres,
    intervalle_s=300,
    historique_h=HISTORIQUE_H,
):
    """Construit les centres réguliers du plus grand groupe de la ferme principale."""
    if intervalle_s <= 0:
        raise ValueError("L'intervalle de trajectoire doit être positif.")

    maintenant = (
        maintenant.replace(tzinfo=timezone.utc)
        if maintenant.tzinfo is None
        else maintenant.astimezone(timezone.utc)
    )
    intervalle_s = int(intervalle_s)
    fraicheur_s = int(parametres.get("fraicheur_s", 600))
    debut = int((maintenant - timedelta(hours=historique_h)).timestamp())
    premier_pas = debut - debut % intervalle_s
    dernier_pas = int(maintenant.timestamp())
    dernier_pas -= dernier_pas % intervalle_s

    ordonnees = []
    for position in positions:
        if (
            position.get("ferme") != FERME_PRINCIPALE
            or not position.get(CHAMP_ANIMAL)
            or position.get(CHAMP_TEMPS) is None
            or position.get(CHAMP_LAT) is None
            or position.get(CHAMP_LON) is None
        ):
            continue
        instant = position[CHAMP_TEMPS]
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        else:
            instant = instant.astimezone(timezone.utc)
        ordonnees.append((instant, position))
    ordonnees.sort(key=lambda element: element[0])

    dernier_par_animal = {}
    index = 0
    signature_precedente = None
    centres = []
    pas = premier_pas
    while pas <= dernier_pas:
        instant_pas = datetime.fromtimestamp(pas, timezone.utc)
        while index < len(ordonnees) and ordonnees[index][0] <= instant_pas:
            instant, position = ordonnees[index]
            dernier_par_animal[position[CHAMP_ANIMAL]] = (instant, position)
            index += 1

        actifs = [
            {
                "animal_id": animal_id,
                "lat": float(position[CHAMP_LAT]),
                "lon": float(position[CHAMP_LON]),
            }
            for animal_id, (instant, position) in dernier_par_animal.items()
            if 0 <= (instant_pas - instant).total_seconds() <= fraicheur_s
        ]
        signature = tuple(sorted(
            (point["animal_id"], dernier_par_animal[point["animal_id"]][0])
            for point in actifs
        ))

        if (
            len(actifs) >= int(parametres["min_animaux_troupeau"])
            and signature != signature_precedente
        ):
            principal, _, _ = analyser_troupeaux(actifs, parametres)
            if principal is not None:
                centres.append({
                    "timestamp": instant_pas,
                    "lat": principal["centre_lat"],
                    "lon": principal["centre_lon"],
                    "nb_animaux": principal["nb_animaux"],
                })
        signature_precedente = signature if actifs else None
        pas += intervalle_s
    return centres


# ----------------------------------------------------------------------------- cycle complet (MongoDB, pymongo synchrone)
def executer_cycle(db, predire=None, seuil=None, client_id=None, parametres=None, maintenant=None):
    """Évalue une trajectoire pour le centre du plus grand groupe de la ferme principale."""
    modele = predire or charger_modele()
    seuil = seuil if seuil is not None else modele.seuil
    maintenant = maintenant or datetime.now(timezone.utc)
    if maintenant.tzinfo is None:
        maintenant = maintenant.replace(tzinfo=timezone.utc)
    else:
        maintenant = maintenant.astimezone(timezone.utc)
    parametres = parametres or {
        "eps_troupeau_m": 30,
        "min_animaux_troupeau": 5,
        "distance_alerte_m": 120,
        "fraicheur_s": 600,
    }
    intervalle_s = int(getattr(modele, "intervalle_s", 300))
    depuis = maintenant - timedelta(
        hours=HISTORIQUE_H,
        seconds=intervalle_s + int(parametres.get("fraicheur_s", 600)),
    )
    filtre = {CHAMP_TEMPS: {"$gte": depuis}, "ferme": FERME_PRINCIPALE}
    if client_id is not None:
        filtre["client_id"] = client_id
    documents = list(db[COLL_POSITIONS].find(
        filtre,
        {
            CHAMP_ANIMAL: 1,
            "client_id": 1,
            "numero_client": 1,
            "ferme": 1,
            CHAMP_TEMPS: 1,
            CHAMP_LAT: 1,
            CHAMP_LON: 1,
        },
    ).sort([(CHAMP_TEMPS, 1)]))

    par_tenant = {}
    for document in documents:
        par_tenant.setdefault(document.get("client_id"), []).append(document)

    bilan = {"troupeaux": 0, "alertes": 0}
    for tenant, docs in par_tenant.items():
        centres = construire_centres_troupeau(
            docs, maintenant, parametres, intervalle_s=intervalle_s
        )
        if not centres:
            continue
        numero_client = docs[-1].get("numero_client")
        identifiant_etat = f"{tenant}:{ID_TROUPEAU_PRINCIPAL}"
        ancien = db[COLL_ETAT].find_one({"_id": identifiant_etat}) or {}
        dc = ancien.get("derniere_cible")
        if dc is not None and dc.tzinfo is not None:      # relue « aware » (client tz_aware) : on revient à l'heure locale naïve du service
            ancien["derniere_cible"] = dc.replace(tzinfo=None)
        etat, alertes = evaluer_animal(centres, ancien, modele, seuil)
        db[COLL_ETAT].replace_one(
            {"_id": identifiant_etat},
            {**etat, "_id": identifiant_etat, "client_id": tenant,
             "animal_id": ID_TROUPEAU_PRINCIPAL,
             "nom_trajectoire": "Troupeau principal",
             "maj": datetime.now(timezone.utc)},
            upsert=True)
        if alertes:
            enregistrer_alertes(
                db,
                ID_TROUPEAU_PRINCIPAL,
                alertes,
                tenant,
                numero_client,
                nom_trajectoire="Troupeau principal",
            )
        bilan["troupeaux"] += 1
        bilan["alertes"] += len(alertes)
    log.info("cycle LSTM : %s", bilan)
    return bilan


def enregistrer_alertes(
    db, animal, alertes, client_id, numero_client, nom_trajectoire=None
):
    docs = [{"animal_id": animal, "type": "mouvement_agite_lstm",
             "client_id": client_id, "numero_client": numero_client,
             **({"nom_trajectoire": nom_trajectoire} if nom_trajectoire else {}),
             **a, "cree_le": datetime.now(timezone.utc)}
            for a in alertes]
    if MODE_SILENCIEUX:
        db[COLL_ALERTES_SILENCIEUSES].insert_many(docs)     # visibles seulement pour analyse, aucune notification
    else:
        db[COLL_ALERTES].insert_many(docs)                  # À ADAPTER au format de vos alertes existantes