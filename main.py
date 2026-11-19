import asyncio
import base64
import binascii
import io
import json
import math
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import paho.mqtt.client as mqtt
from fastapi import (BackgroundTasks, Depends, FastAPI, Header, HTTPException,
                     Response, WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from bson import ObjectId
from gridfs import GridFSBucket
from gridfs.errors import NoFile
from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError

from DBSCAN import analyser_troupeaux, deplacement_inhabituel, est_immobile
from geofence import analyser_point, valider_sommets
from IA.ia_routes import creer_router, calculer_bilan, SEUIL_HEURES_ALERTE   # AJOUT IA
from params import BORNES, DEFAUTS, FIXES, valider_parametres
import trajectoire_service                                   # AJOUT LSTM
from auth import creer_dependance_courant, creer_routeur     # AJOUT AUTH
from messagerie import creer_routeur_messagerie

MQTT_HOST = os.getenv("MQTT_HOST", "ycf81881.ala.eu-central-1.emqxsl.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.environ["MQTT_USER"]
MQTT_PASSWORD = os.environ["MQTT_PASSWORD"]
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "farm/+/data")
CA_FILE = os.getenv("CA_FILE", "emqxsl-ca.crt")

MONGO_URI = os.environ["MONGO_URI"]

RAYON_TERRE_M = 6_371_000          # AJOUT ZONE AUTO

# AJOUT ZONE AUTO : la zone de pâturage est gérée uniquement par le serveur.
ZONE_AUTO = os.getenv("ZONE_AUTO", "true").lower() != "false"   # ZONE_AUTO=false : plus de création auto
SURFACE_PAR_ANIMAL_M2 = float(os.getenv("SURFACE_PAR_ANIMAL_M2", "5"))   # place estimée par animal
ADMIN_KEY = os.getenv("ADMIN_KEY")                              # clé pour modifier la zone à la main (facultatif)

INTERVALLE_VERIF_COLLIERS_S = 60   # vérification des colliers silencieux

# AJOUT IA AUTO : bilan Isolation Forest lancé automatiquement
IA_AUTO = os.getenv("IA_AUTO", "true").lower() != "false"          # IA_AUTO=false pour le désactiver
INTERVALLE_BILAN_IA_S = int(os.getenv("IA_INTERVALLE_S", "3600"))  # une fois par heure
DELAI_PREMIER_BILAN_S = 60                                         # laisse le serveur démarrer
DUREE_CONSERVATION_MESURES_S = 14 * 24 * 3600                      # TTL de mesures_activite : 14 jours
REFERENCE_JOURS = 4

# AJOUT LSTM : alerte « mouvement agité », évaluée toutes les 15 min
LSTM_AUTO = os.getenv("LSTM_AUTO", "true").lower() != "false"
INTERVALLE_LSTM_S = int(os.getenv("LSTM_INTERVALLE_S", "900"))
DELAI_PREMIER_LSTM_S = 90

# AJOUT NETTOYAGE : suppression automatique des anciennes données (économie de stockage)
#   - alertes (TOUS types, actives ou résolues) : supprimées uniquement quand leur âge
#     dépasse DUREE_ALERTES_H heures (24 h par défaut), jamais avant
#   - autres données (positions, activité, événements LSTM, anciennes zones) :
#     supprimées au-delà de DUREE_DONNEES_H heures (170 h par défaut)
#   - JAMAIS touchés : fiches des animaux, paramètres, état courant des animaux, zone active,
#     et comptes utilisateurs (collection « utilisateurs »)
NETTOYAGE_AUTO = os.getenv("NETTOYAGE_AUTO", "true").lower() != "false"   # NETTOYAGE_AUTO=false pour le désactiver
DUREE_ALERTES_H = float(os.getenv("DUREE_ALERTES_H", "24"))
DUREE_DONNEES_H = float(os.getenv("DUREE_DONNEES_H", "170"))
INTERVALLE_NETTOYAGE_S = int(os.getenv("NETTOYAGE_INTERVALLE_S", "600"))   # toutes les 10 min
DELAI_PREMIER_NETTOYAGE_S = 120                                              # laisse le serveur démarrer

mongo_client = MongoClient(MONGO_URI, tz_aware=True)
base = mongo_client["traceur_gps"]
positions = base["positions"]
alerts = base["alerts"]
etat_animaux = base["etat_animaux"]
animaux = base["animaux"]                      # fiches des animaux (type, sexe, âge, poids)
sante_animaux = base["sante_animaux"]
photos_animaux = GridFSBucket(base, bucket_name="photos_animaux")
pasture = base["pastures"]
parametres_col = base["parametres"]
mesures_activite = base["mesures_activite"]    # AJOUT IA : un document par message (mi, steps, duree_s)
lstm_evenements = base["lstm_alertes_silencieuses"]   # AJOUT NETTOYAGE
clients_col = base["clients"]
courant = creer_dependance_courant(
    base["utilisateurs"],
    base["sessions_authentification"],
)

_cache_params = {}
_cache_zone = {}


def P(client_id=None) -> dict:
    """Paramètres actifs (défauts + valeurs enregistrées). reference_jours est fixé à 4.
    Les paramètres de FIXES (ex. marge_pature_m) gardent toujours leur valeur par défaut."""
    cle = str(client_id or "legacy")
    if cle not in _cache_params:
        filtre = {"client_id": client_id} if client_id is not None else {"_id": "global"}
        doc = parametres_col.find_one(filtre) or {}
        _cache_params[cle] = {
            **DEFAUTS,
            **{k: doc[k] for k in DEFAUTS if k in doc and k not in FIXES},
            "reference_jours": REFERENCE_JOURS,
        }
    return _cache_params[cle]


def zone_active(client_id=None):
    """Zone de pâturage active (mise en cache)."""
    cle = str(client_id or "legacy")
    if cle not in _cache_zone:
        filtre = {"active": True}
        if client_id is not None:
            filtre["client_id"] = client_id
        _cache_zone[cle] = pasture.find_one(filtre, {"_id": 0})
    return _cache_zone[cle]


# ---------------------------------------------------------------------------
# Diffusion WebSocket
# ---------------------------------------------------------------------------
def _json(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


class Diffuseur:
    """Envoie les événements à tous les clients WebSocket connectés."""

    def __init__(self):
        self.clients: dict[WebSocket, ObjectId] = {}
        self.boucle = None

    async def _envoyer(self, texte: str, client_id):
        for ws, proprietaire in list(self.clients.items()):
            if proprietaire != client_id:
                continue
            try:
                await ws.send_text(texte)
            except Exception:
                self.clients.pop(ws, None)

    def publier(self, evenement: dict, client_id):
        """Appelable depuis le thread MQTT."""
        if self.boucle is None or not self.clients:
            return
        texte = json.dumps(evenement, default=_json)
        asyncio.run_coroutine_threadsafe(self._envoyer(texte, client_id), self.boucle)


diffuseur = Diffuseur()


def _client_id(valeur):
    if isinstance(valeur, ObjectId):
        return valeur
    return ObjectId(str(valeur))


class Position(BaseModel):
    """Format attendu pour chaque message MQTT."""
    animal_id: str
    lat: float
    lon: float
    vitesse: float
    activite: str
    timestamp: datetime
    numero_client: Optional[str] = None
    ferme: Optional[str] = None
    client_id: Optional[str] = None


class InfoAnimal(BaseModel):
    """Fiche d'un animal, telle qu'envoyée par l'application."""
    type: Optional[str] = Field(None, max_length=80)
    sexe: Optional[Literal["femelle", "male"]] = None
    age_ans: Optional[float] = Field(None, ge=0, le=40)
    poids_kg: Optional[float] = Field(None, gt=0, le=1500)
    ferme: str = Field(default="Ferme principale", min_length=1, max_length=80)


class PhotoAnimal(BaseModel):
    contenu_base64: str = Field(..., max_length=5_600_000)
    type_contenu: Literal["image/jpeg", "image/png", "image/webp"]


class EvenementSante(BaseModel):
    type: Literal["vaccination", "traitement", "consultation", "poids", "note"]
    titre: str = Field(min_length=1, max_length=100)
    date: datetime
    note: str = Field(default="", max_length=1000)
    prochaine_echeance: Optional[datetime] = None
    poids_kg: Optional[float] = Field(None, gt=0, le=1500)


TAILLE_MAX_PHOTO_ANIMAL = 4 * 1024 * 1024


def decoder_photo_animal(photo: PhotoAnimal) -> bytes:
    try:
        contenu = base64.b64decode(photo.contenu_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "Le fichier image est invalide.")
    if not contenu:
        raise HTTPException(400, "Le fichier image est vide.")
    if len(contenu) > TAILLE_MAX_PHOTO_ANIMAL:
        raise HTTPException(413, "La photo ne doit pas dépasser 4 Mo.")

    signatures = {
        "image/jpeg": contenu.startswith(b"\xff\xd8\xff"),
        "image/png": contenu.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/webp": (
            len(contenu) >= 12
            and contenu[:4] == b"RIFF"
            and contenu[8:12] == b"WEBP"
        ),
    }
    if not signatures[photo.type_contenu]:
        raise HTTPException(400, "Le type du fichier ne correspond pas à son contenu.")
    return contenu


# ---------------------------------------------------------------------------
# Alertes
# ---------------------------------------------------------------------------
def gerer_alerte(animal_id, type_alerte, condition, message, position):
    """Ouvre une alerte quand le problème apparaît, la ferme quand il disparaît."""
    client_id = _client_id(position.client_id)
    filtre = {"client_id": client_id, "animal_id": animal_id, "type": type_alerte}
    ouverte = alerts.find_one(
        {**filtre, "resolue": False}
    )
    if condition and ouverte is None:
        doc = {
            "animal_id": animal_id,
            "type": type_alerte,
            "message": message,
            "lat": position.lat,
            "lon": position.lon,
            "timestamp": position.timestamp,
            "resolue": False,
            "client_id": client_id,
            "numero_client": position.numero_client,
        }
        alerts.insert_one(dict(doc))
        print(f"ALERTE [{type_alerte}] {animal_id} : {message}")
        diffuseur.publier({"type": "alerte", "data": doc}, client_id)
    elif not condition and ouverte is not None:
        alerts.update_one(
            {"_id": ouverte["_id"], "client_id": client_id},
            {"$set": {"resolue": True, "resolue_le": position.timestamp}},
        )
        print(f"Alerte résolue [{type_alerte}] {animal_id}")
        diffuseur.publier({"type": "alerte_resolue", "data": {
            "animal_id": animal_id,
            "type": type_alerte,
            "resolue_le": position.timestamp,
        }}, client_id)


def appliquer_alerte_ia(animal_id, condition, message, client_id):
    """AJOUT IA : branche le bilan Isolation Forest sur le système d'alertes existant
    (ouverture, résolution automatique, diffusion WebSocket), à la dernière position connue."""
    client_id = _client_id(client_id)
    doc = positions.find_one(
        {"client_id": client_id, "animal_id": animal_id},
        {"_id": 0}, sort=[("timestamp", -1)])
    if not doc:
        return
    doc["timestamp"] = datetime.now(timezone.utc)
    gerer_alerte(
        animal_id, "comportement_inhabituel", condition, message,
        Position(**{**doc, "client_id": str(client_id)}))


def executer_bilan_ia():
    """AJOUT IA AUTO : calcule le bilan des 24 dernières heures pour tous les animaux
    et ouvre / résout les alertes « comportement_inhabituel » (même logique que
    GET /ia/bilan?enregistrer=true). La référence dure reference_jours jours (réglage)."""
    for client in clients_col.find({}, {"_id": 1, "numero_client": 1}):
        client_id = client["_id"]
        animaux_ia, ignores = calculer_bilan(
            mesures_activite, P(client_id)["reference_jours"] * 24, client_id)
        for x in animaux_ia:
            try:
                appliquer_alerte_ia(
                    x["animal_id"], x["statut"] == "a_surveiller",
                    f"Comportement inhabituel : {x['heures_alerte']} h en alerte sur les "
                    f"dernières 24 h (à vérifier)", client_id)
            except Exception as erreur:
                print(f"Erreur alerte IA {x['animal_id']} : {erreur}")
        nb = sum(1 for x in animaux_ia if x["statut"] == "a_surveiller")
        print(f"Bilan IA client {client['numero_client']} : {len(animaux_ia)} animaux analysés, "
              f"{nb} à surveiller (seuil {SEUIL_HEURES_ALERTE} h), {len(ignores)} ignorés")


def recalculer_alertes_ia():
    """Recalcul des alertes IA en tâche de fond après un changement de reference_jours.
    Ne lève jamais d'exception (ne doit pas faire échouer la réponse HTTP).
    Ignoré si le bilan automatique est désactivé (IA_AUTO=false)."""
    if not IA_AUTO:
        return
    try:
        executer_bilan_ia()
    except FileNotFoundError:
        print("Recalcul IA ignoré : modèle introuvable")
    except Exception as erreur:
        print(f"Erreur recalcul IA : {erreur}")


async def surveillance_ia():
    """AJOUT IA AUTO : lance le bilan à intervalle régulier."""
    await asyncio.sleep(DELAI_PREMIER_BILAN_S)
    while True:
        try:
            await asyncio.to_thread(executer_bilan_ia)
        except FileNotFoundError:
            print("Bilan IA ignoré : modèle introuvable")
        except Exception as erreur:
            print(f"Erreur bilan IA : {erreur}")
        await asyncio.sleep(INTERVALLE_BILAN_IA_S)


def appliquer_alerte_lstm(animal_id, condition, message, client_id):
    """AJOUT LSTM : branche l'alerte sur le système d'alertes existant (mode actif seulement)."""
    client_id = _client_id(client_id)
    doc = positions.find_one(
        {"client_id": client_id, "animal_id": animal_id},
        {"_id": 0}, sort=[("timestamp", -1)])
    if not doc:
        return
    doc["timestamp"] = datetime.now(timezone.utc)
    gerer_alerte(
        animal_id, "mouvement_agite", condition, message,
        Position(**{**doc, "client_id": str(client_id)}))


def executer_cycle_lstm():
    """AJOUT LSTM : un cycle d'analyse des trajectoires."""
    seuil = os.getenv("LSTM_SEUIL_M")          # vide = seuil du modèle
    bilan = {"animaux": 0, "alertes": 0}
    for client in clients_col.find({}, {"_id": 1, "numero_client": 1}):
        resultat = trajectoire_service.executer_cycle(
            base, seuil=float(seuil) if seuil else None,
            client_id=client["_id"])
        bilan["animaux"] += resultat["animaux"]
        bilan["alertes"] += resultat["alertes"]
    print(f"Cycle LSTM : {bilan}")
    return bilan


async def surveillance_lstm():
    """AJOUT LSTM : lance le cycle à intervalle régulier."""
    await asyncio.sleep(DELAI_PREMIER_LSTM_S)
    while True:
        try:
            await asyncio.to_thread(executer_cycle_lstm)
        except Exception as erreur:
            print(f"Erreur cycle LSTM : {erreur}")
        await asyncio.sleep(INTERVALLE_LSTM_S)


def creer_index_ttl():
    """AJOUT IA AUTO : supprime automatiquement les mesures d'activité de plus de 14 jours.

    Si un index simple « timestamp_1 » existe déjà sans TTL (code 85), on le remplace."""
    try:
        mesures_activite.create_index(
            "timestamp", expireAfterSeconds=DUREE_CONSERVATION_MESURES_S)
    except OperationFailure as erreur:
        if erreur.code != 85:
            print(f"Index TTL mesures_activite non créé : {erreur}")
            return
        try:
            mesures_activite.drop_index("timestamp_1")
            mesures_activite.create_index(
                "timestamp", expireAfterSeconds=DUREE_CONSERVATION_MESURES_S)
            print("Index TTL mesures_activite converti (14 jours)")
        except Exception as erreur2:
            print(f"Index TTL mesures_activite non converti : {erreur2}")
    except Exception as erreur:
        print(f"Index TTL mesures_activite non créé : {erreur}")


def creer_index_animaux():
    """Un seul document par animal. Un échec (ex. doublons déjà présents dans la
    collection) ne doit pas empêcher le serveur de démarrer."""
    try:
        indexes = animaux.index_information()
        if "animal_id_1" in indexes:
            animaux.drop_index("animal_id_1")
        animaux.create_index([("client_id", 1), ("animal_id", 1)], unique=True)
    except Exception as erreur:
        print(f"Index animaux non créé : {erreur}")


def creer_index_etat_animaux():
    try:
        if "animal_id_1" in etat_animaux.index_information():
            etat_animaux.drop_index("animal_id_1")
        etat_animaux.create_index(
            [("client_id", 1), ("animal_id", 1)], unique=True)
    except Exception as erreur:
        print(f"Index état animaux non créé : {erreur}")


def creer_index_nettoyage():
    """AJOUT NETTOYAGE : index sur la date pour que les suppressions par âge soient rapides.
    Un échec ne doit pas empêcher le serveur de démarrer."""
    for collection in (positions, alerts):
        try:
            collection.create_index("timestamp")
        except Exception as erreur:
            print(f"Index timestamp non créé ({collection.name}) : {erreur}")


# ---------------------------------------------------------------------------
# Nettoyage automatique de la base (AJOUT NETTOYAGE)
# ---------------------------------------------------------------------------
def nettoyer_base(client_id=None):
    """Supprime les anciennes données pour économiser le stockage.

    - alertes de TOUS types, actives ou résolues : supprimées uniquement quand leur
      âge dépasse DUREE_ALERTES_H (24 h). Avant 24 h, elles ne sont jamais touchées.
    - positions, mesures d'activité, événements LSTM et anciennes zones de pâturage
      plus vieux que DUREE_DONNEES_H (170 h)

    Ne touche jamais : fiches des animaux (animaux), paramètres, état courant des
    animaux (etat_animaux), zone de pâturage active et comptes utilisateurs."""
    maintenant = datetime.now(timezone.utc)
    limite_alertes = maintenant - timedelta(hours=DUREE_ALERTES_H)
    limite_donnees = maintenant - timedelta(hours=DUREE_DONNEES_H)
    bilan = {}

    # --- Alertes : uniquement celles dont l'âge dépasse DUREE_ALERTES_H ---
    filtre_alertes = {"timestamp": {"$lt": limite_alertes}}
    filtre_donnees = {}
    if client_id is not None:
        client_id = _client_id(client_id)
        filtre_alertes["client_id"] = client_id
        filtre_donnees["client_id"] = client_id

    # Les alertes encore actives qui vont disparaître sont retirées de l'application.
    # Si le problème persiste, une nouvelle alerte sera créée au prochain contrôle.
    for a in alerts.find({**filtre_alertes, "resolue": False},
                         {"animal_id": 1, "type": 1, "client_id": 1}):
        diffuseur.publier({"type": "alerte_resolue", "data": {
            "animal_id": a["animal_id"],
            "type": a["type"],
            "resolue_le": maintenant,
        }}, a["client_id"])
    bilan["alertes"] = alerts.delete_many(filtre_alertes).deleted_count

    # --- Autres données ---
    bilan["positions"] = positions.delete_many(
        {**filtre_donnees, "timestamp": {"$lt": limite_donnees}}).deleted_count
    bilan["mesures_activite"] = mesures_activite.delete_many(
        {**filtre_donnees, "timestamp": {"$lt": limite_donnees}}).deleted_count
    bilan["evenements_lstm"] = lstm_evenements.delete_many(
        {**filtre_donnees, "cree_le": {"$lt": limite_donnees}}).deleted_count
    bilan["anciennes_zones"] = pasture.delete_many(
        {**filtre_donnees, "active": False, "cree_le": {"$lt": limite_donnees}}).deleted_count

    total = sum(bilan.values())
    print(f"Nettoyage : {total} documents supprimés {bilan} "
          f"(alertes > {DUREE_ALERTES_H:g} h, données > {DUREE_DONNEES_H:g} h)")
    return bilan


async def surveillance_nettoyage():
    """AJOUT NETTOYAGE : lance le nettoyage à intervalle régulier."""
    await asyncio.sleep(DELAI_PREMIER_NETTOYAGE_S)
    while True:
        try:
            await asyncio.to_thread(nettoyer_base)
        except Exception as erreur:
            print(f"Erreur nettoyage : {erreur}")
        await asyncio.sleep(INTERVALLE_NETTOYAGE_S)


# ---------------------------------------------------------------------------
# Troupeau
# ---------------------------------------------------------------------------
def positions_actuelles(reference, client_id):
    client_id = _client_id(client_id)
    limite = reference - timedelta(seconds=P(client_id)["fraicheur_s"])
    resultat = []
    for animal_id in positions.distinct("animal_id", {"client_id": client_id}):
        doc = positions.find_one(
            {"client_id": client_id, "animal_id": animal_id},
            {"_id": 0}, sort=[("timestamp", -1)]
        )
        if doc and doc["timestamp"] >= limite:
            resultat.append(doc)
    return resultat


def nb_animaux_en_alerte(reference, client_id):
    limite = reference - timedelta(seconds=P(client_id)["fraicheur_s"])
    return etat_animaux.count_documents(
        {"client_id": _client_id(client_id), "alerte": True,
         "derniere_mesure": {"$gte": limite}}
    )


def verifier_troupeau(p, reevaluation=False):
    """Sortie de zone du troupeau, confirmée sur plusieurs messages.

    reevaluation=True : appelé après un changement de paramètres.
    - pas de diffusion du troupeau ici (faite une seule fois à la fin)
    - la confirmation est immédiate (on juge la dernière position telle quelle)
    """
    client_id = _client_id(p.client_id)
    prm = P(client_id)
    actuelles = positions_actuelles(p.timestamp, client_id)
    if len(actuelles) < prm["min_animaux_troupeau"]:
        if not reevaluation:
            diffuseur.publier({"type": "troupeau", "data": {
                "statut": "pas assez d'animaux actifs",
                "animaux_actifs": len(actuelles)}}, client_id)
        return
    principal, autres, ecarts = analyser_troupeaux(actuelles, prm)

    # AJOUT ZONE AUTO : la zone de pâturage automatique suit le grand troupeau
    try:
        maj_zone_auto(principal, client_id, p.numero_client)
    except Exception as erreur:
        print(f"Erreur maj_zone_auto : {erreur}")

    if not reevaluation:
        diffuseur.publier({"type": "troupeau", "data": {
            "statut": "ok", "animaux_actifs": len(actuelles),
            "grand_troupeau": principal, "autres_groupes": autres}}, client_id)

    ecart = ecarts.get(p.animal_id)
    hors = ecart is not None and ecart > prm["distance_alerte_m"]

    etat = etat_animaux.find_one(
        {"client_id": client_id, "animal_id": p.animal_id}) or {}
    if reevaluation:
        nb_hors = prm["nb_confirmations"] if hors else 0
    else:
        nb_hors = etat.get("nb_hors_zone", 0) + 1 if hors else 0
    alerte = nb_hors >= prm["nb_confirmations"]

    etat_animaux.update_one(
        {"client_id": client_id, "animal_id": p.animal_id},
        {"$set": {"nb_hors_zone": nb_hors, "alerte": alerte,
                  "ecart_m": ecart if ecart is not None else 0.0,
                  "derniere_mesure": p.timestamp,
                  "numero_client": p.numero_client}},
        upsert=True,
    )
    total = nb_animaux_en_alerte(p.timestamp, client_id)
    message = (f"À {round(ecart or 0)} m du grand troupeau "
               f"(confirmé sur {prm['nb_confirmations']} mesures). "
               f"Animaux hors zone : {total}")
    gerer_alerte(p.animal_id, "sortie_zone", alerte, message, p)


# ---------------------------------------------------------------------------
# Géorepérage
# ---------------------------------------------------------------------------
def verifier_pature(p):
    client_id = _client_id(p.client_id)
    zone = zone_active(client_id)
    if not zone:
        return
    sommets = [(c["latitude"], c["longitude"]) for c in zone["coordinates"]]
    dedans, bord = analyser_point(p.lat, p.lon, sommets)
    hors = (not dedans) and bord > P(client_id)["marge_pature_m"]
    gerer_alerte(
        p.animal_id, "hors_pature", hors,
        f"Hors de la zone « {zone['name']} », à {round(bord)} m du bord", p,
    )


# ---------------------------------------------------------------------------
# Alertes individuelles
# ---------------------------------------------------------------------------
def verifier_alertes_individuelles(p):
    client_id = _client_id(p.client_id)
    prm = P(client_id)
    duree = prm["duree_immobilite_s"]
    debut = p.timestamp - timedelta(seconds=duree)
    fenetre = list(positions.find(
        {"client_id": client_id, "animal_id": p.animal_id,
         "timestamp": {"$gte": debut, "$lte": p.timestamp}},
        {"_id": 0}).sort("timestamp", 1))
    repere = positions.find_one(
        {"client_id": client_id, "animal_id": p.animal_id,
         "timestamp": {"$lt": debut}},
        {"_id": 0}, sort=[("timestamp", -1)])
    if repere:
        fenetre.insert(0, repere)
    gerer_alerte(
        p.animal_id, "immobilite",
        est_immobile(fenetre, duree, prm["vitesse_repos_max"]),
        f"Au repos depuis plus de {duree // 60} min", p)

    nb = prm["nb_mesures_vitesse"]
    dernieres = list(positions.find(
                     {"client_id": client_id, "animal_id": p.animal_id}, {"_id": 0})
                     .sort("timestamp", -1).limit(nb))[::-1]
    gerer_alerte(
        p.animal_id, "deplacement_inhabituel",
        deplacement_inhabituel(dernieres, prm["vitesse_max"], nb),
        f"Vitesse supérieure à {prm['vitesse_max']} km/h sur {nb} mesures", p)


# ---------------------------------------------------------------------------
# Collier silencieux
# ---------------------------------------------------------------------------
def verifier_colliers(client_id=None):
    maintenant = datetime.now(timezone.utc)
    filtre = {"client_id": _client_id(client_id)} if client_id is not None else {}
    for animal in animaux.find(
        filtre, {"animal_id": 1, "client_id": 1, "numero_client": 1, "ferme": 1}):
        animal_client_id = animal["client_id"]
        animal_id = animal["animal_id"]
        delai = P(animal_client_id)["delai_collier_s"]
        doc = positions.find_one(
            {"client_id": animal_client_id, "animal_id": animal_id},
            {"_id": 0}, sort=[("timestamp", -1)])
        if not doc:
            continue
        age = (maintenant - doc["timestamp"]).total_seconds()
        doc["timestamp"] = maintenant
        gerer_alerte(animal_id, "collier_silencieux", age > delai,
                     f"Aucun message depuis plus de {delai // 60} min",
                     Position(**{**doc, "client_id": str(animal_client_id),
                                 "numero_client": animal["numero_client"],
                                 "ferme": animal.get("ferme")}))


async def surveillance_colliers():
    while True:
        try:
            await asyncio.to_thread(verifier_colliers)
        except Exception as erreur:
            print(f"Erreur vérification colliers : {erreur}")
        await asyncio.sleep(INTERVALLE_VERIF_COLLIERS_S)


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connecté au broker MQTT")
        client.subscribe(MQTT_TOPIC, qos=1)
        print(f"Abonné à {MQTT_TOPIC}")
    else:
        print(f"Échec de connexion : {reason_code}")


def on_message(mqtt_client_instance, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        position = Position(**data)
    except (json.JSONDecodeError, ValidationError) as erreur:
        print(f"Message rejeté sur {msg.topic} : {erreur}")
        return

    if position.timestamp.tzinfo is None:
        position.timestamp = position.timestamp.replace(tzinfo=timezone.utc)

    if position.numero_client:
        client_doc = clients_col.find_one({"numero_client": position.numero_client.strip()})
        if client_doc is None:
            print(f"Message rejeté : numéro client inconnu ({position.numero_client})")
            return
        ferme = (position.ferme or "Ferme principale").strip() or "Ferme principale"
        animal = animaux.find_one({
            "client_id": client_doc["_id"],
            "animal_id": position.animal_id,
        })
        if animal is None:
            animaux.insert_one({
                "client_id": client_doc["_id"],
                "numero_client": client_doc["numero_client"],
                "animal_id": position.animal_id,
                "ferme": ferme,
                "cree_le": datetime.now(timezone.utc),
            })
        else:
            ferme = animal.get("ferme") or ferme
    else:
        proprietaires = list(animaux.find(
            {"animal_id": position.animal_id},
            {"client_id": 1, "numero_client": 1, "ferme": 1},
        ).limit(2))
        if len(proprietaires) != 1:
            print(f"Message rejeté : numéro client manquant ou animal ambigu ({position.animal_id})")
            return
        animal = proprietaires[0]
        client_doc = clients_col.find_one({"_id": animal["client_id"]})
        if client_doc is None:
            print(f"Message rejeté : client introuvable pour {position.animal_id}")
            return
        ferme = animal.get("ferme", "Ferme principale")

    position = position.model_copy(update={
        "client_id": str(client_doc["_id"]),
        "numero_client": client_doc["numero_client"],
        "ferme": ferme,
    })
    document = position.model_dump()
    document["client_id"] = client_doc["_id"]
    positions.insert_one(document)
    print(f"Enregistré : {position.animal_id} ({position.lat}, {position.lon})")
    diffuseur.publier({"type": "position",
                       "data": position.model_dump(exclude={"client_id"})},
                      client_doc["_id"])

    # AJOUT IA : activité du MPU6050 (mi, steps, duree_s) conservée pour le bilan Isolation Forest
    if "mi" in data and "steps" in data:
        try:
            mesures_activite.insert_one({
                "animal_id": position.animal_id, "timestamp": position.timestamp,
                "client_id": client_doc["_id"], "numero_client": client_doc["numero_client"],
                "mi": float(data["mi"]), "steps": float(data["steps"]),
                "duree_s": float(data["duree_s"]) if data.get("duree_s") is not None else None})
        except Exception as erreur:
            print(f"Erreur mesures_activite : {erreur}")

    try:
        gerer_alerte(position.animal_id, "collier_silencieux", False, "", position)
    except Exception as erreur:
        print(f"Erreur collier_silencieux : {erreur}")

    for verification in (verifier_troupeau, verifier_pature,
                         verifier_alertes_individuelles):
        try:
            verification(position)
        except Exception as erreur:
            print(f"Erreur {verification.__name__} : {erreur}")


mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
mqtt_client.tls_set(ca_certs=CA_FILE)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    diffuseur.boucle = asyncio.get_running_loop()
    await asyncio.to_thread(
        positions.create_index, [("animal_id", 1), ("timestamp", -1)]
    )
    await asyncio.to_thread(
        creer_index_etat_animaux
    )
    await asyncio.to_thread(creer_index_animaux)        # AJOUT FICHES ANIMAUX (sans bloquer le démarrage)
    await asyncio.to_thread(
        mesures_activite.create_index, [("animal_id", 1), ("timestamp", -1)]
    )      # AJOUT IA
    await asyncio.to_thread(creer_index_ttl)            # AJOUT IA AUTO
    await asyncio.to_thread(creer_index_nettoyage)      # AJOUT NETTOYAGE
    mqtt_client.connect(MQTT_HOST, MQTT_PORT)
    mqtt_client.loop_start()
    tache = asyncio.create_task(surveillance_colliers())
    tache_ia = asyncio.create_task(surveillance_ia()) if IA_AUTO else None   # AJOUT IA AUTO
    tache_lstm = asyncio.create_task(surveillance_lstm()) if LSTM_AUTO else None   # AJOUT LSTM
    tache_nettoyage = (asyncio.create_task(surveillance_nettoyage())
                       if NETTOYAGE_AUTO else None)     # AJOUT NETTOYAGE
    yield
    tache.cancel()
    if tache_ia:
        tache_ia.cancel()
    if tache_lstm:
        tache_lstm.cancel()
    if tache_nettoyage:
        tache_nettoyage.cancel()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    mongo_client.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["*"],
)
# AJOUT IA : /ia/bilan, /ia/historique ; la durée de référence est lue à chaque appel
app.include_router(creer_router(mesures_activite, appliquer_alerte_ia,
                                lambda client_id: P(client_id)["reference_jours"],
                                courant))
# AJOUT AUTH : /auth/etat, /auth/inscription, /auth/connexion, /auth/moi, /utilisateurs
app.include_router(creer_routeur(base))
app.include_router(creer_routeur_messagerie(base))


# ---------------------------------------------------------------------------
# Zone de pâturage automatique (AJOUT ZONE AUTO)
#   côté du carré = 2 × rayon(grand troupeau) + distance_alerte_m
#   - rayon : fonction du NOMBRE d'animaux du grand troupeau (surface estimée par
#     animal), jamais inférieur au rayon réellement mesuré
#   - distance_alerte_m : réglage de l'utilisateur (page Réglages), relu à chaque calcul
# ---------------------------------------------------------------------------
INTERVALLE_MAJ_ZONE_S = 30          # recalcul périodique de la zone
TOLERANCE_REDUCTION_M = 10          # la zone ne rétrécit pas pour moins de 10 m
_maj_zone = {}


def _zone_publique(zone):
    """Ce que voient les clients (REST + WebSocket)."""
    return {"name": zone["name"], "coordinates": zone["coordinates"],
            "mode": zone.get("mode", "manuel"), "marge_m": zone.get("marge_m")}


def _demi_cote_m(principal, client_id):
    """Demi-côté du carré : rayon du troupeau + moitié de distance_alerte_m."""
    n = max(int(principal.get("nb_animaux", 1)), 1)
    rayon_estime = math.sqrt(n * SURFACE_PAR_ANIMAL_M2 / math.pi)
    rayon = max(rayon_estime, float(principal.get("rayon_m", 0.0)))
    return rayon + P(client_id)["distance_alerte_m"] / 2


def _rectangle(clat, clon, demi_cote_m):
    """Carré (4 sommets) centré sur (clat, clon)."""
    d_lat = math.degrees(demi_cote_m / RAYON_TERRE_M)
    d_lon = math.degrees(demi_cote_m / (RAYON_TERRE_M * math.cos(math.radians(clat))))
    return [(clat - d_lat, clon - d_lon), (clat + d_lat, clon - d_lon),
            (clat + d_lat, clon + d_lon), (clat - d_lat, clon + d_lon)]


def _demi_cote_actuel(zone):
    lats = [c["latitude"] for c in zone["coordinates"]]
    return math.radians(max(lats) - min(lats)) * RAYON_TERRE_M / 2


def _grand_troupeau(client_id):
    """Zone (centre, rayon, nb_animaux) du grand troupeau actuel, ou None."""
    client_id = _client_id(client_id)
    derniere = positions.find_one(
        {"client_id": client_id}, {"_id": 0, "client_id": 0},
        sort=[("timestamp", -1)])
    if not derniere:
        return None
    actuelles = positions_actuelles(derniere["timestamp"], client_id)
    if len(actuelles) < P(client_id)["min_animaux_troupeau"]:
        return None
    return analyser_troupeaux(actuelles, P(client_id))[0]


def _sommets_auto(client_id):
    principal = _grand_troupeau(client_id)
    if principal is None:
        raise ValueError("Pas assez d'animaux actifs pour calculer la zone automatique.")
    return _rectangle(principal["centre_lat"], principal["centre_lon"],
                      _demi_cote_m(principal, client_id))


def _reverifier_pature_tous(client_id):
    client_id = _client_id(client_id)
    for animal_id in positions.distinct("animal_id", {"client_id": client_id}):
        d = positions.find_one(
            {"client_id": client_id, "animal_id": animal_id},
            sort=[("timestamp", -1)])
        if d:
            try:
                verifier_pature(Position(**{
                    **{k: d[k] for k in Position.model_fields if k in d},
                    "client_id": str(client_id),
                }))
            except Exception as e:
                print(f"Vérification pâturage ignorée : {e}")


def _poser_zone(client_id, numero_client, nom, sommets, mode):
    """Remplace la zone active et prévient les clients."""
    client_id = _client_id(client_id)
    cle = str(client_id)
    zone = {
        "name": nom,
        "coordinates": [{"latitude": a, "longitude": b} for a, b in sommets],
        "cree_le": datetime.now(timezone.utc),
        "active": True,
        "mode": mode,
        "marge_m": P(client_id)["distance_alerte_m"] if mode == "auto" else None,
        "client_id": client_id,
        "numero_client": numero_client,
    }
    pasture.update_many({"client_id": client_id, "active": True},
                        {"$set": {"active": False}})
    pasture.insert_one(dict(zone))
    _cache_zone[cle] = zone
    _maj_zone[cle] = time.monotonic()
    diffuseur.publier({"type": "pature", "data": _zone_publique(zone)}, client_id)
    return zone


def maj_zone_auto(principal, client_id, numero_client, forcer=False):
    """Appelée depuis verifier_troupeau (au plus toutes les 30 s) et, avec
    forcer=True, immédiatement après un changement de réglages.
    - aucune zone : en crée une automatique ;
    - zone automatique : la recentre et la redimensionne ;
    Une zone manuelle (administrateur) n'est jamais touchée."""
    if not principal:
        return
    client_id = _client_id(client_id)
    cle = str(client_id)
    if not forcer and time.monotonic() - _maj_zone.get(cle, 0) < INTERVALLE_MAJ_ZONE_S:
        return
    zone = zone_active(client_id)
    demi = _demi_cote_m(principal, client_id)
    if not zone:
        if ZONE_AUTO:
            sommets = _rectangle(principal["centre_lat"], principal["centre_lon"], demi)
            _poser_zone(client_id, numero_client, "Zone de pâturage", sommets, "auto")
            _reverifier_pature_tous(client_id)
        return
    if zone.get("mode") != "auto":
        return
    _maj_zone[cle] = time.monotonic()
    ancien = _demi_cote_actuel(zone)
    if not forcer and demi < ancien and ancien - demi < TOLERANCE_REDUCTION_M:
        demi = ancien                       # évite que la zone « respire »
    sommets = _rectangle(principal["centre_lat"], principal["centre_lon"], demi)
    coords = [{"latitude": a, "longitude": b} for a, b in sommets]
    marge = P(client_id)["distance_alerte_m"]
    pasture.update_one({"client_id": client_id, "active": True},
                       {"$set": {"coordinates": coords, "marge_m": marge}})
    zone = {**zone, "coordinates": coords, "marge_m": marge}
    _cache_zone[cle] = zone
    diffuseur.publier({"type": "pature", "data": _zone_publique(zone)}, client_id)


def recalculer_zone_auto(client_id, numero_client):
    """Recalcul immédiat (changement de distance_alerte_m ou d'autres réglages)."""
    try:
        maj_zone_auto(
            _grand_troupeau(client_id), client_id, numero_client, forcer=True)
    except Exception as e:
        print(f"Erreur recalcul zone : {e}")


def exiger_admin(x_admin_key: Optional[str] = Header(None)):
    """Modifier la zone à la main est réservé à l'administrateur (en-tête X-Admin-Key).
    Sans ADMIN_KEY définie, toute modification manuelle est refusée."""
    if not ADMIN_KEY or not secrets.compare_digest(x_admin_key or "", ADMIN_KEY):
        raise HTTPException(status_code=403,
                            detail="Zone gérée automatiquement par le serveur.")


@app.post("/pastures", dependencies=[Depends(exiger_admin)])
def creer_pature(donnees: dict, u: dict = Depends(courant)):
    """Administrateur. {"mode":"auto"} ou {"coordinates":[{"latitude":..,"longitude":..}, ...]}"""
    mode = "auto" if donnees.get("mode") == "auto" else "manuel"
    try:
        sommets = (_sommets_auto(u["client_id"]) if mode == "auto"
                   else valider_sommets(donnees.get("coordinates")))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _poser_zone(u["client_id"], u["numero_client"],
                str(donnees.get("name") or "Zone de pâturage")[:80], sommets, mode)
    _reverifier_pature_tous(u["client_id"])
    return {"statut": "ok", "mode": mode, "sommets": len(sommets)}


@app.put("/pastures", dependencies=[Depends(exiger_admin)])
def modifier_pature(donnees: dict, u: dict = Depends(courant)):
    """Administrateur. {"mode":"auto"} | {"mode":"manuel"} (fige) | {"coordinates":[...]}"""
    actuelle = zone_active(u["client_id"])
    if not actuelle:
        raise HTTPException(status_code=404, detail="Aucune zone enregistrée.")
    if "coordinates" in donnees:
        mode = "manuel"
    else:
        mode = "auto" if donnees.get("mode", actuelle.get("mode")) == "auto" else "manuel"
    try:
        if mode == "auto":
            sommets = _sommets_auto(u["client_id"])
        elif "coordinates" in donnees:
            sommets = valider_sommets(donnees["coordinates"])
        else:
            sommets = [(c["latitude"], c["longitude"]) for c in actuelle["coordinates"]]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _poser_zone(u["client_id"], u["numero_client"],
                str(donnees.get("name") or actuelle["name"])[:80], sommets, mode)
    _reverifier_pature_tous(u["client_id"])
    return {"statut": "ok", "mode": mode, "sommets": len(sommets)}


@app.get("/pastures")
def lire_pature(u: dict = Depends(courant)):
    zone = zone_active(u["client_id"])
    if not zone:
        return {"statut": "aucune zone"}
    return _zone_publique(zone)


@app.delete("/pastures", dependencies=[Depends(exiger_admin)])
def supprimer_pature(u: dict = Depends(courant)):
    """Désactive la zone, résout les alertes hors_pature, prévient les clients."""
    client_id = u["client_id"]
    pasture.update_many({"client_id": client_id, "active": True},
                        {"$set": {"active": False}})
    _cache_zone[str(client_id)] = None

    maintenant = datetime.now(timezone.utc)
    nb_resolues = 0
    for a in list(alerts.find({
        "client_id": client_id, "type": "hors_pature", "resolue": False,
    })):
        alerts.update_one(
            {"_id": a["_id"], "client_id": client_id},
            {"$set": {"resolue": True, "resolue_le": maintenant}},
        )
        diffuseur.publier({"type": "alerte_resolue", "data": {
            "animal_id": a["animal_id"],
            "type": "hors_pature",
            "resolue_le": maintenant,
        }}, client_id)
        nb_resolues += 1

    diffuseur.publier({"type": "pature", "data": None}, client_id)
    return {"statut": "ok", "alertes_resolues": nb_resolues}


@app.get("/positions")
def lire_positions(animal_id: Optional[str] = None, limite: int = 50,
                   u: dict = Depends(courant)):
    filtre = {"client_id": u["client_id"]}
    if animal_id:
        filtre["animal_id"] = animal_id
    return list(positions.find(
        filtre, {"_id": 0, "client_id": 0}).sort("timestamp", -1).limit(limite))


@app.get("/alertes")
def lire_alertes(animal_id: Optional[str] = None, actives: bool = False, limite: int = 50,
                 u: dict = Depends(courant)):
    filtre = {"client_id": u["client_id"]}
    if animal_id:
        filtre["animal_id"] = animal_id
    if actives:
        filtre["resolue"] = False
    return list(alerts.find(
        filtre, {"_id": 0, "client_id": 0}).sort("timestamp", -1).limit(limite))


@app.get("/lstm/evenements")
def lire_evenements_lstm(animal_id: Optional[str] = None, limite: int = 50,
                         u: dict = Depends(courant)):
    filtre = {"client_id": u["client_id"]}
    if animal_id:
        filtre["animal_id"] = animal_id
    return list(base["lstm_alertes_silencieuses"].find(
        filtre, {"_id": 0, "client_id": 0})
                .sort("cree_le", -1).limit(limite))


# AJOUT NETTOYAGE : lancer le nettoyage à la demande (administrateur, en-tête X-Admin-Key)
@app.post("/nettoyage", dependencies=[Depends(exiger_admin)])
def lancer_nettoyage(u: dict = Depends(courant)):
    """Exécute nettoyer_base() immédiatement et renvoie le nombre de documents
    supprimés par collection."""
    return nettoyer_base(u["client_id"])


# ---------------------------------------------------------------------------
# Fiches des animaux (type, sexe, âge, poids)
# ---------------------------------------------------------------------------
@app.get("/animaux")
def lire_animaux(u: dict = Depends(courant)):
    """Liste des fiches enregistrées (utilisée par la page « Mon troupeau »)."""
    fiches = animaux.find(
        {"client_id": u["client_id"]},
        {"_id": 0, "modifie_le": 0, "client_id": 0},
    )
    resultat = []
    for fiche in fiches:
        image_id = fiche.pop("image_id", None)
        fiche.pop("image_content_type", None)
        photo_modifiee = fiche.pop("image_modifie_le", None)
        fiche["photo_disponible"] = image_id is not None
        fiche["photo_version"] = (
            photo_modifiee.isoformat() if photo_modifiee is not None else None
        )
        resultat.append(fiche)
    return resultat


@app.put("/animaux/{animal_id}")
def enregistrer_animal(animal_id: str, info: InfoAnimal,
                       u: dict = Depends(courant)):
    """Crée ou met à jour la fiche d'un animal."""
    fiche = info.model_dump()
    filtre = {"client_id": u["client_id"], "animal_id": animal_id}
    animaux.update_one(
        filtre,
        {"$set": {**fiche, "numero_client": u["numero_client"],
                  "modifie_le": datetime.now(timezone.utc)},
         "$setOnInsert": {"client_id": u["client_id"], "animal_id": animal_id}},
        upsert=True,
    )
    actuel = animaux.find_one(filtre, {"image_id": 1, "image_modifie_le": 1}) or {}
    photo_modifiee = actuel.get("image_modifie_le")
    return {
        "animal_id": animal_id,
        **fiche,
        "numero_client": u["numero_client"],
        "photo_disponible": actuel.get("image_id") is not None,
        "photo_version": (
            photo_modifiee.isoformat() if photo_modifiee is not None else None
        ),
    }


@app.put("/animaux/{animal_id}/image")
def enregistrer_photo_animal(
    animal_id: str,
    photo: PhotoAnimal,
    u: dict = Depends(courant),
):
    contenu = decoder_photo_animal(photo)
    filtre = {"client_id": u["client_id"], "animal_id": animal_id}
    actuel = animaux.find_one(filtre, {"image_id": 1}) or {}
    image_id = photos_animaux.upload_from_stream(
        f"{animal_id}",
        io.BytesIO(contenu),
        metadata={
            "client_id": str(u["client_id"]),
            "animal_id": animal_id,
            "content_type": photo.type_contenu,
        },
    )
    photo_modifiee = datetime.now(timezone.utc)
    try:
        animaux.update_one(
            filtre,
            {
                "$set": {
                    "image_id": image_id,
                    "image_content_type": photo.type_contenu,
                    "image_modifie_le": photo_modifiee,
                    "numero_client": u["numero_client"],
                },
                "$setOnInsert": {
                    "client_id": u["client_id"],
                    "animal_id": animal_id,
                },
            },
            upsert=True,
        )
    except PyMongoError:
        photos_animaux.delete(image_id)
        raise

    ancienne_image = actuel.get("image_id")
    if ancienne_image is not None:
        try:
            photos_animaux.delete(ancienne_image)
        except NoFile:
            pass

    return {
        "animal_id": animal_id,
        "photo_disponible": True,
        "photo_version": photo_modifiee.isoformat(),
    }


@app.get("/animaux/{animal_id}/image")
def lire_photo_animal(animal_id: str, u: dict = Depends(courant)):
    fiche = animaux.find_one(
        {"client_id": u["client_id"], "animal_id": animal_id},
        {"image_id": 1, "image_content_type": 1},
    )
    if fiche is None or fiche.get("image_id") is None:
        raise HTTPException(404, "Aucune photo n’est enregistrée pour cet animal.")
    try:
        photo = photos_animaux.open_download_stream(fiche["image_id"])
    except NoFile:
        raise HTTPException(404, "La photo de cet animal est introuvable.")
    return Response(
        content=photo.read(),
        media_type=fiche.get("image_content_type", "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/animaux/{animal_id}/sante")
def lire_evenements_sante(animal_id: str, u: dict = Depends(courant)):
    filtre = {"client_id": u["client_id"], "animal_id": animal_id}
    evenements = sante_animaux.find(filtre, {"_id": 0, "client_id": 0})
    resultat = []
    for evenement in evenements:
        evenement["id"] = evenement.pop("event_id")
        resultat.append(evenement)
    resultat.sort(key=lambda evenement: evenement["date"], reverse=True)
    return resultat


@app.put("/animaux/{animal_id}/sante/{evenement_id}")
def enregistrer_evenement_sante(
    animal_id: str,
    evenement_id: str,
    evenement: EvenementSante,
    u: dict = Depends(courant),
):
    filtre = {
        "client_id": u["client_id"],
        "animal_id": animal_id,
        "event_id": evenement_id,
    }
    donnees = evenement.model_dump()
    sante_animaux.update_one(
        filtre,
        {
            "$set": {
                **donnees,
                "animal_id": animal_id,
                "event_id": evenement_id,
                "client_id": u["client_id"],
                "numero_client": u["numero_client"],
            }
        },
        upsert=True,
    )
    return {"id": evenement_id, "animal_id": animal_id, **donnees}


@app.get("/troupeau")
def etat_troupeau(u: dict = Depends(courant)):
    client_id = u["client_id"]
    prm = P(client_id)
    derniere = positions.find_one(
        {"client_id": client_id}, {"_id": 0}, sort=[("timestamp", -1)])
    if not derniere:
        return {"statut": "aucune donnée"}
    reference = derniere["timestamp"]
    actuelles = positions_actuelles(reference, client_id)
    if len(actuelles) < prm["min_animaux_troupeau"]:
        return {"statut": "pas assez d'animaux actifs", "animaux_actifs": len(actuelles)}
    principal, autres, _ = analyser_troupeaux(actuelles, prm)
    limite = reference - timedelta(seconds=prm["fraicheur_s"])
    en_alerte = list(etat_animaux.find(
        {"client_id": client_id, "alerte": True,
         "derniere_mesure": {"$gte": limite}},
        {"_id": 0, "animal_id": 1, "ecart_m": 1},
    ))
    return {
        "statut": "ok",
        "animaux_actifs": len(actuelles),
        "grand_troupeau": principal,
        "autres_groupes": autres,
        "nb_animaux_en_alerte": len(en_alerte),
        "animaux_en_alerte": en_alerte,
    }


def rediffuser_troupeau(u):
    """Recalcule le troupeau avec les paramètres actuels et prévient les clients."""
    try:
        diffuseur.publier({"type": "troupeau", "data": etat_troupeau(u)}, u["client_id"])
    except Exception as e:
        print(f"Rediffusion troupeau ignorée : {e}")


def reevaluer_tous(client_id, numero_client):
    """AJOUT : rejoue toutes les vérifications sur la dernière position de
    chaque animal avec les nouveaux paramètres, sans attendre le prochain
    message MQTT. Alertes ouvertes/fermées et diffusées immédiatement."""
    recalculer_zone_auto(client_id, numero_client)
    client_id = _client_id(client_id)
    for animal_id in positions.distinct("animal_id", {"client_id": client_id}):
        d = positions.find_one(
            {"client_id": client_id, "animal_id": animal_id},
            sort=[("timestamp", -1)])
        if not d:
            continue
        try:
            p = Position(**{
                **{k: d[k] for k in Position.model_fields if k in d},
                "client_id": str(client_id),
            })
        except Exception as e:
            print(f"Réévaluation ignorée ({animal_id}) : {e}")
            continue
        try:
            verifier_troupeau(p, reevaluation=True)
        except Exception as e:
            print(f"Erreur verifier_troupeau : {e}")
        for verification in (verifier_pature, verifier_alertes_individuelles):
            try:
                verification(p)
            except Exception as e:
                print(f"Erreur {verification.__name__} : {e}")
    try:
        verifier_colliers(client_id)
    except Exception as e:
        print(f"Erreur verifier_colliers : {e}")
    utilisateur = {"client_id": client_id}
    rediffuser_troupeau(utilisateur)


@app.get("/parametres")
def lire_parametres(u: dict = Depends(courant)):
    return {"valeurs": P(u["client_id"]), "defauts": DEFAUTS,
            "bornes": {k: [b[0], b[1]] for k, b in BORNES.items()}}


@app.put("/parametres")
def modifier_parametres(donnees: dict, taches: BackgroundTasks,
                        u: dict = Depends(courant)):
    try:
        propres = valider_parametres(donnees)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    propres["reference_jours"] = REFERENCE_JOURS
    client_id = u["client_id"]
    ancienne_reference = P(client_id)["reference_jours"]
    nouveaux = {**P(client_id), **propres}
    parametres_col.update_one(
        {"client_id": client_id},
        {"$set": {**nouveaux, "numero_client": u["numero_client"]},
         "$setOnInsert": {"client_id": client_id}},
        upsert=True,
    )
    _cache_params[str(client_id)] = nouveaux
    taches.add_task(reevaluer_tous, client_id, u["numero_client"])
    if nouveaux["reference_jours"] != ancienne_reference:
        taches.add_task(recalculer_alertes_ia)  # AJOUT : alertes IA recalculées tout de suite
    return {"statut": "ok", "valeurs": nouveaux}


def etat_complet(u: dict) -> dict:
    """Photo de départ envoyée à chaque nouveau client WebSocket."""
    dernieres = []
    client_id = u["client_id"]
    for animal_id in positions.distinct("animal_id", {"client_id": client_id}):
        doc = positions.find_one(
            {"client_id": client_id, "animal_id": animal_id},
            {"_id": 0, "client_id": 0}, sort=[("timestamp", -1)]
        )
        if doc:
            dernieres.append(doc)
    actives = list(
        alerts.find({"client_id": client_id, "resolue": False},
                    {"_id": 0, "client_id": 0})
        .sort("timestamp", -1).limit(100)
    )
    zone = zone_active(client_id)
    return {
        "type": "etat",
        "positions": dernieres,
        "alertes_actives": actives,
        "troupeau": etat_troupeau(u),
        "pature": _zone_publique(zone) if zone else None,   # MODIFIÉ : mode + marge
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        identifiants = await ws.receive_json()
    except (WebSocketDisconnect, json.JSONDecodeError):
        return
    if not isinstance(identifiants, dict) or not isinstance(identifiants.get("token"), str):
        await ws.close(code=4401)
        return
    try:
        u = courant(authorization=f"Bearer {identifiants['token']}")
    except HTTPException:
        await ws.close(code=4401)
        return
    try:
        etat = await asyncio.to_thread(etat_complet, u)
        await ws.send_text(json.dumps(etat, default=_json))
        diffuseur.clients[ws] = u["client_id"]
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        diffuseur.clients.pop(ws, None)
        
        
# À COLLER dans main.py, juste après la route /lstm/evenements (lire_evenements_lstm).
# Aucun import à ajouter : datetime, timedelta, timezone, HTTPException,
# trajectoire_service et la collection `positions` existent déjà dans main.py.


@app.get("/lstm/trajectoire")
def lire_trajectoire_lstm(animal_id: str, instant_local: str,
                          avant_min: int = 90, apres_min: int = 15,
                          u: dict = Depends(courant)):
    """Positions de l'animal autour d'une alerte (heures renvoyées en heure locale, sans fuseau)."""
    tz = trajectoire_service.TZ_LOCAL
    try:
        local = datetime.fromisoformat(instant_local).replace(tzinfo=None)   # déjà local
    except ValueError:
        raise HTTPException(status_code=422, detail="instant_local invalide")
    centre = local.replace(tzinfo=tz).astimezone(timezone.utc)
    docs = positions.find(
        {"client_id": u["client_id"], "animal_id": animal_id,
         "timestamp": {"$gte": centre - timedelta(minutes=avant_min),
                       "$lte": centre + timedelta(minutes=apres_min)}},
        {"_id": 0, "timestamp": 1, "lat": 1, "lon": 1},
    ).sort("timestamp", 1)

    points = []
    for d in docs:
        if d.get("lat") is None or d.get("lon") is None:
            continue
        t = d["timestamp"]
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        points.append({
            "t": t.astimezone(tz).replace(tzinfo=None).isoformat(timespec="seconds"),
            "lat": d["lat"],
            "lon": d["lon"],
        })
    return points