import asyncio
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
                     WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from pymongo import MongoClient
from pymongo.errors import OperationFailure

from DBSCAN import analyser_troupeaux, deplacement_inhabituel, est_immobile
from geofence import analyser_point, valider_sommets
from IA.ia_routes import creer_router, calculer_bilan, SEUIL_HEURES_ALERTE   # AJOUT IA
from params import BORNES, DEFAUTS, valider_parametres
import trajectoire_service                                   # AJOUT LSTM

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
SURFACE_PAR_ANIMAL_M2 = float(os.getenv("SURFACE_PAR_ANIMAL_M2", "400"))   # place estimée par animal
ADMIN_KEY = os.getenv("ADMIN_KEY")                              # clé pour modifier la zone à la main (facultatif)

INTERVALLE_VERIF_COLLIERS_S = 60   # vérification des colliers silencieux

# AJOUT IA AUTO : bilan Isolation Forest lancé automatiquement
IA_AUTO = os.getenv("IA_AUTO", "true").lower() != "false"          # IA_AUTO=false pour le désactiver
INTERVALLE_BILAN_IA_S = int(os.getenv("IA_INTERVALLE_S", "3600"))  # une fois par heure
DELAI_PREMIER_BILAN_S = 60                                         # laisse le serveur démarrer
DUREE_CONSERVATION_MESURES_S = 30 * 24 * 3600                      # TTL de mesures_activite : 30 jours

# AJOUT LSTM : alerte « mouvement agité », évaluée toutes les 15 min
LSTM_AUTO = os.getenv("LSTM_AUTO", "true").lower() != "false"
INTERVALLE_LSTM_S = int(os.getenv("LSTM_INTERVALLE_S", "900"))
DELAI_PREMIER_LSTM_S = 90

mongo_client = MongoClient(MONGO_URI, tz_aware=True)
base = mongo_client["traceur_gps"]
positions = base["positions"]
alerts = base["alerts"]
etat_animaux = base["etat_animaux"]
animaux = base["animaux"]                      # fiches des animaux (type, sexe, âge, poids)
pasture = base["pastures"]
parametres_col = base["parametres"]
mesures_activite = base["mesures_activite"]    # AJOUT IA : un document par message (mi, steps, duree_s)

_cache_params = {"v": None}
_cache_zone = {"zone": None, "charge": False}


def P() -> dict:
    """Paramètres actifs (défauts + valeurs enregistrées par l'utilisateur)."""
    if _cache_params["v"] is None:
        doc = parametres_col.find_one({"_id": "global"}) or {}
        _cache_params["v"] = {**DEFAUTS, **{k: doc[k] for k in DEFAUTS if k in doc}}
    return _cache_params["v"]


def zone_active():
    """Zone de pâturage active (mise en cache)."""
    if not _cache_zone["charge"]:
        _cache_zone["zone"] = pasture.find_one({"active": True}, {"_id": 0})
        _cache_zone["charge"] = True
    return _cache_zone["zone"]


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
        self.clients: set[WebSocket] = set()
        self.boucle = None

    async def _envoyer(self, texte: str):
        for ws in list(self.clients):
            try:
                await ws.send_text(texte)
            except Exception:
                self.clients.discard(ws)

    def publier(self, evenement: dict):
        """Appelable depuis le thread MQTT."""
        if self.boucle is None or not self.clients:
            return
        texte = json.dumps(evenement, default=_json)
        asyncio.run_coroutine_threadsafe(self._envoyer(texte), self.boucle)


diffuseur = Diffuseur()


class Position(BaseModel):
    """Format attendu pour chaque message MQTT."""
    animal_id: str
    lat: float
    lon: float
    vitesse: float
    activite: str
    timestamp: datetime


class InfoAnimal(BaseModel):
    """Fiche d'un animal, telle qu'envoyée par l'application."""
    type: Optional[str] = Field(None, max_length=80)
    sexe: Optional[Literal["femelle", "male"]] = None
    age_ans: Optional[float] = Field(None, ge=0, le=40)
    poids_kg: Optional[float] = Field(None, gt=0, le=1500)


# ---------------------------------------------------------------------------
# Alertes
# ---------------------------------------------------------------------------
def gerer_alerte(animal_id, type_alerte, condition, message, position):
    """Ouvre une alerte quand le problème apparaît, la ferme quand il disparaît."""
    ouverte = alerts.find_one(
        {"animal_id": animal_id, "type": type_alerte, "resolue": False}
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
        }
        alerts.insert_one(dict(doc))
        print(f"ALERTE [{type_alerte}] {animal_id} : {message}")
        diffuseur.publier({"type": "alerte", "data": doc})
    elif not condition and ouverte is not None:
        alerts.update_one(
            {"_id": ouverte["_id"]},
            {"$set": {"resolue": True, "resolue_le": position.timestamp}},
        )
        print(f"Alerte résolue [{type_alerte}] {animal_id}")
        diffuseur.publier({"type": "alerte_resolue", "data": {
            "animal_id": animal_id,
            "type": type_alerte,
            "resolue_le": position.timestamp,
        }})


def appliquer_alerte_ia(animal_id, condition, message):
    """AJOUT IA : branche le bilan Isolation Forest sur le système d'alertes existant
    (ouverture, résolution automatique, diffusion WebSocket), à la dernière position connue."""
    doc = positions.find_one({"animal_id": animal_id}, {"_id": 0}, sort=[("timestamp", -1)])
    if not doc:
        return
    doc["timestamp"] = datetime.now(timezone.utc)
    gerer_alerte(animal_id, "comportement_inhabituel", condition, message, Position(**doc))


def executer_bilan_ia():
    """AJOUT IA AUTO : calcule le bilan des 24 dernières heures pour tous les animaux
    et ouvre / résout les alertes « comportement_inhabituel » (même logique que
    GET /ia/bilan?enregistrer=true). La référence dure reference_jours jours (réglage)."""
    animaux_ia, ignores = calculer_bilan(mesures_activite, P()["reference_jours"] * 24)
    for x in animaux_ia:
        try:
            appliquer_alerte_ia(
                x["animal_id"], x["statut"] == "a_surveiller",
                f"Comportement inhabituel : {x['heures_alerte']} h en alerte sur les "
                f"dernières 24 h (à vérifier)")
        except Exception as erreur:
            print(f"Erreur alerte IA {x['animal_id']} : {erreur}")
    nb = sum(1 for x in animaux_ia if x["statut"] == "a_surveiller")
    print(f"Bilan IA : {len(animaux_ia)} animaux analysés, {nb} à surveiller "
          f"(seuil {SEUIL_HEURES_ALERTE} h), {len(ignores)} ignorés")


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


def appliquer_alerte_lstm(animal_id, condition, message):
    """AJOUT LSTM : branche l'alerte sur le système d'alertes existant (mode actif seulement)."""
    doc = positions.find_one({"animal_id": animal_id}, {"_id": 0}, sort=[("timestamp", -1)])
    if not doc:
        return
    doc["timestamp"] = datetime.now(timezone.utc)
    gerer_alerte(animal_id, "mouvement_agite", condition, message, Position(**doc))


def executer_cycle_lstm():
    """AJOUT LSTM : un cycle d'analyse des trajectoires."""
    bilan = trajectoire_service.executer_cycle(base)
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
    """AJOUT IA AUTO : supprime automatiquement les mesures d'activité de plus de 30 jours.

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
            print("Index TTL mesures_activite converti (30 jours)")
        except Exception as erreur2:
            print(f"Index TTL mesures_activite non converti : {erreur2}")
    except Exception as erreur:
        print(f"Index TTL mesures_activite non créé : {erreur}")


def creer_index_animaux():
    """Un seul document par animal. Un échec (ex. doublons déjà présents dans la
    collection) ne doit pas empêcher le serveur de démarrer."""
    try:
        animaux.create_index("animal_id", unique=True)
    except Exception as erreur:
        print(f"Index animaux non créé : {erreur}")


# ---------------------------------------------------------------------------
# Troupeau
# ---------------------------------------------------------------------------
def positions_actuelles(reference):
    limite = reference - timedelta(seconds=P()["fraicheur_s"])
    resultat = []
    for animal_id in positions.distinct("animal_id"):
        doc = positions.find_one(
            {"animal_id": animal_id}, {"_id": 0}, sort=[("timestamp", -1)]
        )
        if doc and doc["timestamp"] >= limite:
            resultat.append(doc)
    return resultat


def nb_animaux_en_alerte(reference):
    limite = reference - timedelta(seconds=P()["fraicheur_s"])
    return etat_animaux.count_documents(
        {"alerte": True, "derniere_mesure": {"$gte": limite}}
    )


def verifier_troupeau(p, reevaluation=False):
    """Sortie de zone du troupeau, confirmée sur plusieurs messages.

    reevaluation=True : appelé après un changement de paramètres.
    - pas de diffusion du troupeau ici (faite une seule fois à la fin)
    - la confirmation est immédiate (on juge la dernière position telle quelle)
    """
    prm = P()
    actuelles = positions_actuelles(p.timestamp)
    if len(actuelles) < prm["min_animaux_troupeau"]:
        if not reevaluation:
            diffuseur.publier({"type": "troupeau", "data": {
                "statut": "pas assez d'animaux actifs",
                "animaux_actifs": len(actuelles)}})
        return
    principal, autres, ecarts = analyser_troupeaux(actuelles, prm)

    # AJOUT ZONE AUTO : la zone de pâturage automatique suit le grand troupeau
    try:
        maj_zone_auto(principal)
    except Exception as erreur:
        print(f"Erreur maj_zone_auto : {erreur}")

    if not reevaluation:
        diffuseur.publier({"type": "troupeau", "data": {
            "statut": "ok", "animaux_actifs": len(actuelles),
            "grand_troupeau": principal, "autres_groupes": autres}})

    ecart = ecarts.get(p.animal_id)
    hors = ecart is not None and ecart > prm["distance_alerte_m"]

    etat = etat_animaux.find_one({"animal_id": p.animal_id}) or {}
    if reevaluation:
        nb_hors = prm["nb_confirmations"] if hors else 0
    else:
        nb_hors = etat.get("nb_hors_zone", 0) + 1 if hors else 0
    alerte = nb_hors >= prm["nb_confirmations"]

    etat_animaux.update_one(
        {"animal_id": p.animal_id},
        {"$set": {"nb_hors_zone": nb_hors, "alerte": alerte,
                  "ecart_m": ecart if ecart is not None else 0.0,
                  "derniere_mesure": p.timestamp}},
        upsert=True,
    )
    total = nb_animaux_en_alerte(p.timestamp)
    message = (f"À {round(ecart or 0)} m du grand troupeau "
               f"(confirmé sur {prm['nb_confirmations']} mesures). "
               f"Animaux hors zone : {total}")
    gerer_alerte(p.animal_id, "sortie_zone", alerte, message, p)


# ---------------------------------------------------------------------------
# Géorepérage
# ---------------------------------------------------------------------------
def verifier_pature(p):
    zone = zone_active()
    if not zone:
        return
    sommets = [(c["latitude"], c["longitude"]) for c in zone["coordinates"]]
    dedans, bord = analyser_point(p.lat, p.lon, sommets)
    hors = (not dedans) and bord > P()["marge_pature_m"]
    gerer_alerte(
        p.animal_id, "hors_pature", hors,
        f"Hors de la zone « {zone['name']} », à {round(bord)} m du bord", p,
    )


# ---------------------------------------------------------------------------
# Alertes individuelles
# ---------------------------------------------------------------------------
def verifier_alertes_individuelles(p):
    prm = P()
    duree = prm["duree_immobilite_s"]
    debut = p.timestamp - timedelta(seconds=duree)
    fenetre = list(positions.find(
        {"animal_id": p.animal_id, "timestamp": {"$gte": debut, "$lte": p.timestamp}},
        {"_id": 0}).sort("timestamp", 1))
    repere = positions.find_one(
        {"animal_id": p.animal_id, "timestamp": {"$lt": debut}},
        {"_id": 0}, sort=[("timestamp", -1)])
    if repere:
        fenetre.insert(0, repere)
    gerer_alerte(
        p.animal_id, "immobilite",
        est_immobile(fenetre, duree, prm["vitesse_repos_max"]),
        f"Au repos depuis plus de {duree // 60} min", p)

    nb = prm["nb_mesures_vitesse"]
    dernieres = list(positions.find({"animal_id": p.animal_id}, {"_id": 0})
                     .sort("timestamp", -1).limit(nb))[::-1]
    gerer_alerte(
        p.animal_id, "deplacement_inhabituel",
        deplacement_inhabituel(dernieres, prm["vitesse_max"], nb),
        f"Vitesse supérieure à {prm['vitesse_max']} km/h sur {nb} mesures", p)


# ---------------------------------------------------------------------------
# Collier silencieux
# ---------------------------------------------------------------------------
def verifier_colliers():
    delai = P()["delai_collier_s"]
    maintenant = datetime.now(timezone.utc)
    for animal_id in positions.distinct("animal_id"):
        doc = positions.find_one(
            {"animal_id": animal_id}, {"_id": 0}, sort=[("timestamp", -1)])
        if not doc:
            continue
        age = (maintenant - doc["timestamp"]).total_seconds()
        doc["timestamp"] = maintenant
        gerer_alerte(animal_id, "collier_silencieux", age > delai,
                     f"Aucun message depuis plus de {delai // 60} min",
                     Position(**doc))


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


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        position = Position(**data)
    except (json.JSONDecodeError, ValidationError) as erreur:
        print(f"Message rejeté sur {msg.topic} : {erreur}")
        return

    if position.timestamp.tzinfo is None:
        position.timestamp = position.timestamp.replace(tzinfo=timezone.utc)

    positions.insert_one(position.model_dump())
    print(f"Enregistré : {position.animal_id} ({position.lat}, {position.lon})")
    diffuseur.publier({"type": "position", "data": position.model_dump()})

    # AJOUT IA : activité du MPU6050 (mi, steps, duree_s) conservée pour le bilan Isolation Forest
    if "mi" in data and "steps" in data:
        try:
            mesures_activite.insert_one({
                "animal_id": position.animal_id, "timestamp": position.timestamp,
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
        etat_animaux.create_index, "animal_id", unique=True
    )
    await asyncio.to_thread(creer_index_animaux)        # AJOUT FICHES ANIMAUX (sans bloquer le démarrage)
    await asyncio.to_thread(
        mesures_activite.create_index, [("animal_id", 1), ("timestamp", -1)]
    )      # AJOUT IA
    await asyncio.to_thread(creer_index_ttl)            # AJOUT IA AUTO
    mqtt_client.connect(MQTT_HOST, MQTT_PORT)
    mqtt_client.loop_start()
    tache = asyncio.create_task(surveillance_colliers())
    tache_ia = asyncio.create_task(surveillance_ia()) if IA_AUTO else None   # AJOUT IA AUTO
    tache_lstm = asyncio.create_task(surveillance_lstm()) if LSTM_AUTO else None   # AJOUT LSTM
    yield
    tache.cancel()
    if tache_ia:
        tache_ia.cancel()
    if tache_lstm:
        tache_lstm.cancel()
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
                                lambda: P()["reference_jours"]))


# ---------------------------------------------------------------------------
# Zone de pâturage automatique (AJOUT ZONE AUTO)
#   côté du carré = 2 × rayon(grand troupeau) + distance_alerte_m
#   - rayon : fonction du NOMBRE d'animaux du grand troupeau (surface estimée par
#     animal), jamais inférieur au rayon réellement mesuré
#   - distance_alerte_m : réglage de l'utilisateur (page Réglages), relu à chaque calcul
# ---------------------------------------------------------------------------
INTERVALLE_MAJ_ZONE_S = 30          # recalcul périodique de la zone
TOLERANCE_REDUCTION_M = 10          # la zone ne rétrécit pas pour moins de 10 m
_maj_zone = {"t": 0.0}


def _zone_publique(zone):
    """Ce que voient les clients (REST + WebSocket)."""
    return {"name": zone["name"], "coordinates": zone["coordinates"],
            "mode": zone.get("mode", "manuel"), "marge_m": zone.get("marge_m")}


def _demi_cote_m(principal):
    """Demi-côté du carré : rayon du troupeau + moitié de distance_alerte_m."""
    n = max(int(principal.get("nb_animaux", 1)), 1)
    rayon_estime = math.sqrt(n * SURFACE_PAR_ANIMAL_M2 / math.pi)
    rayon = max(rayon_estime, float(principal.get("rayon_m", 0.0)))
    return rayon + P()["distance_alerte_m"] / 2


def _rectangle(clat, clon, demi_cote_m):
    """Carré (4 sommets) centré sur (clat, clon)."""
    d_lat = math.degrees(demi_cote_m / RAYON_TERRE_M)
    d_lon = math.degrees(demi_cote_m / (RAYON_TERRE_M * math.cos(math.radians(clat))))
    return [(clat - d_lat, clon - d_lon), (clat + d_lat, clon - d_lon),
            (clat + d_lat, clon + d_lon), (clat - d_lat, clon + d_lon)]


def _demi_cote_actuel(zone):
    lats = [c["latitude"] for c in zone["coordinates"]]
    return math.radians(max(lats) - min(lats)) * RAYON_TERRE_M / 2


def _grand_troupeau():
    """Zone (centre, rayon, nb_animaux) du grand troupeau actuel, ou None."""
    derniere = positions.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
    if not derniere:
        return None
    actuelles = positions_actuelles(derniere["timestamp"])
    if len(actuelles) < P()["min_animaux_troupeau"]:
        return None
    return analyser_troupeaux(actuelles, P())[0]


def _sommets_auto():
    principal = _grand_troupeau()
    if principal is None:
        raise ValueError("Pas assez d'animaux actifs pour calculer la zone automatique.")
    return _rectangle(principal["centre_lat"], principal["centre_lon"],
                      _demi_cote_m(principal))


def _reverifier_pature_tous():
    for animal_id in positions.distinct("animal_id"):
        d = positions.find_one({"animal_id": animal_id}, sort=[("timestamp", -1)])
        if d:
            try:
                verifier_pature(Position(**{k: d[k] for k in Position.model_fields}))
            except Exception as e:
                print(f"Vérification pâturage ignorée : {e}")


def _poser_zone(nom, sommets, mode):
    """Remplace la zone active et prévient les clients."""
    zone = {
        "name": nom,
        "coordinates": [{"latitude": a, "longitude": b} for a, b in sommets],
        "cree_le": datetime.now(timezone.utc),
        "active": True,
        "mode": mode,
        "marge_m": P()["distance_alerte_m"] if mode == "auto" else None,
    }
    pasture.update_many({"active": True}, {"$set": {"active": False}})
    pasture.insert_one(dict(zone))
    _cache_zone["zone"], _cache_zone["charge"] = zone, True
    _maj_zone["t"] = time.monotonic()
    diffuseur.publier({"type": "pature", "data": _zone_publique(zone)})
    return zone


def maj_zone_auto(principal, forcer=False):
    """Appelée depuis verifier_troupeau (au plus toutes les 30 s) et, avec
    forcer=True, immédiatement après un changement de réglages.
    - aucune zone : en crée une automatique ;
    - zone automatique : la recentre et la redimensionne ;
    Une zone manuelle (administrateur) n'est jamais touchée."""
    if not principal:
        return
    if not forcer and time.monotonic() - _maj_zone["t"] < INTERVALLE_MAJ_ZONE_S:
        return
    zone = zone_active()
    demi = _demi_cote_m(principal)
    if not zone:
        if ZONE_AUTO:
            sommets = _rectangle(principal["centre_lat"], principal["centre_lon"], demi)
            _poser_zone("Zone de pâturage", sommets, "auto")
            _reverifier_pature_tous()
        return
    if zone.get("mode") != "auto":
        return
    _maj_zone["t"] = time.monotonic()
    ancien = _demi_cote_actuel(zone)
    if not forcer and demi < ancien and ancien - demi < TOLERANCE_REDUCTION_M:
        demi = ancien                       # évite que la zone « respire »
    sommets = _rectangle(principal["centre_lat"], principal["centre_lon"], demi)
    coords = [{"latitude": a, "longitude": b} for a, b in sommets]
    marge = P()["distance_alerte_m"]
    pasture.update_one({"active": True},
                       {"$set": {"coordinates": coords, "marge_m": marge}})
    zone = {**zone, "coordinates": coords, "marge_m": marge}
    _cache_zone["zone"] = zone
    diffuseur.publier({"type": "pature", "data": _zone_publique(zone)})


def recalculer_zone_auto():
    """Recalcul immédiat (changement de distance_alerte_m ou d'autres réglages)."""
    try:
        maj_zone_auto(_grand_troupeau(), forcer=True)
    except Exception as e:
        print(f"Erreur recalcul zone : {e}")


def exiger_admin(x_admin_key: Optional[str] = Header(None)):
    """Modifier la zone à la main est réservé à l'administrateur (en-tête X-Admin-Key).
    Sans ADMIN_KEY définie, toute modification manuelle est refusée."""
    if not ADMIN_KEY or not secrets.compare_digest(x_admin_key or "", ADMIN_KEY):
        raise HTTPException(status_code=403,
                            detail="Zone gérée automatiquement par le serveur.")


@app.post("/pastures", dependencies=[Depends(exiger_admin)])
def creer_pature(donnees: dict):
    """Administrateur. {"mode":"auto"} ou {"coordinates":[{"latitude":..,"longitude":..}, ...]}"""
    mode = "auto" if donnees.get("mode") == "auto" else "manuel"
    try:
        sommets = _sommets_auto() if mode == "auto" else valider_sommets(donnees.get("coordinates"))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _poser_zone(str(donnees.get("name") or "Zone de pâturage")[:80], sommets, mode)
    _reverifier_pature_tous()
    return {"statut": "ok", "mode": mode, "sommets": len(sommets)}


@app.put("/pastures", dependencies=[Depends(exiger_admin)])
def modifier_pature(donnees: dict):
    """Administrateur. {"mode":"auto"} | {"mode":"manuel"} (fige) | {"coordinates":[...]}"""
    actuelle = zone_active()
    if not actuelle:
        raise HTTPException(status_code=404, detail="Aucune zone enregistrée.")
    if "coordinates" in donnees:
        mode = "manuel"
    else:
        mode = "auto" if donnees.get("mode", actuelle.get("mode")) == "auto" else "manuel"
    try:
        if mode == "auto":
            sommets = _sommets_auto()
        elif "coordinates" in donnees:
            sommets = valider_sommets(donnees["coordinates"])
        else:
            sommets = [(c["latitude"], c["longitude"]) for c in actuelle["coordinates"]]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _poser_zone(str(donnees.get("name") or actuelle["name"])[:80], sommets, mode)
    _reverifier_pature_tous()
    return {"statut": "ok", "mode": mode, "sommets": len(sommets)}


@app.get("/pastures")
def lire_pature():
    zone = zone_active()
    if not zone:
        return {"statut": "aucune zone"}
    return _zone_publique(zone)


@app.delete("/pastures", dependencies=[Depends(exiger_admin)])
def supprimer_pature():
    """Désactive la zone, résout les alertes hors_pature, prévient les clients."""
    pasture.update_many({"active": True}, {"$set": {"active": False}})
    _cache_zone["zone"], _cache_zone["charge"] = None, True

    maintenant = datetime.now(timezone.utc)
    nb_resolues = 0
    for a in list(alerts.find({"type": "hors_pature", "resolue": False})):
        alerts.update_one(
            {"_id": a["_id"]},
            {"$set": {"resolue": True, "resolue_le": maintenant}},
        )
        diffuseur.publier({"type": "alerte_resolue", "data": {
            "animal_id": a["animal_id"],
            "type": "hors_pature",
            "resolue_le": maintenant,
        }})
        nb_resolues += 1

    diffuseur.publier({"type": "pature", "data": None})
    return {"statut": "ok", "alertes_resolues": nb_resolues}


@app.get("/positions")
def lire_positions(animal_id: Optional[str] = None, limite: int = 50):
    filtre = {"animal_id": animal_id} if animal_id else {}
    return list(positions.find(filtre, {"_id": 0}).sort("timestamp", -1).limit(limite))


@app.get("/alertes")
def lire_alertes(animal_id: Optional[str] = None, actives: bool = False, limite: int = 50):
    filtre = {}
    if animal_id:
        filtre["animal_id"] = animal_id
    if actives:
        filtre["resolue"] = False
    return list(alerts.find(filtre, {"_id": 0}).sort("timestamp", -1).limit(limite))


@app.get("/lstm/evenements")
def lire_evenements_lstm(animal_id: Optional[str] = None, limite: int = 50):
    filtre = {"animal_id": animal_id} if animal_id else {}
    return list(base["lstm_alertes_silencieuses"].find(filtre, {"_id": 0})
                .sort("cree_le", -1).limit(limite))


# ---------------------------------------------------------------------------
# Fiches des animaux (type, sexe, âge, poids)
# ---------------------------------------------------------------------------
@app.get("/animaux")
def lire_animaux():
    """Liste des fiches enregistrées (utilisée par la page « Mon troupeau »)."""
    return list(animaux.find({}, {"_id": 0, "modifie_le": 0}))


@app.put("/animaux/{animal_id}")
def enregistrer_animal(animal_id: str, info: InfoAnimal):
    """Crée ou met à jour la fiche d'un animal."""
    fiche = info.model_dump()
    animaux.update_one(
        {"animal_id": animal_id},
        {"$set": {**fiche, "modifie_le": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return {"animal_id": animal_id, **fiche}


@app.get("/troupeau")
def etat_troupeau():
    prm = P()
    derniere = positions.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
    if not derniere:
        return {"statut": "aucune donnée"}
    reference = derniere["timestamp"]
    actuelles = positions_actuelles(reference)
    if len(actuelles) < prm["min_animaux_troupeau"]:
        return {"statut": "pas assez d'animaux actifs", "animaux_actifs": len(actuelles)}
    principal, autres, _ = analyser_troupeaux(actuelles, prm)
    limite = reference - timedelta(seconds=prm["fraicheur_s"])
    en_alerte = list(etat_animaux.find(
        {"alerte": True, "derniere_mesure": {"$gte": limite}},
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


def rediffuser_troupeau():
    """Recalcule le troupeau avec les paramètres actuels et prévient les clients."""
    try:
        diffuseur.publier({"type": "troupeau", "data": etat_troupeau()})
    except Exception as e:
        print(f"Rediffusion troupeau ignorée : {e}")


def reevaluer_tous():
    """AJOUT : rejoue toutes les vérifications sur la dernière position de
    chaque animal avec les nouveaux paramètres, sans attendre le prochain
    message MQTT. Alertes ouvertes/fermées et diffusées immédiatement."""
    recalculer_zone_auto()   # AJOUT ZONE AUTO : la zone suit distance_alerte_m tout de suite
    for animal_id in positions.distinct("animal_id"):
        d = positions.find_one({"animal_id": animal_id}, sort=[("timestamp", -1)])
        if not d:
            continue
        try:
            p = Position(**{k: d[k] for k in Position.model_fields})
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
        verifier_colliers()
    except Exception as e:
        print(f"Erreur verifier_colliers : {e}")
    rediffuser_troupeau()   # une seule diffusion, après les alertes


@app.get("/parametres")
def lire_parametres():
    return {"valeurs": P(), "defauts": DEFAUTS,
            "bornes": {k: [b[0], b[1]] for k, b in BORNES.items()}}


@app.put("/parametres")
def modifier_parametres(donnees: dict, taches: BackgroundTasks):
    try:
        propres = valider_parametres(donnees)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    ancienne_reference = P()["reference_jours"]
    nouveaux = {**P(), **propres}
    parametres_col.update_one({"_id": "global"}, {"$set": nouveaux}, upsert=True)
    _cache_params["v"] = nouveaux
    taches.add_task(reevaluer_tous)            # MODIFIÉ : après la réponse
    if nouveaux["reference_jours"] != ancienne_reference:
        taches.add_task(recalculer_alertes_ia)  # AJOUT : alertes IA recalculées tout de suite
    return {"statut": "ok", "valeurs": nouveaux}


@app.post("/parametres/reinitialiser")
def reinitialiser_parametres(taches: BackgroundTasks):
    ancienne_reference = P()["reference_jours"]
    parametres_col.delete_one({"_id": "global"})
    _cache_params["v"] = dict(DEFAUTS)
    taches.add_task(reevaluer_tous)            # MODIFIÉ : après la réponse
    if DEFAUTS["reference_jours"] != ancienne_reference:
        taches.add_task(recalculer_alertes_ia)  # AJOUT : alertes IA recalculées tout de suite
    return {"statut": "ok", "valeurs": DEFAUTS}


def etat_complet() -> dict:
    """Photo de départ envoyée à chaque nouveau client WebSocket."""
    dernieres = []
    for animal_id in positions.distinct("animal_id"):
        doc = positions.find_one(
            {"animal_id": animal_id}, {"_id": 0}, sort=[("timestamp", -1)]
        )
        if doc:
            dernieres.append(doc)
    actives = list(
        alerts.find({"resolue": False}, {"_id": 0}).sort("timestamp", -1).limit(100)
    )
    zone = zone_active()
    return {
        "type": "etat",
        "positions": dernieres,
        "alertes_actives": actives,
        "troupeau": etat_troupeau(),
        "pature": _zone_publique(zone) if zone else None,   # MODIFIÉ : mode + marge
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        etat = await asyncio.to_thread(etat_complet)
        await ws.send_text(json.dumps(etat, default=_json))
        diffuseur.clients.add(ws)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        diffuseur.clients.discard(ws)