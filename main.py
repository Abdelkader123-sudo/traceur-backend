import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from pymongo import MongoClient

from analyse import (
    DELAI_COLLIER_S,
    DISTANCE_ALERTE_M,
    DUREE_IMMOBILITE_S,
    FRAICHEUR_S,
    MIN_ANIMAUX_TROUPEAU,
    NB_CONFIRMATIONS,
    NB_MESURES_VITESSE,
    VITESSE_MAX,
    analyser_troupeaux,
    deplacement_inhabituel,
    est_immobile,
)
from geofence import analyser_point, valider_sommets
from sante import JOURS_EVALUATION, JOURS_REFERENCE, evaluer_sante

MQTT_HOST = os.getenv("MQTT_HOST", "ycf81881.ala.eu-central-1.emqxsl.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.environ["MQTT_USER"]
MQTT_PASSWORD = os.environ["MQTT_PASSWORD"]
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "farm/+/data")
CA_FILE = os.getenv("CA_FILE", "emqxsl-ca.crt")

MONGO_URI = os.environ["MONGO_URI"]

INTERVALLE_VERIF_COLLIERS_S = 60   # vérification des colliers silencieux
MARGE_PATURE_M = 10   # tolérance GPS hors zone (EXPÉRIMENTAL)

mongo_client = MongoClient(MONGO_URI, tz_aware=True)
base = mongo_client["traceur_gps"]
positions = base["positions"]
alerts = base["alerts"]
etat_animaux = base["etat_animaux"]
animaux = base["animaux"]
pasture = base["pastures"]

_cache_zone = {"zone": None, "charge": False}


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


# ---------------------------------------------------------------------------
# Troupeau
# ---------------------------------------------------------------------------
def positions_actuelles(reference):
    """Dernière position de chaque animal, si elle date de moins de FRAICHEUR_S."""
    limite = reference - timedelta(seconds=FRAICHEUR_S)
    resultat = []
    for animal_id in positions.distinct("animal_id"):
        doc = positions.find_one(
            {"animal_id": animal_id}, {"_id": 0}, sort=[("timestamp", -1)]
        )
        if doc and doc["timestamp"] >= limite:
            resultat.append(doc)
    return resultat


def nb_animaux_en_alerte(reference):
    limite = reference - timedelta(seconds=FRAICHEUR_S)
    return etat_animaux.count_documents(
        {"alerte": True, "derniere_mesure": {"$gte": limite}}
    )


def verifier_troupeau(p):
    """Sortie de zone du troupeau, confirmée sur plusieurs messages."""
    actuelles = positions_actuelles(p.timestamp)
    if len(actuelles) < MIN_ANIMAUX_TROUPEAU:
        return
    _, _, ecarts = analyser_troupeaux(actuelles)

    ecart = ecarts.get(p.animal_id)
    hors = ecart is not None and ecart > DISTANCE_ALERTE_M

    etat = etat_animaux.find_one({"animal_id": p.animal_id}) or {}
    nb_hors = etat.get("nb_hors_zone", 0) + 1 if hors else 0
    alerte = nb_hors >= NB_CONFIRMATIONS

    etat_animaux.update_one(
        {"animal_id": p.animal_id},
        {"$set": {
            "nb_hors_zone": nb_hors,
            "alerte": alerte,
            "ecart_m": ecart if ecart is not None else 0.0,
            "derniere_mesure": p.timestamp,
        }},
        upsert=True,
    )

    total = nb_animaux_en_alerte(p.timestamp)
    message = (
        f"À {round(ecart or 0)} m du grand troupeau "
        f"(confirmé sur {NB_CONFIRMATIONS} mesures). "
        f"Animaux hors zone : {total}"
    )
    gerer_alerte(p.animal_id, "sortie_zone", alerte, message, p)


# ---------------------------------------------------------------------------
# Géorepérage
# ---------------------------------------------------------------------------
def verifier_pature(p):
    """Alerte hors_pature si l'animal est à plus de MARGE_PATURE_M hors de la zone."""
    zone = zone_active()
    if not zone:
        return
    sommets = [(c["latitude"], c["longitude"]) for c in zone["coordinates"]]
    dedans, bord = analyser_point(p.lat, p.lon, sommets)
    hors = (not dedans) and bord > MARGE_PATURE_M
    gerer_alerte(
        p.animal_id, "hors_pature", hors,
        f"Hors de la zone « {zone['name']} », à {round(bord)} m du bord", p,
    )


# ---------------------------------------------------------------------------
# Alertes individuelles
# ---------------------------------------------------------------------------
def verifier_alertes_individuelles(p):
    """Immobilité prolongée et déplacement inhabituel."""
    debut = p.timestamp - timedelta(seconds=DUREE_IMMOBILITE_S)
    fenetre = list(
        positions.find(
            {"animal_id": p.animal_id, "timestamp": {"$gte": debut, "$lte": p.timestamp}},
            {"_id": 0},
        ).sort("timestamp", 1)
    )
    repere = positions.find_one(
        {"animal_id": p.animal_id, "timestamp": {"$lt": debut}},
        {"_id": 0},
        sort=[("timestamp", -1)],
    )
    if repere:
        fenetre.insert(0, repere)
    gerer_alerte(
        p.animal_id, "immobilite", est_immobile(fenetre, DUREE_IMMOBILITE_S),
        f"Au repos depuis plus de {DUREE_IMMOBILITE_S // 60} min", p,
    )

    dernieres = list(
        positions.find({"animal_id": p.animal_id}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(NB_MESURES_VITESSE)
    )[::-1]
    gerer_alerte(
        p.animal_id, "deplacement_inhabituel",
        deplacement_inhabituel(dernieres),
        f"Vitesse supérieure à {VITESSE_MAX} km/h sur {NB_MESURES_VITESSE} mesures", p,
    )


# ---------------------------------------------------------------------------
# Collier silencieux
# ---------------------------------------------------------------------------
def verifier_colliers():
    maintenant = datetime.now(timezone.utc)
    for animal_id in positions.distinct("animal_id"):
        doc = positions.find_one(
            {"animal_id": animal_id}, {"_id": 0}, sort=[("timestamp", -1)]
        )
        if not doc:
            continue
        age = (maintenant - doc["timestamp"]).total_seconds()
        silencieux = age > DELAI_COLLIER_S
        doc["timestamp"] = maintenant
        gerer_alerte(
            animal_id, "collier_silencieux", silencieux,
            f"Aucun message depuis plus de {DELAI_COLLIER_S // 60} min",
            Position(**doc),
        )


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
    mqtt_client.connect(MQTT_HOST, MQTT_PORT)
    mqtt_client.loop_start()
    tache = asyncio.create_task(surveillance_colliers())
    yield
    tache.cancel()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    mongo_client.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.post("/pastures")
def creer_pature(donnees: dict):
    """Enregistre la zone (remplace l'ancienne) et revérifie tous les animaux."""
    try:
        sommets = valider_sommets(donnees.get("coordinates"))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    zone = {
        "name": str(donnees.get("name") or "Zone de pâturage")[:80],
        "coordinates": [{"latitude": a, "longitude": b} for a, b in sommets],
        "cree_le": datetime.now(timezone.utc),
        "active": True,
    }
    pasture.update_many({"active": True}, {"$set": {"active": False}})
    pasture.insert_one(dict(zone))
    _cache_zone["zone"], _cache_zone["charge"] = zone, True

    for animal_id in positions.distinct("animal_id"):
        d = positions.find_one({"animal_id": animal_id}, sort=[("timestamp", -1)])
        if d:
            try:
                verifier_pature(Position(**{k: d[k] for k in Position.model_fields}))
            except Exception as e:
                print(f"Vérification initiale ignorée : {e}")
    diffuseur.publier({"type": "pature", "data": {
        "name": zone["name"], "coordinates": zone["coordinates"]}})
    return {"statut": "ok", "sommets": len(sommets)}


@app.get("/pastures")
def lire_pature():
    zone = zone_active()
    if not zone:
        return {"statut": "aucune zone"}
    return {"name": zone["name"], "coordinates": zone["coordinates"]}


@app.delete("/pastures")
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


@app.get("/troupeau")
def etat_troupeau():
    derniere = positions.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
    if not derniere:
        return {"statut": "aucune donnée"}
    reference = derniere["timestamp"]
    actuelles = positions_actuelles(reference)
    if len(actuelles) < MIN_ANIMAUX_TROUPEAU:
        return {"statut": "pas assez d'animaux actifs", "animaux_actifs": len(actuelles)}
    principal, autres, _ = analyser_troupeaux(actuelles)
    limite = reference - timedelta(seconds=FRAICHEUR_S)
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


@app.get("/sante/{animal_id}")
def sante(animal_id: str):
    maintenant = datetime.now(timezone.utc)
    fin = maintenant.replace(hour=0, minute=0, second=0, microsecond=0)
    debut = fin - timedelta(days=JOURS_REFERENCE + JOURS_EVALUATION)
    mesures = list(positions.find(
        {"animal_id": animal_id, "timestamp": {"$gte": debut, "$lt": fin}},
        {"_id": 0},
    ).sort("timestamp", 1))
    fiche = animaux.find_one({"animal_id": animal_id}, {"_id": 0})
    resultat = evaluer_sante(mesures, fiche, maintenant)
    resultat["animal_id"] = animal_id
    return resultat


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
        "pature": ({"name": zone["name"], "coordinates": zone["coordinates"]}
                   if zone else None),
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