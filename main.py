import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import paho.mqtt.client as mqtt
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
from pymongo import MongoClient

from analyse import (
    DUREE_IMMOBILITE_S,
    MARGE_ZONE_M,
    NB_MESURES_VITESSE,
    VITESSE_MAX,
    calculer_zones,
    deplacement_inhabituel,
    ecart_a_la_zone,
    est_immobile,
)

# --- Paramètres du broker MQTT (identiques à ceux de Node-RED) ---
# --- Paramètres du broker MQTT ---
MQTT_HOST = os.getenv("MQTT_HOST", "ycf81881.ala.eu-central-1.emqxsl.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.environ["MQTT_USER"]
MQTT_PASSWORD = os.environ["MQTT_PASSWORD"]
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "farm/+/data")  # + = n'importe quel animal
CA_FILE = os.getenv("CA_FILE", "emqxsl-ca.crt")

# --- Paramètres MongoDB Atlas ---
MONGO_URI = os.environ["MONGO_URI"]

# --- Paramètres de l'analyse ---
MIN_POSITIONS_ANALYSE = 100      # minimum de positions par animal pour lancer DBSCAN
MAX_POSITIONS_ANALYSE = 5000     # on n'analyse que les positions les plus récentes
INTERVALLE_RECALCUL_S = 86400    # recalcul automatique des zones : 1 fois par jour

mongo_client = MongoClient(MONGO_URI, tz_aware=True)
base = mongo_client["traceur_gps"]
positions = base["positions"]
geofences = base["geofences"]
alerts = base["alerts"]


class Position(BaseModel):
    """Format attendu pour chaque message MQTT (le « contrat »)."""
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
    """Ouvre une alerte quand le problème apparaît, la ferme quand il disparaît.

    Une seule alerte est créée par épisode : pas de doublon à chaque message
    tant que le problème dure.
    """
    ouverte = alerts.find_one(
        {"animal_id": animal_id, "type": type_alerte, "resolue": False}
    )
    if condition and ouverte is None:
        alerts.insert_one({
            "animal_id": animal_id,
            "type": type_alerte,
            "message": message,
            "lat": position.lat,
            "lon": position.lon,
            "timestamp": position.timestamp,
            "resolue": False,
        })
        print(f"ALERTE [{type_alerte}] {animal_id} : {message}")
    elif not condition and ouverte is not None:
        alerts.update_one(
            {"_id": ouverte["_id"]},
            {"$set": {"resolue": True, "resolue_le": position.timestamp}},
        )
        print(f"Alerte résolue [{type_alerte}] {animal_id}")


def verifier_alertes(p):
    """Applique les trois détections à la position qui vient d'arriver."""
    # 1. Sortie de zone (si les zones de l'animal sont connues)
    doc = geofences.find_one({"animal_id": p.animal_id})
    if doc and doc.get("zones"):
        ecart = ecart_a_la_zone(p.lat, p.lon, doc["zones"])
        gerer_alerte(
            p.animal_id, "sortie_zone", ecart > MARGE_ZONE_M,
            f"À {round(ecart)} m de sa zone habituelle la plus proche", p,
        )

    # 2. Immobilité prolongée : tout est « repos » sur la fenêtre de temps
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
    if repere:  # dernier message avant la fenêtre : l'animal y était-il déjà au repos ?
        fenetre.insert(0, repere)
    gerer_alerte(
        p.animal_id, "immobilite", est_immobile(fenetre, DUREE_IMMOBILITE_S),
        f"Au repos depuis plus de {DUREE_IMMOBILITE_S // 60} min", p,
    )

    # 3. Déplacement inhabituel : plusieurs mesures rapides d'affilée
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
# Zones habituelles (DBSCAN)
# ---------------------------------------------------------------------------
def recalculer_zones(animal_id):
    """Applique DBSCAN sur l'historique d'un animal et enregistre ses zones."""
    docs = list(
        positions.find({"animal_id": animal_id}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(MAX_POSITIONS_ANALYSE)
    )
    if len(docs) < MIN_POSITIONS_ANALYSE:
        return {
            "animal_id": animal_id,
            "statut": "pas assez de données",
            "positions": len(docs),
            "minimum": MIN_POSITIONS_ANALYSE,
        }
    zones, anomalies = calculer_zones(docs)
    geofences.replace_one(
        {"animal_id": animal_id},
        {
            "animal_id": animal_id,
            "zones": zones,
            "calcule_le": datetime.now(timezone.utc),
        },
        upsert=True,
    )
    return {
        "animal_id": animal_id,
        "statut": "ok",
        "zones": zones,
        "nb_anomalies": len(anomalies),
        "anomalies": anomalies[:20],
    }


def recalculer_toutes_les_zones():
    """Relance DBSCAN pour chaque animal connu."""
    for animal_id in positions.distinct("animal_id"):
        resultat = recalculer_zones(animal_id)
        print(f"Recalcul des zones de {animal_id} : {resultat['statut']}")


async def recalcul_periodique():
    """Recalcule les zones au démarrage, puis à intervalle régulier."""
    while True:
        try:
            await asyncio.to_thread(recalculer_toutes_les_zones)
        except Exception as erreur:
            print(f"Erreur pendant le recalcul des zones : {erreur}")
        await asyncio.sleep(INTERVALLE_RECALCUL_S)


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
        position = Position(**data)  # validation du format
    except (json.JSONDecodeError, ValidationError) as erreur:
        print(f"Message rejeté sur {msg.topic} : {erreur}")
        return

    if position.timestamp.tzinfo is None:  # horodatage sans fuseau : on suppose UTC
        position.timestamp = position.timestamp.replace(tzinfo=timezone.utc)

    positions.insert_one(position.model_dump())
    print(f"Enregistré : {position.animal_id} ({position.lat}, {position.lon})")

    try:
        verifier_alertes(position)
    except Exception as erreur:  # ne jamais arrêter l'écoute MQTT
        print(f"Erreur pendant la détection : {erreur}")


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
    # Démarrage : index MongoDB, connexion au broker, recalcul périodique
    await asyncio.to_thread(
        positions.create_index, [("animal_id", 1), ("timestamp", -1)]
    )
    mqtt_client.connect(MQTT_HOST, MQTT_PORT)
    mqtt_client.loop_start()
    tache = asyncio.create_task(recalcul_periodique())
    yield
    # Arrêt : fermeture propre
    tache.cancel()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    mongo_client.close()


app = FastAPI(lifespan=lifespan)


@app.get("/positions")
def lire_positions(animal_id: Optional[str] = None, limite: int = 50):
    """Retourne les dernières positions enregistrées (filtrables par animal)."""
    filtre = {"animal_id": animal_id} if animal_id else {}
    resultats = positions.find(filtre, {"_id": 0}).sort("timestamp", -1).limit(limite)
    return list(resultats)


@app.get("/alertes")
def lire_alertes(animal_id: Optional[str] = None, actives: bool = False, limite: int = 50):
    """Retourne les dernières alertes (filtrables par animal, ou seulement les actives)."""
    filtre = {}
    if animal_id:
        filtre["animal_id"] = animal_id
    if actives:
        filtre["resolue"] = False
    resultats = alerts.find(filtre, {"_id": 0}).sort("timestamp", -1).limit(limite)
    return list(resultats)


@app.post("/analyse/{animal_id}")
def analyser(animal_id: str):
    """Lance DBSCAN pour un animal et enregistre ses zones habituelles."""
    return recalculer_zones(animal_id)