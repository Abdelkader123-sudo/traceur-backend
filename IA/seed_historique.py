"""Remplit mesures_activite avec N jours d'historique simulé (rythme jour/nuit), même format que Node-RED.
Relançable sans risque : supprime d'abord les anciennes mesures du seed (source="seed", ou duree_s == 300),
sans toucher aux mesures envoyées par Node-RED.
Usage : MONGO_URI=... python seed_historique.py [jours=7] [animal_malade=cow04]
Le dernier ~12 h de l'animal malade est ramené à 1/3 de son activité (anomalie à détecter)."""
import os, random, sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pymongo import MongoClient

JOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
MALADE = sys.argv[2] if len(sys.argv) > 2 else "cow04"
PAS_S = 300                                   # un message toutes les 5 min (12 messages/heure > 5 requis)
TZ = ZoneInfo("Africa/Tunis")
ANIMAUX = [f"cow{i:02d}" for i in range(1, 21)]
PROFILS = {"repos": (0.02, 0), "marche": (0.12, 1.0)}

def rythme(h):
    return 0.15 if (h >= 22 or h < 5) else 1.0 if (6 <= h < 10 or 16 <= h < 19) else 0.6

col = MongoClient("mongodb+srv://abdelkaderjaziri4_db_user:abdelkaderjaziriSEM3@cluster0.mokdsar.mongodb.net/?appName=Cluster0", tz_aware=True)["traceur_gps"]["mesures_activite"]
avant = col.delete_many({"$or": [{"source": "seed"}, {"duree_s": float(PAS_S)}]}).deleted_count
print(f"{avant} anciennes mesures du seed supprimées")
fin = datetime.now(timezone.utc).replace(second=0, microsecond=0)
debut = fin - timedelta(days=JOURS)
docs = []
for a in ANIMAUX:
    echelle = random.uniform(0.7, 1.4)
    t = debut
    while t < fin:
        f = rythme(t.astimezone(TZ).hour)
        if a == MALADE and t > fin - timedelta(hours=12):
            f *= 0.33
        marche = random.random() < min(0.95, max(0.01, 0.4 * f))
        acc, cad = PROFILS["marche" if marche else "repos"]
        docs.append({"animal_id": a, "timestamp": t,
                     "mi": float(max(0, round(acc * 100 * echelle * random.uniform(0.8, 1.2)))),
                     "steps": float(max(0, round(cad * echelle * PAS_S * random.uniform(0.8, 1.2)))),
                     "duree_s": float(PAS_S), "source": "seed"})
        t += timedelta(seconds=PAS_S)
for i in range(0, len(docs), 5000):
    col.insert_many(docs[i:i + 5000])
print(f"{len(docs)} mesures insérées ({JOURS} j, {len(ANIMAUX)} animaux) ; anomalie sur {MALADE}")