"""Ajoute 7 jours calendaires de mesures pour tester deux anomalies sans suppression.

Les dernières 24 heures simulent une activité très basse pour cow08 et très élevée
pour cow09. Les données existantes et les autres animaux ne sont jamais modifiés.
"""

import argparse
import os
import random
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from pymongo import MongoClient

PAS_S = 300
TZ = ZoneInfo("Africa/Tunis")
PROFILS = {"repos": (0.02, 0), "marche": (0.12, 1.0)}
ANOMALIES = {"cow08": "calme", "cow09": "forte"}
DECALAGE_TEST_S = 150


def rythme(heure: int) -> float:
    if heure >= 22 or heure < 5:
        return 0.15
    if 6 <= heure < 10 or 16 <= heure < 19:
        return 1.0
    return 0.6


def debut_periode(jours: int, fin: datetime) -> datetime:
    premier_jour = fin.astimezone(TZ).date() - timedelta(days=jours - 1)
    return datetime.combine(premier_jour, time.min, TZ).astimezone(timezone.utc)


def creer_mesures(jours: int, animal_id: str, fin: datetime) -> list[dict]:
    debut = debut_periode(jours, fin) + timedelta(seconds=DECALAGE_TEST_S)
    rng = random.Random(42 + int(animal_id[-2:]))
    echelle = rng.uniform(0.7, 1.4)
    docs = []
    t = debut

    while t < fin:
        dans_anomalie = t >= fin - timedelta(hours=24)
        facteur = rythme(t.astimezone(TZ).hour)
        marche = rng.random() < min(0.95, max(0.01, 0.4 * facteur))
        acc, cad = PROFILS["marche" if marche else "repos"]

        if dans_anomalie and ANOMALIES[animal_id] == "calme":
            reduction = 0.01
            mi = float(max(0, round(acc * 100 * echelle * reduction * rng.uniform(0.8, 1.2))))
            steps = float(max(0, round(cad * echelle * PAS_S * reduction * rng.uniform(0.8, 1.2))))
        elif dans_anomalie and ANOMALIES[animal_id] == "forte":
            acc, cad = PROFILS["marche"]
            mi = float(round(acc * 100 * echelle * 8 * rng.uniform(0.9, 1.1)))
            steps = float(round(cad * echelle * PAS_S * 8 * rng.uniform(0.9, 1.1)))
        else:
            mi = float(max(0, round(acc * 100 * echelle * rng.uniform(0.8, 1.2))))
            steps = float(max(0, round(cad * echelle * PAS_S * rng.uniform(0.8, 1.2))))

        docs.append({
            "animal_id": animal_id,
            "timestamp": t,
            "mi": mi,
            "steps": steps,
            "duree_s": float(PAS_S),
            "source": "seed",
        })
        t += timedelta(seconds=PAS_S)

    return docs


def filtrer_mesures_existantes(
    docs: list[dict], cles_existantes: set[tuple[str, datetime]]
) -> list[dict]:
    return [
        doc for doc in docs
        if (doc["animal_id"], doc["timestamp"]) not in cles_existantes
    ]


def fin_periode(maintenant: datetime) -> datetime:
    minute = maintenant.minute - maintenant.minute % (PAS_S // 60)
    return maintenant.replace(minute=minute, second=0, microsecond=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ajouter les 7 derniers jours calendaires : cow08 plus calme et "
            "cow09 plus active sur les dernières 24 h. Aucune mesure n'est supprimée."
        )
    )
    parser.add_argument(
        "--jours",
        type=int,
        default=7,
        help="Nombre de jours calendaires à générer, aujourd'hui inclus (4 à 30)",
    )
    args = parser.parse_args()
    jours = max(4, min(args.jours, 30))

    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        parser.error("MONGO_URI est requis (définissez la variable d'environnement avant le lancement).")

    collection = MongoClient(mongo_uri, tz_aware=True)["traceur_gps"]["mesures_activite"]
    fin = fin_periode(datetime.now(timezone.utc))
    debut = debut_periode(jours, fin)
    docs = [
        doc
        for animal_id in ANOMALIES
        for doc in creer_mesures(jours, animal_id, fin)
    ]
    cles_existantes = {
        (doc["animal_id"], doc["timestamp"])
        for doc in collection.find(
            {
                "animal_id": {"$in": list(ANOMALIES)},
                "timestamp": {"$gte": debut, "$lt": fin},
            },
            {"_id": 0, "animal_id": 1, "timestamp": 1},
        )
    }
    nouvelles = filtrer_mesures_existantes(docs, cles_existantes)

    for i in range(0, len(nouvelles), 5000):
        collection.insert_many(nouvelles[i:i + 5000])

    print(
        f"{len(nouvelles)} mesures ajoutées sur {len(docs)} prévues pour cow08 et cow09 "
        f"({jours} jours calendaires, du {debut.astimezone(TZ):%d/%m} au "
        f"{(fin - timedelta(seconds=PAS_S - DECALAGE_TEST_S)).astimezone(TZ):%d/%m %H:%M} "
        f"heure locale) ; {len(cles_existantes)} mesures existantes conservées. "
        "Anomalies des dernières 24 h : cow08 plus calme, cow09 plus active."
    )


if __name__ == "__main__":
    main()
