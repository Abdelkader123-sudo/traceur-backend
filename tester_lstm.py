"""Injecte dans MongoDB un animal fictif pour tester l'alerte « mouvement agité ».

Scénario : 16 positions calmes (une toutes les 5 min, dérive d'environ 10 m),
puis 6 positions agitées (environ 200 m entre deux positions, directions aléatoires).
La dernière position est datée de maintenant.

Utilisation (depuis le dossier du backend, avec le venv actif) :
    $env:MONGO_URI = "mongodb+srv://..."           # PowerShell
    python tester_lstm.py                          # injecte
    python tester_lstm.py --supprimer              # nettoie les données de test
Options : --animal NOM  --calmes 16  --agitees 6  --pas 200
"""
import argparse
import math
import os
import random
from datetime import datetime, timedelta, timezone

from pymongo import MongoClient

INTERVALLE_MIN = 5
LAT_PAR_M = 1 / 111_320.0


def deplacer(lat, lon, distance_m, angle):
    """Déplace un point de distance_m mètres dans la direction angle (radians)."""
    dlat = distance_m * math.cos(angle) * LAT_PAR_M
    dlon = distance_m * math.sin(angle) * LAT_PAR_M / math.cos(math.radians(lat))
    return lat + dlat, lon + dlon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="test_lstm")
    ap.add_argument("--calmes", type=int, default=16)
    ap.add_argument("--agitees", type=int, default=6)
    ap.add_argument("--pas", type=float, default=200.0, help="distance (m) entre positions agitées")
    ap.add_argument("--supprimer", action="store_true")
    args = ap.parse_args()

    client = MongoClient(os.environ["MONGO_URI"], tz_aware=True)
    base = client["traceur_gps"]

    if args.supprimer:
        for nom in ("positions", "alerts", "etat_animaux", "lstm_alertes_silencieuses"):
            n = base[nom].delete_many({"animal_id": args.animal}).deleted_count
            print(f"{nom} : {n} document(s) supprimé(s)")
        # l'état du modèle est indexé par _id (= identifiant de l'animal)
        n = base["lstm_etat"].delete_many({"_id": args.animal}).deleted_count
        print(f"lstm_etat : {n} document(s) supprimé(s)")
        return

    # point de départ : dernière position d'un vrai animal (sinon Tunis)
    ref = base["positions"].find_one({"animal_id": {"$ne": args.animal}},
                                     sort=[("timestamp", -1)])
    lat, lon = (ref["lat"], ref["lon"]) if ref else (36.8065, 10.1815)

    total = args.calmes + args.agitees
    fin = datetime.now(timezone.utc).replace(microsecond=0)
    debut = fin - timedelta(minutes=INTERVALLE_MIN * (total - 1))

    docs = []
    for i in range(total):
        if i < args.calmes:
            lat, lon = deplacer(lat, lon, random.uniform(0, 10), random.uniform(0, 2 * math.pi))
            vitesse, activite = 0.3, "repos"
        else:
            lat, lon = deplacer(lat, lon, args.pas, random.uniform(0, 2 * math.pi))
            vitesse = round(args.pas / (INTERVALLE_MIN * 60) * 3.6, 1)   # km/h
            activite = "marche"
        docs.append({"animal_id": args.animal, "lat": lat, "lon": lon,
                     "vitesse": vitesse, "activite": activite,
                     "timestamp": debut + timedelta(minutes=INTERVALLE_MIN * i)})

    base["positions"].insert_many(docs)
    print(f"{len(docs)} positions injectées pour « {args.animal} » "
          f"({args.calmes} calmes puis {args.agitees} agitées, de {docs[0]['timestamp']:%H:%M} "
          f"à {docs[-1]['timestamp']:%H:%M} UTC).")
    print("Attendez le prochain cycle LSTM (toutes les 15 min), puis regardez les logs Render, "
          "la collection lstm_evenements et la page « Mouvements agités ».")


if __name__ == "__main__":
    main()