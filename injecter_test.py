import argparse
import os

from pymongo import MongoClient

import tester_modele as m
import trajectoire_service as t

ANIMAL = "TEST-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supprimer", action="store_true", help="efface les données de test")
    ap.add_argument("--pas", type=float, default=300.0, help="distance (m) des sauts agités")
    args = ap.parse_args()

    nom = "traceur_gps"
    uri = "mongodb+srv://abdelkaderjaziri4_db_user:abdelkaderjaziriSEM3@cluster0.mokdsar.mongodb.net/?appName=Cluster0"
    if not uri or not nom:
        raise SystemExit("Définissez MONGO_URI et MONGO_DB (voir l'en-tête du script).")
    db = MongoClient(uri)[nom]

    if args.supprimer:
        a = db[t.COLL_POSITIONS].delete_many({t.CHAMP_ANIMAL: ANIMAL}).deleted_count
        db[t.COLL_ETAT].delete_one({"_id": ANIMAL})
        b = db[t.COLL_ALERTES_SILENCIEUSES].delete_many({"animal_id": ANIMAL}).deleted_count
        print(f"Supprimé : {a} positions, {b} alertes.")
        return

    # 16 positions calmes puis 6 agitées, la dernière = maintenant (le service ne juge que la dernière heure
    # au premier cycle d'un animal, donc l'agitation doit être récente).
    positions, _ = m.fabriquer(16, 6, 0, args.pas)
    docs = [{t.CHAMP_ANIMAL: ANIMAL, t.CHAMP_TEMPS: p["timestamp"],
             t.CHAMP_LAT: p["lat"], t.CHAMP_LON: p["lon"]} for p in positions]
    db[t.COLL_POSITIONS].insert_many(docs)
    print(f"{len(docs)} positions insérées pour {ANIMAL}.")
    print("Attendez le prochain cycle du backend (LSTM_INTERVALLE_S), puis rafraîchissez la page.")


if __name__ == "__main__":
    main()
