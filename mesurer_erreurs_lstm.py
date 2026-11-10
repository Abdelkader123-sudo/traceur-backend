"""Mesure les erreurs du modèle LSTM sur les vraies positions (LECTURE SEULE).

Ne modifie rien dans MongoDB : il lit les positions, rejoue le modèle fenêtre par fenêtre
(comme le fait le service) et affiche la distribution des erreurs en mètres, par animal.
Cela permet de choisir un seuil adapté à VOTRE troupeau.

Utilisation (depuis le dossier du backend, venv actif) :
    $env:MONGO_URI = "mongodb+srv://..."
    python mesurer_erreurs_lstm.py                 # dernières 24 h
    python mesurer_erreurs_lstm.py --heures 72     # fenêtre plus longue
    python mesurer_erreurs_lstm.py --ignorer test_lstm
"""
import argparse
import os
from datetime import datetime, timedelta, timezone
from itertools import groupby

import numpy as np
from pymongo import MongoClient

import trajectoire_service as t

SEUILS_TEST = (70, 100, 150, 195)


def erreurs_animal(modele, positions):
    """Erreurs (m) du modèle pour toutes les fenêtres d'un animal."""
    pts = t.preparer_positions(positions)
    if len(pts) < t.N + 2:
        return np.array([])
    F, _ = t.deplacements(pts)
    X, Y, idx = t.fenetres(F)
    if len(idx) == 0:
        return np.array([])
    return np.linalg.norm(modele(X) - Y, axis=1)


def resume(nom, e):
    pct = lambda p: float(np.percentile(e, p))
    ligne = (f"{nom:<14}{len(e):>7}{np.median(e):>9.1f}{pct(95):>9.1f}"
             f"{pct(99):>9.1f}{e.max():>9.1f}")
    ligne += "".join(f"{100 * np.mean(e > s):>9.1f}" for s in SEUILS_TEST)
    return ligne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heures", type=int, default=24)
    ap.add_argument("--ignorer", nargs="*", default=["test_lstm"])
    args = ap.parse_args()

    db = MongoClient("mongodb+srv://abdelkaderjaziri4_db_user:abdelkaderjaziriSEM3@cluster0.mokdsar.mongodb.net/?appName=Cluster0", tz_aware=True)["traceur_gps"]
    modele = t.charger_modele()
    depuis = datetime.now(timezone.utc) - timedelta(hours=args.heures)

    curseur = db["positions"].find(
        {"timestamp": {"$gte": depuis}, "animal_id": {"$nin": args.ignorer}},
        {"animal_id": 1, "timestamp": 1, "lat": 1, "lon": 1},
    ).sort([("animal_id", 1), ("timestamp", 1)])

    entete = (f"{'animal':<14}{'fenêtres':>7}{'médiane':>9}{'p95':>9}{'p99':>9}{'max':>9}"
              + "".join(f"{'>' + str(s) + 'm %':>9}" for s in SEUILS_TEST))
    print(f"Erreurs du modèle (m) sur les {args.heures} dernières heures — "
          f"seuil actuel du modèle : {modele.seuil:.1f} m\n")
    print(entete)
    print("-" * len(entete))

    toutes = []
    for animal, docs in groupby(curseur, key=lambda d: d["animal_id"]):
        positions = []
        for d in docs:
            ts = d["timestamp"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            positions.append({"timestamp": ts, "lat": d.get("lat"), "lon": d.get("lon")})
        e = erreurs_animal(modele, positions)
        if len(e) == 0:
            print(f"{animal:<14}  pas assez de positions ({len(positions)})")
            continue
        toutes.append(e)
        print(resume(str(animal), e))

    if toutes:
        print("-" * len(entete))
        print(resume("TOUS", np.concatenate(toutes)))
        print("\nLecture : la colonne « > X m % » donne le pourcentage de positions "
              "NORMALES qui dépasseraient un seuil de X mètres (= fausses alertes par position).")
        print("Un seuil raisonnable laisse environ 1 % ou moins au-dessus.")
    else:
        print("\nAucun animal avec assez de positions sur cette période.")


if __name__ == "__main__":
    main()
