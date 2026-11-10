"""Teste le modèle LSTM HORS LIGNE : aucune connexion à MongoDB, rien n'est écrit nulle part.

Le script fabrique les positions d'un animal fictif (une toutes les 5 min), fait tourner le vrai modèle
(trajectoire.onnx) et affiche, pour chaque position jugée :
    - le déplacement RÉEL (m) depuis la position précédente,
    - le déplacement PRÉVU (m) par le modèle d'après les 12 déplacements précédents,
    - l'ERREUR (m) = distance entre les deux,
    - la décision : normale / suspecte, et le moment où l'alerte se déclenche.

À lancer depuis le dossier du backend (à côté de trajectoire_service.py), venv actif :
    python tester_modele.py                           # 16 positions calmes puis 6 agitées
    python tester_modele.py --agitees 0               # animal toujours calme : aucune alerte attendue
    python tester_modele.py --apres 4 --agitees 3     # agitation brève puis retour au calme
    python tester_modele.py --k 3                     # exige 3 positions suspectes de suite
    python tester_modele.py --seuil 120               # force un autre seuil (m)
    python tester_modele.py --pas 400                 # agitation plus forte
"""
import argparse
import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np

import trajectoire_service as t

M_PAR_DEG = 111320.0


def avancer(lat, lon, distance_m, cap):
    """Déplace un point de distance_m mètres dans la direction cap (radians, 0 = nord)."""
    lat += distance_m * math.cos(cap) / M_PAR_DEG
    lon += distance_m * math.sin(cap) / (M_PAR_DEG * math.cos(math.radians(lat)))
    return lat, lon


def fabriquer(calmes, agitees, apres, pas, intervalle_min=5):
    """Positions : marche calme (cap qui change peu, 5 à 12 m), puis sauts de `pas` m dans des
    directions aléatoires, puis retour au calme."""
    total = calmes + agitees + apres
    fin = datetime.now(timezone.utc).replace(microsecond=0)
    debut = fin - timedelta(minutes=intervalle_min * (total - 1))
    lat, lon, cap = 36.8065, 10.1815, random.uniform(0, 2 * math.pi)
    positions, phases = [], []
    for i in range(total):
        agite = calmes <= i < calmes + agitees
        if agite:
            cap = random.uniform(0, 2 * math.pi)
            d = pas
        else:
            cap += random.gauss(0, 0.25)
            d = random.uniform(5, 12)
        lat, lon = avancer(lat, lon, d, cap)
        positions.append({"timestamp": debut + timedelta(minutes=intervalle_min * i),
                          "lat": lat, "lon": lon})
        phases.append("agité" if agite else "calme")
    return positions, phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calmes", type=int, default=16, help="positions calmes au début")
    ap.add_argument("--agitees", type=int, default=6, help="positions agitées ensuite")
    ap.add_argument("--apres", type=int, default=0, help="positions calmes après l'agitation")
    ap.add_argument("--pas", type=float, default=200.0, help="distance (m) des sauts agités")
    ap.add_argument("--seuil", type=float, default=None, help="seuil en m (défaut : celui du modèle)")
    ap.add_argument("--k", type=int, default=t.K, help="positions suspectes de suite avant alerte")
    ap.add_argument("--graine", type=int, default=7, help="graine aléatoire (résultats reproductibles)")
    args = ap.parse_args()

    random.seed(args.graine)
    modele = t.charger_modele()
    seuil = args.seuil if args.seuil is not None else modele.seuil

    positions, phases = fabriquer(args.calmes, args.agitees, args.apres, args.pas)
    n_pos = len(positions)
    print(f"{n_pos} positions : {args.calmes} calmes, {args.agitees} agitées (sauts de {args.pas:.0f} m), "
          f"{args.apres} calmes ensuite.")
    print(f"Seuil : {seuil:.1f} m | K = {args.k} | fenêtre du modèle : {t.N} déplacements "
          f"(il faut au moins {t.N + 2} positions).\n")

    pts = t.preparer_positions(positions)
    if len(pts) < t.N + 2:
        print(f"Pas assez de positions : {len(pts)} (minimum {t.N + 2}). Augmentez --calmes.")
        return
    F, instants = t.deplacements(pts)
    X, Y, idx = t.fenetres(F)
    if len(idx) == 0:
        print("Aucune fenêtre exploitable.")
        return
    prevu = modele(X)
    erreurs = np.linalg.norm(prevu - Y, axis=1)

    # Rejoue la logique exacte du service (toutes les fenêtres jugées, comme après plusieurs cycles)
    _, alertes = t.evaluer_animal(
        positions, {"derniere_cible": datetime(1970, 1, 1), "serie": 0}, modele, seuil, k=args.k)
    instants_alerte = {a["instant_local"] for a in alertes}

    print(f"{'position':>9} {'heure':>6} {'phase':>6} {'réel (m)':>9} {'prévu (m)':>10} "
          f"{'erreur (m)':>11}  décision")
    print("-" * 75)
    serie = 0
    for j, i in enumerate(idx):
        i = int(i)
        e = float(erreurs[j])
        numero = i + 2                      # F[i] va de la position i+1 à la position i+2 (numérotées dès 1)
        if e > seuil:
            serie += 1
            decision = f"SUSPECTE (série : {serie}, alerte à {args.k})"
        else:
            serie = 0
            decision = "normale"
        if instants[i] in instants_alerte:
            decision += "  <== ALERTE"
        print(f"{numero:>9} {instants[i]:%H:%M} {phases[numero - 1]:>6} "
              f"{float(np.linalg.norm(Y[j])):>9.1f} {float(np.linalg.norm(prevu[j])):>10.1f} "
              f"{e:>11.1f}  {decision}")

    print("-" * 75)
    print(f"{len(idx)} positions jugées (les {t.N + 1} premières servent seulement d'historique).")
    print(f"Erreur médiane : {np.median(erreurs):.1f} m | maximale : {erreurs.max():.1f} m.")
    print(f"Alertes déclenchées : {len(alertes)}")
    if args.agitees > 0 and not alertes:
        print("Aucune alerte : l'agitation n'a pas atteint K positions suspectes de suite, "
              "ou le modèle l'a jugée trop faible. Essayez --pas plus grand, --k plus petit "
              "ou --seuil plus bas.")
    if args.agitees == 0 and alertes:
        print("Fausse alerte sur un animal calme : le seuil est probablement trop bas.")


if __name__ == "__main__":
    main()
