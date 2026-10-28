import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

RAYON_TERRE_M = 6_371_000

# --- Paramètres des troupeaux ---
EPS_TROUPEAU_M = 30        # deux animaux à moins de 30 m sont voisins
MIN_ANIMAUX_TROUPEAU = 5   # en dessous, pas d'analyse de zone
FRAICHEUR_S = 420          # position ignorée au-delà de 7 min (envoi toutes les 2 min)
DISTANCE_ALERTE_M = 120    # écart bord à bord avec le grand troupeau
NB_CONFIRMATIONS = 2       # messages successifs du même animal hors zone avant alerte

# --- Collier silencieux ---
DELAI_COLLIER_S = 600      # aucun message depuis 10 min = collier en panne    

# --- Paramètres de détection individuelle ---
DUREE_IMMOBILITE_S = 1800  # 30 min au repos (mettre 300 pour tester)
VITESSE_REPOS_MAX = 0.3    # km/h
VITESSE_MAX = 5.0          # km/h
NB_MESURES_VITESSE = 3


def distance_m(lat1, lon1, lat2, lon2):
    """Distance en mètres entre deux points GPS (formule haversine)."""
    p1 = np.radians([[lat1, lon1]])
    p2 = np.radians([[lat2, lon2]])
    return float(haversine_distances(p1, p2)[0][0] * RAYON_TERRE_M)


def _zone(coords, derniers, indices):
    """Zone (centre, rayon) d'un groupe d'animaux."""
    points = coords[indices]
    centre = points.mean(axis=0)
    rayon = (haversine_distances(points, centre.reshape(1, -1)) * RAYON_TERRE_M).max()
    return {
        "centre_lat": round(float(np.degrees(centre[0])), 6),
        "centre_lon": round(float(np.degrees(centre[1])), 6),
        "rayon_m": round(float(rayon), 1),
        "nb_animaux": len(indices),
        "animaux": [derniers[i]["animal_id"] for i in indices],
    }


def analyser_troupeaux(derniers, p):
    """p : dict de paramètres (eps_troupeau_m, min_animaux_troupeau, distance_alerte_m)."""
    if len(derniers) < p["min_animaux_troupeau"]:
        return None, [], {}

    coords = np.radians([[d["lat"], d["lon"]] for d in derniers])
    labels = DBSCAN(
        eps=p["eps_troupeau_m"] / RAYON_TERRE_M,
        min_samples=2,
        metric="haversine",
    ).fit_predict(coords)

    suivant = int(labels.max()) + 1
    labels = list(labels)
    for i, l in enumerate(labels):
        if l == -1:
            labels[i] = suivant
            suivant += 1

    groupes = {}
    for i, l in enumerate(labels):
        groupes.setdefault(l, []).append(i)

    id_principal = max(groupes, key=lambda g: len(groupes[g]))
    principal = _zone(coords, derniers, groupes[id_principal])

    autres, ecarts = [], {}
    for g, indices in groupes.items():
        if g == id_principal:
            continue
        zone = _zone(coords, derniers, indices)
        ecart = distance_m(
            principal["centre_lat"], principal["centre_lon"],
            zone["centre_lat"], zone["centre_lon"],
        ) - principal["rayon_m"] - zone["rayon_m"]
        ecart = round(max(ecart, 0.0), 1)
        autres.append({
            "type": "troupeau" if len(indices) > 1 else "individuel",
            "zone": zone,
            "ecart_m": ecart,
            "depasse_seuil": ecart > p["distance_alerte_m"],
        })
        for animal_id in zone["animaux"]:
            ecarts[animal_id] = ecart
    return principal, autres, ecarts


def est_immobile(mesures, duree_min_s, vitesse_repos_max):
    if len(mesures) < 2:
        return False
    couverture = (mesures[-1]["timestamp"] - mesures[0]["timestamp"]).total_seconds()
    if couverture < duree_min_s:
        return False
    return all(
        m["activite"] == "repos" and m["vitesse"] <= vitesse_repos_max
        for m in mesures
    )


def deplacement_inhabituel(mesures, vitesse_max, nb):
    if len(mesures) < nb:
        return False
    return all(m["vitesse"] >= vitesse_max for m in mesures[-nb:])