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


def analyser_troupeaux(derniers):
    """Trouve le grand troupeau, les autres groupes et leur écart au grand troupeau.

    Retourne (principal, autres, ecarts) :
      - principal : zone du plus grand groupe
      - autres : liste des autres groupes (type, zone, écart, dépasse le seuil)
      - ecarts : dict animal_id -> écart (m) pour les animaux hors du grand troupeau
    """
    if len(derniers) < MIN_ANIMAUX_TROUPEAU:
        return None, [], {}

    coords = np.radians([[p["lat"], p["lon"]] for p in derniers])
    labels = DBSCAN(
        eps=EPS_TROUPEAU_M / RAYON_TERRE_M,
        min_samples=2,
        metric="haversine",
    ).fit_predict(coords)

    # Chaque animal isolé (-1) devient son propre groupe d'un animal
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
            "depasse_seuil": ecart > DISTANCE_ALERTE_M,
        })
        for animal_id in zone["animaux"]:
            ecarts[animal_id] = ecart
    return principal, autres, ecarts


def est_immobile(mesures, duree_min_s=DUREE_IMMOBILITE_S):
    """Vrai si l'animal est resté au repos sur toute la durée donnée."""
    if len(mesures) < 2:
        return False
    couverture = (mesures[-1]["timestamp"] - mesures[0]["timestamp"]).total_seconds()
    if couverture < duree_min_s:
        return False
    return all(
        m["activite"] == "repos" and m["vitesse"] <= VITESSE_REPOS_MAX
        for m in mesures
    )


def deplacement_inhabituel(mesures, vitesse_max=VITESSE_MAX, nb=NB_MESURES_VITESSE):
    """Vrai si les 'nb' dernières mesures dépassent toutes la vitesse maximale."""
    if len(mesures) < nb:
        return False
    return all(m["vitesse"] >= vitesse_max for m in mesures[-nb:])