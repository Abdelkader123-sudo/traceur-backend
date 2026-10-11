import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

RAYON_TERRE_M = 6_371_000

# --- Paramètres de DBSCAN ---
EPS_METRES = 80          # deux points à moins de 80 m sont voisins
MIN_POINTS = 8           # nombre minimum de voisins pour former une zone dense

# --- Paramètres de détection ---
MARGE_ZONE_M = 30        # tolérance au-delà du rayon avant de parler de sortie de zone
DUREE_IMMOBILITE_S = 1800  # 30 min au repos = immobilité prolongée (mettre 300 pour tester)
VITESSE_REPOS_MAX = 0.3  # km/h : en dessous, l'animal est considéré à l'arrêt
VITESSE_MAX = 5.0        # km/h : au-dessus, déplacement rapide
NB_MESURES_VITESSE = 3   # nombre de mesures consécutives rapides avant alerte


def distance_m(lat1, lon1, lat2, lon2):
    """Distance en mètres entre deux points GPS (formule haversine)."""
    p1 = np.radians([[lat1, lon1]])
    p2 = np.radians([[lat2, lon2]])
    return float(haversine_distances(p1, p2)[0][0] * RAYON_TERRE_M)


def calculer_zones(positions):
    """Applique DBSCAN sur l'historique GPS d'un animal.

    positions : liste de dicts contenant au moins 'lat', 'lon' et 'timestamp'.
    Retourne (zones, anomalies) :
      - zones : zones habituelles (centre, rayon en mètres, nombre de points)
      - anomalies : positions isolées, hors de toute zone dense
    """
    coords = np.radians([[p["lat"], p["lon"]] for p in positions])

    modele = DBSCAN(
        eps=EPS_METRES / RAYON_TERRE_M,   # distance en radians
        min_samples=MIN_POINTS,
        metric="haversine",               # distance réelle sur la sphère terrestre
    )
    labels = modele.fit_predict(coords)

    zones = []
    for label in sorted(set(labels) - {-1}):
        points = coords[labels == label]
        centre = points.mean(axis=0)
        distances = haversine_distances(points, centre.reshape(1, -1)) * RAYON_TERRE_M
        zones.append({
            "centre_lat": round(float(np.degrees(centre[0])), 6),
            "centre_lon": round(float(np.degrees(centre[1])), 6),
            "rayon_m": round(float(distances.max()), 1),
            "nb_points": int(len(points)),
        })

    anomalies = [
        {"lat": p["lat"], "lon": p["lon"], "timestamp": p["timestamp"]}
        for p, label in zip(positions, labels) if label == -1
    ]
    return zones, anomalies


def ecart_a_la_zone(lat, lon, zones):
    """Distance (m) entre la position et le bord de la zone la plus proche.

    Négatif ou nul : l'animal est dans une zone. Positif : il est dehors.
    """
    return min(
        distance_m(lat, lon, z["centre_lat"], z["centre_lon"]) - z["rayon_m"]
        for z in zones
    )


def est_immobile(mesures, duree_min_s=DUREE_IMMOBILITE_S):
    """Vrai si l'animal est resté au repos sur toute la durée donnée.

    mesures : positions triées de la plus ancienne à la plus récente. La première
    est le dernier message reçu juste AVANT la fenêtre de temps : elle prouve que
    l'animal était déjà au repos au début de la fenêtre, même si des messages ont
    été espacés ou perdus.
    """
    if len(mesures) < 2:
        return False
    couverture = (mesures[-1]["timestamp"] - mesures[0]["timestamp"]).total_seconds()
    if couverture < duree_min_s:
        return False  # historique trop court pour conclure
    return all(
        m["activite"] == "repos" and m["vitesse"] <= VITESSE_REPOS_MAX
        for m in mesures
    )


def deplacement_inhabituel(mesures, vitesse_max=VITESSE_MAX, nb=NB_MESURES_VITESSE):
    """Vrai si les 'nb' dernières mesures dépassent toutes la vitesse maximale.

    mesures : positions triées de la plus ancienne à la plus récente.
    """
    if len(mesures) < nb:
        return False
    return all(m["vitesse"] >= vitesse_max for m in mesures[-nb:])