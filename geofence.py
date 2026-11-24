"""Géorepérage : un point est-il à l'intérieur d'un polygone de pâturage ?"""
import math

RAYON_TERRE_M = 6_371_000


def valider_sommets(coordonnees: list[dict]) -> list[tuple[float, float]]:
    """Retourne [(lat, lon), ...] ou lève ValueError."""
    if not isinstance(coordonnees, list) or len(coordonnees) < 3:
        raise ValueError("Une zone demande au moins 3 sommets.")
    sommets = []
    for c in coordonnees:
        try:
            lat, lon = float(c["latitude"]), float(c["longitude"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Sommet invalide.")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Coordonnées hors limites.")
        sommets.append((lat, lon))
    return sommets


def _vers_metres(lat, lon, lat0, lon0):
    """Projection locale (x est, y nord) en mètres, valable sur quelques km."""
    x = math.radians(lon - lon0) * RAYON_TERRE_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * RAYON_TERRE_M
    return x, y


def _dist_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    t = 0 if n == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / n))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def analyser_point(lat: float, lon: float, sommets: list) -> tuple[bool, float]:
    """Retourne (a_l_interieur, distance_au_bord_en_m)."""
    lat0 = sum(s[0] for s in sommets) / len(sommets)
    lon0 = sum(s[1] for s in sommets) / len(sommets)
    pts = [_vers_metres(a, b, lat0, lon0) for a, b in sommets]
    px, py = _vers_metres(lat, lon, lat0, lon0)

    dedans = False                       # lancer de rayon
    n = len(pts)
    for i in range(n):
        (ax, ay), (bx, by) = pts[i], pts[(i + 1) % n]
        if (ay > py) != (by > py):
            if px < (bx - ax) * (py - ay) / (by - ay) + ax:
                dedans = not dedans

    bord = min(_dist_segment(px, py, *pts[i], *pts[(i + 1) % n]) for i in range(n))
    return dedans, bord


def evaluer_hors_zone(lat: float, lon: float, sommets: list) -> tuple[bool, float]:
    """Retourne si le point est hors du polygone et sa distance au bord."""
    dedans, distance_bord = analyser_point(lat, lon, sommets)
    return not dedans, distance_bord