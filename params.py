"""Paramètres réglables depuis l'application (valeurs par défaut et bornes)."""

DEFAUTS = {
    "eps_troupeau_m": 30,          # distance max entre deux animaux voisins (m)
    "distance_alerte_m": 120,      # écart au grand troupeau déclenchant l'alerte (m)
    "nb_confirmations": 2,         # messages consécutifs hors zone avant alerte
    "min_animaux_troupeau": 5,     # animaux actifs minimum pour analyser
    "fraicheur_s": 420,            # âge max d'une position prise en compte (s)
    "vitesse_max": 5.0,            # km/h : déplacement rapide
    "nb_mesures_vitesse": 3,       # mesures rapides consécutives avant alerte
    "vitesse_repos_max": 0.3,      # km/h : en dessous, animal à l'arrêt
    "duree_immobilite_s": 1800,    # repos continu avant alerte (s)
    "delai_collier_s": 600,        # silence du collier avant alerte (s)
    "marge_pature_m": 10,          # tolérance hors zone dessinée (m)
}

# clé: (minimum, maximum, entier ?)
BORNES = {
    "eps_troupeau_m": (2, 500, False),
    "distance_alerte_m": (5, 5000, False),
    "nb_confirmations": (1, 20, True),
    "min_animaux_troupeau": (2, 1000, True),
    "fraicheur_s": (30, 7200, True),
    "vitesse_max": (0.5, 60, False),
    "nb_mesures_vitesse": (1, 20, True),
    "vitesse_repos_max": (0, 5, False),
    "duree_immobilite_s": (60, 86400, True),
    "delai_collier_s": (60, 86400, True),
    "marge_pature_m": (0, 500, False),
}


def valider_parametres(donnees: dict) -> dict:
    """Retourne les paramètres valides ou lève ValueError."""
    if not isinstance(donnees, dict):
        raise ValueError("Format invalide.")
    propres = {}
    for cle, valeur in donnees.items():
        if cle not in BORNES:
            raise ValueError(f"Paramètre inconnu : {cle}")
        mini, maxi, entier = BORNES[cle]
        try:
            v = int(valeur) if entier else float(valeur)
        except (TypeError, ValueError):
            raise ValueError(f"{cle} : nombre attendu.")
        if not (mini <= v <= maxi):
            raise ValueError(f"{cle} doit être entre {mini} et {maxi}.")
        propres[cle] = v
    return propres