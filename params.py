"""Paramètres réglables depuis l'application (valeurs par défaut et bornes)."""

DEFAUTS = {
    # --- Modifiables depuis la page Réglages ---
    "eps_troupeau_m": 30,          # distance max entre deux animaux voisins (m)
    "distance_alerte_m": 120,      # écart au grand troupeau déclenchant l'alerte (m)
    "min_animaux_troupeau": 5,     # animaux minimum pour former / analyser un troupeau
    "vitesse_max": 5.0,            # km/h : déplacement rapide
    "duree_immobilite_s": 1800,    # repos continu avant alerte (s)
    "delai_collier_s": 600,        # silence du collier avant alerte (s)
    "marge_pature_m": 10,          # tolérance hors zone dessinée (m)

    # --- Retirés de l'interface : valeurs fixes utilisées par le backend ---
    "nb_confirmations": 2,         # messages consécutifs hors zone avant alerte
    "fraicheur_s": 600,            # âge max d'une position prise en compte (s)
    "nb_mesures_vitesse": 3,       # mesures rapides consécutives avant alerte
    "vitesse_repos_max": 0.5,      # km/h : en dessous, animal à l'arrêt

    # --- Réglé depuis la page Historique d'activité ---
    "reference_jours": 7,          # durée de la référence IA (jours)
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
    # minimum 5 j : avec 72 h valides requises, 3 j donneraient toujours « historique insuffisant » ;
    # à 4 j (96 h) il ne reste aucune marge pour les heures sans mesure, car le modèle perd ~18 h
    # au démarrage de ses moyennes glissantes.
    # maximum 14 j : le serveur supprime les mesures de plus de 30 jours
    "reference_jours": (5, 14, True),
}

# Paramètres entiers pour lesquels une valeur décimale (4.5) est refusée au lieu d'être tronquée.
# (Les autres entiers gardent leur comportement actuel.)
ENTIERS_STRICTS = {"reference_jours"}


def valider_parametres(donnees: dict) -> dict:
    """Retourne les paramètres valides ou lève ValueError."""
    if not isinstance(donnees, dict):
        raise ValueError("Format invalide.")
    propres = {}
    for cle, valeur in donnees.items():
        if cle not in BORNES:
            raise ValueError(f"Paramètre inconnu : {cle}")
        mini, maxi, entier = BORNES[cle]
        if cle in ENTIERS_STRICTS:
            if isinstance(valeur, bool):
                raise ValueError(f"{cle} : nombre entier attendu.")
            if isinstance(valeur, float) and not valeur.is_integer():
                raise ValueError(f"{cle} : nombre entier attendu.")
        try:
            v = int(valeur) if entier else float(valeur)
        except (TypeError, ValueError):
            raise ValueError(f"{cle} : nombre attendu.")
        if not (mini <= v <= maxi):
            raise ValueError(f"{cle} doit être entre {mini} et {maxi}.")
        propres[cle] = v
    return propres