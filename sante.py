"""Surveillance de l'état général : règles statistiques, sans machine learning.

IMPORTANT
- Ce module NE POSE PAS de diagnostic et ne calcule aucune probabilité de maladie.
- « Aucun écart détecté » ne signifie jamais « animal en bonne santé ».
- Tous les seuils ci-dessous sont EXPÉRIMENTAUX : ce sont des hypothèses techniques,
  pas des normes vétérinaires validées. Ils doivent être calibrés sur des données
  réelles et validés par un vétérinaire avant tout usage réel.
"""
import math
from datetime import datetime, timedelta, timezone
from statistics import median

RAYON_TERRE_M = 6_371_000

# --- Paramètres techniques (liés au matériel) ---
INTERVALLE_ATTENDU_S = 120                          # un message toutes les 2 min
MESURES_PAR_JOUR = 86400 // INTERVALLE_ATTENDU_S    # 720

# --- Paramètres EXPÉRIMENTAUX (aucune source validée) ---
QUALITE_MIN = 0.7            # part minimale de mesures reçues pour qu'un jour soit fiable
VITESSE_GPS_MAX_KMH = 20.0   # au-delà, le saut GPS est jugé irréaliste et ignoré
SAUTS_MAX_RATIO = 0.05       # au-delà de 5 % de sauts, la distance du jour n'est pas fiable
JOURS_REFERENCE = 14         # fenêtre maximale de la référence individuelle
JOURS_REFERENCE_MIN = 5      # nombre minimal de jours fiables dans la référence
JOURS_EVALUATION = 3         # derniers jours comparés à la référence
SEUIL_SCORE = 3.0            # écart robuste (en « MAD ») considéré inhabituel
JOURS_SURVEILLANCE = 2       # jours consécutifs pour « surveillance recommandée »
JOURS_VETERINAIRE = 3        # jours consécutifs, distance ET activité, pour « évaluation »
PLANCHER_MAD_RELATIF = 0.10  # évite une échelle nulle si l'animal est très régulier
POIDS_ANCIEN_JOURS = 90      # poids saisi trop ancien

NIVEAUX = {
    "donnees_insuffisantes": "Données insuffisantes",
    "aucun_ecart": "Aucun écart inhabituel détecté dans les indicateurs disponibles",
    "surveillance": "Surveillance recommandée",
    "veterinaire": "Évaluation vétérinaire à envisager",
}

LIMITES = [
    "Ce système ne diagnostique aucune maladie.",
    "Une absence d'écart ne prouve pas que l'animal est en bonne santé.",
    "Les seuils sont expérimentaux et non validés par des données vétérinaires.",
    "L'activité repose sur le champ 'activite' reçu, non validé sur le terrain.",
    "Une dégradation lente et continue peut contaminer la période de référence.",
    "Distance GPS : la formule de Haversine suppose une Terre sphérique et le bruit "
    "GPS gonfle légèrement la distance d'un animal immobile.",
]


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en mètres entre deux points GPS (formule de Haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * RAYON_TERRE_M * math.asin(math.sqrt(a))


def _valider(mesures: list[dict]) -> tuple[list[dict], int]:
    """Écarte les mesures invalides et les doublons. Retourne (mesures, nb_rejetées)."""
    propres, vus, rejets = [], set(), 0
    for m in mesures:
        try:
            ts, lat, lon, v = m["timestamp"], m["lat"], m["lon"], m["vitesse"]
            ok = (
                isinstance(ts, datetime)
                and -90 <= lat <= 90 and -180 <= lon <= 180
                and v >= 0
                and ts not in vus
            )
        except (KeyError, TypeError):
            ok = False
        if ok:
            vus.add(ts)
            propres.append(m)
        else:
            rejets += 1
    propres.sort(key=lambda m: m["timestamp"])
    return propres, rejets


def indicateurs_jour(mesures: list[dict]) -> dict:
    """Indicateurs d'une journée (mesures valides, triées, du même jour)."""
    n = len(mesures)
    qualite = min(n / MESURES_PAR_JOUR, 1.0)
    min_par_mesure = INTERVALLE_ATTENDU_S / 60
    minutes = {"repos": 0.0, "marche": 0.0, "course": 0.0}
    for m in mesures:
        if m.get("activite") in minutes:
            minutes[m["activite"]] += min_par_mesure

    distance, sauts, paires, trous = 0.0, 0, 0, 0
    for a, b in zip(mesures, mesures[1:]):
        dt = (b["timestamp"] - a["timestamp"]).total_seconds()
        if dt <= 0:
            continue
        if dt > 3 * INTERVALLE_ATTENDU_S:     # trou de données : on ne relie pas
            trous += 1
            continue
        d = distance_m(a["lat"], a["lon"], b["lat"], b["lon"])
        if d / dt * 3.6 > VITESSE_GPS_MAX_KMH:  # saut GPS irréaliste
            sauts += 1
            continue
        distance += d
        paires += 1

    ratio_sauts = sauts / (sauts + paires) if (sauts + paires) else 0.0
    return {
        "nb_mesures": n,
        "qualite": round(qualite, 2),
        "distance_m": round(distance, 1),
        "minutes_repos": minutes["repos"],
        "minutes_marche": minutes["marche"],
        "minutes_course": minutes["course"],
        "minutes_actives": minutes["marche"] + minutes["course"],
        "sauts_gps_ignores": sauts,
        "trous_de_donnees": trous,
        "fiable": qualite >= QUALITE_MIN and ratio_sauts <= SAUTS_MAX_RATIO,
    }


def reference_robuste(valeurs: list[float]) -> dict | None:
    """Médiane et échelle robuste (1,4826 × MAD) d'une série de valeurs."""
    if len(valeurs) < JOURS_REFERENCE_MIN:
        return None
    med = median(valeurs)
    mad = median(abs(v - med) for v in valeurs)
    if med == 0 and mad == 0:
        return None                            # série nulle : score non défini
    echelle = max(1.4826 * mad, PLANCHER_MAD_RELATIF * abs(med))
    return {"mediane": round(med, 1), "echelle": round(echelle, 2)}


def _serie_consecutive(scores: list[float | None]) -> int:
    """Nombre de jours consécutifs (en finissant par le plus récent) sous -SEUIL_SCORE."""
    n = 0
    for s in reversed(scores):
        if s is not None and s <= -SEUIL_SCORE:
            n += 1
        else:
            break
    return n


def contexte_fiche(fiche: dict | None, maintenant: datetime) -> dict:
    """Informations de la fiche animal. Aucune référence pondérale validée n'est utilisée."""
    fiche = fiche or {}
    champs = ["espece", "race", "age_mois", "sexe", "poids_kg", "stade_physiologique"]
    manquants = [c for c in champs if fiche.get(c) in (None, "")]
    remarques = ["Évaluation pondérale impossible : aucune table de référence validée "
                 "(race, âge, sexe, stade) n'est configurée."]
    date_poids = fiche.get("poids_date")
    if isinstance(date_poids, datetime):
        if date_poids.tzinfo is None:
            date_poids = date_poids.replace(tzinfo=timezone.utc)
        if (maintenant - date_poids).days > POIDS_ANCIEN_JOURS:
            remarques.append("Le poids saisi est ancien : à mettre à jour.")
    elif fiche.get("poids_kg") is not None:
        remarques.append("Date de saisie du poids inconnue.")
    return {"champs_manquants": manquants, "remarques": remarques}


def evaluer_sante(mesures: list[dict], fiche: dict | None = None,
                  maintenant: datetime | None = None) -> dict:
    """Évalue les écarts de comportement d'un animal par rapport à sa propre habitude.

    mesures : dicts avec timestamp (datetime), lat, lon, vitesse, activite.
    Seuls les jours COMPLETS (avant aujourd'hui, UTC) sont analysés.
    """
    maintenant = maintenant or datetime.now(timezone.utc)
    debut_aujourdhui = maintenant.replace(hour=0, minute=0, second=0, microsecond=0)
    nb_jours = JOURS_REFERENCE + JOURS_EVALUATION
    jours = [(debut_aujourdhui - timedelta(days=k)).date() for k in range(nb_jours, 0, -1)]

    valides, rejets = _valider(mesures)
    par_jour: dict = {j: [] for j in jours}
    for m in valides:
        j = m["timestamp"].astimezone(timezone.utc).date()
        if j in par_jour:
            par_jour[j].append(m)
    ind = {j: indicateurs_jour(ms) if ms else None for j, ms in par_jour.items()}

    jours_eval, jours_ref = jours[-JOURS_EVALUATION:], jours[:-JOURS_EVALUATION]
    fiables_ref = [ind[j] for j in jours_ref if ind[j] and ind[j]["fiable"]]
    fiables_eval = [j for j in jours_eval if ind[j] and ind[j]["fiable"]]

    raisons, detail, series = [], {}, {}
    for cle, libelle in (("distance_m", "distance quotidienne"),
                         ("minutes_actives", "activité quotidienne (marche + course)")):
        ref = reference_robuste([i[cle] for i in fiables_ref])
        if ref is None:
            detail[cle] = {"reference": None}
            series[cle] = None
            continue
        scores = []
        for j in jours_eval:
            if ind[j] and ind[j]["fiable"]:
                scores.append(round((ind[j][cle] - ref["mediane"]) / ref["echelle"], 2))
            else:
                scores.append(None)
        series[cle] = _serie_consecutive(scores)
        detail[cle] = {
            "reference": ref,
            "valeurs_recentes": [ind[j][cle] if ind[j] else None for j in jours_eval],
            "scores_robustes": scores,
            "jours_consecutifs_en_baisse": series[cle],
            "libelle": libelle,
        }

    qualite = {
        "jours_reference_fiables": len(fiables_ref),
        "jours_recents_fiables": len(fiables_eval),
        "mesures_rejetees": rejets,
    }

    if any(s is None for s in series.values()):
        niveau = "donnees_insuffisantes"
        raisons.append(f"Référence individuelle insuffisante : {len(fiables_ref)} jour(s) "
                       f"fiable(s), minimum {JOURS_REFERENCE_MIN}.")
    elif len(fiables_eval) < 2:
        niveau = "donnees_insuffisantes"
        raisons.append("Moins de 2 jours récents fiables (mesures manquantes ou GPS aberrant).")
    else:
        sd, sa = series["distance_m"], series["minutes_actives"]
        if sd >= JOURS_VETERINAIRE and sa >= JOURS_VETERINAIRE:
            niveau = "veterinaire"
            raisons.append(f"Distance ET activité nettement sous l'habitude de l'animal "
                           f"depuis {min(sd, sa)} jours consécutifs.")
        elif max(sd, sa) >= JOURS_SURVEILLANCE:
            niveau = "surveillance"
            if sd >= JOURS_SURVEILLANCE:
                raisons.append(f"Distance quotidienne inhabituellement basse depuis {sd} jours.")
            if sa >= JOURS_SURVEILLANCE:
                raisons.append(f"Activité quotidienne inhabituellement basse depuis {sa} jours.")
        else:
            niveau = "aucun_ecart"
            raisons.append("Aucun écart inhabituel dans les indicateurs disponibles.")

    return {
        "niveau": niveau,
        "libelle": NIVEAUX[niveau],
        "raisons": raisons,
        "indicateurs": detail,
        "qualite_des_donnees": qualite,
        "contexte_fiche": contexte_fiche(fiche, maintenant),
        "parametres_experimentaux": {
            "seuil_score": SEUIL_SCORE,
            "jours_surveillance": JOURS_SURVEILLANCE,
            "jours_veterinaire": JOURS_VETERINAIRE,
            "jours_reference": JOURS_REFERENCE,
            "avertissement": "Seuils expérimentaux, non validés par un vétérinaire.",
        },
        "limites": LIMITES,
    }