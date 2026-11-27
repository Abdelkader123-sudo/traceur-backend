from datetime import date, datetime, timedelta, timezone
import math
from typing import Literal, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from auth import creer_dependance_courant
from fcm_notifications import envoyer_notifications, fcm_prete


class EvenementTravail(BaseModel):
    evenement_id: str = Field(min_length=1, max_length=128)
    type: Literal["journee_debut", "journee_fin", "tache_debut", "tache_fin"]
    journee_id: str = Field(min_length=1, max_length=128)
    tache_id: Optional[str] = Field(default=None, max_length=128)
    nom_tache: Optional[str] = Field(default=None, max_length=120)
    horodatage_utc: datetime
    date_locale: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    decalage_minutes: int = Field(ge=-840, le=840)


class RapportPresence(BaseModel):
    journee_id: str = Field(min_length=1, max_length=128)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    precision_m: float = Field(gt=0, le=10000)
    est_simulee: bool = False


class RapportPositionDemandee(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    precision_m: float = Field(gt=0, le=10000)
    est_simulee: bool = False


class ZoneTravail(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    rayon_m: int = Field(ge=30, le=5000)


class TauxHoraireTravail(BaseModel):
    taux_horaire: float = Field(gt=0, allow_inf_nan=False)


def _utc(instant: datetime) -> datetime:
    if instant.tzinfo is None:
        return instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def _iso(instant: datetime) -> str:
    return _utc(instant).isoformat()


def _distance_metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rayon_terre = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    a = min(1.0, max(0.0, a))
    return 2 * rayon_terre * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _valider_evenement(evenement: dict):
    type_evenement = evenement["type"]
    tache_id = evenement.get("tache_id")
    nom_tache = evenement.get("nom_tache")
    if type_evenement in ("tache_debut", "tache_fin") and not tache_id:
        raise HTTPException(422, "Identifiant de tâche requis.")
    if type_evenement == "tache_debut" and not (nom_tache or "").strip():
        raise HTTPException(422, "Nom de tâche requis.")
    if type_evenement in ("journee_debut", "journee_fin") and tache_id:
        raise HTTPException(422, "Une journée ne doit pas référencer une tâche.")


def _fin_journee_utc(date_jour: date, decalage_minutes: int) -> datetime:
    fuseau = timezone(timedelta(minutes=decalage_minutes))
    minuit_local_suivant = datetime.combine(
        date_jour + timedelta(days=1),
        datetime.min.time(),
        tzinfo=fuseau,
    )
    return minuit_local_suivant.astimezone(timezone.utc)


def _duree_jusqua_fin_journee(
    debut: datetime,
    fin: datetime,
    date_jour: date,
    decalage_minutes: int,
) -> int:
    fin_effective = min(
        _utc(fin),
        _fin_journee_utc(date_jour, decalage_minutes),
    )
    return max(0, int((fin_effective - _utc(debut)).total_seconds()))


def _rejouer(evenements: list[dict]) -> dict:
    etat = {
        "journee_active": None,
        "tache_active": None,
        "journees_du_jour": [],
        "taches_du_jour": [],
        "evenements_locaux_appliques": [],
    }
    journees_terminees = set()
    taches_terminees = set()
    ordre = sorted(
        evenements,
        key=lambda evenement: (
            _utc(evenement["horodatage_utc"]),
            evenement.get("cree_le", datetime.min.replace(tzinfo=timezone.utc)),
            evenement["evenement_id"],
        ),
    )
    for evenement in ordre:
        instant = _utc(evenement["horodatage_utc"])
        type_evenement = evenement["type"]
        journee_id = evenement["journee_id"]
        journee = etat["journee_active"]
        tache = etat["tache_active"]

        if type_evenement == "journee_debut":
            if journee is not None:
                if journee["journee_id"] == journee_id:
                    etat["evenements_locaux_appliques"].append(
                        evenement["evenement_id"]
                    )
                    continue
                if tache is not None:
                    debut_tache = datetime.fromisoformat(tache["debut_utc"])
                    if instant < debut_tache:
                        raise HTTPException(409, "Le début de journée est invalide.")
                    etat["taches_du_jour"].append(
                        {
                            "tache_id": tache["tache_id"],
                            "nom": tache["nom"],
                            "debut_utc": tache["debut_utc"],
                            "fin_utc": _iso(instant),
                            "decalage_minutes": tache["decalage_minutes"],
                            "duree_secondes": int(
                                (instant - debut_tache).total_seconds()
                            ),
                        }
                    )
                    taches_terminees.add(tache["tache_id"])
                    etat["tache_active"] = None
                debut_journee = datetime.fromisoformat(journee["debut_utc"])
                if instant < debut_journee:
                    raise HTTPException(409, "Le début de journée est invalide.")
                etat["journees_du_jour"].append(
                    {
                        "journee_id": journee["journee_id"],
                        "debut_utc": journee["debut_utc"],
                        "fin_utc": _iso(instant),
                        "date_locale": journee["date_locale"],
                        "decalage_minutes": journee["decalage_minutes"],
                        "duree_secondes": int(
                            (instant - debut_journee).total_seconds()
                        ),
                    }
                )
                journees_terminees.add(journee["journee_id"])
                etat["journee_active"] = None
            etat["journee_active"] = {
                "journee_id": journee_id,
                "debut_utc": _iso(instant),
                "date_locale": evenement["date_locale"],
                "decalage_minutes": evenement["decalage_minutes"],
            }
        elif type_evenement == "tache_debut":
            if tache is not None:
                if tache["tache_id"] == evenement.get("tache_id"):
                    etat["evenements_locaux_appliques"].append(
                        evenement["evenement_id"]
                    )
                    continue
                debut_tache = datetime.fromisoformat(tache["debut_utc"])
                if instant < debut_tache:
                    raise HTTPException(409, "Le début de tâche est invalide.")
                etat["taches_du_jour"].append(
                    {
                        "tache_id": tache["tache_id"],
                        "nom": tache["nom"],
                        "debut_utc": tache["debut_utc"],
                        "fin_utc": _iso(instant),
                        "decalage_minutes": tache["decalage_minutes"],
                        "duree_secondes": int(
                            (instant - debut_tache).total_seconds()
                        ),
                    }
                )
                taches_terminees.add(tache["tache_id"])
                etat["tache_active"] = None
            if (
                journee is None
                or journee["journee_id"] != journee_id
                or instant < datetime.fromisoformat(journee["debut_utc"])
            ):
                raise HTTPException(409, "Le début de tâche ne correspond pas.")
            etat["tache_active"] = {
                "tache_id": evenement["tache_id"],
                "nom": evenement["nom_tache"].strip(),
                "debut_utc": _iso(instant),
                "journee_id": journee_id,
                "decalage_minutes": journee["decalage_minutes"],
            }
        elif type_evenement == "tache_fin":
            if (
                journee is None
                or journee["journee_id"] != journee_id
                or tache is None
                or tache["tache_id"] != evenement.get("tache_id")
                or instant < datetime.fromisoformat(tache["debut_utc"])
            ):
                if evenement.get("tache_id") in taches_terminees:
                    etat["evenements_locaux_appliques"].append(
                        evenement["evenement_id"]
                    )
                    continue
                raise HTTPException(409, "La fin de tâche ne correspond pas.")
            debut_tache = datetime.fromisoformat(tache["debut_utc"])
            etat["taches_du_jour"].append(
                {
                    "tache_id": tache["tache_id"],
                    "nom": tache["nom"],
                    "debut_utc": tache["debut_utc"],
                    "fin_utc": _iso(instant),
                    "decalage_minutes": tache["decalage_minutes"],
                    "duree_secondes": int((instant - debut_tache).total_seconds()),
                }
            )
            taches_terminees.add(tache["tache_id"])
            etat["tache_active"] = None
        elif type_evenement == "journee_fin":
            if (
                journee is None
                or journee["journee_id"] != journee_id
            ):
                if journee_id in journees_terminees:
                    etat["evenements_locaux_appliques"].append(
                        evenement["evenement_id"]
                    )
                    continue
                raise HTTPException(409, "La fin de journée ne correspond pas.")
            if tache is not None:
                debut_tache = datetime.fromisoformat(tache["debut_utc"])
                if instant < debut_tache:
                    raise HTTPException(409, "La fin de journée est invalide.")
                etat["taches_du_jour"].append(
                    {
                        "tache_id": tache["tache_id"],
                        "nom": tache["nom"],
                        "debut_utc": tache["debut_utc"],
                        "fin_utc": _iso(instant),
                        "decalage_minutes": tache["decalage_minutes"],
                        "duree_secondes": int(
                            (instant - debut_tache).total_seconds()
                        ),
                    }
                )
                taches_terminees.add(tache["tache_id"])
                etat["tache_active"] = None
            debut_journee = datetime.fromisoformat(journee["debut_utc"])
            if instant < debut_journee:
                raise HTTPException(409, "La fin de journée ne correspond pas.")
            etat["journees_du_jour"].append(
                {
                    "journee_id": journee_id,
                    "debut_utc": journee["debut_utc"],
                    "fin_utc": _iso(instant),
                    "date_locale": journee["date_locale"],
                    "decalage_minutes": journee["decalage_minutes"],
                    "duree_secondes": int(
                        (instant - debut_journee).total_seconds()
                    ),
                }
            )
            journees_terminees.add(journee_id)
            etat["journee_active"] = None
        etat["evenements_locaux_appliques"].append(evenement["evenement_id"])
    return etat


def _statistique_jour(
    evenements: list[dict],
    date_jour: date,
    maintenant: datetime,
) -> dict:
    etat = _rejouer(evenements)
    journees = list(etat["journees_du_jour"])
    taches = list(etat["taches_du_jour"])
    duree = sum(
        _duree_jusqua_fin_journee(
            datetime.fromisoformat(journee["debut_utc"]),
            datetime.fromisoformat(journee["fin_utc"]),
            date_jour,
            journee.get("decalage_minutes", 0),
        )
        for journee in journees
    )

    active = etat["journee_active"]
    if active is not None:
        debut_journee = datetime.fromisoformat(active["debut_utc"])
        decalage_minutes = active.get("decalage_minutes", 0)
        fin_journee = min(
            _utc(maintenant),
            _fin_journee_utc(date_jour, decalage_minutes),
        )
        if active["date_locale"] != date_jour.isoformat():
            fin_journee = debut_journee
        duree += _duree_jusqua_fin_journee(
            debut_journee,
            fin_journee,
            date_jour,
            decalage_minutes,
        )
        tache = etat["tache_active"]
        if tache is not None:
            debut_tache = datetime.fromisoformat(tache["debut_utc"])
            fin_tache = max(debut_tache, fin_journee)
            taches.append(
                {
                    "nom": tache["nom"],
                    "debut_utc": tache["debut_utc"],
                    "fin_utc": _iso(fin_tache),
                    "decalage_minutes": tache.get("decalage_minutes", 0),
                    "duree_secondes": int(
                        (fin_tache - debut_tache).total_seconds()
                    ),
                }
            )

    return {
        "date_locale": date_jour.isoformat(),
        "duree_secondes": duree,
        "taches": [
            {
                "nom": tache["nom"],
                "debut_utc": tache["debut_utc"],
                "fin_utc": _iso(
                    min(
                        _utc(datetime.fromisoformat(tache["fin_utc"])),
                        _fin_journee_utc(
                            date_jour,
                            tache.get("decalage_minutes", 0),
                        ),
                    )
                ),
                "duree_secondes": _duree_jusqua_fin_journee(
                    datetime.fromisoformat(tache["debut_utc"]),
                    datetime.fromisoformat(tache["fin_utc"]),
                    date_jour,
                    tache.get("decalage_minutes", 0),
                ),
            }
            for tache in taches
        ],
    }


def creer_routeur_temps_travail(db):
    users = db["utilisateurs"]
    evenements_col = db["temps_travail_evenements"]
    presence_col = db["temps_travail_presence"]
    demandes_presence_col = db["temps_travail_demandes_presence"]
    zones_col = db["temps_travail_zones"]
    salaires_col = db["temps_travail_salaires"]
    evenements_col.create_index(
        [("client_id", 1), ("utilisateur_id", 1), ("evenement_id", 1)],
        unique=True,
        name="temps_travail_evenement_unique",
    )
    evenements_col.create_index(
        [("client_id", 1), ("date_locale", 1), ("utilisateur_id", 1)],
        name="temps_travail_par_jour",
    )
    presence_col.create_index(
        [("client_id", 1), ("utilisateur_id", 1)],
        unique=True,
        name="temps_travail_presence_unique",
    )
    presence_col.create_index(
        "recu_le_utc",
        expireAfterSeconds=24 * 60 * 60,
        name="temps_travail_presence_expiration",
    )
    demandes_presence_col.create_index(
        [("client_id", 1), ("utilisateur_id", 1), ("statut", 1)],
        unique=True,
        partialFilterExpression={"statut": "en_attente"},
        name="temps_travail_demandes_presence_utilisateur",
    )
    demandes_presence_col.create_index(
        "expire_le_utc",
        expireAfterSeconds=0,
        name="temps_travail_demandes_presence_expiration",
    )
    zones_col.create_index(
        [("client_id", 1)],
        unique=True,
        name="temps_travail_zone_unique",
    )
    salaires_col.create_index(
        [("client_id", 1), ("utilisateur_id", 1)],
        unique=True,
        name="temps_travail_salaire_utilisateur_unique",
    )

    router = APIRouter(tags=["temps-travail"])
    courant = creer_dependance_courant(
        users,
        db["sessions_authentification"],
    )

    def _etat_courant(utilisateur: dict) -> dict:
        documents = list(
            evenements_col.find(
                {
                    "client_id": utilisateur["client_id"],
                    "utilisateur_id": utilisateur["_id"],
                }
            )
        )
        if not documents:
            return _rejouer([])
        dernier = max(
            documents,
            key=lambda evenement: _utc(evenement["horodatage_utc"]),
        )
        decalage = dernier["decalage_minutes"]
        aujourd_hui = (
            datetime.now(timezone.utc) + timedelta(minutes=decalage)
        ).date().isoformat()
        return _rejouer(
            [
                evenement
                for evenement in documents
                if evenement["date_locale"] == aujourd_hui
            ]
        )

    def _verifier_proprietaire(utilisateur: dict):
        if utilisateur.get("role") != "superadmin":
            raise HTTPException(403, "Réservé au client propriétaire.")

    @router.post("/temps-travail/evenements")
    def enregistrer_evenement(
        corps: EvenementTravail,
        utilisateur: dict = Depends(courant),
    ):
        evenement = corps.model_dump()
        _valider_evenement(evenement)
        evenement["horodatage_utc"] = _utc(corps.horodatage_utc)
        filtre = {
            "client_id": utilisateur["client_id"],
            "utilisateur_id": utilisateur["_id"],
            "evenement_id": corps.evenement_id,
        }
        existant = evenements_col.find_one(filtre)
        if existant:
            if any(existant.get(cle) != valeur for cle, valeur in evenement.items()):
                raise HTTPException(409, "Identifiant d’événement déjà utilisé.")
            if corps.type == "journee_fin":
                presence_col.delete_one(
                    {
                        "client_id": utilisateur["client_id"],
                        "utilisateur_id": utilisateur["_id"],
                    }
                )
            return {"evenement_id": corps.evenement_id}

        journees_existantes = list(
            evenements_col.find(
                {
                    "client_id": utilisateur["client_id"],
                    "utilisateur_id": utilisateur["_id"],
                    "date_locale": corps.date_locale,
                }
            )
        )
        _rejouer([*journees_existantes, evenement])
        document = {
            **evenement,
            "client_id": utilisateur["client_id"],
            "utilisateur_id": utilisateur["_id"],
            "numero_client": utilisateur["numero_client"],
            "cree_le": datetime.now(timezone.utc),
        }
        try:
            evenements_col.insert_one(document)
        except DuplicateKeyError:
            existant = evenements_col.find_one(filtre)
            if not existant or any(
                existant.get(cle) != valeur for cle, valeur in evenement.items()
            ):
                raise HTTPException(409, "Identifiant d’événement déjà utilisé.")
        if corps.type == "journee_fin":
            presence_col.delete_one(
                {
                    "client_id": utilisateur["client_id"],
                    "utilisateur_id": utilisateur["_id"],
                }
            )
        return {"evenement_id": corps.evenement_id}

    @router.get("/temps-travail/etat")
    def lire_etat(utilisateur: dict = Depends(courant)):
        filtre = {
            "client_id": utilisateur["client_id"],
            "utilisateur_id": utilisateur["_id"],
        }
        documents = list(evenements_col.find(filtre))
        decalage = 0
        if documents:
            dernier = max(
                documents,
                key=lambda evenement: _utc(evenement["horodatage_utc"]),
            )
            decalage = dernier["decalage_minutes"]
        aujourd_hui = (
            datetime.now(timezone.utc) + timedelta(minutes=decalage)
        ).date().isoformat()
        return _rejouer(
            [
                evenement
                for evenement in documents
                if evenement["date_locale"] == aujourd_hui
            ]
        )

    @router.get("/admin/temps-travail")
    def statistiques_admin(
        debut: date = Query(...),
        fin: date = Query(...),
        utilisateur: dict = Depends(courant),
    ):
        if utilisateur.get("role") != "superadmin":
            raise HTTPException(403, "Réservé au client propriétaire.")
        if fin < debut:
            raise HTTPException(422, "La date de fin précède la date de début.")
        if (fin - debut).days > 365:
            raise HTTPException(422, "La période ne peut pas dépasser 366 jours.")

        client_id = utilisateur["client_id"]
        membres = list(
            users.find(
                {
                    "client_id": client_id,
                    "role": {"$ne": "superadmin"},
                }
            )
        )
        documents = list(
            evenements_col.find(
                {
                    "client_id": client_id,
                    "date_locale": {
                        "$gte": debut.isoformat(),
                        "$lte": fin.isoformat(),
                    },
                }
            )
        )
        evenements_par_utilisateur_jour = {}
        for evenement in documents:
            cle = (evenement["utilisateur_id"], evenement["date_locale"])
            evenements_par_utilisateur_jour.setdefault(cle, []).append(evenement)

        maintenant = datetime.now(timezone.utc)
        utilisateurs = []
        for membre in sorted(
            membres,
            key=lambda personne: (
                personne.get("nom_cle", personne.get("nom", "")).lower(),
                personne.get("prenom_cle", personne.get("prenom", "")).lower(),
            ),
        ):
            journees = []
            date_jour = debut
            while date_jour <= fin:
                journees.append(
                    _statistique_jour(
                        evenements_par_utilisateur_jour.get(
                            (membre["_id"], date_jour.isoformat()),
                            [],
                        ),
                        date_jour,
                        maintenant,
                    )
                )
                date_jour += timedelta(days=1)
            nom = " ".join(
                partie
                for partie in (
                    membre.get("prenom", "").strip(),
                    membre.get("nom", "").strip(),
                )
                if partie
            )
            utilisateurs.append(
                {
                    "utilisateur_id": str(membre["_id"]),
                    "nom": nom,
                    "journees": journees,
                }
            )
        return {"utilisateurs": utilisateurs}

    @router.get("/admin/temps-travail/salaires")
    def salaires_admin(utilisateur: dict = Depends(courant)):
        _verifier_proprietaire(utilisateur)
        membres = {
            membre["_id"]
            for membre in users.find(
                {
                    "client_id": utilisateur["client_id"],
                    "role": {"$ne": "superadmin"},
                }
            )
        }
        return {
            "salaires": [
                {
                    "utilisateur_id": str(document["utilisateur_id"]),
                    "taux_horaire": document["taux_horaire"],
                }
                for document in salaires_col.find(
                    {"client_id": utilisateur["client_id"]}
                )
                if document.get("utilisateur_id") in membres
            ]
        }

    @router.put("/admin/temps-travail/salaires/{utilisateur_id}")
    def definir_salaire_admin(
        utilisateur_id: str,
        corps: TauxHoraireTravail,
        utilisateur: dict = Depends(courant),
    ):
        _verifier_proprietaire(utilisateur)
        if not ObjectId.is_valid(utilisateur_id):
            raise HTTPException(404, "Utilisateur introuvable.")
        membre_id = ObjectId(utilisateur_id)
        membre = users.find_one(
            {
                "_id": membre_id,
                "client_id": utilisateur["client_id"],
                "role": {"$ne": "superadmin"},
            }
        )
        if membre is None:
            raise HTTPException(404, "Utilisateur introuvable.")

        salaires_col.update_one(
            {
                "client_id": utilisateur["client_id"],
                "utilisateur_id": membre_id,
            },
            {
                "$set": {
                    "client_id": utilisateur["client_id"],
                    "utilisateur_id": membre_id,
                    "taux_horaire": corps.taux_horaire,
                    "mis_a_jour_utc": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )
        return {
            "utilisateur_id": utilisateur_id,
            "taux_horaire": corps.taux_horaire,
        }

    @router.get("/temps-travail/presence/configuration")
    def configuration_presence(utilisateur: dict = Depends(courant)):
        zone = zones_col.find_one({"client_id": utilisateur["client_id"]})
        return {
            "zone_configuree": zone is not None,
            "intervalle_secondes": INTERVALLE_PRESENCE_SECONDES,
        }

    @router.put("/admin/temps-travail/presence/zone")
    def definir_zone_travail(
        corps: ZoneTravail,
        utilisateur: dict = Depends(courant),
    ):
        _verifier_proprietaire(utilisateur)
        zone = {
            **corps.model_dump(),
            "client_id": utilisateur["client_id"],
            "mis_a_jour_utc": datetime.now(timezone.utc),
        }
        zones_col.update_one(
            {"client_id": utilisateur["client_id"]},
            {"$set": zone},
            upsert=True,
        )
        return {
            "latitude": corps.latitude,
            "longitude": corps.longitude,
            "rayon_m": corps.rayon_m,
        }

    @router.post("/temps-travail/presence")
    def enregistrer_presence(
        corps: RapportPresence,
        utilisateur: dict = Depends(courant),
    ):
        if corps.est_simulee:
            raise HTTPException(
                422,
                "Une position simulée ne peut pas être utilisée pour la présence.",
            )
        if corps.latitude == 0.0 and corps.longitude == 0.0:
            raise HTTPException(
                422,
                "La position obtenue est invalide.",
            )
        etat = _etat_courant(utilisateur)
        journee = etat["journee_active"]
        if journee is None or journee["journee_id"] != corps.journee_id:
            raise HTTPException(
                409,
                "Aucune journée de travail correspondante n’est active.",
            )

        zone = zones_col.find_one({"client_id": utilisateur["client_id"]})
        if zone is None:
            raise HTTPException(
                409,
                "Le propriétaire n’a pas encore défini la zone de travail.",
            )

        precision_fiable = (
            0 < corps.precision_m <= MAX_PRECISION_PRESENCE_METRES
        )
        dans_ferme = None
        if precision_fiable and not corps.est_simulee:
            distance = _distance_metres(
                corps.latitude,
                corps.longitude,
                zone["latitude"],
                zone["longitude"],
            )
            borne_interieure = max(0, distance - corps.precision_m)
            borne_exterieure = distance + corps.precision_m
            if borne_exterieure <= zone["rayon_m"]:
                dans_ferme = True
            elif borne_interieure > zone["rayon_m"]:
                dans_ferme = False

        recu = datetime.now(timezone.utc)
        presence_col.update_one(
            {
                "client_id": utilisateur["client_id"],
                "utilisateur_id": utilisateur["_id"],
            },
            {
                "$set": {
                    "client_id": utilisateur["client_id"],
                    "utilisateur_id": utilisateur["_id"],
                    "journee_id": corps.journee_id,
                    "recu_le_utc": recu,
                    "precision_fiable": precision_fiable,
                    "est_simulee": corps.est_simulee,
                    "dans_ferme": dans_ferme,
                    "latitude": corps.latitude,
                    "longitude": corps.longitude,
                    "precision_m": corps.precision_m,
                }
            },
            upsert=True,
        )
        return {
            "recu_le_utc": _iso(recu),
            "presence_verifiable": dans_ferme is not None,
            "dans_ferme": dans_ferme,
        }

    @router.post(
        "/admin/temps-travail/presence/demandes/{utilisateur_id}",
        status_code=201,
    )
    def demander_position(
        utilisateur_id: str,
        utilisateur: dict = Depends(courant),
    ):
        _verifier_proprietaire(utilisateur)
        if not ObjectId.is_valid(utilisateur_id):
            raise HTTPException(404, "Utilisateur introuvable.")
        membre = users.find_one(
            {
                "_id": ObjectId(utilisateur_id),
                "client_id": utilisateur["client_id"],
                "role": {"$ne": "superadmin"},
            }
        )
        if membre is None:
            raise HTTPException(404, "Utilisateur introuvable.")

        installations = list(
            dict.fromkeys(
                installation
                for installation in membre.get("fcm_installation_ids", [])
                if isinstance(installation, str) and installation
            )
        )
        if not installations:
            raise HTTPException(
                409,
                "Cet utilisateur n’a pas activé les notifications sur son appareil.",
            )
        if not fcm_prete():
            raise HTTPException(
                503,
                "Le service de notification est momentanément indisponible.",
            )

        maintenant = datetime.now(timezone.utc)
        demandes_existantes = demandes_presence_col.find(
            {
                "client_id": utilisateur["client_id"],
                "utilisateur_id": membre["_id"],
            }
        )
        for demande_existante in demandes_existantes:
            if demande_existante.get("statut") != "en_attente":
                continue
            if _utc(demande_existante["expire_le_utc"]) > maintenant:
                raise HTTPException(
                    409,
                    "Une demande de position est déjà en attente pour cet utilisateur.",
                )
            demandes_presence_col.update_one(
                {
                    "_id": demande_existante["_id"],
                    "statut": "en_attente",
                },
                {"$set": {"statut": "expiree"}},
            )

        demande = {
            "client_id": utilisateur["client_id"],
            "utilisateur_id": membre["_id"],
            "demandeur_id": utilisateur["_id"],
            "statut": "en_attente",
            "cree_le_utc": maintenant,
            "expire_le_utc": maintenant + DUREE_DEMANDE_POSITION,
        }
        try:
            resultat = demandes_presence_col.insert_one(demande)
        except DuplicateKeyError as erreur:
            raise HTTPException(
                409,
                "Une demande de position est déjà en attente pour cet utilisateur.",
            ) from erreur
        demande_id = str(resultat.inserted_id)
        identifiants_invalides = envoyer_notifications(
            installations,
            "Demande de position",
            "Votre administrateur vous demande de partager votre position.",
            {
                "type": "demande_localisation",
                "demande_id": demande_id,
            },
        )
        if len(identifiants_invalides) >= len(installations):
            demandes_presence_col.delete_one(
                {
                    "_id": resultat.inserted_id,
                    "client_id": utilisateur["client_id"],
                }
            )
            raise HTTPException(
                502,
                "La notification n’a pas pu être envoyée à cet utilisateur.",
            )
        return {
            "demande_id": demande_id,
            "expire_le_utc": _iso(demande["expire_le_utc"]),
        }

    @router.post("/temps-travail/presence/demandes/{demande_id}/position")
    def transmettre_position_demandee(
        demande_id: str,
        corps: RapportPositionDemandee,
        utilisateur: dict = Depends(courant),
    ):
        if not ObjectId.is_valid(demande_id):
            raise HTTPException(404, "Demande de position introuvable.")
        filtre = {
            "_id": ObjectId(demande_id),
            "client_id": utilisateur["client_id"],
            "utilisateur_id": utilisateur["_id"],
            "statut": "en_attente",
        }
        demande = demandes_presence_col.find_one(filtre)
        if demande is None:
            raise HTTPException(404, "Demande de position introuvable.")
        maintenant = datetime.now(timezone.utc)
        if _utc(demande["expire_le_utc"]) <= maintenant:
            raise HTTPException(410, "Cette demande de position a expiré.")
        if corps.est_simulee or corps.precision_m > MAX_PRECISION_PRESENCE_METRES:
            raise HTTPException(
                422,
                "Une position réelle et suffisamment précise est nécessaire.",
            )
        if corps.latitude == 0.0 and corps.longitude == 0.0:
            raise HTTPException(
                422,
                "La position obtenue est invalide.",
            )
        resultat = demandes_presence_col.update_one(
            {**filtre, "expire_le_utc": {"$gt": maintenant}},
            {
                "$set": {
                    "statut": "terminee",
                    "latitude": corps.latitude,
                    "longitude": corps.longitude,
                    "precision_m": corps.precision_m,
                    "recu_le_utc": maintenant,
                    "expire_le_utc": maintenant + DUREE_VISIBILITE_POSITION,
                }
            },
        )
        if resultat.matched_count == 0:
            raise HTTPException(409, "Cette demande n’est plus en attente.")
        return {
            "statut": "terminee",
            "recu_le_utc": _iso(maintenant),
        }

    @router.post("/temps-travail/presence/demandes/{demande_id}/refus")
    def refuser_demande_position(
        demande_id: str,
        utilisateur: dict = Depends(courant),
    ):
        if not ObjectId.is_valid(demande_id):
            raise HTTPException(404, "Demande de position introuvable.")
        filtre = {
            "_id": ObjectId(demande_id),
            "client_id": utilisateur["client_id"],
            "utilisateur_id": utilisateur["_id"],
            "statut": "en_attente",
        }
        demande = demandes_presence_col.find_one(filtre)
        if demande is None:
            raise HTTPException(404, "Demande de position introuvable.")
        maintenant = datetime.now(timezone.utc)
        if _utc(demande["expire_le_utc"]) <= maintenant:
            raise HTTPException(410, "Cette demande de position a expiré.")
        resultat = demandes_presence_col.update_one(
            {**filtre, "expire_le_utc": {"$gt": maintenant}},
            {
                "$set": {
                    "statut": "refusee",
                    "refusee_le_utc": maintenant,
                }
            },
        )
        if resultat.matched_count == 0:
            raise HTTPException(409, "Cette demande n’est plus en attente.")
        return {"statut": "refusee"}

    @router.get("/admin/temps-travail/presence")
    def presences_admin(utilisateur: dict = Depends(courant)):
        _verifier_proprietaire(utilisateur)
        client_id = utilisateur["client_id"]
        zone = zones_col.find_one({"client_id": client_id})
        membres = list(
            users.find(
                {
                    "client_id": client_id,
                    "role": {"$ne": "superadmin"},
                }
            )
        )
        maintenant = datetime.now(timezone.utc)
        demandes_par_utilisateur = {}
        for demande in demandes_presence_col.find({"client_id": client_id}):
            if _utc(demande["expire_le_utc"]) <= maintenant:
                continue
            existante = demandes_par_utilisateur.get(demande["utilisateur_id"])
            if (
                existante is None
                or _utc(demande["cree_le_utc"])
                > _utc(existante["cree_le_utc"])
            ):
                demandes_par_utilisateur[demande["utilisateur_id"]] = demande

        utilisateurs = []
        for membre in membres:
            etat = _etat_courant(membre)
            journee = etat["journee_active"]
            presence = presence_col.find_one(
                {
                    "client_id": client_id,
                    "utilisateur_id": membre["_id"],
                }
            )
            derniere_verification = (
                presence.get("recu_le_utc") if presence else None
            )
            presence_recente = (
                derniere_verification is not None
                and timedelta(0)
                <= maintenant - _utc(derniere_verification)
                <= AGE_MAX_PRESENCE
                and journee is not None
                and presence.get("journee_id") == journee["journee_id"]
                and (
                    zone is None
                    or _utc(derniere_verification)
                    >= _utc(zone["mis_a_jour_utc"])
                )
            )
            demande_position = demandes_par_utilisateur.get(membre["_id"])
            latitude = (
                presence.get("latitude")
                if presence_recente
                and presence
                and not presence.get("est_simulee")
                and presence.get("precision_fiable")
                else None
            )
            longitude = (
                presence.get("longitude")
                if presence_recente
                and presence
                and not presence.get("est_simulee")
                and presence.get("precision_fiable")
                else None
            )
            if (
                demande_position is not None
                and demande_position.get("statut") == "terminee"
                and (
                    derniere_verification is None
                    or _utc(demande_position["recu_le_utc"])
                    >= _utc(derniere_verification)
                )
            ):
                latitude = demande_position["latitude"]
                longitude = demande_position["longitude"]
                derniere_verification = demande_position["recu_le_utc"]
            precision_fiable = bool(
                presence
                and presence.get("precision_fiable")
                and not presence.get("est_simulee")
            )
            verifiee = bool(
                zone
                and presence_recente
                and precision_fiable
                and presence.get("dans_ferme") is not None
            )
            if zone is None:
                statut = "zone_non_configuree"
            elif journee is None:
                statut = "hors_service"
            elif not presence_recente:
                statut = "non_verifiee"
            elif presence and presence.get("est_simulee"):
                statut = "non_verifiee"
            elif not precision_fiable:
                statut = "precision_insuffisante"
            elif presence.get("dans_ferme") is True:
                statut = "dans_ferme"
            elif presence.get("dans_ferme") is False:
                statut = "hors_ferme"
            else:
                statut = "precision_insuffisante"

            nom = " ".join(
                partie
                for partie in (
                    membre.get("prenom", "").strip(),
                    membre.get("nom", "").strip(),
                )
                if partie
            )
            utilisateurs.append(
                {
                    "utilisateur_id": str(membre["_id"]),
                    "nom": nom,
                    "journee_active": journee is not None,
                    "tache_active": (
                        etat["tache_active"]["nom"]
                        if etat["tache_active"]
                        else None
                    ),
                    "statut": statut,
                    "presence_verifiee": verifiee,
                    "demande_localisation_statut": (
                        demande_position.get("statut")
                        if demande_position is not None
                        else None
                    ),
                    "derniere_verification_utc": (
                        _iso(derniere_verification)
                        if derniere_verification is not None
                        else None
                    ),
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
        utilisateurs.sort(key=lambda personne: personne["nom"].lower())
        return {
            "zone": (
                {
                    "latitude": zone["latitude"],
                    "longitude": zone["longitude"],
                    "rayon_m": zone["rayon_m"],
                }
                if zone
                else None
            ),
            "utilisateurs": utilisateurs,
        }

    return router


INTERVALLE_PRESENCE_SECONDES = 10 * 60
AGE_MAX_PRESENCE = timedelta(minutes=20)
MAX_PRECISION_PRESENCE_METRES = 100
DUREE_DEMANDE_POSITION = timedelta(minutes=15)
DUREE_VISIBILITE_POSITION = timedelta(minutes=15)
