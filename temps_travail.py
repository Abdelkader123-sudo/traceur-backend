from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from auth import creer_dependance_courant


class EvenementTravail(BaseModel):
    evenement_id: str = Field(min_length=1, max_length=128)
    type: Literal["journee_debut", "journee_fin", "tache_debut", "tache_fin"]
    journee_id: str = Field(min_length=1, max_length=128)
    tache_id: Optional[str] = Field(default=None, max_length=128)
    nom_tache: Optional[str] = Field(default=None, max_length=120)
    horodatage_utc: datetime
    date_locale: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    decalage_minutes: int = Field(ge=-840, le=840)


def _utc(instant: datetime) -> datetime:
    if instant.tzinfo is None:
        return instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def _iso(instant: datetime) -> str:
    return _utc(instant).isoformat()


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
    duree = sum(journee["duree_secondes"] for journee in journees)

    active = etat["journee_active"]
    if active is not None:
        debut_journee = datetime.fromisoformat(active["debut_utc"])
        fin_journee = (
            maintenant
            if active["date_locale"] == date_jour.isoformat()
            else debut_journee
        )
        duree += max(0, int((fin_journee - debut_journee).total_seconds()))
        tache = etat["tache_active"]
        if tache is not None:
            debut_tache = datetime.fromisoformat(tache["debut_utc"])
            fin_tache = max(debut_tache, fin_journee)
            taches.append(
                {
                    "nom": tache["nom"],
                    "debut_utc": tache["debut_utc"],
                    "fin_utc": _iso(fin_tache),
                    "duree_secondes": int((fin_tache - debut_tache).total_seconds()),
                }
            )

    return {
        "date_locale": date_jour.isoformat(),
        "duree_secondes": duree,
        "taches": [
            {
                "nom": tache["nom"],
                "debut_utc": tache["debut_utc"],
                "fin_utc": tache["fin_utc"],
                "duree_secondes": tache["duree_secondes"],
            }
            for tache in taches
        ],
    }


def creer_routeur_temps_travail(db):
    users = db["utilisateurs"]
    evenements_col = db["temps_travail_evenements"]
    evenements_col.create_index(
        [("client_id", 1), ("utilisateur_id", 1), ("evenement_id", 1)],
        unique=True,
        name="temps_travail_evenement_unique",
    )
    evenements_col.create_index(
        [("client_id", 1), ("date_locale", 1), ("utilisateur_id", 1)],
        name="temps_travail_par_jour",
    )

    router = APIRouter(tags=["temps-travail"])
    courant = creer_dependance_courant(
        users,
        db["sessions_authentification"],
    )

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

    return router
