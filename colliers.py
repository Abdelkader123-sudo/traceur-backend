from copy import deepcopy
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError


class AffectationCollier(BaseModel):
    animal_id: str = Field(min_length=1, max_length=128)


def _client_id(u):
    if u.get("role") != "superadmin":
        raise HTTPException(403, "Réservé au propriétaire du compte client.")
    return u["client_id"]


def _filtre_collier(client_id, collier_id):
    return {"client_id": client_id, "collier_id": collier_id}


def _date_utc(valeur):
    if valeur.tzinfo is None:
        return valeur.replace(tzinfo=timezone.utc)
    return valeur.astimezone(timezone.utc)


def animal_pour_collier(collection, client_id, collier_id, instant):
    collier = collection.find_one(
        _filtre_collier(client_id, collier_id),
        {"affectations": 1},
    )
    if not collier:
        return None

    instant = _date_utc(instant)
    for affectation in reversed(collier.get("affectations", [])):
        debut = affectation.get("debut")
        fin = affectation.get("fin")
        if debut is None:
            continue
        debut = _date_utc(debut)
        fin = _date_utc(fin) if fin is not None else None
        if debut <= instant and (fin is None or instant < fin):
            return affectation.get("animal_id")
    return None


def enregistrer_signal(
    collection,
    client,
    collier_id,
    position,
):
    maintenant = datetime.now(timezone.utc)
    collection.update_one(
        _filtre_collier(client["_id"], collier_id),
        {
            "$set": {
                "derniere_reception": maintenant,
                "derniere_position": {
                    "latitude": position.lat,
                    "longitude": position.lon,
                    "vitesse": position.vitesse,
                    "activite": position.activite,
                    "timestamp": position.timestamp,
                },
            },
            "$setOnInsert": {
                "client_id": client["_id"],
                "numero_client": client["numero_client"],
                "collier_id": collier_id,
                "cree_le": maintenant,
                "affectations": [],
                "revision": 0,
            },
        },
        upsert=True,
    )


def _affectation_active(collier):
    for affectation in reversed(collier.get("affectations", [])):
        if affectation.get("fin") is None:
            return affectation
    return None


def _public(collier):
    affectations = collier.get("affectations", [])
    active = _affectation_active(collier)
    return {
        "collier_id": collier["collier_id"],
        "animal_id": active.get("animal_id") if active else None,
        "derniere_reception": collier.get("derniere_reception"),
        "derniere_position": collier.get("derniere_position"),
        "affectations": affectations,
    }


def lister_colliers(collection, u):
    client_id = _client_id(u)
    collier_documents = collection.find({"client_id": client_id})
    return sorted(
        (_public(collier) for collier in collier_documents),
        key=lambda collier: collier["collier_id"],
    )


def _conflit_animal(collection, client_id, collier_id, animal_id):
    autre = collection.find_one({
        "client_id": client_id,
        "animal_id": animal_id,
        "collier_id": {"$ne": collier_id},
    })
    if autre:
        raise HTTPException(
            409,
            "Cet animal possède déjà un collier actif.",
        )


def _actualiser_affectations(collier, animal_id, u, maintenant):
    actuelle = _affectation_active(collier)
    if actuelle and actuelle.get("animal_id") == animal_id:
        return None

    affectations = deepcopy(collier.get("affectations", []))
    for affectation in affectations:
        if affectation.get("fin") is None:
            affectation["fin"] = maintenant
    affectations.append({
        "animal_id": animal_id,
        "debut": maintenant,
        "fin": None,
        "affecte_par": str(u["_id"]),
    })
    return affectations


def _enregistrer_revision(collection, filtre, collier, valeurs):
    revision = collier.get("revision", 0)
    filtre_revision = {
        **filtre,
        "revision": revision,
    } if "revision" in collier else {
        **filtre,
        "revision": {"$exists": False},
    }
    resultat = collection.update_one(
        filtre_revision,
        {"$set": {**valeurs, "revision": revision + 1}},
    )
    if not resultat.matched_count:
        raise HTTPException(
            409,
            "L’affectation du collier a changé simultanément. Réessayez.",
        )
    return collection.find_one(filtre)


def affecter_collier(collection, animaux, collier_id, demande, u):
    client_id = _client_id(u)
    collier_id = collier_id.strip()
    animal_id = demande.animal_id.strip()
    if not collier_id or len(collier_id) > 128 or not animal_id:
        raise HTTPException(422, "Identifiant de collier ou d’animal invalide.")

    filtre = _filtre_collier(client_id, collier_id)
    collier = collection.find_one(filtre)
    if not collier:
        raise HTTPException(
            404,
            "Collier introuvable. Attendez son premier signal puis réessayez.",
        )

    _conflit_animal(collection, client_id, collier_id, animal_id)
    maintenant = datetime.now(timezone.utc)
    affectations = _actualiser_affectations(
        collier,
        animal_id,
        u,
        maintenant,
    )
    if affectations is None:
        return _public(collier)

    animaux.update_one(
        {"client_id": client_id, "animal_id": animal_id},
        {
            "$setOnInsert": {
                "client_id": client_id,
                "numero_client": u["numero_client"],
                "animal_id": animal_id,
                "ferme": "Ferme principale",
                "cree_le": maintenant,
            },
        },
        upsert=True,
    )

    try:
        resultat = _enregistrer_revision(
            collection,
            filtre,
            collier,
            {"affectations": affectations, "animal_id": animal_id},
        )
    except DuplicateKeyError:
        raise HTTPException(
            409,
            "Cet animal possède déjà un collier actif.",
        )
    return _public(resultat)


def dissocier_collier(collection, collier_id, u):
    client_id = _client_id(u)
    collier_id = collier_id.strip()
    if not collier_id or len(collier_id) > 128:
        raise HTTPException(422, "Identifiant de collier invalide.")

    filtre = _filtre_collier(client_id, collier_id)
    collier = collection.find_one(filtre)
    if not collier:
        raise HTTPException(404, "Collier introuvable.")

    actuelle = _affectation_active(collier)
    if not actuelle:
        return _public(collier)

    maintenant = datetime.now(timezone.utc)
    affectations = deepcopy(collier.get("affectations", []))
    for affectation in affectations:
        if affectation.get("fin") is None:
            affectation["fin"] = maintenant
    resultat = _enregistrer_revision(
        collection,
        filtre,
        collier,
        {"affectations": affectations, "animal_id": None},
    )
    return _public(resultat)
