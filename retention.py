from datetime import datetime, timedelta, timezone

from messagerie import nettoyer_conversations


DUREE_GPS_J = 9
DUREE_MPU_J = 9
DUREE_ALERTES_H = 26
DUREE_CONVERSATIONS_J = 3
DUREE_TEMPS_TRAVAIL_J = 30


def nettoyer_donnees(db, client_id=None, maintenant=None, notifier_alerte=None):
    """Applique les durées de conservation du backend et retourne les suppressions."""
    maintenant = maintenant or datetime.now(timezone.utc)
    limite_gps = maintenant - timedelta(days=DUREE_GPS_J)
    limite_mpu = maintenant - timedelta(days=DUREE_MPU_J)
    limite_alertes = maintenant - timedelta(hours=DUREE_ALERTES_H)
    limite_conversations = maintenant - timedelta(days=DUREE_CONVERSATIONS_J)
    limite_temps_travail = maintenant - timedelta(days=DUREE_TEMPS_TRAVAIL_J)

    filtre_tenant = {"client_id": client_id} if client_id is not None else {}
    filtre_alertes = {**filtre_tenant, "timestamp": {"$lt": limite_alertes}}
    bilan = {}

    if notifier_alerte is not None:
        for alerte in db["alerts"].find(
            {**filtre_alertes, "resolue": False},
            {"animal_id": 1, "type": 1, "client_id": 1},
        ):
            notifier_alerte(alerte, maintenant)
    bilan["alertes"] = db["alerts"].delete_many(filtre_alertes).deleted_count
    bilan["positions"] = db["positions"].delete_many(
        {**filtre_tenant, "timestamp": {"$lt": limite_gps}}
    ).deleted_count
    bilan["mesures_activite"] = db["mesures_activite"].delete_many(
        {**filtre_tenant, "timestamp": {"$lt": limite_mpu}}
    ).deleted_count
    bilan["evenements_lstm"] = db["lstm_alertes_silencieuses"].delete_many(
        {**filtre_tenant, "cree_le": {"$lt": limite_alertes}}
    ).deleted_count
    # Keep this field for response compatibility; pasture zone records are retained.
    bilan["anciennes_zones"] = 0
    bilan["evenements_travail"] = db["temps_travail_evenements"].delete_many(
        {**filtre_tenant, "horodatage_utc": {"$lt": limite_temps_travail}}
    ).deleted_count

    conversations = nettoyer_conversations(
        db, limite_conversations, client_id=client_id
    )
    bilan["conversations"] = conversations["conversations"]
    bilan["messages"] = conversations["messages"]
    return bilan
