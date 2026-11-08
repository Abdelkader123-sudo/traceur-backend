# À COLLER dans main.py, juste à côté de la route /lstm/evenements
# (même objet `app`, même base `db` que cette route ; adaptez le nom `db` si besoin).
from datetime import datetime, timedelta, timezone

import trajectoire_service as traj


@app.get("/lstm/trajectoire")
def lstm_trajectoire(animal_id: str, instant_local: str, avant_min: int = 90, apres_min: int = 15):
    """Positions de l'animal autour d'une alerte. Les heures renvoyées sont en heure locale, sans fuseau."""
    local = datetime.fromisoformat(instant_local).replace(tzinfo=None)   # instant_local est déjà local
    centre = local.replace(tzinfo=traj.TZ_LOCAL).astimezone(timezone.utc)
    docs = db[traj.COLL_POSITIONS].find(
        {traj.CHAMP_ANIMAL: animal_id,
         traj.CHAMP_TEMPS: {"$gte": centre - timedelta(minutes=avant_min),
                            "$lte": centre + timedelta(minutes=apres_min)}},
        {traj.CHAMP_TEMPS: 1, traj.CHAMP_LAT: 1, traj.CHAMP_LON: 1, "_id": 0},
    ).sort(traj.CHAMP_TEMPS, 1)

    points = []
    for d in docs:
        if d.get(traj.CHAMP_LAT) is None or d.get(traj.CHAMP_LON) is None:
            continue
        t = d[traj.CHAMP_TEMPS]
        if t.tzinfo is None:                       # pymongo renvoie des dates naïves = UTC
            t = t.replace(tzinfo=timezone.utc)
        points.append({
            "t": t.astimezone(traj.TZ_LOCAL).replace(tzinfo=None).isoformat(timespec="seconds"),
            "lat": d[traj.CHAMP_LAT],
            "lon": d[traj.CHAMP_LON],
        })
    return points
