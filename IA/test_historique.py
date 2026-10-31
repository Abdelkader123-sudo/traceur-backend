import numpy as np
import pandas as pd

from IA import ia_routes
from IA.entrainer_modele_final import entrainer


def _build_dataset(days=10):
    records = []
    start = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    rng = np.random.default_rng(42)
    for animal in ["cow01", "cow04"]:
        for day in range(days):
            for h in range(24):
                for minute in range(0, 60, 5):
                    ts = start + pd.Timedelta(days=day, hours=h, minutes=minute)
                    hour = ts.tz_convert("Africa/Tunis").tz_localize(None).hour
                    base = 220 + 180 * np.sin((hour - 7) / 24 * 2 * np.pi)
                    base = max(30, base)
                    if animal == "cow04" and ts >= pd.Timestamp("2026-01-10 12:00:00", tz="UTC"):
                        base *= 0.33
                    steps = int(max(0, base + rng.normal(0, 25)))
                    records.append({
                        "animal_id": animal,
                        "timestamp": ts,
                        "mi": 1.0,
                        "steps": steps,
                        "duree_s": 300,
                    })
    return pd.DataFrame(records)


class FakeCollection:
    def __init__(self, df):
        self.df = df.sort_values("timestamp").reset_index(drop=True)

    def find(self, filtre, projection=None):
        df = self.df.copy()
        if "animal_id" in filtre:
            df = df[df["animal_id"] == filtre["animal_id"]]
        ts = df["timestamp"]
        if "timestamp" in filtre:
            gte = filtre["timestamp"].get("$gte")
            lt = filtre["timestamp"].get("$lt")
            mask = pd.Series(True, index=df.index)
            if gte is not None:
                mask &= ts >= pd.Timestamp(gte)
            if lt is not None:
                mask &= ts < pd.Timestamp(lt)
            df = df[mask]
        return [{
            "animal_id": row.animal_id,
            "timestamp": row.timestamp,
            "mi": row.mi,
            "steps": row.steps,
            "duree_s": row.duree_s,
        } for _, row in df.iterrows()]


def test_bilan_and_historique_are_consistent():
    df = _build_dataset(days=10)
    fake = FakeCollection(df)
    bundle = entrainer(df)
    ia_routes._bundle = bundle

    ref = pd.Timestamp("2026-01-10 23:00:00", tz="UTC").tz_convert("Africa/Tunis").tz_localize(None)
    animaux, ignores = ia_routes.calculer_bilan(fake, heures=168, reference=ref)

    assert any(a["animal_id"] == "cow04" and a["statut"] == "a_surveiller" for a in animaux)
    assert any(a["animal_id"] == "cow01" and a["statut"] == "normal" for a in animaux)

    hist = ia_routes.calculer_historique(fake, "cow04", heures=168, reference=ref)
    assert hist["statut"] == "a_surveiller"
    assert hist["heures_alerte_24h"] == next(a["heures_alerte"] for a in animaux if a["animal_id"] == "cow04")

    hist_336 = ia_routes.calculer_historique(fake, "cow04", heures=336, reference=ref)
    assert hist_336["statut"] == hist["statut"]
    assert hist_336["heures_alerte_24h"] == hist["heures_alerte_24h"]
    assert len(hist_336["points"]) != len(hist["points"])

    # simuler un collier silencieux sur les 3 derniers jours de cow01
    cutoff = pd.Timestamp("2026-01-08 00:00:00", tz="UTC")
    df2 = df.copy()
    df2 = df2[(df2["animal_id"] != "cow01") | (df2["timestamp"] < cutoff)].copy()
    fake2 = FakeCollection(df2)
    _, ignores2 = ia_routes.calculer_bilan(fake2, heures=168, reference=ref)
    assert any(x["animal_id"] == "cow01" for x in ignores2)

    debut = pd.Timestamp("2026-01-08 00:00:00")
    fin = pd.Timestamp("2026-01-10 00:00:00")
    hist_plage = ia_routes.calculer_historique(fake, "cow04", debut=debut, fin=fin, reference=ref)
    span_hours = (fin - debut).total_seconds() / 3600
    assert hist_plage["heures_valides"] <= span_hours

    res = ia_routes.scorer(bundle, df)
    old = ia_routes.bilan_quotidien(res)
    new = ia_routes.bilan_quotidien(res, reference=res["timestamp"].max())
    pd.testing.assert_frame_equal(old, new)
