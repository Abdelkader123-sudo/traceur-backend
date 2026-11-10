from datetime import datetime, timedelta, timezone

from IA.seed_historique import (
    ANOMALIES,
    TZ,
    creer_mesures,
    debut_periode,
    filtrer_mesures_existantes,
    fin_periode,
)


def test_seed_generates_daily_data_and_both_anomalies():
    fin = datetime(2026, 1, 8, 12, 35, tzinfo=timezone.utc)

    for animal_id, kind in ANOMALIES.items():
        mesures = creer_mesures(7, animal_id, fin)
        debut_anomalie = fin - timedelta(hours=24)
        anomalie = [m for m in mesures if m["timestamp"] >= debut_anomalie]
        normal = [m for m in mesures if m["timestamp"] < debut_anomalie]
        dates = {m["timestamp"].astimezone(TZ).date() for m in mesures}

        assert {m["animal_id"] for m in mesures} == {animal_id}
        assert dates == {
            (fin.astimezone(TZ).date() - timedelta(days=i))
            for i in range(7)
        }
        assert all(m["timestamp"].second == 30 for m in mesures)
        assert all(
            (b["timestamp"] - a["timestamp"]).total_seconds() == 300
            for a, b in zip(mesures, mesures[1:])
        )
        if kind == "calme":
            assert sum(m["steps"] for m in anomalie) < sum(
                m["steps"] for m in normal
            ) / 10
        else:
            assert min(m["steps"] for m in anomalie) > max(
                m["steps"] for m in normal
            )


def test_seed_only_skips_exact_existing_timestamps():
    fin = datetime(2026, 1, 8, 12, 35, tzinfo=timezone.utc)
    mesures = creer_mesures(7, "cow08", fin)
    existing = {
        ("cow08", mesures[0]["timestamp"]),
        ("cow08", mesures[-1]["timestamp"]),
    }

    nouvelles = filtrer_mesures_existantes(mesures, existing)

    assert len(nouvelles) == len(mesures) - len(existing)
    assert not existing.intersection(
        {(m["animal_id"], m["timestamp"]) for m in nouvelles}
    )


def test_period_covers_today_and_only_elapsed_five_minute_slots():
    maintenant = datetime(2026, 1, 8, 12, 31, 12, tzinfo=timezone.utc)
    fin = fin_periode(maintenant)
    debut = debut_periode(7, fin)

    assert fin <= maintenant
    assert fin > maintenant - timedelta(minutes=5)
    assert debut.astimezone(TZ).date() == fin.astimezone(TZ).date() - timedelta(days=6)
