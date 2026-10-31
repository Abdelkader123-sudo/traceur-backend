"""Crée le modèle Isolation Forest FINAL, indépendant de l'identifiant et de l'échelle.

Entrée  : un CSV avec les colonnes timestamp, cow_name (ou animal_id), mi, steps (+ duree_s pour des messages espacés)
Sortie  : modele_isolation_forest_final.joblib (modèle seul, aucune référence liée à un nom)

Usage    : python entrainer_modele_final.py vache.csv [date_de_fin]
Dans le backend :
    bundle = joblib.load("modele_isolation_forest_final.joblib")
    res = scorer(bundle, mesures_recentes)   # historique de l'animal : idéalement >= 7 jours, minimum 3 jours
    print(bilan_quotidien(res))
"""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

CSV = "vache.csv"
DATE_FIN = None
SORTIE = "modele_isolation_forest_final.joblib"
CONTAMINATION = 0.02
SEED = 42
MIN_HEURES_REF = 72       # heures valides minimum (3 jours) pour calculer la référence d'un animal
FEATURES = ["ecart_1h", "ecart_6h", "ecart_24h", "ecart_mi_24h", "heure_sin", "heure_cos"]


def agreger_horaire(df):
    """Mesures -> une ligne par animal et par heure (grille complète).

    Deux formats d'entrée acceptés :
      - minute par minute (mi, steps)            : somme sur l'heure, heure valable si >= 45 min
      - messages espacés (mi, steps, duree_s)    : pas ramenés à un taux (pas/s), heure valable si >= 45 min couvertes et >= 5 messages
    """
    if "animal_id" in df.columns and "cow_name" not in df.columns:
        df = df.rename(columns={"animal_id": "cow_name"})
    df = df.drop_duplicates(["cow_name", "timestamp"])
    par_message = "duree_s" in df.columns and df["duree_s"].notna().any()
    blocs = []
    for animal, g in df.groupby("cow_name"):
        if par_message:
            g = g.assign(taux=g["steps"] / g["duree_s"].clip(lower=1), heure=g["timestamp"].dt.floor("1h"))
            a = g.groupby("heure").agg(mi=("mi", "mean"), taux=("taux", "mean"), n=("mi", "size"),
                                       debut=("timestamp", "min"), fin=("timestamp", "max"))
            r = pd.DataFrame({"mi": a["mi"] * 60.0, "steps": a["taux"] * 3600.0})
            ok = ((a["fin"] - a["debut"]).dt.total_seconds() >= 2700) & (a["n"] >= 5)
            r.loc[~ok, ["mi", "steps"]] = np.nan
            r = r.reindex(pd.date_range(r.index.min(), r.index.max(), freq="1h"))
            r.index.name = "timestamp"
        else:
            g = g.set_index("timestamp").sort_index()
            r = g[["mi", "steps"]].resample("1h").sum().astype(float)
            r["n"] = g["mi"].resample("1h").count()
            r.loc[r["n"] < 45, ["mi", "steps"]] = np.nan
            r = r.drop(columns="n")
        r["cow_name"] = animal
        blocs.append(r.reset_index())
    return pd.concat(blocs, ignore_index=True)


def caracteristiques_base(d):
    """Sommes glissantes sur 6 h et 24 h + heure cyclique (aucune valeur absolue n'est utilisée par le modèle)."""
    d = d.sort_values(["cow_name", "timestamp"]).reset_index(drop=True)
    g = d.groupby("cow_name")
    d["pas_1h_log"] = np.log1p(d["steps"])
    d["pas_6h"] = g["steps"].transform(lambda s: s.rolling(6, min_periods=4).sum())
    d["pas_24h"] = g["steps"].transform(lambda s: s.rolling(24, min_periods=18).sum())
    d["mi_24h"] = g["mi"].transform(lambda s: s.rolling(24, min_periods=18).sum())
    d["pas_6h_log"] = np.log1p(d["pas_6h"])
    d["heure"] = d["timestamp"].dt.hour
    d["heure_sin"] = np.sin(2 * np.pi * d["heure"] / 24)
    d["heure_cos"] = np.cos(2 * np.pi * d["heure"] / 24)
    return d


def construire_references(d):
    """Routine de CHAQUE animal, calculée sur ses propres mesures (médianes : robustes aux anomalies)."""
    med = d.groupby("cow_name").agg(med_pas24=("pas_24h", "median"), med_mi24=("mi_24h", "median")).reset_index()
    prof = (d.groupby(["cow_name", "heure"])
              .agg(prof_1h=("pas_1h_log", "median"), prof_6h=("pas_6h_log", "median")).reset_index())
    return {"med": med, "prof": prof}


def appliquer_references(d, refs):
    d = d.merge(refs["med"], on="cow_name", how="left").merge(refs["prof"], on=["cow_name", "heure"], how="left")
    d["ecart_1h"] = d["pas_1h_log"] - d["prof_1h"]
    d["ecart_6h"] = d["pas_6h_log"] - d["prof_6h"]
    d["ecart_24h"] = np.log1p(d["pas_24h"]) - np.log1p(d["med_pas24"])
    d["ecart_mi_24h"] = np.log1p(d["mi_24h"]) - np.log1p(d["med_mi24"])
    return d.dropna(subset=FEATURES).reset_index(drop=True)


def preparer_depuis_horaire(horaire):
    """Prend un DataFrame déjà agrégé par heure et retourne les features relatives.

    La routine calculée pour le modèle est le profil horaire (prof_1h), complété par les
    profils sur 6 h et les médianes sur 24 h. Le code de sortie reste identique à
    l'ancienne version ; on sépare simplement l'agrégation pour éviter de la refaire.
    """
    d = caracteristiques_base(horaire)
    valides = d.groupby("cow_name")["pas_24h"].count()
    trop_courts = sorted(valides[valides < MIN_HEURES_REF].index)
    if trop_courts:
        print(f"Attention : pas assez de mesures pour {trop_courts} (< {MIN_HEURES_REF} h valides), ignorés")
        d = d[~d["cow_name"].isin(trop_courts)]
    return appliquer_references(d, construire_references(d))


def preparer(df):
    """Données brutes -> tableau de caractéristiques relatives (références propres à chaque animal)."""
    return preparer_depuis_horaire(agreger_horaire(df))


def entrainer(df):
    X = preparer(df)
    modele = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=SEED)
    modele.fit(X[FEATURES])
    return {"modele": modele, "features": FEATURES, "contamination": CONTAMINATION,
            "entrainement": {"n_animaux": int(X["cow_name"].nunique()), "n_heures": len(X),
                             "periode": (str(X["timestamp"].min()), str(X["timestamp"].max()))}}


def scorer(bundle, df):
    """Score d'anomalie par animal et par heure (plus grand = plus inhabituel). Fonctionne pour n'importe quel identifiant."""
    X = preparer(df)
    X["score"] = -bundle["modele"].decision_function(X[bundle["features"]])
    X["alerte"] = bundle["modele"].predict(X[bundle["features"]]) == -1
    return X[["cow_name", "timestamp", "score", "alerte"]]


def bilan_quotidien(res, reference=None):
    """Pour chaque animal : heures en alerte et score maximal sur les 24 dernières heures.

    Par défaut, on garde le comportement historique (fin = max(timestamp) du DataFrame). Si
    `reference` est fourni, la fenêtre est calculée comme ]reference - 24h, reference].
    """
    if res.empty:
        return pd.DataFrame(columns=["heures_alerte", "score_max"])
    ref = pd.Timestamp(reference) if reference is not None else res["timestamp"].max()
    ts_tz = getattr(res["timestamp"].dt, "tz", None)
    if ts_tz is not None and getattr(ref, "tzinfo", None) is None:
        ref = ref.tz_localize(ts_tz)
    elif ts_tz is None and getattr(ref, "tzinfo", None) is not None:
        ref = ref.tz_localize(None)
    dernier = res[(res["timestamp"] > ref - pd.Timedelta(hours=24)) & (res["timestamp"] <= ref)]
    return (dernier.groupby("cow_name").agg(heures_alerte=("alerte", "sum"), score_max=("score", "max"))
            .round(3).sort_values(["heures_alerte", "score_max"], ascending=False))


def test_independance(bundle, df):
    """Contrôle : mêmes données avec autres identifiants et pas/mouvement x3 -> les scores doivent rester proches."""
    fenetre = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=14)]
    ids = {n: f"animal_{i:02d}" for i, n in enumerate(sorted(fenetre["cow_name"].unique()))}
    autre = fenetre.assign(cow_name=fenetre["cow_name"].map(ids), steps=fenetre["steps"] * 3, mi=fenetre["mi"] * 3)
    a = scorer(bundle, fenetre).assign(cow_name=lambda x: x["cow_name"].map(ids))
    b = scorer(bundle, autre)
    m = a.merge(b, on=["cow_name", "timestamp"], suffixes=("_ref", "_x3"))
    print(f"\nTest identifiant + échelle x3 : corrélation des scores = {m['score_ref'].corr(m['score_x3']):.3f}, "
          f"alertes {int(m['alerte_ref'].sum())} -> {int(m['alerte_x3'].sum())}")


if __name__ == "__main__":
    CSV = sys.argv[1] if len(sys.argv) > 1 else CSV
    DATE_FIN = sys.argv[2] if len(sys.argv) > 2 else None
    df = pd.read_csv(CSV, parse_dates=["timestamp"],
                     usecols=lambda c: c in ("timestamp", "cow_name", "animal_id", "mi", "steps", "duree_s"))
    if DATE_FIN:
        df = df[df["timestamp"] < DATE_FIN]
        print(f"Mesures ignorées à partir du {DATE_FIN}")
    bundle = entrainer(df)
    joblib.dump(bundle, SORTIE)
    e = bundle["entrainement"]
    print(f"Modèle entraîné sur {e['n_animaux']} animaux, {e['n_heures']} heures "
          f"({e['periode'][0][:10]} -> {e['periode'][1][:10]})")
    print(f"Sauvegardé : {SORTIE}")

    bundle = joblib.load(SORTIE)
    recent = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=7)]
    print("\nBilan des 24 dernières heures (contrôle de fonctionnement, PAS une évaluation) :")
    print(bilan_quotidien(scorer(bundle, recent)).to_string())
    test_independance(bundle, df)