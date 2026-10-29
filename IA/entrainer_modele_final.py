"""Crée le modèle Isolation Forest FINAL (sans anomalies injectées).

Entrée  : un CSV minute par minute avec les colonnes timestamp, cow_name, mi, steps
Sortie  : modele_isolation_forest_final.joblib (modèle + références personnelles de chaque vache)

Usage    : python entrainer_modele_final.py vache.csv [date_de_fin]
           ex. : python entrainer_modele_final.py vache.csv 2020-08-03   (ignore les mesures à partir de cette date)
Dans le backend :
    bundle = joblib.load("modele_isolation_forest_final.joblib")
    res = scorer(bundle, mesures_recentes)       # au moins 48 h de mesures par vache
    print(bilan_quotidien(res))
"""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

CSV = "vache.csv"
DATE_FIN = None           # lus depuis la ligne de commande dans le bloc principal
SORTIE = "modele_isolation_forest_final.joblib"
CONTAMINATION = 0.02      # 0.01 = moins d'alertes, 0.05 = plus de détections mais plus de fausses alertes
SEED = 42
FEATURES = ["pas_1h_log", "pas_6h_log", "ecart_6h", "ecart_24h", "ecart_mi_24h", "heure_sin", "heure_cos"]


def agreger_horaire(df):
    """Mesures minute par minute -> une ligne par vache et par heure (grille complète)."""
    df = df.drop_duplicates(["cow_name", "timestamp"])
    blocs = []
    for vache, g in df.groupby("cow_name"):
        g = g.set_index("timestamp").sort_index()
        r = g[["mi", "steps"]].resample("1h").sum().astype(float)
        r["n"] = g["mi"].resample("1h").count()
        r.loc[r["n"] < 45, ["mi", "steps"]] = np.nan        # heure incomplète = manquante
        r = r.drop(columns="n")
        r["cow_name"] = vache
        blocs.append(r.reset_index())
    return pd.concat(blocs, ignore_index=True)


def caracteristiques_base(d):
    """Moyennes glissantes sur 1 h, 6 h et 24 h + heure cyclique."""
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
    """Routine personnelle de chaque vache (médiane sur 24 h et profil horaire)."""
    med = d.groupby("cow_name").agg(med_pas24=("pas_24h", "median"), med_mi24=("mi_24h", "median")).reset_index()
    prof = d.groupby(["cow_name", "heure"])["pas_6h_log"].mean().rename("prof_6h").reset_index()
    return {"med": med, "prof": prof}


def appliquer_references(d, refs):
    d = d.merge(refs["med"], on="cow_name", how="left").merge(refs["prof"], on=["cow_name", "heure"], how="left")
    d["ecart_24h"] = np.log1p(d["pas_24h"]) - np.log1p(d["med_pas24"])
    d["ecart_mi_24h"] = np.log1p(d["mi_24h"]) - np.log1p(d["med_mi24"])
    d["ecart_6h"] = d["pas_6h_log"] - d["prof_6h"]
    return d.dropna(subset=FEATURES).reset_index(drop=True)


def entrainer(df_minute):
    d = caracteristiques_base(agreger_horaire(df_minute))
    refs = construire_references(d)
    X = appliquer_references(d, refs)
    modele = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=SEED)
    modele.fit(X[FEATURES])
    return {"modele": modele, "refs": refs, "features": FEATURES, "contamination": CONTAMINATION,
            "vaches": sorted(X["cow_name"].unique()), "n_heures": len(X),
            "periode": (str(X["timestamp"].min()), str(X["timestamp"].max()))}


def scorer(bundle, df_minute):
    """Score d'anomalie par vache et par heure (score plus grand = plus inhabituel)."""
    d = caracteristiques_base(agreger_horaire(df_minute))
    inconnues = set(d["cow_name"]) - set(bundle["vaches"])
    if inconnues:
        print(f"Attention : pas de références pour {sorted(inconnues)} (vaches ignorées, il faut les entraîner)")
    X = appliquer_references(d, bundle["refs"])
    X["score"] = -bundle["modele"].decision_function(X[bundle["features"]])
    X["alerte"] = bundle["modele"].predict(X[bundle["features"]]) == -1
    return X[["cow_name", "timestamp", "score", "alerte"]]


def bilan_quotidien(res):
    """Pour chaque vache : heures en alerte et score maximal sur les 24 dernières heures."""
    dernier = res[res["timestamp"] > res["timestamp"].max() - pd.Timedelta(hours=24)]
    return (dernier.groupby("cow_name").agg(heures_alerte=("alerte", "sum"), score_max=("score", "max"))
            .round(3).sort_values(["heures_alerte", "score_max"], ascending=False))


if __name__ == "__main__":
    CSV = sys.argv[1] if len(sys.argv) > 1 else CSV
    DATE_FIN = sys.argv[2] if len(sys.argv) > 2 else None   # pour écarter une fin d'enregistrement défectueuse
    df = pd.read_csv(CSV, parse_dates=["timestamp"], usecols=["timestamp", "cow_name", "mi", "steps"])
    if DATE_FIN:
        df = df[df["timestamp"] < DATE_FIN]
        print(f"Mesures ignorées à partir du {DATE_FIN}")
    bundle = entrainer(df)
    joblib.dump(bundle, SORTIE)
    print(f"Modèle entraîné sur {len(bundle['vaches'])} vaches, {bundle['n_heures']} heures "
          f"({bundle['periode'][0][:10]} -> {bundle['periode'][1][:10]})")
    print(f"Sauvegardé : {SORTIE}")

    # Vérification du chargement et du scoring (sur les 3 derniers jours des données d'entraînement :
    # ce n'est PAS une évaluation, seulement un contrôle que la chaîne fonctionne)
    bundle = joblib.load(SORTIE)
    recent = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=3)]
    print("\nBilan des 24 dernières heures (contrôle de fonctionnement) :")
    print(bilan_quotidien(scorer(bundle, recent)).to_string())
