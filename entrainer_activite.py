import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

df = pd.read_csv("dataset.csv")          # adapte le nom
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

# Colonnes utiles : accéléromètre et gyroscope bruts + magnitudes (sans _g ni _diff)
bases = ["AX", "AY", "AZ", "GX", "GY", "GZ", "AccMag", "GyrMag"]
stats = ["Mean", "StdDev", "IQR", "Max", "Min", "Kurtosis", "Skewness",
         "ZeroCrossing", "SpectralEntropy", "DominantFrequency"]
colonnes = [f"{s}_{b}" for b in bases for s in stats if f"{s}_{b}" in df.columns]

X = df[colonnes]
y = df["WalkingStandingLying"]
groupes = df["SheepID"]

modele = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
)

# Validation par mouton : aucun mouton n'est à la fois en entraînement et en test
vrais, predits, scores = [], [], []
for train, test in GroupKFold(n_splits=5).split(X, y, groupes):
    modele.fit(X.iloc[train], y.iloc[train])
    pred = modele.predict(X.iloc[test])
    vrais.extend(y.iloc[test]); predits.extend(pred)
    scores.append((pred == y.iloc[test]).mean())

print("Exactitude par pli :", np.round(scores, 3))
print("Moyenne :", round(np.mean(scores), 3))
print(classification_report(vrais, predits))
print(confusion_matrix(vrais, predits, labels=["Standing", "Lying", "Walking"]))

# Modèle final sur toutes les données
modele.fit(X, y)
joblib.dump({"modele": modele, "colonnes": colonnes}, "modele_activite.joblib")
print("Modèle sauvegardé : modele_activite.joblib")

# Variables les plus importantes
importances = pd.Series(modele.feature_importances_, index=colonnes).sort_values(ascending=False)
print(importances.head(15))
# Exactitude par mouton (validation par mouton)
# ---- Exactitude par mouton (validation par mouton) ----
resultats = {}
for train, test in GroupKFold(n_splits=5).split(X, y, groupes):
    m = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
    )
    m.fit(X.iloc[train], y.iloc[train])
    pred = m.predict(X.iloc[test])
    y_test = y.iloc[test].values
    g_test = groupes.iloc[test].values
    for mouton in np.unique(g_test):
        masque = g_test == mouton
        resultats[mouton] = (pred[masque] == y_test[masque]).mean()

print("\nExactitude par mouton :")
print(pd.Series(resultats).sort_values().round(3))