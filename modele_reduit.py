import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

df = pd.read_csv("dataset.csv")          # adapte le nom
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

bases = ["AX", "AY", "AZ", "GX", "GY", "GZ", "AccMag", "GyrMag"]
stats = ["Mean", "StdDev", "IQR", "Max", "Min"]
colonnes = [f"{s}_{b}" for b in bases for s in stats if f"{s}_{b}" in df.columns]

X, y, groupes = df[colonnes], df["WalkingStandingLying"], df["SheepID"]

scores = []
for train, test in GroupKFold(n_splits=5).split(X, y, groupes):
    m = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               n_jobs=-1, random_state=42)
    m.fit(X.iloc[train], y.iloc[train])
    scores.append((m.predict(X.iloc[test]) == y.iloc[test]).mean())

print("Variables :", len(colonnes))
print("Exactitude par pli :", np.round(scores, 3))
print("Moyenne :", round(np.mean(scores), 3), "(modèle complet : 0.865)")

final = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               n_jobs=-1, random_state=42).fit(X, y)
joblib.dump({"modele": final, "colonnes": colonnes}, "modele_activite_reduit.joblib")