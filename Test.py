import pandas as pd

df = pd.read_csv("dataset.csv")   # adapte le nom

print(df.shape)                                  # lignes, colonnes
print(df["WalkingStandingLying"].value_counts()) # répartition des classes
print(df["SheepID"].nunique())                   # nombre de moutons
print(df.groupby("SheepID").size().describe())   # lignes par mouton
print(df["SensorPos"].value_counts())            # positions des capteurs