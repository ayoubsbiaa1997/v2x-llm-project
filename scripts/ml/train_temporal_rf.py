import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

os.makedirs("figures/temporal", exist_ok=True)

df = pd.read_csv("datasets/processed/mobility/temporal_sequences_decision_multi.csv")

drop_cols = [
    "sequence_id",
    "scenario_source",
    "target_window",
    "target_vehicle_density",
    "target_mean_speed",
    "target_stopped_ratio",
    "target_mobility_instability_index",
    "decision"
]

X = df.drop(columns=drop_cols)
y = df["decision"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
pred = rf.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\nAccuracy RF temporel :", acc)

print("\nClassification Report :")
print(classification_report(y_test, pred, target_names=encoder.classes_))

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion - Random Forest temporel")
plt.tight_layout()
plt.savefig("figures/temporal/confusion_temporal_rf.png", dpi=300)
plt.close()

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

importance.to_csv("figures/temporal/temporal_rf_feature_importance.csv", index=False)

print("\nTop 15 features :")
print(importance.head(15))
