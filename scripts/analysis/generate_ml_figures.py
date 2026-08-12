import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# ==========================================
# Création dossier figures
# ==========================================os.makedirs("figures/ml", exist_ok=True)

# ==========================================
# Chargement dataset
# ==========================================

print("Chargement dataset...")

df = pd.read_csv("datasets/processed/final_dataset.csv")

# ==========================================
# Features réduites
# ==========================================
features = [
    "vehicles",
    "distance_gnb_m",
    "traffic_lambda",
    "packet_size",
    "throughput_mbps",
    "jitter_ms"
]

X = df[features]

y = df["decision"]# ==========================================
# Encodage labels
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# ==========================================
# Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42, stratify=y_encoded
)

# ==========================================
# RANDOM FOREST
# ==========================================

print("Entraînement Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# ==========================================
# XGBOOST
# ==========================================

print("Entraînement XGBoost...")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)

# ==========================================
# MATRICE DE CONFUSION RF
# ==========================================

cm_rf = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("figures/ml/confusion_matrix_rf.png", dpi=300)

plt.close()

# ==========================================
# MATRICE DE CONFUSION XGB
# ==========================================

cm_xgb = confusion_matrix(y_test, xgb_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_xgb,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=encoder.classes_,yticklabels=encoder.classes_
)

plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("figures/ml/confusion_matrix_xgb.png", dpi=300)

plt.close()# ==========================================
# FEATURE IMPORTANCE RF
# ==========================================

importance_rf = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
})

importance_rf = importance_rf.sort_values(
    by="importance",
    ascending=True
)
plt.figure(figsize=(8, 5))

plt.barh(
    importance_rf["feature"],
    importance_rf["importance"]
)

plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig("figures/ml/feature_importance_rf.png", dpi=300)

plt.close()
# ==========================================
# COMPARAISON DES MODELES
# ==========================================

models = ["Random Forest", "XGBoost"]
accuracies = [rf_acc, xgb_acc]

plt.figure(figsize=(6, 5))

plt.bar(models, accuracies)

plt.ylim(0.8, 1.0)

plt.ylabel("Accuracy")
plt.title("Comparaison des performances des modèles")

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center')

plt.tight_layout()

plt.savefig("figures/ml/model_comparison.png", dpi=300)

plt.close()

print("\nFigures générées avec succès dans : figures/ml/")
