import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =========================
# Dossier de sortie
# =========================

os.makedirs("figures/ml", exist_ok=True)

# =========================
# Chargement dataset
# =========================

df = pd.read_csv("datasets/processed/final_dataset.csv")

features = [
    "vehicles",
    "distance_gnb_m",
    "traffic_lambda",
    "packet_size",
    "throughput_mbps",
    "jitter_ms"
]

X = df[features]
y = df["decision"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Classes :", encoder.classes_)

# =========================
# Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# Modèle XGBoost
# =========================

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# SHAP Explainer
# =========================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# =========================
# Summary plot global
# =========================

plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=features,
    show=False
)

plt.tight_layout()
plt.savefig("figures/ml/shap_summary_global.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Bar plot global
# =========================

plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=features,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig("figures/ml/shap_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figures SHAP générées avec succès dans figures/ml/")
