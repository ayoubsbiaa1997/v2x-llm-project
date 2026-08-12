import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# Chargement dataset
# ==========================

df = pd.read_csv(
    "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v4_balanced.csv"
)

# ==========================
# Variables utilisées
# ==========================

features = [

    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",

    "mean_speed",
    "max_speed",
    "min_speed",
    "speed_std",

    "vehicle_density",
    "stopped_ratio",
    "lane_changes",
    "mobility_instability_index"

]

X = df[features]

y = df["decision"]

# ==========================
# Split train/test
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# ==========================
# Random Forest
# ==========================

rf = RandomForestClassifier(

    n_estimators=200,

    random_state=42

)

rf.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================

y_pred = rf.predict(X_test)

# ==========================
# Résultats
# ==========================

print()
print("Accuracy RF v4 :")
print(accuracy_score(y_test, y_pred))

print()
print("Classification Report :")
print(classification_report(y_test, y_pred))

print()
print("Confusion Matrix :")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Feature Importance
# ==========================

importance = pd.DataFrame({

    "feature": features,

    "importance": rf.feature_importances_

})

importance = importance.sort_values(

    by="importance",

    ascending=False

)

print()
print("Top Features :")
print(importance.head(15))
