import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# =========================
# Chargement dataset
# =========================

df = pd.read_csv("datasets/processed/final_dataset.csv")

# =========================
# Features
# =========================

features = [
    "vehicles",
    "distance_gnb_m",
    "traffic_lambda",
    "packet_size",
    "throughput_mbps",
    "jitter_ms"
]

X = df[features]

# =========================
# Labels
# =========================

y = df["decision"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# =========================
# Split train/test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =====================================================
# RANDOM FOREST
# =====================================================

print("\n================ RANDOM FOREST ================\n")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Accuracy RF:", accuracy_score(y_test, rf_pred))

print("\nClassification Report RF:\n")
print(classification_report(
    y_test,
    rf_pred,
    target_names=encoder.classes_
))

print("\nConfusion Matrix RF:\n")
print(confusion_matrix(y_test, rf_pred))

# =========================
# Feature importance RF
# =========================

importance_rf = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
})

importance_rf = importance_rf.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance RF:\n")
print(importance_rf)

# =====================================================
# XGBOOST
# =====================================================

print("\n================ XGBOOST ================\n")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("Accuracy XGB:", accuracy_score(y_test, xgb_pred))

print("\nClassification Report XGB:\n")
print(classification_report(
    y_test,
    xgb_pred,
    target_names=encoder.classes_
))

print("\nConfusion Matrix XGB:\n")
print(confusion_matrix(y_test, xgb_pred))
