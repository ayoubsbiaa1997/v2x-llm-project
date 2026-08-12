import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(
    "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v4_balanced.csv"
)

drop_cols = [
    "decision",
    "qos_source"
]

X = df.drop(columns=drop_cols)
X = X.select_dtypes(include=["int64", "float64"])

y = df["decision"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("\nAccuracy RF dataset synchronisé équilibré :")
print(accuracy_score(y_test, pred))

print("\nClassification Report :")
print(classification_report(y_test, pred))

print("\nConfusion Matrix :")
print(confusion_matrix(y_test, pred))

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 15 Features :")
print(importance.head(15))

