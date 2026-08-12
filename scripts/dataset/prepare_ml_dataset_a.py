from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "datasets/processed/mobility/"
    "temporal_v2x_synchronized_dataset_v4.csv"
)

OUTPUT_DIR = Path("datasets/ml/dataset_a")
ARTIFACT_DIR = Path("models/preprocessing/dataset_a")

TARGET_COLUMN = "decision"

FEATURE_COLUMNS = [
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
    "mobility_instability_index",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# Fonctions de validation
# ============================================================

def validate_dataset(df: pd.DataFrame) -> None:
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes absentes du dataset : {missing_columns}"
        )

    missing_values = df[required_columns].isna().sum()

    if missing_values.sum() > 0:
        invalid = missing_values[missing_values > 0]
        raise ValueError(
            "Des valeurs manquantes ont été détectées :\n"
            f"{invalid}"
        )

    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "La variable cible doit contenir au moins deux classes."
        )


# ============================================================
# Chargement
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

validate_dataset(df)

X = df[FEATURE_COLUMNS].copy()
y_text = df[TARGET_COLUMN].copy()


# ============================================================
# Encodage des labels
# ============================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

label_mapping = {
    class_name: int(index)
    for index, class_name in enumerate(label_encoder.classes_)
}

print("Encodage des classes :")
for class_name, class_id in label_mapping.items():
    print(f"  {class_name} -> {class_id}")


# ============================================================
# Séparation stratifiée
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# Reconstruction des fichiers tabulaires
# ============================================================

train_df = X_train.copy()
train_df["target_encoded"] = y_train
train_df["target_label"] = label_encoder.inverse_transform(y_train)

test_df = X_test.copy()
test_df["target_encoded"] = y_test
test_df["target_label"] = label_encoder.inverse_transform(y_test)


# ============================================================
# Sauvegarde
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

train_file = OUTPUT_DIR / "dataset_a_train.csv"
test_file = OUTPUT_DIR / "dataset_a_test.csv"

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

joblib.dump(
    label_encoder,
    ARTIFACT_DIR / "label_encoder.joblib"
)

metadata = {
    "source_dataset": str(INPUT_FILE),
    "target_column": TARGET_COLUMN,
    "feature_columns": FEATURE_COLUMNS,
    "n_features": len(FEATURE_COLUMNS),
    "n_samples_total": int(len(df)),
    "n_samples_train": int(len(train_df)),
    "n_samples_test": int(len(test_df)),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "label_mapping": label_mapping,
    "excluded_columns": [
        "time_s",
        "scenario_source",
    ],
}

with open(
    ARTIFACT_DIR / "metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)


# ============================================================
# Résumé
# ============================================================

print("\nPréparation terminée.")
print(f"Dataset initial : {df.shape}")
print(f"X : {X.shape}")
print(f"Train : {train_df.shape}")
print(f"Test : {test_df.shape}")

print("\nDistribution du train :")
print(train_df["target_label"].value_counts())

print("\nDistribution du test :")
print(test_df["target_label"].value_counts())

print("\nFichiers générés :")
print(f"  {train_file}")
print(f"  {test_file}")
print(
    f"  {ARTIFACT_DIR / 'label_encoder.joblib'}"
)
print(
    f"  {ARTIFACT_DIR / 'metadata.json'}"
)
