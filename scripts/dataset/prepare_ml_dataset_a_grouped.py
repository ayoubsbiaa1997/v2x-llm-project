from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "datasets/processed/mobility/"
    "temporal_v2x_synchronized_dataset_v4.csv"
)

OUTPUT_DIR = Path("datasets/ml/dataset_a_grouped")
ARTIFACT_DIR = Path("models/preprocessing/dataset_a_grouped")

TARGET_COLUMN = "decision"
GROUP_COLUMN = "scenario_source"

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

TEST_SCENARIOS = [
    "qos_D100_STABLE",
    "qos_D250_MEDIUM",
    "qos_D400_HIGH",
]


# ============================================================
# Chargement et validation
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset introuvable : {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = (
    FEATURE_COLUMNS
    + [TARGET_COLUMN, GROUP_COLUMN]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Colonnes absentes : {missing_columns}"
    )

if df[required_columns].isna().sum().sum() > 0:
    raise ValueError(
        "Le dataset contient des valeurs manquantes."
    )

available_scenarios = set(df[GROUP_COLUMN].unique())
missing_scenarios = set(TEST_SCENARIOS) - available_scenarios

if missing_scenarios:
    raise ValueError(
        "Scénarios de test absents du dataset : "
        f"{sorted(missing_scenarios)}"
    )


# ============================================================
# Encodage des labels
# ============================================================

label_encoder = LabelEncoder()
label_encoder.fit(df[TARGET_COLUMN])

label_mapping = {
    class_name: int(index)
    for index, class_name
    in enumerate(label_encoder.classes_)
}

df["target_encoded"] = label_encoder.transform(
    df[TARGET_COLUMN]
)


# ============================================================
# Séparation par groupes
# ============================================================

test_mask = df[GROUP_COLUMN].isin(TEST_SCENARIOS)

train_raw = df.loc[~test_mask].copy()
test_raw = df.loc[test_mask].copy()

train_df = train_raw[
    FEATURE_COLUMNS
    + [GROUP_COLUMN, "target_encoded", TARGET_COLUMN]
].copy()

test_df = test_raw[
    FEATURE_COLUMNS
    + [GROUP_COLUMN, "target_encoded", TARGET_COLUMN]
].copy()

train_df = train_df.rename(
    columns={TARGET_COLUMN: "target_label"}
)

test_df = test_df.rename(
    columns={TARGET_COLUMN: "target_label"}
)


# ============================================================
# Vérifications anti-fuite
# ============================================================

train_groups = set(train_df[GROUP_COLUMN].unique())
test_groups = set(test_df[GROUP_COLUMN].unique())

overlap = train_groups.intersection(test_groups)

if overlap:
    raise RuntimeError(
        f"Fuite de groupes détectée : {sorted(overlap)}"
    )

train_classes = set(train_df["target_label"].unique())
test_classes = set(test_df["target_label"].unique())

if train_classes != test_classes:
    print(
        "Attention : les classes train/test ne sont "
        "pas strictement identiques."
    )
    print("Train :", sorted(train_classes))
    print("Test  :", sorted(test_classes))


# ============================================================
# Sauvegarde
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

train_file = OUTPUT_DIR / "dataset_a_grouped_train.csv"
test_file = OUTPUT_DIR / "dataset_a_grouped_test.csv"

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

joblib.dump(
    label_encoder,
    ARTIFACT_DIR / "label_encoder.joblib"
)

metadata = {
    "source_dataset": str(INPUT_FILE),
    "feature_columns": FEATURE_COLUMNS,
    "target_column": TARGET_COLUMN,
    "group_column": GROUP_COLUMN,
    "test_scenarios": TEST_SCENARIOS,
    "train_scenarios": sorted(train_groups),
    "n_samples_total": int(len(df)),
    "n_samples_train": int(len(train_df)),
    "n_samples_test": int(len(test_df)),
    "n_features": len(FEATURE_COLUMNS),
    "label_mapping": label_mapping,
    "group_overlap": sorted(overlap),
}

with open(
    ARTIFACT_DIR / "metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# Résumé
# ============================================================

print("Préparation groupée terminée.")

print("\nScénarios train :")
for scenario in sorted(train_groups):
    print(" -", scenario)

print("\nScénarios test :")
for scenario in sorted(test_groups):
    print(" -", scenario)

print("\nDimensions :")
print("Train :", train_df.shape)
print("Test  :", test_df.shape)

print("\nDistribution train :")
print(train_df["target_label"].value_counts())

print("\nDistribution test :")
print(test_df["target_label"].value_counts())

print("\nChevauchement des scénarios :", overlap)

print("\nFichiers générés :")
print(train_file)
print(test_file)
print(ARTIFACT_DIR / "metadata.json")
