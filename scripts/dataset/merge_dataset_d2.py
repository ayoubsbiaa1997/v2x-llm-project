#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


# Racine du projet :
# ~/v2x-llm-project/scripts/merge_dataset_d2.py
# parents[1] correspond à ~/v2x-llm-project
PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_DIR
    / "datasets"
    / "raw"
    / "dynamic_dataset_d2"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "datasets"
    / "processed"
    / "dynamic_d2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(
    INPUT_DIR.glob("*_qos_temporal.csv")
)

print(f"Dossier d'entrée : {INPUT_DIR}")
print(f"{len(csv_files)} fichiers détectés")

if not csv_files:
    raise FileNotFoundError(
        f"Aucun fichier *_qos_temporal.csv trouvé dans {INPUT_DIR}"
    )

dfs = []

profiles = [
    "early_degradation",
    "late_degradation",
    "abrupt_degradation",
    "medium_control",
    "stable_control",
    "persistent_high",
]

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    run_id = csv_file.name.replace(
        "_qos_temporal.csv",
        ""
    )

    profile = next(
        (
            profile_name.upper()
            for profile_name in profiles
            if f"d2_{profile_name}_" in run_id
        ),
        "UNKNOWN",
    )

    df["run_id"] = run_id
    df["profile"] = profile
    df["source_file"] = csv_file.name

    dfs.append(df)

merged = pd.concat(
    dfs,
    ignore_index=True
)

output_file = (
    OUTPUT_DIR
    / "dynamic_qos_merged_d2.csv"
)

merged.to_csv(
    output_file,
    index=False
)

print()
print("Fusion terminée")
print(f"Fichier : {output_file}")
print(f"Dimensions : {merged.shape}")

print()
print("Nombre de lignes par profil :")
print(
    merged["profile"]
    .value_counts()
    .sort_index()
)

print()
print("Nombre de runs uniques :")
print(merged["run_id"].nunique())
