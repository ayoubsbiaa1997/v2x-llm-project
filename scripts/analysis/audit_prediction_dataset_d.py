#!/usr/bin/env python3
"""
Audit scientifique du Dataset D.

Contrôles :
- aucune variable future dans X ;
- aucun identifiant ou paramètre de simulation dans X ;
- aucune intersection temporelle ;
- deux classes présentes ;
- chaque run contient des séquences cohérentes ;
- vérification de la baseline majoritaire.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


INPUT_FILE = Path(
    "datasets/processed/dynamic/"
    "prediction_dataset_d.csv"
)


TARGET_COLUMN = "future_degradation"
GROUP_COLUMN = "run_id"


FORBIDDEN_TOKENS = [
    "future",
    "phase",
    "lambda",
    "run",
    "seed",
    "time",
    "target",
    "degradation",
    "audit",
    "ue_count",
    "distance",
]


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset D introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise RuntimeError(
            "Dataset D vide."
        )

    features = [
        column
        for column in df.columns
        if "_lag_" in column
    ]

    forbidden_features = [
        column
        for column in features
        if any(
            token in column.lower()
            for token in FORBIDDEN_TOKENS
        )
    ]

    temporal_overlap = int(
        (
            df["future_start_time_s"]
            <= df["history_end_time_s"]
        ).sum()
    )

    missing_features = int(
        df[features]
        .isna()
        .sum()
        .sum()
    )

    missing_target = int(
        df[TARGET_COLUMN]
        .isna()
        .sum()
    )

    duplicated_sequences = int(
        df.duplicated(
            subset=[
                GROUP_COLUMN,
                "history_start_time_s",
                "future_start_time_s",
            ]
        ).sum()
    )

    class_counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    class_ratios = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
    )

    majority_class = int(
        class_counts.idxmax()
    )

    majority_prediction = [
        majority_class
    ] * len(df)

    majority_accuracy = accuracy_score(
        df[TARGET_COLUMN],
        majority_prediction,
    )

    majority_balanced_accuracy = (
        balanced_accuracy_score(
            df[TARGET_COLUMN],
            majority_prediction,
        )
    )

    majority_f1_macro = f1_score(
        df[TARGET_COLUMN],
        majority_prediction,
        average="macro",
        zero_division=0,
    )

    print("=" * 78)
    print("AUDIT ANTI-LEAKAGE DU DATASET D")
    print("=" * 78)

    print("Shape :", df.shape)
    print(
        "Runs :",
        df[GROUP_COLUMN].nunique(),
    )
    print(
        "Variables ML :",
        len(features),
    )

    print("\nDistribution de la cible :")
    print(class_counts.to_string())

    print("\nProportions :")
    print(
        class_ratios.round(4).to_string()
    )

    print("\nContrôles :")
    print(
        "Variables interdites :",
        forbidden_features,
    )
    print(
        "Chevauchements temporels :",
        temporal_overlap,
    )
    print(
        "Valeurs manquantes dans X :",
        missing_features,
    )
    print(
        "Valeurs manquantes dans y :",
        missing_target,
    )
    print(
        "Séquences dupliquées :",
        duplicated_sequences,
    )

    print("\nBaseline classe majoritaire :")
    print(
        "Accuracy :",
        round(majority_accuracy, 4),
    )
    print(
        "Balanced accuracy :",
        round(
            majority_balanced_accuracy,
            4,
        ),
    )
    print(
        "F1 macro :",
        round(majority_f1_macro, 4),
    )

    errors = []

    if len(features) != 30:
        errors.append(
            f"30 features attendues, "
            f"{len(features)} obtenues."
        )

    if forbidden_features:
        errors.append(
            "Variables interdites présentes dans X."
        )

    if temporal_overlap:
        errors.append(
            "Chevauchement temporel détecté."
        )

    if missing_features:
        errors.append(
            "Valeurs manquantes dans X."
        )

    if missing_target:
        errors.append(
            "Valeurs manquantes dans y."
        )

    if duplicated_sequences:
        errors.append(
            "Séquences dupliquées."
        )

    if df[TARGET_COLUMN].nunique() != 2:
        errors.append(
            "La cible ne contient pas deux classes."
        )

    positive_ratio = float(
        df[TARGET_COLUMN].mean()
    )

    if not 0.05 <= positive_ratio <= 0.95:
        errors.append(
            "Déséquilibre extrême de la cible : "
            f"{positive_ratio:.4f}"
        )

    if df[GROUP_COLUMN].nunique() != 36:
        errors.append(
            "Le dataset ne contient pas les 36 runs."
        )

    if errors:
        print("\nAUDIT ÉCHOUÉ :")

        for error in errors:
            print("-", error)

        raise RuntimeError(
            "Le Dataset D ne peut pas encore "
            "être utilisé pour le ML."
        )

    print(
        "\nAUDIT ANTI-LEAKAGE RÉUSSI."
    )


if __name__ == "__main__":
    main()
