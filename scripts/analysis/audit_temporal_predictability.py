#!/usr/bin/env python3
"""
Audit de la persistance temporelle et des transitions du Dataset V4.

Ce script vérifie si la tâche de prédiction future est réellement
apprenable ou si elle se réduit à recopier l'état précédent.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


INPUT_FILE = Path(
    "datasets/processed/mobility/"
    "temporal_v2x_synchronized_dataset_v4.csv"
)

OUTPUT_DIR = Path("results/ml/temporal_leakage_audit")

GROUP_COLUMN = "scenario_source"
TIME_COLUMN = "time_s"
TARGET_COLUMN = "decision"

CLASS_ORDER = [
    "KEEP_CONNECTION",
    "LOAD_BALANCING",
    "HO_NOW",
]

SEVERITY = {
    "KEEP_CONNECTION": 0,
    "LOAD_BALANCING": 1,
    "HO_NOW": 2,
}

HORIZONS = [1, 3, 5, 10]


def validate_dataset(df: pd.DataFrame) -> None:
    required = {
        GROUP_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes obligatoires absentes : {sorted(missing)}"
        )

    if df[list(required)].isna().sum().sum() > 0:
        raise ValueError(
            "Le dataset contient des valeurs manquantes "
            "dans les colonnes essentielles."
        )

    unknown_classes = (
        set(df[TARGET_COLUMN].unique())
        - set(CLASS_ORDER)
    )

    if unknown_classes:
        raise ValueError(
            f"Classes inconnues : {sorted(unknown_classes)}"
        )


def build_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scenario, group in df.groupby(GROUP_COLUMN):
        group = (
            group
            .sort_values(TIME_COLUMN)
            .reset_index(drop=True)
        )

        decisions = group[TARGET_COLUMN]

        transition_mask = (
            decisions.iloc[1:].to_numpy()
            != decisions.iloc[:-1].to_numpy()
        )

        transition_count = int(transition_mask.sum())

        counts = decisions.value_counts()

        dominant_class = counts.idxmax()
        dominant_ratio = float(
            counts.max() / len(group)
        )

        severity = decisions.map(SEVERITY)

        degradation_count = int(
            (
                severity.iloc[1:].to_numpy()
                > severity.iloc[:-1].to_numpy()
            ).sum()
        )

        recovery_count = int(
            (
                severity.iloc[1:].to_numpy()
                < severity.iloc[:-1].to_numpy()
            ).sum()
        )

        row = {
            "scenario_source": scenario,
            "n_samples": int(len(group)),
            "transition_count": transition_count,
            "degradation_count": degradation_count,
            "recovery_count": recovery_count,
            "dominant_class": dominant_class,
            "dominant_ratio": dominant_ratio,
        }

        for class_name in CLASS_ORDER:
            row[f"count_{class_name}"] = int(
                counts.get(class_name, 0)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_persistence(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for horizon in HORIZONS:
        y_true_all = []
        y_pred_all = []
        degradation_labels = []

        for _, group in df.groupby(GROUP_COLUMN):
            group = (
                group
                .sort_values(TIME_COLUMN)
                .reset_index(drop=True)
            )

            current_decision = group[TARGET_COLUMN]
            future_decision = current_decision.shift(-horizon)

            valid = future_decision.notna()

            current_valid = current_decision.loc[valid]
            future_valid = future_decision.loc[valid]

            y_pred_all.extend(
                current_valid.tolist()
            )

            y_true_all.extend(
                future_valid.tolist()
            )

            current_severity = current_valid.map(SEVERITY)
            future_severity = future_valid.map(SEVERITY)

            degradation_labels.extend(
                (
                    future_severity.to_numpy()
                    > current_severity.to_numpy()
                ).astype(int).tolist()
            )

        accuracy = accuracy_score(
            y_true_all,
            y_pred_all,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_true_all,
            y_pred_all,
        )

        f1_macro = f1_score(
            y_true_all,
            y_pred_all,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )

        rows.append({
            "horizon_steps": horizon,
            "n_evaluated_samples": len(y_true_all),
            "persistence_accuracy": float(accuracy),
            "persistence_balanced_accuracy": float(
                balanced_accuracy
            ),
            "persistence_f1_macro": float(f1_macro),
            "future_degradation_count": int(
                sum(degradation_labels)
            ),
            "future_degradation_ratio": float(
                sum(degradation_labels)
                / len(degradation_labels)
            ),
        })

    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    validate_dataset(df)

    df = df.sort_values(
        [GROUP_COLUMN, TIME_COLUMN]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenario_summary = build_scenario_summary(df)
    persistence = evaluate_persistence(df)

    scenario_summary.to_csv(
        OUTPUT_DIR / "scenario_transition_summary.csv",
        index=False,
    )

    persistence.to_csv(
        OUTPUT_DIR / "persistence_baselines.csv",
        index=False,
    )

    total_transitions = int(
        scenario_summary["transition_count"].sum()
    )

    total_degradations = int(
        scenario_summary["degradation_count"].sum()
    )

    total_recoveries = int(
        scenario_summary["recovery_count"].sum()
    )

    summary = {
        "dataset": str(INPUT_FILE),
        "n_samples": int(len(df)),
        "n_scenarios": int(
            df[GROUP_COLUMN].nunique()
        ),
        "total_transitions": total_transitions,
        "total_degradations": total_degradations,
        "total_recoveries": total_recoveries,
        "maximum_scenario_dominant_ratio": float(
            scenario_summary["dominant_ratio"].max()
        ),
        "minimum_scenario_dominant_ratio": float(
            scenario_summary["dominant_ratio"].min()
        ),
    }

    with open(
        OUTPUT_DIR / "audit_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("=" * 75)
    print("RÉSUMÉ PAR SCÉNARIO")
    print("=" * 75)

    print(
        scenario_summary.to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("BASELINES DE PERSISTANCE")
    print("=" * 75)

    print(
        persistence.to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("SYNTHÈSE")
    print("=" * 75)

    print("Échantillons :", len(df))
    print(
        "Scénarios :",
        df[GROUP_COLUMN].nunique(),
    )
    print(
        "Transitions totales :",
        total_transitions,
    )
    print(
        "Dégradations totales :",
        total_degradations,
    )
    print(
        "Récupérations totales :",
        total_recoveries,
    )

    print(
        "\nRésultats sauvegardés dans :",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
