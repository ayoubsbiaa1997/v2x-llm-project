#!/usr/bin/env python3
"""
Construction du Dataset D sans fuite temporelle.

Les variables d'entrée utilisent exclusivement les fenêtres historiques.
La cible utilise exclusivement les fenêtres futures.

Tâche :
    historique QoS -> apparition d'une dégradation future
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "datasets/processed/dynamic/"
    "dynamic_qos_merged.csv"
)

OUTPUT_FILE = Path(
    "datasets/processed/dynamic/"
    "prediction_dataset_d.csv"
)

AUDIT_FILE = Path(
    "results/ml/dynamic_dataset_d/"
    "prediction_dataset_d_audit.csv"
)


GROUP_COLUMN = "run_id"
TIME_COLUMN = "time_s"

SEQUENCE_LENGTH = 5
PREDICTION_HORIZON = 5


FEATURE_COLUMNS = [
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
]


METADATA_COLUMNS = [
    "phase",
    "lambda_be",
    "ue_count",
    "distance_m",
    "seed",
    "run_number",
    "packet_size_be",
    "phase1_end",
    "phase2_end",
    "phase3_end",
    "sim_time",
]


def validate_source(df: pd.DataFrame) -> None:
    required = {
        GROUP_COLUMN,
        TIME_COLUMN,
        *FEATURE_COLUMNS,
        *METADATA_COLUMNS,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes absentes : {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Le dataset fusionné est vide."
        )

    if df[list(required)].isna().any().any():
        null_counts = (
            df[list(required)]
            .isna()
            .sum()
        )

        raise ValueError(
            "Valeurs manquantes :\n"
            f"{null_counts[null_counts > 0]}"
        )

    duplicated = df.duplicated(
        subset=[GROUP_COLUMN, TIME_COLUMN]
    )

    if duplicated.any():
        raise ValueError(
            "Des couples run_id/time_s sont dupliqués."
        )

    for run_id, group in df.groupby(GROUP_COLUMN):
        if not group[
            TIME_COLUMN
        ].is_monotonic_increasing:
            raise ValueError(
                f"{run_id}: timestamps non croissants."
            )


def safe_relative_increase(
    future_value: float,
    history_value: float,
    epsilon: float = 1e-9,
) -> float:
    denominator = max(abs(history_value), epsilon)

    return float(
        (future_value - history_value)
        / denominator
    )


def evaluate_future_degradation(
    history: pd.DataFrame,
    future: pd.DataFrame,
) -> dict:
    """
    Détecte une aggravation future relativement à l'état historique.

    La cible vaut 1 uniquement lorsqu'une hausse significative et
    suffisamment soutenue du risque QoS apparaît dans la fenêtre future.

    Les seuils absolus de type delay >= 10 ms ont été supprimés :
    ils classaient comme dégradation un réseau déjà dégradé mais stable.
    """

    recent_history = history.tail(3)

    history_delay = float(
        recent_history["delay_ms"].median()
    )
    history_jitter = float(
        recent_history["jitter_ms"].median()
    )
    history_loss = float(
        recent_history["packet_loss"].median()
    )
    history_pdr = float(
        recent_history["pdr"].median()
    )
    history_congestion = float(
        recent_history["congestion_index"].median()
    )

    future_delay = float(
        future["delay_ms"].median()
    )
    future_jitter = float(
        future["jitter_ms"].median()
    )
    future_loss = float(
        future["packet_loss"].median()
    )
    future_pdr = float(
        future["pdr"].median()
    )
    future_congestion = float(
        future["congestion_index"].median()
    )

    delay_increase = (
        future_delay - history_delay
    )
    jitter_increase = (
        future_jitter - history_jitter
    )
    loss_increase = (
        future_loss - history_loss
    )
    pdr_drop = (
        history_pdr - future_pdr
    )
    congestion_increase = (
        future_congestion
        - history_congestion
    )

    delay_relative_increase = (
        safe_relative_increase(
            future_delay,
            history_delay,
        )
    )

    jitter_relative_increase = (
        safe_relative_increase(
            future_jitter,
            history_jitter,
        )
    )

    degradation_signals = {
        "delay_signal": bool(
            delay_increase >= 0.50
            and delay_relative_increase >= 0.25
        ),
        "jitter_signal": bool(
            jitter_increase >= 0.20
            and jitter_relative_increase >= 0.25
        ),
        "loss_signal": bool(
            loss_increase >= 0.20
        ),
        "pdr_signal": bool(
            pdr_drop >= 0.20
        ),
        "congestion_signal": bool(
            congestion_increase >= 0.01
        ),
    }

    number_of_signals = int(
        sum(degradation_signals.values())
    )

    def qos_risk(frame: pd.DataFrame) -> np.ndarray:
        delay_component = np.clip(
            frame["delay_ms"].to_numpy() / 20.0,
            0.0,
            1.0,
        )

        jitter_component = np.clip(
            frame["jitter_ms"].to_numpy() / 10.0,
            0.0,
            1.0,
        )

        loss_component = np.clip(
            frame["packet_loss"].to_numpy() / 10.0,
            0.0,
            1.0,
        )

        pdr_component = np.clip(
            (100.0 - frame["pdr"].to_numpy()) / 10.0,
            0.0,
            1.0,
        )

        congestion_component = np.clip(
            frame["congestion_index"].to_numpy(),
            0.0,
            1.0,
        )

        return (
            0.30 * delay_component
            + 0.15 * jitter_component
            + 0.25 * loss_component
            + 0.10 * pdr_component
            + 0.20 * congestion_component
        )

    history_risk_values = qos_risk(
        recent_history
    )

    future_risk_values = qos_risk(
        future
    )

    history_risk = float(
        np.median(history_risk_values)
    )

    future_risk = float(
        np.median(future_risk_values)
    )

    risk_increase = (
        future_risk - history_risk
    )

    future_windows_above_baseline = int(
        (
            future_risk_values
            >= history_risk + 0.02
        ).sum()
    )

    sustained_degradation = bool(
        future_windows_above_baseline >= 3
    )

    future_degradation = int(
        sustained_degradation
        and risk_increase >= 0.02
        and number_of_signals >= 2
    )

    return {
        "future_degradation": future_degradation,
        "number_of_degradation_signals": (
            number_of_signals
        ),
        "severe_degradation": 0,
        "sustained_degradation_audit": int(
            sustained_degradation
        ),
        "future_windows_above_baseline_audit": (
            future_windows_above_baseline
        ),
        "history_risk_audit": history_risk,
        "future_risk_audit": future_risk,
        "risk_increase_audit": risk_increase,
        "history_delay_audit": history_delay,
        "future_delay_audit": future_delay,
        "delay_increase_audit": delay_increase,
        "history_jitter_audit": history_jitter,
        "future_jitter_audit": future_jitter,
        "jitter_increase_audit": jitter_increase,
        "history_loss_audit": history_loss,
        "future_loss_audit": future_loss,
        "loss_increase_audit": loss_increase,
        "history_pdr_audit": history_pdr,
        "future_pdr_audit": future_pdr,
        "pdr_drop_audit": pdr_drop,
        "history_congestion_audit": (
            history_congestion
        ),
        "future_congestion_audit": (
            future_congestion
        ),
        "congestion_increase_audit": (
            congestion_increase
        ),
        **{
            f"{name}_audit": int(value)
            for name, value
            in degradation_signals.items()
        },
    }


def build_prediction_dataset(
    source: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for run_id, group in source.groupby(
        GROUP_COLUMN
    ):
        group = (
            group
            .sort_values(TIME_COLUMN)
            .reset_index(drop=True)
        )

        minimum_length = (
            SEQUENCE_LENGTH
            + PREDICTION_HORIZON
        )

        if len(group) < minimum_length:
            print(
                f"Run ignoré, trop court : "
                f"{run_id} ({len(group)} fenêtres)"
            )
            continue

        last_start = (
            len(group)
            - SEQUENCE_LENGTH
            - PREDICTION_HORIZON
            + 1
        )

        for start in range(last_start):
            history_end = (
                start + SEQUENCE_LENGTH
            )

            future_end = (
                history_end
                + PREDICTION_HORIZON
            )

            history = group.iloc[
                start:history_end
            ].copy()

            future = group.iloc[
                history_end:future_end
            ].copy()

            target_information = (
                evaluate_future_degradation(
                    history=history,
                    future=future,
                )
            )

            row = {
                "run_id": run_id,
                "history_start_time_s": float(
                    history.iloc[0][TIME_COLUMN]
                ),
                "history_end_time_s": float(
                    history.iloc[-1][TIME_COLUMN]
                ),
                "future_start_time_s": float(
                    future.iloc[0][TIME_COLUMN]
                ),
                "future_end_time_s": float(
                    future.iloc[-1][TIME_COLUMN]
                ),
                "history_last_phase_audit": str(
                    history.iloc[-1]["phase"]
                ),
                "future_phases_audit": "|".join(
                    future["phase"]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
                "ue_count_audit": int(
                    group.iloc[0]["ue_count"]
                ),
                "distance_m_audit": float(
                    group.iloc[0]["distance_m"]
                ),
                "seed_audit": int(
                    group.iloc[0]["seed"]
                ),
                **target_information,
            }

            for position in range(
                SEQUENCE_LENGTH
            ):
                source_row = history.iloc[
                    position
                ]

                lag = (
                    SEQUENCE_LENGTH
                    - position
                    - 1
                )

                for feature in FEATURE_COLUMNS:
                    feature_name = (
                        f"{feature}_lag_{lag}"
                    )

                    row[feature_name] = float(
                        source_row[feature]
                    )

            rows.append(row)

    return pd.DataFrame(rows)


def get_ml_feature_columns(
    dataset: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in dataset.columns
        if "_lag_" in column
    ]


def validate_prediction_dataset(
    dataset: pd.DataFrame,
) -> None:
    if dataset.empty:
        raise RuntimeError(
            "Dataset D vide."
        )

    expected_feature_count = (
        len(FEATURE_COLUMNS)
        * SEQUENCE_LENGTH
    )

    feature_columns = get_ml_feature_columns(
        dataset
    )

    if len(feature_columns) != expected_feature_count:
        raise RuntimeError(
            "Nombre de variables ML incorrect : "
            f"attendu={expected_feature_count}, "
            f"obtenu={len(feature_columns)}"
        )

    temporal_overlap = (
        dataset["future_start_time_s"]
        <= dataset["history_end_time_s"]
    )

    if temporal_overlap.any():
        raise RuntimeError(
            "Chevauchement entre historique et futur."
        )

    forbidden_tokens = [
        "future",
        "phase",
        "lambda",
        "run_id",
        "seed",
        "time_s",
        "target",
        "degradation",
        "audit",
    ]

    forbidden_features = [
        column
        for column in feature_columns
        if any(
            token in column.lower()
            for token in forbidden_tokens
        )
    ]

    if forbidden_features:
        raise RuntimeError(
            "Variables interdites dans X : "
            f"{forbidden_features}"
        )

    if dataset[
        feature_columns
    ].isna().any().any():
        raise RuntimeError(
            "Valeurs manquantes dans les variables ML."
        )

    if dataset[
        "future_degradation"
    ].isna().any():
        raise RuntimeError(
            "Valeurs manquantes dans la cible."
        )

    if not set(
        dataset["future_degradation"].unique()
    ).issubset({0, 1}):
        raise RuntimeError(
            "La cible doit être binaire."
        )

    duplicates = dataset.duplicated(
        subset=[
            "run_id",
            "history_start_time_s",
            "future_start_time_s",
        ]
    )

    if duplicates.any():
        raise RuntimeError(
            "Séquences dupliquées."
        )


def build_run_audit(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    audit = (
        dataset
        .groupby("run_id")
        .agg(
            n_sequences=(
                "future_degradation",
                "size",
            ),
            positive_sequences=(
                "future_degradation",
                "sum",
            ),
            degradation_ratio=(
                "future_degradation",
                "mean",
            ),
            mean_signals=(
                "number_of_degradation_signals",
                "mean",
            ),
            max_delay_increase=(
                "delay_increase_audit",
                "max",
            ),
            max_loss_increase=(
                "loss_increase_audit",
                "max",
            ),
            max_congestion_increase=(
                "congestion_increase_audit",
                "max",
            ),
        )
        .reset_index()
    )

    return audit


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset fusionné introuvable : "
            f"{INPUT_FILE}"
        )

    source = pd.read_csv(
        INPUT_FILE
    )

    source = (
        source
        .sort_values(
            [GROUP_COLUMN, TIME_COLUMN]
        )
        .reset_index(drop=True)
    )

    validate_source(source)

    dataset = build_prediction_dataset(
        source
    )

    validate_prediction_dataset(
        dataset
    )

    audit = build_run_audit(
        dataset
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    feature_columns = get_ml_feature_columns(
        dataset
    )

    class_counts = (
        dataset["future_degradation"]
        .value_counts()
        .sort_index()
    )

    class_ratios = (
        dataset["future_degradation"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("=" * 78)
    print("CONSTRUCTION DU DATASET D")
    print("=" * 78)

    print("Fichier :", OUTPUT_FILE)
    print("Shape :", dataset.shape)
    print(
        "Nombre de runs :",
        dataset["run_id"].nunique(),
    )
    print(
        "Variables ML :",
        len(feature_columns),
    )

    print("\nDistribution de la cible :")
    print(class_counts.to_string())

    print("\nProportions :")
    print(
        class_ratios.round(4).to_string()
    )

    print("\nSéquences par run :")
    print(
        dataset.groupby("run_id")
        .size()
        .describe()
        .to_string()
    )

    print("\nContrôles temporels :")
    print(
        "Chevauchements historique/futur :",
        int(
            (
                dataset["future_start_time_s"]
                <= dataset["history_end_time_s"]
            ).sum()
        ),
    )

    print(
        "Valeurs manquantes :",
        int(dataset.isna().sum().sum()),
    )

    print(
        "Doublons de séquences :",
        int(
            dataset.duplicated(
                subset=[
                    "run_id",
                    "history_start_time_s",
                    "future_start_time_s",
                ]
            ).sum()
        ),
    )

    print("\nPremières variables ML :")
    print(feature_columns[:12])

    print("\nAudit par run :", AUDIT_FILE)
    print("\nDataset D construit avec succès.")


if __name__ == "__main__":
    main()
