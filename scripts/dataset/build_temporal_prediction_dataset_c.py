from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "datasets/processed/mobility/"
    "temporal_v2x_synchronized_dataset_v4.csv"
)

OUTPUT_FILE = Path(
    "datasets/processed/mobility/"
    "temporal_prediction_dataset_c.csv"
)

GROUP_COLUMN = "scenario_source"
TIME_COLUMN = "time_s"
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

SEQUENCE_LENGTH = 5
PREDICTION_HORIZON = 1


def validate_source(df: pd.DataFrame) -> None:
    required = (
        FEATURE_COLUMNS
        + [GROUP_COLUMN, TIME_COLUMN, TARGET_COLUMN]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes absentes : {missing}"
        )

    if df[required].isna().sum().sum() > 0:
        raise ValueError(
            "Le dataset source contient des valeurs manquantes."
        )


def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scenario, group in df.groupby(GROUP_COLUMN):
        group = (
            group
            .sort_values(TIME_COLUMN)
            .reset_index(drop=True)
        )

        max_start = (
            len(group)
            - SEQUENCE_LENGTH
            - PREDICTION_HORIZON
            + 1
        )

        for start in range(max_start):
            end = start + SEQUENCE_LENGTH
            target_index = (
                end
                + PREDICTION_HORIZON
                - 1
            )

            sequence = group.iloc[start:end]
            target_row = group.iloc[target_index]

            row = {
                "scenario_source": scenario,
                "sequence_start_time_s": float(
                    sequence.iloc[0][TIME_COLUMN]
                ),
                "sequence_end_time_s": float(
                    sequence.iloc[-1][TIME_COLUMN]
                ),
                "target_time_s": float(
                    target_row[TIME_COLUMN]
                ),
                "decision_future": target_row[TARGET_COLUMN],
            }

            for step in range(SEQUENCE_LENGTH):
                source_row = sequence.iloc[step]

                for feature in FEATURE_COLUMNS:
                    row[f"{feature}_t{step}"] = source_row[feature]

            rows.append(row)

    return pd.DataFrame(rows)


def validate_temporal_integrity(dataset: pd.DataFrame) -> None:
    invalid = dataset[
        dataset["target_time_s"]
        <= dataset["sequence_end_time_s"]
    ]

    if not invalid.empty:
        raise RuntimeError(
            "Erreur temporelle : la cible n'est pas strictement future."
        )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {INPUT_FILE}"
        )

    source = pd.read_csv(INPUT_FILE)

    validate_source(source)

    dataset = build_sequences(source)

    if dataset.empty:
        raise RuntimeError(
            "Aucune séquence temporelle n'a été générée."
        )

    validate_temporal_integrity(dataset)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("Dataset C généré :", OUTPUT_FILE)
    print("Shape :", dataset.shape)

    print("\nRépartition des classes futures :")
    print(
        dataset["decision_future"]
        .value_counts()
    )

    print("\nSéquences par scénario :")
    print(
        dataset["scenario_source"]
        .value_counts()
        .sort_index()
    )

    print("\nContrôle temporel :")
    print(
        dataset[
            [
                "scenario_source",
                "sequence_start_time_s",
                "sequence_end_time_s",
                "target_time_s",
                "decision_future",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()

