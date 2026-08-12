#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "datasets/processed/dynamic_d2/"
    "dynamic_qos_merged_d2_enriched.csv"
)

OUTPUT_DIR = Path(
    "datasets/processed/dynamic_d2/target_analysis"
)

NUMERIC_QOS_COLUMNS = [
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
]

QUANTILES = [
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]


def fail(message: str) -> None:
    print(f"ERREUR : {message}", file=sys.stderr)
    sys.exit(1)


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        fail(f"Fichier introuvable : {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = {
        "run_id",
        "profile",
        "time_s",
        *NUMERIC_QOS_COLUMNS,
    }

    missing = required - set(df.columns)

    if missing:
        fail(
            "Colonnes absentes : "
            + ", ".join(sorted(missing))
        )

    for column in NUMERIC_QOS_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[NUMERIC_QOS_COLUMNS].isna().any().any():
        missing_counts = (
            df[NUMERIC_QOS_COLUMNS]
            .isna()
            .sum()
        )

        fail(
            "Valeurs non numériques ou manquantes :\n"
            + missing_counts.to_string()
        )

    return (
        df.sort_values(["run_id", "time_s"])
        .reset_index(drop=True)
    )


def compute_global_quantiles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    quantiles = (
        df[NUMERIC_QOS_COLUMNS]
        .quantile(QUANTILES)
        .transpose()
    )

    quantiles.columns = [
        f"q{int(q * 100):02d}"
        for q in QUANTILES
    ]

    quantiles.index.name = "metric"

    return quantiles.reset_index()


def compute_profile_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    aggregations = {}

    for column in NUMERIC_QOS_COLUMNS:
        aggregations[f"{column}_mean"] = (
            column,
            "mean",
        )
        aggregations[f"{column}_median"] = (
            column,
            "median",
        )
        aggregations[f"{column}_std"] = (
            column,
            "std",
        )
        aggregations[f"{column}_min"] = (
            column,
            "min",
        )
        aggregations[f"{column}_max"] = (
            column,
            "max",
        )

    return (
        df.groupby("profile", as_index=False)
        .agg(**aggregations)
        .sort_values("profile")
        .reset_index(drop=True)
    )


def compute_phase_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "phase" not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby(
            ["profile", "phase"],
            as_index=False,
        )
        .agg(
            observations=("time_s", "size"),
            runs=("run_id", "nunique"),
            throughput_mean=(
                "throughput_mbps",
                "mean",
            ),
            throughput_median=(
                "throughput_mbps",
                "median",
            ),
            delay_mean=("delay_ms", "mean"),
            delay_median=("delay_ms", "median"),
            jitter_mean=("jitter_ms", "mean"),
            packet_loss_mean=("packet_loss", "mean"),
            pdr_mean=("pdr", "mean"),
            congestion_mean=(
                "congestion_index",
                "mean",
            ),
            congestion_median=(
                "congestion_index",
                "median",
            ),
        )
        .sort_values(["profile", "phase"])
        .reset_index(drop=True)
    )


def compute_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df[NUMERIC_QOS_COLUMNS]
        .corr(method="spearman")
        .round(4)
    )


def future_rolling_mean(
    series: pd.Series,
    horizon: int,
) -> pd.Series:
    shifted = series.shift(-1)

    return (
        shifted.iloc[::-1]
        .rolling(
            window=horizon,
            min_periods=horizon,
        )
        .mean()
        .iloc[::-1]
    )


def add_future_metrics(
    df: pd.DataFrame,
    horizon: int = 3,
) -> pd.DataFrame:
    result = df.copy()

    grouped = result.groupby(
        "run_id",
        group_keys=False,
    )

    for column in NUMERIC_QOS_COLUMNS:
        result[f"future_{column}"] = (
            grouped[column]
            .apply(
                lambda series: future_rolling_mean(
                    series,
                    horizon,
                )
            )
        )

    return result


def evaluate_candidate_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    congestion_thresholds = [
        0.50,
        0.60,
        0.70,
    ]

    delay_quantiles = [
        0.75,
        0.85,
        0.90,
    ]

    loss_quantiles = [
        0.75,
        0.85,
        0.90,
    ]

    pdr_quantiles = [
        0.10,
        0.15,
        0.25,
    ]

    valid = df.dropna(
        subset=[
            f"future_{column}"
            for column in NUMERIC_QOS_COLUMNS
        ]
    ).copy()

    for congestion_threshold in congestion_thresholds:
        for delay_q in delay_quantiles:
            for loss_q in loss_quantiles:
                for pdr_q in pdr_quantiles:
                    delay_threshold = float(
                        valid["future_delay_ms"]
                        .quantile(delay_q)
                    )

                    loss_threshold = float(
                        valid["future_packet_loss"]
                        .quantile(loss_q)
                    )

                    pdr_threshold = float(
                        valid["future_pdr"]
                        .quantile(pdr_q)
                    )

                    congestion_bad = (
                        valid[
                            "future_congestion_index"
                        ]
                        >= congestion_threshold
                    )

                    delay_bad = (
                        valid["future_delay_ms"]
                        >= delay_threshold
                    )

                    loss_bad = (
                        valid["future_packet_loss"]
                        >= loss_threshold
                    )

                    pdr_bad = (
                        valid["future_pdr"]
                        <= pdr_threshold
                    )

                    score = (
                        congestion_bad.astype(int)
                        + delay_bad.astype(int)
                        + loss_bad.astype(int)
                        + pdr_bad.astype(int)
                    )

                    target = (score >= 2).astype(int)

                    positive = int(target.sum())
                    total = int(len(target))
                    positive_rate = (
                        positive / total
                        if total
                        else np.nan
                    )

                    profile_rates = (
                        valid.assign(
                            candidate_target=target
                        )
                        .groupby("profile")[
                            "candidate_target"
                        ]
                        .mean()
                    )

                    row = {
                        "congestion_threshold": (
                            congestion_threshold
                        ),
                        "delay_quantile": delay_q,
                        "delay_threshold": (
                            delay_threshold
                        ),
                        "loss_quantile": loss_q,
                        "loss_threshold": (
                            loss_threshold
                        ),
                        "pdr_quantile": pdr_q,
                        "pdr_threshold": (
                            pdr_threshold
                        ),
                        "positive_count": positive,
                        "negative_count": (
                            total - positive
                        ),
                        "positive_rate": (
                            positive_rate
                        ),
                    }

                    for profile, rate in (
                        profile_rates.items()
                    ):
                        row[
                            f"positive_rate_{profile}"
                        ] = float(rate)

                    rows.append(row)

    candidates = pd.DataFrame(rows)

    candidates["distance_from_35_percent"] = (
        candidates["positive_rate"] - 0.35
    ).abs()

    return (
        candidates.sort_values(
            [
                "distance_from_35_percent",
                "positive_rate",
            ]
        )
        .reset_index(drop=True)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    global_quantiles = compute_global_quantiles(df)
    profile_summary = compute_profile_summary(df)
    phase_summary = compute_phase_summary(df)
    correlations = compute_correlations(df)

    future_df = add_future_metrics(
        df,
        horizon=3,
    )

    candidates = evaluate_candidate_targets(
        future_df
    )

    global_quantiles.to_csv(
        OUTPUT_DIR / "global_qos_quantiles.csv",
        index=False,
    )

    profile_summary.to_csv(
        OUTPUT_DIR / "qos_summary_by_profile.csv",
        index=False,
    )

    if not phase_summary.empty:
        phase_summary.to_csv(
            OUTPUT_DIR / "qos_summary_by_phase.csv",
            index=False,
        )

    correlations.to_csv(
        OUTPUT_DIR / "qos_spearman_correlations.csv"
    )

    candidates.to_csv(
        OUTPUT_DIR / "candidate_composite_targets.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("ANALYSE DES SEUILS — DATASET D2")
    print("=" * 72)

    print(f"Forme du dataset : {df.shape}")
    print(f"Runs : {df['run_id'].nunique()}")
    print(f"Profils : {df['profile'].nunique()}")

    print("\nQuantiles globaux :")
    print(
        global_quantiles.to_string(
            index=False
        )
    )

    print("\nCorrélations de Spearman :")
    print(correlations.to_string())

    print(
        "\nDix meilleures configurations "
        "candidates :"
    )

    columns_to_show = [
        "congestion_threshold",
        "delay_quantile",
        "delay_threshold",
        "loss_quantile",
        "loss_threshold",
        "pdr_quantile",
        "pdr_threshold",
        "positive_count",
        "negative_count",
        "positive_rate",
    ]

    print(
        candidates[
            columns_to_show
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nFichiers créés dans :")
    print(OUTPUT_DIR)
    print("=" * 72)


if __name__ == "__main__":
    main()
