#!/usr/bin/env python3
"""
Analyse des simulations dynamiques de calibration.

Le script vérifie les fichiers, résume les métriques par phase et classe
les configurations selon leur capacité à produire une dégradation suivie
d'une récupération.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path(
    "datasets/raw/dynamic_qos_calibration"
)

OUTPUT_DIR = Path(
    "results/ml/dynamic_calibration"
)

REQUIRED_COLUMNS = {
    "time_s",
    "phase",
    "lambda_be",
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
}

EXPECTED_PHASES = {
    "STABLE",
    "MEDIUM",
    "HIGH",
    "RECOVERY",
}

METRIC_COLUMNS = [
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
]


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if abs(denominator) < 1e-12:
        return np.nan

    return float(numerator / denominator)


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: colonnes absentes {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"{path.name}: fichier vide."
        )

    if df[list(REQUIRED_COLUMNS)].isna().sum().sum() > 0:
        raise ValueError(
            f"{path.name}: valeurs manquantes."
        )

    if not df["time_s"].is_monotonic_increasing:
        raise ValueError(
            f"{path.name}: timestamps non croissants."
        )

    observed_phases = set(df["phase"].unique())
    missing_phases = EXPECTED_PHASES - observed_phases

    if missing_phases:
        raise ValueError(
            f"{path.name}: phases absentes "
            f"{sorted(missing_phases)}"
        )

    invalid_congestion = (
        (df["congestion_index"] < 0)
        | (df["congestion_index"] > 1)
    ).sum()

    if invalid_congestion:
        raise ValueError(
            f"{path.name}: congestion hors [0,1]."
        )

    return df


def summarize_run(
    run_id: str,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    phase_summary = (
        df.groupby("phase", sort=False)[METRIC_COLUMNS]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    means = (
        df.groupby("phase", sort=False)[METRIC_COLUMNS]
        .mean()
    )

    stable = means.loc["STABLE"]
    medium = means.loc["MEDIUM"]
    high = means.loc["HIGH"]
    recovery = means.loc["RECOVERY"]

    high_delay_increase = float(
        high["delay_ms"] - stable["delay_ms"]
    )

    high_loss_increase = float(
        high["packet_loss"] - stable["packet_loss"]
    )

    high_congestion_increase = float(
        high["congestion_index"]
        - stable["congestion_index"]
    )

    recovery_delay_improvement = float(
        high["delay_ms"] - recovery["delay_ms"]
    )

    recovery_loss_improvement = float(
        high["packet_loss"]
        - recovery["packet_loss"]
    )

    recovery_congestion_improvement = float(
        high["congestion_index"]
        - recovery["congestion_index"]
    )

    degradation_detected = bool(
        high["packet_loss"] >= 0.5
        or high["delay_ms"] >= 5.0
        or high["congestion_index"] >= 0.10
        or high_delay_increase >= 2.0
    )

    recovery_detected = bool(
        recovery["delay_ms"] < high["delay_ms"]
        or recovery["packet_loss"] < high["packet_loss"]
        or recovery["congestion_index"]
        < high["congestion_index"]
    )

    # Score utilisé seulement pour classer les calibrations.
    stress_score = float(
        min(high["delay_ms"] / 10.0, 3.0)
        + min(high["packet_loss"] / 5.0, 3.0)
        + min(high["congestion_index"] / 0.2, 3.0)
    )

    row = {
        "run_id": run_id,
        "n_windows": int(len(df)),
        "stable_lambda": int(
            df.loc[df["phase"] == "STABLE", "lambda_be"].mode()[0]
        ),
        "medium_lambda": int(
            df.loc[df["phase"] == "MEDIUM", "lambda_be"].mode()[0]
        ),
        "high_lambda": int(
            df.loc[df["phase"] == "HIGH", "lambda_be"].mode()[0]
        ),
        "recovery_lambda": int(
            df.loc[df["phase"] == "RECOVERY", "lambda_be"].mode()[0]
        ),
        "stable_throughput_mean": float(
            stable["throughput_mbps"]
        ),
        "high_throughput_mean": float(
            high["throughput_mbps"]
        ),
        "stable_delay_mean": float(
            stable["delay_ms"]
        ),
        "high_delay_mean": float(
            high["delay_ms"]
        ),
        "recovery_delay_mean": float(
            recovery["delay_ms"]
        ),
        "stable_packet_loss_mean": float(
            stable["packet_loss"]
        ),
        "high_packet_loss_mean": float(
            high["packet_loss"]
        ),
        "recovery_packet_loss_mean": float(
            recovery["packet_loss"]
        ),
        "stable_congestion_mean": float(
            stable["congestion_index"]
        ),
        "high_congestion_mean": float(
            high["congestion_index"]
        ),
        "recovery_congestion_mean": float(
            recovery["congestion_index"]
        ),
        "high_delay_increase": high_delay_increase,
        "high_loss_increase": high_loss_increase,
        "high_congestion_increase": (
            high_congestion_increase
        ),
        "high_to_stable_delay_ratio": safe_ratio(
            high["delay_ms"],
            stable["delay_ms"],
        ),
        "recovery_delay_improvement": (
            recovery_delay_improvement
        ),
        "recovery_loss_improvement": (
            recovery_loss_improvement
        ),
        "recovery_congestion_improvement": (
            recovery_congestion_improvement
        ),
        "degradation_detected": degradation_detected,
        "recovery_detected": recovery_detected,
        "stress_score": stress_score,
    }

    return phase_summary, row


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Répertoire introuvable : {INPUT_DIR}"
        )

    files = sorted(
        INPUT_DIR.glob("*_qos_temporal.csv")
    )

    if not files:
        raise FileNotFoundError(
            "Aucun fichier temporel de calibration trouvé."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_rows = []

    for path in files:
        run_id = path.name.removesuffix(
            "_qos_temporal.csv"
        )

        df = load_and_validate(path)

        phase_summary, comparison_row = summarize_run(
            run_id=run_id,
            df=df,
        )

        phase_summary.to_csv(
            OUTPUT_DIR / f"{run_id}_phase_summary.csv",
            index=False,
        )

        comparison_rows.append(comparison_row)

    comparison = pd.DataFrame(comparison_rows)

    comparison = comparison.sort_values(
        by=[
            "degradation_detected",
            "recovery_detected",
            "stress_score",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    comparison.to_csv(
        OUTPUT_DIR / "calibration_comparison.csv",
        index=False,
    )

    display_columns = [
        "run_id",
        "high_throughput_mean",
        "high_delay_mean",
        "high_packet_loss_mean",
        "high_congestion_mean",
        "recovery_delay_mean",
        "degradation_detected",
        "recovery_detected",
        "stress_score",
    ]

    print("=" * 110)
    print("COMPARAISON DES PROFILS DYNAMIQUES")
    print("=" * 110)

    print(
        comparison[display_columns]
        .to_string(index=False)
    )

    valid = comparison[
        comparison["degradation_detected"]
        & comparison["recovery_detected"]
    ]

    print("\n" + "=" * 110)
    print("DÉCISION")
    print("=" * 110)

    if valid.empty:
        print(
            "Aucune configuration ne produit encore une "
            "dégradation suffisamment visible suivie d'une récupération."
        )
        print(
            "Il faudra réaliser une calibration renforcée."
        )
    else:
        best = valid.iloc[0]

        print("Configuration recommandée :")
        print(best[display_columns].to_string())

    print(
        "\nRésultats sauvegardés dans :",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
