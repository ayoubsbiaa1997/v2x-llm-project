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
    "datasets/processed/dynamic_d2/"
    "profile_diagnostics"
)

QOS_COLUMNS = [
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
]


def fail(message: str) -> None:
    print(f"ERREUR : {message}", file=sys.stderr)
    sys.exit(1)


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        fail(f"Fichier introuvable : {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = {
        "run_id",
        "profile",
        "time_s",
        "phase",
        "lambda_be",
        "ue_count",
        "distance_m",
        "seed",
        *QOS_COLUMNS,
    }

    missing = required - set(df.columns)

    if missing:
        fail(
            "Colonnes absentes : "
            + ", ".join(sorted(missing))
        )

    return (
        df.sort_values(["run_id", "time_s"])
        .reset_index(drop=True)
    )


def build_profile_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("profile", as_index=False)
        .agg(
            runs=("run_id", "nunique"),
            observations=("time_s", "size"),
            lambda_min=("lambda_be", "min"),
            lambda_median=("lambda_be", "median"),
            lambda_max=("lambda_be", "max"),
            throughput_mean=("throughput_mbps", "mean"),
            throughput_median=("throughput_mbps", "median"),
            throughput_min=("throughput_mbps", "min"),
            throughput_max=("throughput_mbps", "max"),
            delay_mean=("delay_ms", "mean"),
            delay_median=("delay_ms", "median"),
            delay_max=("delay_ms", "max"),
            jitter_mean=("jitter_ms", "mean"),
            loss_mean=("packet_loss", "mean"),
            loss_median=("packet_loss", "median"),
            loss_min=("packet_loss", "min"),
            loss_max=("packet_loss", "max"),
            pdr_mean=("pdr", "mean"),
            pdr_median=("pdr", "median"),
            congestion_mean=("congestion_index", "mean"),
            congestion_median=("congestion_index", "median"),
            congestion_min=("congestion_index", "min"),
            congestion_max=("congestion_index", "max"),
        )
        .sort_values("profile")
        .reset_index(drop=True)
    )


def build_phase_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["profile", "phase", "lambda_be"],
            as_index=False,
        )
        .agg(
            runs=("run_id", "nunique"),
            observations=("time_s", "size"),
            throughput_mean=("throughput_mbps", "mean"),
            delay_mean=("delay_ms", "mean"),
            jitter_mean=("jitter_ms", "mean"),
            loss_mean=("packet_loss", "mean"),
            pdr_mean=("pdr", "mean"),
            congestion_mean=("congestion_index", "mean"),
        )
        .sort_values(
            ["profile", "lambda_be", "phase"]
        )
        .reset_index(drop=True)
    )


def build_run_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "run_id",
                "profile",
                "ue_count",
                "distance_m",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            observations=("time_s", "size"),
            lambda_min=("lambda_be", "min"),
            lambda_max=("lambda_be", "max"),
            throughput_mean=("throughput_mbps", "mean"),
            delay_mean=("delay_ms", "mean"),
            jitter_mean=("jitter_ms", "mean"),
            loss_mean=("packet_loss", "mean"),
            pdr_mean=("pdr", "mean"),
            congestion_mean=("congestion_index", "mean"),
        )
        .sort_values(
            ["profile", "ue_count", "distance_m", "seed"]
        )
        .reset_index(drop=True)
    )


def check_metric_identities(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "run_id": df["run_id"],
            "profile": df["profile"],
            "time_s": df["time_s"],
        }
    )

    result["loss_plus_pdr"] = (
        df["packet_loss"] + df["pdr"]
    )

    result["loss_pdr_identity_error"] = (
        result["loss_plus_pdr"] - 100.0
    ).abs()

    result["pdr_expected_from_loss"] = (
        100.0 - df["packet_loss"]
    )

    return result


def detect_counterintuitive_profiles(
    profile_summary: pd.DataFrame,
) -> None:
    indexed = profile_summary.set_index("profile")

    stable_name = "STABLE_CONTROL"
    high_name = "PERSISTENT_HIGH"

    if (
        stable_name in indexed.index
        and high_name in indexed.index
    ):
        stable_loss = indexed.loc[
            stable_name,
            "loss_mean",
        ]

        high_loss = indexed.loc[
            high_name,
            "loss_mean",
        ]

        stable_congestion = indexed.loc[
            stable_name,
            "congestion_mean",
        ]

        high_congestion = indexed.loc[
            high_name,
            "congestion_mean",
        ]

        print("\nContrôle de cohérence :")

        print(
            f"  Perte moyenne STABLE_CONTROL : "
            f"{stable_loss:.6f}"
        )

        print(
            f"  Perte moyenne PERSISTENT_HIGH : "
            f"{high_loss:.6f}"
        )

        print(
            f"  Congestion moyenne STABLE_CONTROL : "
            f"{stable_congestion:.6f}"
        )

        print(
            f"  Congestion moyenne PERSISTENT_HIGH : "
            f"{high_congestion:.6f}"
        )

        if stable_loss > high_loss:
            print(
                "\n  ALERTE : STABLE_CONTROL présente "
                "davantage de pertes que PERSISTENT_HIGH."
            )

        if stable_congestion > high_congestion:
            print(
                "  ALERTE : STABLE_CONTROL présente "
                "davantage de congestion que PERSISTENT_HIGH."
            )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    profile_summary = build_profile_summary(df)
    phase_summary = build_phase_summary(df)
    run_summary = build_run_summary(df)
    identity_check = check_metric_identities(df)

    profile_summary.to_csv(
        OUTPUT_DIR / "profile_summary.csv",
        index=False,
    )

    phase_summary.to_csv(
        OUTPUT_DIR / "profile_phase_lambda_summary.csv",
        index=False,
    )

    run_summary.to_csv(
        OUTPUT_DIR / "run_summary.csv",
        index=False,
    )

    identity_check.to_csv(
        OUTPUT_DIR / "metric_identity_check.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("DIAGNOSTIC DES PROFILS D2")
    print("=" * 100)

    columns = [
        "profile",
        "runs",
        "lambda_min",
        "lambda_median",
        "lambda_max",
        "throughput_mean",
        "delay_mean",
        "loss_mean",
        "pdr_mean",
        "congestion_mean",
    ]

    print("\nRésumé par profil :")
    print(
        profile_summary[columns]
        .to_string(index=False)
    )

    print("\nRésumé par profil, phase et lambda :")
    print(
        phase_summary.to_string(index=False)
    )

    print("\nVérification packet_loss + pdr :")
    print(
        identity_check[
            "loss_pdr_identity_error"
        ].describe().to_string()
    )

    max_identity_error = float(
        identity_check[
            "loss_pdr_identity_error"
        ].max()
    )

    if max_identity_error < 1e-6:
        print(
            "\nCONFIRMÉ : pdr = 100 - packet_loss "
            "pour toutes les observations."
        )
    else:
        print(
            "\npacket_loss et pdr ne sont pas "
            "strictement complémentaires partout."
        )

    detect_counterintuitive_profiles(
        profile_summary
    )

    print("\nFichiers créés dans :")
    print(OUTPUT_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()
