#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path.home() / "v2x-llm-project"

RAW_DIR = (
    PROJECT_DIR
    / "datasets"
    / "raw"
    / "dynamic_dataset_d2_pilot"
)

MANIFEST_PATH = RAW_DIR / "campaign_manifest.csv"

OUTPUT_DIR = (
    PROJECT_DIR
    / "datasets"
    / "processed"
    / "dynamic_d2"
)

MERGED_OUTPUT = OUTPUT_DIR / "dynamic_qos_merged_d2_enriched.csv"
AUDIT_JSON = OUTPUT_DIR / "audit_dataset_d2_raw.json"
RUN_SUMMARY_OUTPUT = OUTPUT_DIR / "run_summary_d2.csv"
PROFILE_SUMMARY_OUTPUT = OUTPUT_DIR / "profile_summary_d2.csv"
PHASE_SUMMARY_OUTPUT = OUTPUT_DIR / "phase_summary_d2.csv"


REQUIRED_QOS_COLUMNS = {
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

REQUIRED_MANIFEST_COLUMNS = {
    "run_id",
    "profile",
    "ue_count",
    "distance_m",
    "seed",
    "run_number",
    "stable_lambda",
    "medium_lambda",
    "high_lambda",
    "recovery_lambda",
    "packet_size_be",
    "phase1_end",
    "phase2_end",
    "phase3_end",
    "sim_time",
    "status",
    "exit_code",
}


def fail(message: str) -> None:
    print(f"\nERREUR : {message}", file=sys.stderr)
    sys.exit(1)


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        fail(f"Manifest introuvable : {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)

    if missing:
        fail(
            "Colonnes absentes du manifest : "
            + ", ".join(sorted(missing))
        )

    if manifest["run_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["run_id"].duplicated(keep=False),
            "run_id",
        ].tolist()

        fail(
            "run_id dupliqués dans le manifest : "
            + ", ".join(duplicates[:10])
        )

    return manifest


def load_qos_files() -> pd.DataFrame:
    csv_files = sorted(
        RAW_DIR.glob("*_qos_temporal.csv")
    )

    if not csv_files:
        fail(f"Aucun CSV QoS trouvé dans {RAW_DIR}")

    print(f"Fichiers QoS détectés : {len(csv_files)}")

    frames: list[pd.DataFrame] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        missing = REQUIRED_QOS_COLUMNS - set(df.columns)

        if missing:
            fail(
                f"{csv_path.name} : colonnes absentes : "
                + ", ".join(sorted(missing))
            )

        run_id = csv_path.stem.removesuffix(
            "_qos_temporal"
        )

        df["run_id"] = run_id
        df["source_file"] = csv_path.name

        frames.append(df)

    merged = pd.concat(
        frames,
        ignore_index=True,
    )

    return merged


def validate_runs(
    qos: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    qos_runs = set(qos["run_id"].unique())
    manifest_runs = set(manifest["run_id"].unique())

    missing_in_manifest = qos_runs - manifest_runs
    missing_qos_files = manifest_runs - qos_runs

    if missing_in_manifest:
        fail(
            "Runs QoS absents du manifest : "
            + ", ".join(sorted(missing_in_manifest))
        )

    if missing_qos_files:
        fail(
            "Runs du manifest sans fichier QoS : "
            + ", ".join(sorted(missing_qos_files))
        )


def merge_with_manifest(
    qos: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    merged = qos.merge(
        manifest,
        on="run_id",
        how="left",
        validate="many_to_one",
    )

    merged = merged.sort_values(
        ["run_id", "time_s"],
        kind="stable",
    ).reset_index(drop=True)

    merged["sequence_index"] = (
        merged.groupby("run_id")
        .cumcount()
        .astype(int)
    )

    merged["time_normalized"] = (
        merged["time_s"]
        / merged["sim_time"]
    )

    return merged


def inspect_duplicate_rows(
    df: pd.DataFrame,
) -> dict[str, int]:
    exact_duplicates = int(
        df.duplicated().sum()
    )

    temporal_duplicates = int(
        df.duplicated(
            subset=["run_id", "time_s"]
        ).sum()
    )

    return {
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_run_time_rows": temporal_duplicates,
    }


def build_run_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
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
            min_time_s=("time_s", "min"),
            max_time_s=("time_s", "max"),
            phases=("phase", "nunique"),
            lambda_values=("lambda_be", "nunique"),
            mean_throughput_mbps=(
                "throughput_mbps",
                "mean",
            ),
            mean_delay_ms=("delay_ms", "mean"),
            mean_jitter_ms=("jitter_ms", "mean"),
            mean_packet_loss=("packet_loss", "mean"),
            mean_pdr=("pdr", "mean"),
            mean_congestion_index=(
                "congestion_index",
                "mean",
            ),
            max_congestion_index=(
                "congestion_index",
                "max",
            ),
        )
    )

    return summary


def build_profile_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        df.groupby("profile", as_index=False)
        .agg(
            runs=("run_id", "nunique"),
            observations=("time_s", "size"),
            mean_throughput_mbps=(
                "throughput_mbps",
                "mean",
            ),
            std_throughput_mbps=(
                "throughput_mbps",
                "std",
            ),
            mean_delay_ms=("delay_ms", "mean"),
            std_delay_ms=("delay_ms", "std"),
            mean_jitter_ms=("jitter_ms", "mean"),
            mean_packet_loss=("packet_loss", "mean"),
            mean_pdr=("pdr", "mean"),
            mean_congestion_index=(
                "congestion_index",
                "mean",
            ),
            max_congestion_index=(
                "congestion_index",
                "max",
            ),
        )
        .sort_values("profile")
        .reset_index(drop=True)
    )

    return summary


def build_phase_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["profile", "phase"],
            as_index=False,
        )
        .agg(
            runs=("run_id", "nunique"),
            observations=("time_s", "size"),
            lambda_min=("lambda_be", "min"),
            lambda_max=("lambda_be", "max"),
            mean_throughput_mbps=(
                "throughput_mbps",
                "mean",
            ),
            mean_delay_ms=("delay_ms", "mean"),
            mean_jitter_ms=("jitter_ms", "mean"),
            mean_packet_loss=("packet_loss", "mean"),
            mean_pdr=("pdr", "mean"),
            mean_congestion_index=(
                "congestion_index",
                "mean",
            ),
        )
        .sort_values(["profile", "phase"])
        .reset_index(drop=True)
    )

    return summary


def build_audit_report(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict:
    duplicates = inspect_duplicate_rows(df)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    constant_numeric_columns = [
        column
        for column in numeric_columns
        if df[column].nunique(dropna=False) <= 1
    ]

    missing_values = {
        column: int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }

    rows_per_run = (
        df.groupby("run_id")
        .size()
    )

    phases_by_profile = (
        df.groupby("profile")["phase"]
        .apply(
            lambda values: sorted(
                values.astype(str).unique().tolist()
            )
        )
        .to_dict()
    )

    lambda_by_profile = (
        df.groupby("profile")["lambda_be"]
        .apply(
            lambda values: sorted(
                values.dropna().unique().tolist()
            )
        )
        .to_dict()
    )

    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "runs": {
            "manifest_runs": int(
                manifest["run_id"].nunique()
            ),
            "merged_runs": int(
                df["run_id"].nunique()
            ),
        },
        "profiles": {
            "count": int(df["profile"].nunique()),
            "values": sorted(
                df["profile"].astype(str).unique().tolist()
            ),
            "observations_per_profile": {
                str(key): int(value)
                for key, value in (
                    df["profile"]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
        },
        "phases": {
            "count": int(df["phase"].nunique()),
            "values": sorted(
                df["phase"].astype(str).unique().tolist()
            ),
            "phases_by_profile": phases_by_profile,
        },
        "lambda_values_by_profile": lambda_by_profile,
        "rows_per_run": {
            "minimum": int(rows_per_run.min()),
            "maximum": int(rows_per_run.max()),
            "mean": float(rows_per_run.mean()),
            "unique_counts": sorted(
                rows_per_run.unique().astype(int).tolist()
            ),
        },
        "missing_values": missing_values,
        "constant_numeric_columns": (
            constant_numeric_columns
        ),
        "duplicates": duplicates,
        "manifest_status_counts": {
            str(key): int(value)
            for key, value in (
                manifest["status"]
                .value_counts(dropna=False)
                .items()
            )
        },
    }

    return report


def print_report(
    df: pd.DataFrame,
    report: dict,
) -> None:
    print("\n" + "=" * 70)
    print("AUDIT DATASET D2 BRUT")
    print("=" * 70)

    print(
        f"Lignes                 : "
        f"{report['dataset_shape']['rows']}"
    )

    print(
        f"Colonnes               : "
        f"{report['dataset_shape']['columns']}"
    )

    print(
        f"Runs                   : "
        f"{report['runs']['merged_runs']}"
    )

    print(
        f"Profils                : "
        f"{report['profiles']['count']}"
    )

    print(
        f"Phases                 : "
        f"{report['phases']['values']}"
    )

    print(
        f"Lignes par run         : "
        f"{report['rows_per_run']['unique_counts']}"
    )

    print(
        f"Valeurs manquantes     : "
        f"{sum(report['missing_values'].values())}"
    )

    print(
        f"Doublons exacts        : "
        f"{report['duplicates']['exact_duplicate_rows']}"
    )

    print(
        f"Doublons run/time      : "
        f"{report['duplicates']['duplicate_run_time_rows']}"
    )

    print(
        f"Colonnes constantes    : "
        f"{report['constant_numeric_columns']}"
    )

    print("\nObservations par profil :")
    print(
        df["profile"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nRuns par profil :")
    print(
        df.groupby("profile")["run_id"]
        .nunique()
        .sort_index()
        .to_string()
    )

    print("\nDistribution des phases :")
    print(
        pd.crosstab(
            df["profile"],
            df["phase"],
        ).to_string()
    )

    print("\nValeurs lambda par profil :")

    for profile, values in (
        report["lambda_values_by_profile"].items()
    ):
        print(f"  {profile}: {values}")

    print("\nFichier fusionné :")
    print(MERGED_OUTPUT)

    print("\nRapport d'audit :")
    print(AUDIT_JSON)

    print("=" * 70)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()
    qos = load_qos_files()

    validate_runs(qos, manifest)

    merged = merge_with_manifest(
        qos,
        manifest,
    )

    run_summary = build_run_summary(merged)
    profile_summary = build_profile_summary(merged)
    phase_summary = build_phase_summary(merged)

    report = build_audit_report(
        merged,
        manifest,
    )

    merged.to_csv(
        MERGED_OUTPUT,
        index=False,
    )

    run_summary.to_csv(
        RUN_SUMMARY_OUTPUT,
        index=False,
    )

    profile_summary.to_csv(
        PROFILE_SUMMARY_OUTPUT,
        index=False,
    )

    phase_summary.to_csv(
        PHASE_SUMMARY_OUTPUT,
        index=False,
    )

    with AUDIT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_report(
        merged,
        report,
    )


if __name__ == "__main__":
    main()
