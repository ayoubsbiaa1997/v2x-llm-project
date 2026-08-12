#!/usr/bin/env python3
"""
Fusion et validation des runs dynamiques ns-3 du Dataset D.

Chaque CSV temporel est associé aux métadonnées du manifest.
Les colonnes de simulation sont conservées pour l'audit, mais elles
ne seront pas automatiquement utilisées comme variables ML.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_DIR = Path(
    "datasets/raw/dynamic_dataset_d"
)

MANIFEST_FILE = (
    INPUT_DIR / "campaign_manifest.csv"
)

OUTPUT_FILE = Path(
    "datasets/processed/dynamic/"
    "dynamic_qos_merged.csv"
)

AUDIT_FILE = Path(
    "results/ml/dynamic_dataset_d/"
    "merged_dataset_audit.csv"
)


REQUIRED_MANIFEST_COLUMNS = {
    "run_id",
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


EXPECTED_PHASES = {
    "STABLE",
    "MEDIUM",
    "HIGH",
    "RECOVERY",
}


METADATA_COLUMNS = [
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
]


def validate_manifest(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    missing = (
        REQUIRED_MANIFEST_COLUMNS
        - set(manifest.columns)
    )

    if missing:
        raise ValueError(
            "Colonnes absentes du manifest : "
            f"{sorted(missing)}"
        )

    if manifest.empty:
        raise ValueError(
            "Le manifest est vide."
        )

    if manifest["run_id"].duplicated().any():
        duplicates = (
            manifest.loc[
                manifest["run_id"].duplicated(),
                "run_id",
            ]
            .tolist()
        )

        raise ValueError(
            f"Run IDs dupliqués : {duplicates}"
        )

    successful = manifest[
        manifest["status"] == "SUCCESS"
    ].copy()

    if successful.empty:
        raise RuntimeError(
            "Aucun run marqué SUCCESS."
        )

    if len(successful) != len(manifest):
        failed = manifest[
            manifest["status"] != "SUCCESS"
        ][["run_id", "status", "exit_code"]]

        print(
            "Attention : certains runs seront exclus :"
        )
        print(failed.to_string(index=False))

    return successful


def load_and_validate_run(
    run_id: str,
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{run_id}: fichier absent : {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{run_id}: fichier vide : {path}"
        )

    df = pd.read_csv(path)

    missing = (
        REQUIRED_QOS_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{run_id}: colonnes absentes : "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"{run_id}: aucune observation."
        )

    if df[list(REQUIRED_QOS_COLUMNS)].isna().any().any():
        null_counts = (
            df[list(REQUIRED_QOS_COLUMNS)]
            .isna()
            .sum()
        )

        raise ValueError(
            f"{run_id}: valeurs manquantes :\n"
            f"{null_counts[null_counts > 0]}"
        )

    if not df["time_s"].is_monotonic_increasing:
        raise ValueError(
            f"{run_id}: temps non croissant."
        )

    if df["time_s"].duplicated().any():
        raise ValueError(
            f"{run_id}: timestamps dupliqués."
        )

    observed_phases = set(
        df["phase"].unique()
    )

    missing_phases = (
        EXPECTED_PHASES
        - observed_phases
    )

    if missing_phases:
        raise ValueError(
            f"{run_id}: phases absentes : "
            f"{sorted(missing_phases)}"
        )

    invalid_pdr = (
        (df["pdr"] < 0)
        | (df["pdr"] > 100)
    )

    if invalid_pdr.any():
        raise ValueError(
            f"{run_id}: valeurs PDR hors [0,100]."
        )

    invalid_loss = (
        (df["packet_loss"] < 0)
        | (df["packet_loss"] > 100)
    )

    if invalid_loss.any():
        raise ValueError(
            f"{run_id}: pertes hors [0,100]."
        )

    invalid_congestion = (
        (df["congestion_index"] < 0)
        | (df["congestion_index"] > 1)
    )

    if invalid_congestion.any():
        raise ValueError(
            f"{run_id}: congestion hors [0,1]."
        )

    return df


def build_run_audit(
    run_id: str,
    df: pd.DataFrame,
) -> dict:
    phase_counts = (
        df["phase"]
        .value_counts()
        .to_dict()
    )

    return {
        "run_id": run_id,
        "n_windows": int(len(df)),
        "time_min_s": float(df["time_s"].min()),
        "time_max_s": float(df["time_s"].max()),
        "stable_windows": int(
            phase_counts.get("STABLE", 0)
        ),
        "medium_windows": int(
            phase_counts.get("MEDIUM", 0)
        ),
        "high_windows": int(
            phase_counts.get("HIGH", 0)
        ),
        "recovery_windows": int(
            phase_counts.get("RECOVERY", 0)
        ),
        "throughput_mean": float(
            df["throughput_mbps"].mean()
        ),
        "delay_mean": float(
            df["delay_ms"].mean()
        ),
        "jitter_mean": float(
            df["jitter_ms"].mean()
        ),
        "packet_loss_mean": float(
            df["packet_loss"].mean()
        ),
        "pdr_mean": float(
            df["pdr"].mean()
        ),
        "congestion_mean": float(
            df["congestion_index"].mean()
        ),
    }


def validate_merged(
    merged: pd.DataFrame,
    expected_runs: int,
) -> None:
    if merged.empty:
        raise RuntimeError(
            "Le dataset fusionné est vide."
        )

    actual_runs = int(
        merged["run_id"].nunique()
    )

    if actual_runs != expected_runs:
        raise RuntimeError(
            "Nombre de runs incorrect : "
            f"attendu={expected_runs}, obtenu={actual_runs}"
        )

    duplicates = merged.duplicated(
        subset=["run_id", "time_s"]
    )

    if duplicates.any():
        raise RuntimeError(
            "Des couples run_id/time_s sont dupliqués."
        )

    if merged.isna().any().any():
        null_counts = merged.isna().sum()
        null_counts = null_counts[
            null_counts > 0
        ]

        raise RuntimeError(
            "Valeurs manquantes après fusion :\n"
            f"{null_counts}"
        )

    for run_id, group in merged.groupby(
        "run_id"
    ):
        if not group[
            "time_s"
        ].is_monotonic_increasing:
            raise RuntimeError(
                f"{run_id}: ordre temporel invalide "
                "après fusion."
            )


def main() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest introuvable : {MANIFEST_FILE}"
        )

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    successful = validate_manifest(
        manifest
    )

    merged_frames = []
    audit_rows = []

    for _, metadata in successful.iterrows():
        run_id = str(
            metadata["run_id"]
        )

        csv_path = (
            INPUT_DIR
            / f"{run_id}_qos_temporal.csv"
        )

        run_df = load_and_validate_run(
            run_id=run_id,
            path=csv_path,
        )

        run_df = run_df.copy()

        run_df.insert(
            0,
            "run_id",
            run_id,
        )

        for column in METADATA_COLUMNS:
            run_df[column] = metadata[column]

        merged_frames.append(run_df)

        audit_rows.append(
            build_run_audit(
                run_id=run_id,
                df=run_df,
            )
        )

    merged = pd.concat(
        merged_frames,
        ignore_index=True,
    )

    merged = (
        merged
        .sort_values(
            ["run_id", "time_s"]
        )
        .reset_index(drop=True)
    )

    validate_merged(
        merged=merged,
        expected_runs=len(successful),
    )

    audit = pd.DataFrame(
        audit_rows
    ).sort_values("run_id")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    print("=" * 75)
    print("FUSION DU DATASET D")
    print("=" * 75)

    print("Fichier :", OUTPUT_FILE)
    print("Shape :", merged.shape)
    print(
        "Nombre de runs :",
        merged["run_id"].nunique(),
    )
    print(
        "Fenêtres par run :"
    )
    print(
        merged.groupby("run_id")
        .size()
        .describe()
        .to_string()
    )

    print("\nDistribution des phases :")
    print(
        merged["phase"]
        .value_counts()
        .to_string()
    )

    print("\nRésumé QoS moyen par phase :")
    summary = (
        merged
        .groupby("phase")
        [
            [
                "throughput_mbps",
                "delay_ms",
                "jitter_ms",
                "packet_loss",
                "pdr",
                "congestion_index",
            ]
        ]
        .mean()
    )

    print(summary.to_string())

    print("\nContrôles :")
    print(
        "Valeurs manquantes :",
        int(merged.isna().sum().sum()),
    )
    print(
        "Doublons run_id/time_s :",
        int(
            merged.duplicated(
                subset=["run_id", "time_s"]
            ).sum()
        ),
    )

    print("\nAudit par run :", AUDIT_FILE)
    print("\nFusion validée avec succès.")


if __name__ == "__main__":
    main()
