#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path.home() / "v2x-llm-project"

DEFAULT_INPUT = (
    ROOT
    / "external_validation"
    / "ngsim"
    / "raw"
    / "us101.csv"
)

DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "external_validation"
    / "ngsim"
    / "dataset_audit.json"
)

NGSIM_COLUMNS = [
    "Vehicle_ID",
    "Frame_ID",
    "Total_Frames",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "Global_X",
    "Global_Y",
    "v_Length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auditer un fichier de trajectoires NGSIM "
            "sans en-tête et séparé par des espaces."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100_000,
    )

    return parser.parse_args()


def load_ngsim(
    path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=NGSIM_COLUMNS,
        nrows=nrows,
        engine="c",
    )


def main() -> int:
    args = parse_args()

    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()

    if not input_file.exists():
        print(
            f"Fichier introuvable : {input_file}",
            file=sys.stderr,
        )
        return 1

    try:
        sample = load_ngsim(
            input_file,
            nrows=args.sample_rows,
        )
    except Exception as exc:
        print(
            f"Lecture impossible : {exc}",
            file=sys.stderr,
        )
        return 1

    if sample.empty:
        print(
            "Le fichier ne contient aucune observation.",
            file=sys.stderr,
        )
        return 1

    if len(sample.columns) != len(NGSIM_COLUMNS):
        print(
            "Nombre de colonnes inattendu : "
            f"{len(sample.columns)} au lieu de "
            f"{len(NGSIM_COLUMNS)}.",
            file=sys.stderr,
        )
        return 1

    numeric_columns = [
        "Vehicle_ID",
        "Frame_ID",
        "Total_Frames",
        "Global_Time",
        "Local_X",
        "Local_Y",
        "Global_X",
        "Global_Y",
        "v_Length",
        "v_Width",
        "v_Class",
        "v_Vel",
        "v_Acc",
        "Lane_ID",
        "Preceding",
        "Following",
        "Space_Headway",
        "Time_Headway",
    ]

    for column in numeric_columns:
        sample[column] = pd.to_numeric(
            sample[column],
            errors="coerce",
        )

    missing_values = {
        column: int(count)
        for column, count in (
            sample.isna()
            .sum()
            .sort_values(ascending=False)
            .items()
        )
        if count > 0
    }

    vehicle_count = int(
        sample["Vehicle_ID"].nunique()
    )

    frame_min = int(
        sample["Frame_ID"].min()
    )

    frame_max = int(
        sample["Frame_ID"].max()
    )

    global_time_min = int(
        sample["Global_Time"].min()
    )

    global_time_max = int(
        sample["Global_Time"].max()
    )

    duration_seconds = (
        global_time_max - global_time_min
    ) / 1000.0

    lane_distribution = {
        str(int(lane)): int(count)
        for lane, count in (
            sample["Lane_ID"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    vehicle_class_distribution = {
        str(int(vehicle_class)): int(count)
        for vehicle_class, count in (
            sample["v_Class"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    speed_fps = sample["v_Vel"]
    acceleration_fps2 = sample["v_Acc"]

    feet_to_meters = 0.3048

    speed_mps = speed_fps * feet_to_meters
    acceleration_mps2 = (
        acceleration_fps2 * feet_to_meters
    )

    report = {
        "input_file": str(input_file),
        "file_size_mb": round(
            input_file.stat().st_size
            / 1024
            / 1024,
            3,
        ),
        "format": {
            "separator": "whitespace",
            "header_present": False,
            "sampling_frequency_hz": 10,
            "column_count": len(NGSIM_COLUMNS),
            "columns": NGSIM_COLUMNS,
        },
        "sample": {
            "requested_rows": args.sample_rows,
            "loaded_rows": int(len(sample)),
            "duplicate_rows": int(
                sample.duplicated().sum()
            ),
            "missing_values": missing_values,
        },
        "temporal_coverage": {
            "frame_min": frame_min,
            "frame_max": frame_max,
            "global_time_min_ms": global_time_min,
            "global_time_max_ms": global_time_max,
            "duration_seconds_in_sample": round(
                duration_seconds,
                3,
            ),
        },
        "vehicles": {
            "unique_vehicle_count": vehicle_count,
            "vehicle_id_min": int(
                sample["Vehicle_ID"].min()
            ),
            "vehicle_id_max": int(
                sample["Vehicle_ID"].max()
            ),
            "class_distribution": (
                vehicle_class_distribution
            ),
        },
        "lanes": {
            "lane_distribution": lane_distribution,
            "lane_min": int(
                sample["Lane_ID"].min()
            ),
            "lane_max": int(
                sample["Lane_ID"].max()
            ),
        },
        "mobility_statistics": {
            "mean_speed_fps": round(
                float(speed_fps.mean()),
                6,
            ),
            "mean_speed_mps": round(
                float(speed_mps.mean()),
                6,
            ),
            "mean_speed_kmh": round(
                float(speed_mps.mean() * 3.6),
                6,
            ),
            "speed_std_mps": round(
                float(speed_mps.std()),
                6,
            ),
            "mean_acceleration_mps2": round(
                float(acceleration_mps2.mean()),
                6,
            ),
            "mean_space_headway_ft": round(
                float(
                    sample["Space_Headway"].mean()
                ),
                6,
            ),
            "mean_space_headway_m": round(
                float(
                    sample["Space_Headway"].mean()
                    * feet_to_meters
                ),
                6,
            ),
            "mean_time_headway_s": round(
                float(
                    sample["Time_Headway"].mean()
                ),
                6,
            ),
        },
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("AUDIT DU DATASET NGSIM US-101")
    print("=" * 76)
    print(f"Fichier        : {input_file}")
    print(f"Taille         : {report['file_size_mb']} MB")
    print(f"Lignes lues    : {len(sample)}")
    print(f"Colonnes       : {len(sample.columns)}")
    print(f"Véhicules      : {vehicle_count}")
    print(
        f"Frames         : {frame_min} -> {frame_max}"
    )
    print(
        f"Durée observée : {duration_seconds:.2f} s"
    )
    print()

    print("Statistiques de mobilité :")
    print(
        "  Vitesse moyenne : "
        f"{report['mobility_statistics']['mean_speed_mps']:.3f} m/s"
    )
    print(
        "  Vitesse moyenne : "
        f"{report['mobility_statistics']['mean_speed_kmh']:.3f} km/h"
    )
    print(
        "  Accélération moyenne : "
        f"{report['mobility_statistics']['mean_acceleration_mps2']:.3f} "
        "m/s²"
    )
    print(
        "  Espacement moyen : "
        f"{report['mobility_statistics']['mean_space_headway_m']:.3f} m"
    )
    print(
        "  Temps inter-véhicules moyen : "
        f"{report['mobility_statistics']['mean_time_headway_s']:.3f} s"
    )
    print()

    print("Distribution des voies :")
    for lane, count in lane_distribution.items():
        print(
            f"  Voie {lane:>2s} : {count} observations"
        )

    print()

    if missing_values:
        print("Valeurs manquantes détectées :")
        for column, count in missing_values.items():
            print(
                f"  {column}: {count}"
            )
    else:
        print(
            "Aucune valeur manquante détectée "
            "dans l'échantillon."
        )

    print()
    print(f"Rapport JSON : {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
