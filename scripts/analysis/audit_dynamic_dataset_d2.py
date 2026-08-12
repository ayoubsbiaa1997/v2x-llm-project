#!/usr/bin/env python3

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration des chemins
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "datasets"
    / "processed"
    / "dynamic_d2"
    / "dynamic_qos_merged_d2.csv"
)

ANALYSIS_DIR = (
    PROJECT_DIR
    / "datasets"
    / "analysis"
    / "dynamic_d2"
)

FIGURES_DIR = ANALYSIS_DIR / "figures"
DISTRIBUTIONS_DIR = FIGURES_DIR / "distributions"
PROFILES_DIR = FIGURES_DIR / "profiles"

for directory in [
    ANALYSIS_DIR,
    FIGURES_DIR,
    DISTRIBUTIONS_DIR,
    PROFILES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Paramètres d’analyse
# ============================================================

QOS_COLUMNS = [
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
]

PACKET_COLUMNS = [
    "selected_flow_count",
    "tx_packets_window",
    "rx_packets_window",
    "rx_bytes_window",
]

PROFILE_ORDER = [
    "STABLE_CONTROL",
    "MEDIUM_CONTROL",
    "EARLY_DEGRADATION",
    "LATE_DEGRADATION",
    "ABRUPT_DEGRADATION",
    "PERSISTENT_HIGH",
]

PHASE_ORDER = [
    "STABLE",
    "MEDIUM",
    "HIGH",
    "RECOVERY",
]


# ============================================================
# Fonctions utilitaires
# ============================================================

def parse_run_metadata(run_id: str) -> dict:
    """
    Extrait ue_count, distance_m et seed depuis un identifiant tel que :
    d2_early_degradation_u16_d300_s222
    """

    pattern = re.compile(
        r"_u(?P<ue_count>\d+)"
        r"_d(?P<distance_m>\d+)"
        r"_s(?P<seed>\d+)$"
    )

    match = pattern.search(str(run_id))

    if match is None:
        return {
            "ue_count": np.nan,
            "distance_m": np.nan,
            "seed": np.nan,
        }

    return {
        "ue_count": int(match.group("ue_count")),
        "distance_m": int(match.group("distance_m")),
        "seed": int(match.group("seed")),
    }


def safe_percentage(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan

    return 100.0 * numerator / denominator


def save_line_plot(
    dataframe: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """
    Trace la moyenne temporelle par profil.
    L’écart-type est ajouté sous forme de bande.
    """

    figure, axis = plt.subplots(figsize=(11, 6))

    for profile in PROFILE_ORDER:
        profile_data = dataframe[
            dataframe["profile"] == profile
        ]

        if profile_data.empty:
            continue

        temporal_stats = (
            profile_data
            .groupby("time_s", as_index=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = temporal_stats["time_s"].to_numpy()
        mean_values = temporal_stats["mean"].to_numpy()
        std_values = (
            temporal_stats["std"]
            .fillna(0.0)
            .to_numpy()
        )

        axis.plot(
            x,
            mean_values,
            label=profile,
            linewidth=1.8,
        )

        axis.fill_between(
            x,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.12,
        )

    axis.set_title(title)
    axis.set_xlabel("Temps de simulation (s)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_distribution_plot(
    dataframe: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:

    figure, axis = plt.subplots(figsize=(9, 5))

    values = (
        pd.to_numeric(dataframe[metric], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if not values.empty:
        axis.hist(values, bins=35, edgecolor="black")

    axis.set_title(title)
    axis.set_xlabel(metric)
    axis.set_ylabel("Nombre d’observations")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_boxplot_by_profile(
    dataframe: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:

    values_by_profile = []
    labels = []

    for profile in PROFILE_ORDER:
        values = (
            dataframe.loc[
                dataframe["profile"] == profile,
                metric,
            ]
            .dropna()
            .to_numpy()
        )

        if len(values) > 0:
            values_by_profile.append(values)
            labels.append(profile)

    figure, axis = plt.subplots(figsize=(12, 6))

    if values_by_profile:
        axis.boxplot(
            values_by_profile,
            tick_labels=labels,
            showfliers=True,
        )

    axis.set_title(title)
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


# ============================================================
# Chargement
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Dataset chargé : {INPUT_FILE}")
print(f"Dimensions initiales : {df.shape}")


# ============================================================
# Vérification des colonnes
# ============================================================

required_columns = [
    "time_s",
    "phase",
    "lambda_be",
    "run_id",
    "profile",
] + QOS_COLUMNS + PACKET_COLUMNS

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes manquantes : "
        + ", ".join(missing_columns)
    )


# ============================================================
# Nettoyage des types
# ============================================================

numeric_columns = [
    "time_s",
    "lambda_be",
] + QOS_COLUMNS + PACKET_COLUMNS

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df["phase"] = (
    df["phase"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["profile"] = (
    df["profile"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["run_id"] = df["run_id"].astype(str)


# ============================================================
# Extraction des variables statiques depuis run_id
# ============================================================

metadata = pd.DataFrame(
    [
        parse_run_metadata(run_id)
        for run_id in df["run_id"]
    ]
)

df = pd.concat(
    [
        df.reset_index(drop=True),
        metadata.reset_index(drop=True),
    ],
    axis=1,
)

df["packet_size_be"] = 1200


# ============================================================
# Variables de contrôle supplémentaires
# ============================================================

df["calculated_pdr"] = np.where(
    df["tx_packets_window"] > 0,
    100.0
    * df["rx_packets_window"]
    / df["tx_packets_window"],
    np.nan,
)

df["calculated_packet_loss"] = (
    100.0 - df["calculated_pdr"]
)

df["pdr_absolute_error"] = (
    df["pdr"] - df["calculated_pdr"]
).abs()

df["packet_loss_absolute_error"] = (
    df["packet_loss"]
    - df["calculated_packet_loss"]
).abs()

df["expected_throughput_mbps"] = (
    df["rx_bytes_window"]
    * 8.0
    / (0.5 * 1_000_000.0)
)

df["throughput_absolute_error"] = (
    df["throughput_mbps"]
    - df["expected_throughput_mbps"]
).abs()

df["sequence_index"] = (
    df
    .sort_values(["run_id", "time_s"])
    .groupby("run_id")
    .cumcount()
)


# ============================================================
# Sauvegarde du dataset enrichi pour audit
# ============================================================

enriched_file = (
    ANALYSIS_DIR
    / "dynamic_qos_merged_d2_audited.csv"
)

df.to_csv(enriched_file, index=False)


# ============================================================
# Statistiques générales
# ============================================================

summary_columns = (
    numeric_columns
    + [
        "ue_count",
        "distance_m",
        "seed",
        "packet_size_be",
        "calculated_pdr",
        "pdr_absolute_error",
        "throughput_absolute_error",
    ]
)

statistics = (
    df[summary_columns]
    .describe()
    .transpose()
)

statistics["missing_count"] = (
    df[summary_columns]
    .isna()
    .sum()
)

statistics["missing_percent"] = (
    100.0
    * statistics["missing_count"]
    / len(df)
)

statistics.to_csv(
    ANALYSIS_DIR / "audit_statistics.csv"
)


# ============================================================
# Valeurs manquantes
# ============================================================

missing_values = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": (
        100.0 * df.isna().sum().values / len(df)
    ),
})

missing_values.to_csv(
    ANALYSIS_DIR / "missing_values.csv",
    index=False,
)


# ============================================================
# Doublons
# ============================================================

duplicate_mask = df.duplicated(
    subset=["run_id", "time_s"],
    keep=False,
)

duplicates = df.loc[duplicate_mask].copy()

duplicates.to_csv(
    ANALYSIS_DIR / "duplicate_rows.csv",
    index=False,
)


# ============================================================
# Lignes physiquement incohérentes
# ============================================================

invalid_conditions = (
    (df["pdr"] < 0)
    | (df["pdr"] > 100)
    | (df["packet_loss"] < 0)
    | (df["packet_loss"] > 100)
    | (df["throughput_mbps"] < 0)
    | (df["delay_ms"] < 0)
    | (df["jitter_ms"] < 0)
    | (df["congestion_index"] < 0)
    | (df["congestion_index"] > 1)
    | (
        df["rx_packets_window"]
        > df["tx_packets_window"]
    )
)

invalid_rows = df.loc[invalid_conditions].copy()

invalid_rows.to_csv(
    ANALYSIS_DIR / "invalid_physical_values.csv",
    index=False,
)


# ============================================================
# Contrôle des formules PDR / loss / throughput
# ============================================================

calculation_errors = df.loc[
    (df["pdr_absolute_error"] > 1e-6)
    | (df["packet_loss_absolute_error"] > 1e-6)
    | (df["throughput_absolute_error"] > 1e-6)
].copy()

calculation_errors.to_csv(
    ANALYSIS_DIR / "metric_calculation_errors.csv",
    index=False,
)


# ============================================================
# Nombre de lignes et runs par profil
# ============================================================

profile_counts = (
    df
    .groupby("profile")
    .agg(
        observation_count=("run_id", "size"),
        unique_runs=("run_id", "nunique"),
        mean_samples_per_run=("run_id", lambda x: len(x) / x.nunique()),
    )
    .reindex(PROFILE_ORDER)
)

profile_counts.to_csv(
    ANALYSIS_DIR / "profile_counts.csv"
)


# ============================================================
# Statistiques par profil
# ============================================================

profile_statistics = (
    df
    .groupby("profile")[QOS_COLUMNS]
    .agg(["count", "mean", "std", "min", "median", "max"])
)

profile_statistics.to_csv(
    ANALYSIS_DIR / "statistics_by_profile.csv"
)


# ============================================================
# Statistiques par profil et phase
# ============================================================

profile_phase_statistics = (
    df
    .groupby(["profile", "phase"])[
        ["lambda_be"] + QOS_COLUMNS
    ]
    .agg(["count", "mean", "std", "min", "median", "max"])
)

profile_phase_statistics.to_csv(
    ANALYSIS_DIR / "statistics_by_profile_phase.csv"
)


# ============================================================
# Statistiques par UE et distance
# ============================================================

scenario_statistics = (
    df
    .groupby(
        [
            "profile",
            "ue_count",
            "distance_m",
        ]
    )[QOS_COLUMNS]
    .agg(["mean", "std", "min", "max"])
)

scenario_statistics.to_csv(
    ANALYSIS_DIR / "statistics_by_scenario.csv"
)


# ============================================================
# Corrélations
# ============================================================

correlation_columns = [
    "lambda_be",
    "ue_count",
    "distance_m",
    "packet_size_be",
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
]

correlation_matrix = (
    df[correlation_columns]
    .corr(method="pearson")
)

correlation_matrix.to_csv(
    ANALYSIS_DIR / "correlation_matrix_pearson.csv"
)

spearman_matrix = (
    df[correlation_columns]
    .corr(method="spearman")
)

spearman_matrix.to_csv(
    ANALYSIS_DIR / "correlation_matrix_spearman.csv"
)


# ============================================================
# Valeurs aberrantes avec méthode IQR
# ============================================================

outlier_frames = []

for metric in QOS_COLUMNS:
    q1 = df[metric].quantile(0.25)
    q3 = df[metric].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    metric_outliers = df.loc[
        (df[metric] < lower_bound)
        | (df[metric] > upper_bound)
    ].copy()

    if metric_outliers.empty:
        continue

    metric_outliers["outlier_metric"] = metric
    metric_outliers["lower_bound"] = lower_bound
    metric_outliers["upper_bound"] = upper_bound

    outlier_frames.append(metric_outliers)

if outlier_frames:
    outliers = pd.concat(
        outlier_frames,
        ignore_index=True,
    )
else:
    outliers = pd.DataFrame()

outliers.to_csv(
    ANALYSIS_DIR / "outliers_iqr.csv",
    index=False,
)


# ============================================================
# Variabilité temporelle par run
# ============================================================

run_variability = (
    df
    .groupby("run_id")
    .agg(
        profile=("profile", "first"),
        ue_count=("ue_count", "first"),
        distance_m=("distance_m", "first"),
        seed=("seed", "first"),
        sample_count=("time_s", "size"),
        time_min=("time_s", "min"),
        time_max=("time_s", "max"),
        lambda_unique=("lambda_be", "nunique"),
        throughput_std=("throughput_mbps", "std"),
        delay_std=("delay_ms", "std"),
        jitter_std=("jitter_ms", "std"),
        loss_std=("packet_loss", "std"),
        pdr_std=("pdr", "std"),
        congestion_std=("congestion_index", "std"),
    )
    .reset_index()
)

run_variability.to_csv(
    ANALYSIS_DIR / "temporal_variability_by_run.csv",
    index=False,
)


# ============================================================
# Vérification des phases
# ============================================================

phase_counts = (
    df
    .groupby(["profile", "phase"])
    .size()
    .rename("observation_count")
    .reset_index()
)

phase_counts.to_csv(
    ANALYSIS_DIR / "phase_counts.csv",
    index=False,
)

phase_sequences = (
    df
    .sort_values(["run_id", "time_s"])
    .groupby("run_id")
    .agg(
        profile=("profile", "first"),
        phase_sequence=(
            "phase",
            lambda values: " -> ".join(
                pd.Series(values)
                .drop_duplicates()
                .astype(str)
            ),
        ),
        lambda_sequence=(
            "lambda_be",
            lambda values: " -> ".join(
                pd.Series(values)
                .drop_duplicates()
                .astype(str)
            ),
        ),
    )
    .reset_index()
)

phase_sequences.to_csv(
    ANALYSIS_DIR / "phase_sequences_by_run.csv",
    index=False,
)


# ============================================================
# Figures temporelles par profil
# ============================================================

plot_labels = {
    "throughput_mbps": "Débit (Mbit/s)",
    "delay_ms": "Délai moyen (ms)",
    "jitter_ms": "Gigue moyenne (ms)",
    "packet_loss": "Pertes de paquets (%)",
    "pdr": "PDR (%)",
    "congestion_index": "Indice de congestion",
}

for metric in QOS_COLUMNS:
    save_line_plot(
        dataframe=df,
        metric=metric,
        output_path=(
            PROFILES_DIR
            / f"{metric}_temporal_profiles.png"
        ),
        title=(
            f"Évolution temporelle de {plot_labels[metric]} "
            "selon le profil de trafic"
        ),
        ylabel=plot_labels[metric],
    )

    save_distribution_plot(
        dataframe=df,
        metric=metric,
        output_path=(
            DISTRIBUTIONS_DIR
            / f"{metric}_distribution.png"
        ),
        title=f"Distribution globale de {plot_labels[metric]}",
    )

    save_boxplot_by_profile(
        dataframe=df,
        metric=metric,
        output_path=(
            PROFILES_DIR
            / f"{metric}_boxplot_by_profile.png"
        ),
        title=(
            f"Comparaison de {plot_labels[metric]} "
            "selon le profil"
        ),
    )


# ============================================================
# Heatmap de corrélation sans seaborn
# ============================================================

figure, axis = plt.subplots(figsize=(11, 9))

image = axis.imshow(
    correlation_matrix.to_numpy(),
    aspect="auto",
    vmin=-1,
    vmax=1,
)

axis.set_xticks(
    range(len(correlation_matrix.columns))
)

axis.set_xticklabels(
    correlation_matrix.columns,
    rotation=45,
    ha="right",
)

axis.set_yticks(
    range(len(correlation_matrix.index))
)

axis.set_yticklabels(
    correlation_matrix.index
)

for row in range(len(correlation_matrix.index)):
    for column in range(len(correlation_matrix.columns)):
        value = correlation_matrix.iloc[row, column]

        axis.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

figure.colorbar(
    image,
    ax=axis,
    label="Coefficient de corrélation de Pearson",
)

axis.set_title("Matrice de corrélation des variables du dataset D2")
figure.tight_layout()
figure.savefig(
    FIGURES_DIR / "correlation_matrix.png",
    dpi=300,
)
plt.close(figure)


# ============================================================
# Rapport textuel
# ============================================================

report_file = ANALYSIS_DIR / "audit_report.txt"

with report_file.open(
    "w",
    encoding="utf-8",
) as report:

    report.write("AUDIT DU DATASET D2\n")
    report.write("=" * 70 + "\n\n")

    report.write(f"Fichier analysé : {INPUT_FILE}\n")
    report.write(f"Nombre de lignes : {len(df)}\n")
    report.write(f"Nombre de colonnes : {len(df.columns)}\n")
    report.write(
        f"Nombre de runs uniques : {df['run_id'].nunique()}\n"
    )
    report.write(
        f"Nombre de profils : {df['profile'].nunique()}\n\n"
    )

    report.write("RÉPARTITION PAR PROFIL\n")
    report.write("-" * 70 + "\n")
    report.write(profile_counts.to_string())
    report.write("\n\n")

    report.write("VALEURS MANQUANTES\n")
    report.write("-" * 70 + "\n")
    report.write(
        missing_values[
            missing_values["missing_count"] > 0
        ].to_string(index=False)
    )
    report.write("\n\n")

    report.write("CONTRÔLES DE COHÉRENCE\n")
    report.write("-" * 70 + "\n")
    report.write(
        f"Lignes dupliquées run_id/time_s : "
        f"{len(duplicates)}\n"
    )
    report.write(
        f"Lignes avec valeurs physiques invalides : "
        f"{len(invalid_rows)}\n"
    )
    report.write(
        f"Lignes avec erreurs de calcul QoS : "
        f"{len(calculation_errors)}\n"
    )
    report.write(
        f"Nombre de valeurs aberrantes IQR : "
        f"{len(outliers)}\n\n"
    )

    report.write("PLAGES OBSERVÉES\n")
    report.write("-" * 70 + "\n")

    for metric in QOS_COLUMNS:
        report.write(
            f"{metric}: "
            f"min={df[metric].min():.6f}, "
            f"moyenne={df[metric].mean():.6f}, "
            f"max={df[metric].max():.6f}\n"
        )

    report.write("\n")

    report.write("VARIABILITÉ TEMPORELLE\n")
    report.write("-" * 70 + "\n")

    constant_throughput_runs = run_variability[
        run_variability["throughput_std"].fillna(0) < 1e-12
    ]

    constant_congestion_runs = run_variability[
        run_variability["congestion_std"].fillna(0) < 1e-12
    ]

    report.write(
        "Runs dont le throughput est constant : "
        f"{len(constant_throughput_runs)} / "
        f"{len(run_variability)}\n"
    )

    report.write(
        "Runs dont l’indice de congestion est constant : "
        f"{len(constant_congestion_runs)} / "
        f"{len(run_variability)}\n\n"
    )

    report.write("SÉQUENCES DE PHASES\n")
    report.write("-" * 70 + "\n")
    report.write(
        phase_sequences[
            [
                "run_id",
                "profile",
                "phase_sequence",
                "lambda_sequence",
            ]
        ].to_string(index=False)
    )

    report.write("\n\n")

    report.write("INTERPRÉTATION AUTOMATIQUE\n")
    report.write("-" * 70 + "\n")

    if len(invalid_rows) == 0:
        report.write(
            "[OK] Aucune valeur QoS physiquement impossible "
            "n’a été détectée.\n"
        )
    else:
        report.write(
            "[ATTENTION] Certaines valeurs QoS sont hors des "
            "plages physiques attendues.\n"
        )

    if len(calculation_errors) == 0:
        report.write(
            "[OK] Les calculs du PDR, des pertes et du débit "
            "sont cohérents avec les compteurs par fenêtre.\n"
        )
    else:
        report.write(
            "[ATTENTION] Des écarts existent entre les métriques "
            "et les compteurs par fenêtre.\n"
        )

    if df["run_id"].nunique() == 48:
        report.write(
            "[OK] Les 48 runs attendus sont présents.\n"
        )
    else:
        report.write(
            "[ATTENTION] Le nombre de runs diffère des "
            "48 simulations attendues.\n"
        )

    if (
        df.groupby("profile")["run_id"].nunique()
        .reindex(PROFILE_ORDER)
        .fillna(0)
        .eq(8)
        .all()
    ):
        report.write(
            "[OK] Chaque profil contient les 8 configurations "
            "attendues.\n"
        )
    else:
        report.write(
            "[ATTENTION] Au moins un profil ne contient pas "
            "8 runs.\n"
        )


# ============================================================
# Résumé terminal
# ============================================================

print()
print("=" * 70)
print("AUDIT D2 TERMINÉ")
print("=" * 70)
print(f"Dataset enrichi : {enriched_file}")
print(f"Rapport          : {report_file}")
print(f"Figures          : {FIGURES_DIR}")
print()
print(f"Lignes                    : {len(df)}")
print(f"Runs uniques              : {df['run_id'].nunique()}")
print(f"Profils                   : {df['profile'].nunique()}")
print(f"Doublons run_id/time_s    : {len(duplicates)}")
print(f"Valeurs physiques invalides : {len(invalid_rows)}")
print(f"Erreurs de calcul QoS     : {len(calculation_errors)}")
print(f"Valeurs aberrantes IQR    : {len(outliers)}")
print("=" * 70)
