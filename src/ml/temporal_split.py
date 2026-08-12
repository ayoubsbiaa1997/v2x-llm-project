"""
Partitions groupées du Dataset C pour éviter toute fuite entre scénarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


GROUP_COLUMN = "scenario_source"
TARGET_COLUMN = "decision_future"

METADATA_COLUMNS = {
    GROUP_COLUMN,
    "sequence_start_time_s",
    "sequence_end_time_s",
    "target_time_s",
    TARGET_COLUMN,
}


@dataclass
class TemporalSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_scenarios: list[str]
    test_scenarios: list[str]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retourne uniquement les colonnes historiques utilisables par le ML."""

    features = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    if not features:
        raise ValueError("Aucune variable temporelle trouvée.")

    return features


def build_group_holdout(
    df: pd.DataFrame,
    test_scenarios: list[str],
) -> TemporalSplit:
    """Construit un hold-out sans chevauchement de scénarios."""

    available = set(df[GROUP_COLUMN].unique())
    requested = set(test_scenarios)

    missing = requested - available
    if missing:
        raise ValueError(
            f"Scénarios de test absents : {sorted(missing)}"
        )

    test_mask = df[GROUP_COLUMN].isin(test_scenarios)

    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    train_groups = sorted(train_df[GROUP_COLUMN].unique())
    test_groups = sorted(test_df[GROUP_COLUMN].unique())

    overlap = set(train_groups) & set(test_groups)

    if overlap:
        raise RuntimeError(
            f"Fuite de scénarios détectée : {sorted(overlap)}"
        )

    return TemporalSplit(
        train_df=train_df,
        test_df=test_df,
        train_scenarios=train_groups,
        test_scenarios=test_groups,
    )


def iter_leave_one_scenario_out(df: pd.DataFrame):
    """Génère un fold par scénario de test."""

    scenarios = sorted(df[GROUP_COLUMN].unique())

    for scenario in scenarios:
        split = build_group_holdout(
            df=df,
            test_scenarios=[scenario],
        )

        yield scenario, split
