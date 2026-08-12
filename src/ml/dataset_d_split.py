"""
Découpage strict du Dataset D par runs complets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


GROUP_COLUMN = "run_id"
TARGET_COLUMN = "future_degradation"


@dataclass
class DatasetDSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_runs: list[str]
    test_runs: list[str]


def get_dataset_d_features(
    df: pd.DataFrame,
) -> list[str]:
    """Retourne uniquement les variables historiques laggées."""

    features = [
        column
        for column in df.columns
        if "_lag_" in column
    ]

    if not features:
        raise ValueError(
            "Aucune variable historique trouvée."
        )

    forbidden_tokens = [
        "future",
        "phase",
        "lambda",
        "run",
        "seed",
        "time",
        "audit",
        "target",
        "degradation",
    ]

    forbidden = [
        column
        for column in features
        if any(
            token in column.lower()
            for token in forbidden_tokens
        )
    ]

    if forbidden:
        raise ValueError(
            f"Variables interdites détectées : {forbidden}"
        )

    return features


def validate_dataset_d(
    df: pd.DataFrame,
) -> None:
    required = {
        GROUP_COLUMN,
        TARGET_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes absentes : {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Dataset D vide."
        )

    if df[GROUP_COLUMN].isna().any():
        raise ValueError(
            "run_id contient des valeurs manquantes."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "La cible contient des valeurs manquantes."
        )

    if set(df[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError(
            "La cible doit contenir exactement les classes 0 et 1."
        )

    features = get_dataset_d_features(df)

    if df[features].isna().any().any():
        raise ValueError(
            "Les variables ML contiennent des valeurs manquantes."
        )


def build_group_holdout_d1(
    df: pd.DataFrame,
    test_runs: list[str],
) -> DatasetDSplit:
    """Construit un hold-out strict par runs."""

    validate_dataset_d(df)

    available_runs = set(
        df[GROUP_COLUMN].unique()
    )

    requested_runs = set(test_runs)

    missing_runs = (
        requested_runs - available_runs
    )

    if missing_runs:
        raise ValueError(
            f"Runs absents : {sorted(missing_runs)}"
        )

    test_mask = df[
        GROUP_COLUMN
    ].isin(test_runs)

    train_df = df.loc[
        ~test_mask
    ].copy()

    test_df = df.loc[
        test_mask
    ].copy()

    train_runs = sorted(
        train_df[GROUP_COLUMN].unique()
    )

    test_runs_actual = sorted(
        test_df[GROUP_COLUMN].unique()
    )

    overlap = (
        set(train_runs)
        & set(test_runs_actual)
    )

    if overlap:
        raise RuntimeError(
            f"Fuite de runs : {sorted(overlap)}"
        )

    if train_df[TARGET_COLUMN].nunique() != 2:
        raise RuntimeError(
            "Le train D1 ne contient pas les deux classes."
        )

    if test_df[TARGET_COLUMN].nunique() != 2:
        raise RuntimeError(
            "Le test D1 ne contient pas les deux classes."
        )

    return DatasetDSplit(
        train_df=train_df,
        test_df=test_df,
        train_runs=train_runs,
        test_runs=test_runs_actual,
    )


def iter_group_kfold_d2(
    df: pd.DataFrame,
    n_splits: int = 6,
) -> Iterator[
    tuple[int, DatasetDSplit]
]:
    """Génère les folds GroupKFold sans chevauchement de runs."""

    validate_dataset_d(df)

    groups = df[
        GROUP_COLUMN
    ].to_numpy()

    splitter = GroupKFold(
        n_splits=n_splits
    )

    dummy_x = np.zeros(
        shape=(len(df), 1)
    )

    y = df[
        TARGET_COLUMN
    ].to_numpy()

    for fold_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            dummy_x,
            y,
            groups=groups,
        ),
        start=1,
    ):
        train_df = df.iloc[
            train_indices
        ].copy()

        test_df = df.iloc[
            test_indices
        ].copy()

        train_runs = sorted(
            train_df[
                GROUP_COLUMN
            ].unique()
        )

        test_runs = sorted(
            test_df[
                GROUP_COLUMN
            ].unique()
        )

        overlap = (
            set(train_runs)
            & set(test_runs)
        )

        if overlap:
            raise RuntimeError(
                f"Fold {fold_index}: fuite de runs."
            )

        if train_df[TARGET_COLUMN].nunique() != 2:
            raise RuntimeError(
                f"Fold {fold_index}: "
                "train avec une seule classe."
            )

        if test_df[TARGET_COLUMN].nunique() != 2:
            raise RuntimeError(
                f"Fold {fold_index}: "
                "test avec une seule classe."
            )

        yield fold_index, DatasetDSplit(
            train_df=train_df,
            test_df=test_df,
            train_runs=train_runs,
            test_runs=test_runs,
        )


def permute_training_labels(
    train_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permute uniquement les labels du train.

    Les labels du test restent inchangés.
    """

    output = train_df.copy()

    rng = np.random.default_rng(
        random_state
    )

    output[TARGET_COLUMN] = rng.permutation(
        output[TARGET_COLUMN].to_numpy()
    )

    return output
