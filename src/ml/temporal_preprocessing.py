"""
Prétraitement du Dataset C temporel pour les modèles tabulaires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.ml.temporal_split import (
    TARGET_COLUMN,
    get_feature_columns,
)


@dataclass
class PreparedTemporalData:
    """Données temporelles préparées pour un modèle tabulaire."""

    X_train: pd.DataFrame | np.ndarray
    X_test: pd.DataFrame | np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    label_encoder: LabelEncoder
    scaler: Optional[StandardScaler]


def _validate_temporal_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Vérifie les DataFrames avant prétraitement."""

    if train_df.empty:
        raise ValueError("Le jeu d'entraînement temporel est vide.")

    if test_df.empty:
        raise ValueError("Le jeu de test temporel est vide.")

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"La cible {TARGET_COLUMN!r} est absente du train."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"La cible {TARGET_COLUMN!r} est absente du test."
        )

    if train_df.isna().sum().sum() > 0:
        raise ValueError(
            "Le train temporel contient des valeurs manquantes."
        )

    if test_df.isna().sum().sum() > 0:
        raise ValueError(
            "Le test temporel contient des valeurs manquantes."
        )


def prepare_temporal_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale: bool = False,
) -> PreparedTemporalData:
    """
    Prépare un split temporel pour un modèle tabulaire.

    Le LabelEncoder est ajusté uniquement sur les classes du train.
    Une erreur est levée si le test contient une classe inconnue.
    """

    _validate_temporal_split(
        train_df=train_df,
        test_df=test_df,
    )

    feature_names = get_feature_columns(train_df)

    test_feature_names = get_feature_columns(test_df)

    if feature_names != test_feature_names:
        raise ValueError(
            "Les colonnes historiques du train et du test diffèrent."
        )

    X_train_df = train_df[feature_names].copy()
    X_test_df = test_df[feature_names].copy()

    non_numeric_train = (
        X_train_df
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    non_numeric_test = (
        X_test_df
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    if non_numeric_train or non_numeric_test:
        raise TypeError(
            "Variables non numériques détectées. "
            f"Train={non_numeric_train}, Test={non_numeric_test}"
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[TARGET_COLUMN])

    unknown_test_classes = (
        set(test_df[TARGET_COLUMN].unique())
        - set(label_encoder.classes_)
    )

    if unknown_test_classes:
        raise ValueError(
            "Le test contient des classes absentes du train : "
            f"{sorted(unknown_test_classes)}"
        )

    y_train = label_encoder.transform(
        train_df[TARGET_COLUMN]
    ).astype(np.int64)

    y_test = label_encoder.transform(
        test_df[TARGET_COLUMN]
    ).astype(np.int64)

    scaler: Optional[StandardScaler] = None

    if scale:
        scaler = StandardScaler()

        X_train_output = scaler.fit_transform(
            X_train_df
        )

        X_test_output = scaler.transform(
            X_test_df
        )
    else:
        X_train_output = X_train_df
        X_test_output = X_test_df

    return PreparedTemporalData(
        X_train=X_train_output,
        X_test=X_test_output,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        label_encoder=label_encoder,
        scaler=scaler,
    )
