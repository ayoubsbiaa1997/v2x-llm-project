"""
Prétraitement du Dataset D pour les modèles tabulaires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ml.dataset_d_split import (
    TARGET_COLUMN,
    get_dataset_d_features,
)


@dataclass
class PreparedDatasetD:
    """Données prêtes pour l'entraînement sur Dataset D."""

    X_train: pd.DataFrame | np.ndarray
    X_test: pd.DataFrame | np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: Optional[StandardScaler]


def _validate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    if train_df.empty:
        raise ValueError(
            "Le jeu d'entraînement D est vide."
        )

    if test_df.empty:
        raise ValueError(
            "Le jeu de test D est vide."
        )

    for name, frame in [
        ("train", train_df),
        ("test", test_df),
    ]:
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(
                f"La cible {TARGET_COLUMN!r} "
                f"est absente du {name}."
            )

        if frame[TARGET_COLUMN].isna().any():
            raise ValueError(
                f"La cible contient des valeurs "
                f"manquantes dans le {name}."
            )

        if set(
            frame[TARGET_COLUMN].unique()
        ) != {0, 1}:
            raise ValueError(
                f"Le {name} doit contenir "
                "les classes 0 et 1."
            )


def prepare_dataset_d(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale: bool = False,
) -> PreparedDatasetD:
    """
    Prépare le Dataset D.

    Le scaler est ajusté uniquement sur le train afin d'éviter
    toute fuite statistique provenant du jeu de test.
    """

    _validate_split(
        train_df=train_df,
        test_df=test_df,
    )

    feature_names = get_dataset_d_features(
        train_df
    )

    test_features = get_dataset_d_features(
        test_df
    )

    if feature_names != test_features:
        raise ValueError(
            "Les variables du train et du test diffèrent."
        )

    X_train_df = (
        train_df[feature_names]
        .copy()
        .astype(float)
    )

    X_test_df = (
        test_df[feature_names]
        .copy()
        .astype(float)
    )

    if X_train_df.isna().any().any():
        raise ValueError(
            "X_train contient des valeurs manquantes."
        )

    if X_test_df.isna().any().any():
        raise ValueError(
            "X_test contient des valeurs manquantes."
        )

    if not np.isfinite(
        X_train_df.to_numpy()
    ).all():
        raise ValueError(
            "X_train contient des valeurs non finies."
        )

    if not np.isfinite(
        X_test_df.to_numpy()
    ).all():
        raise ValueError(
            "X_test contient des valeurs non finies."
        )

    y_train = (
        train_df[TARGET_COLUMN]
        .to_numpy(dtype=np.int64)
    )

    y_test = (
        test_df[TARGET_COLUMN]
        .to_numpy(dtype=np.int64)
    )

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

    return PreparedDatasetD(
        X_train=X_train_output,
        X_test=X_test_output,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )
