"""
Prétraitement et validation des variables ML du projet V2X.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = {
    "target_encoded",
    "target_label",
}

NON_FEATURE_COLUMNS = {
    "scenario_source",
}


@dataclass
class PreparedData:
    """Conteneur des données prêtes pour l'entraînement."""

    X_train: pd.DataFrame | np.ndarray
    X_test: pd.DataFrame | np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: Optional[StandardScaler]


def _validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """Vérifie la structure et la qualité d'un DataFrame."""

    if df.empty:
        raise ValueError(f"{name} est vide.")

    missing_targets = TARGET_COLUMNS - set(df.columns)
    if missing_targets:
        raise ValueError(
            f"{name} ne contient pas les colonnes cibles requises : "
            f"{sorted(missing_targets)}"
        )

    if df.isna().sum().sum() > 0:
        missing = df.isna().sum()
        raise ValueError(
            f"{name} contient des valeurs manquantes :\n"
            f"{missing[missing > 0]}"
        )


def _extract_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sépare les variables d'entrée et la cible encodée."""

    excluded_columns = TARGET_COLUMNS | NON_FEATURE_COLUMNS

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("Aucune variable d'entrée disponible.")

    X = df[feature_columns].copy()
    y = df["target_encoded"].to_numpy(dtype=np.int64)

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()

    if non_numeric:
        raise TypeError(
            "Les variables d'entrée suivantes ne sont pas numériques : "
            f"{non_numeric}"
        )

    return X, y


def prepare_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale: bool = False,
) -> PreparedData:
    """
    Prépare les ensembles d'entraînement et de test.

    Parameters
    ----------
    train_df:
        Jeu d'entraînement préparé.

    test_df:
        Jeu de test préparé.

    scale:
        Si True, applique StandardScaler ajusté uniquement sur le train.
    """

    _validate_dataframe(train_df, "train_df")
    _validate_dataframe(test_df, "test_df")

    X_train, y_train = _extract_features_and_target(train_df)
    X_test, y_test = _extract_features_and_target(test_df)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Les colonnes d'entrée du train et du test sont différentes."
        )

    feature_names = X_train.columns.tolist()
    scaler: Optional[StandardScaler] = None

    if scale:
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_output: pd.DataFrame | np.ndarray = X_train_scaled
        X_test_output: pd.DataFrame | np.ndarray = X_test_scaled
    else:
        X_train_output = X_train
        X_test_output = X_test

    return PreparedData(
        X_train=X_train_output,
        X_test=X_test_output,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )
