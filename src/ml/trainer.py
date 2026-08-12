"""
Entraînement générique des modèles de classification du projet V2X.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass
class TrainingResult:
    """Résultats bruts produits après entraînement et prédiction."""

    model: Any
    predictions: np.ndarray
    probabilities: np.ndarray | None
    training_time_s: float
    prediction_time_s: float
    n_train_samples: int
    n_test_samples: int
def validate_training_data(
    X_train: Any,
    X_test: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Vérifie la cohérence des jeux d'entraînement et de test."""

    if len(X_train) == 0:
        raise ValueError("X_train est vide.")

    if len(X_test) == 0:
        raise ValueError("X_test est vide.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train et y_train n'ont pas le même nombre d'observations."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test et y_test n'ont pas le même nombre d'observations."
        )

    if np.unique(y_train).size < 2:
        raise ValueError(
            "y_train doit contenir au moins deux classes."
        )
def train_and_predict(
    model: Any,
    X_train: Any,
    X_test: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> TrainingResult:
    """
    Entraîne un modèle, génère les prédictions et mesure les temps.
    """

    validate_training_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    start_training = perf_counter()
    model.fit(X_train, y_train)
    training_time_s = perf_counter() - start_training

    start_prediction = perf_counter()
    predictions = model.predict(X_test)
    prediction_time_s = perf_counter() - start_prediction

    predictions = np.asarray(predictions, dtype=np.int64)

    if len(predictions) != len(y_test):
        raise RuntimeError(
            "Le nombre de prédictions ne correspond pas à la taille du test."
        )

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(X_test),
            dtype=float,
        )

    return TrainingResult(
        model=model,
        predictions=predictions,
        probabilities=probabilities,
        training_time_s=float(training_time_s),
        prediction_time_s=float(prediction_time_s),
        n_train_samples=int(len(X_train)),
        n_test_samples=int(len(X_test)),
    )
