"""
Évaluation des modèles de classification du projet V2X.
"""

from __future__ import annotations

from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    """Résultats complets de l'évaluation d'un classifieur."""

    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    mcc: float
    confusion_matrix: np.ndarray
    classification_report: dict


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> EvaluationResult:
    """
    Calcule les métriques principales d'une classification multiclasse.
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true et y_pred doivent avoir la même forme."
        )

    if y_true.size == 0:
        raise ValueError(
            "Les vecteurs d'évaluation ne peuvent pas être vides."
        )

    labels = np.arange(len(class_names))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return EvaluationResult(
        accuracy=float(
            accuracy_score(y_true, y_pred)
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        precision_macro=float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        recall_macro=float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        f1_macro=float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        mcc=float(
            matthews_corrcoef(y_true, y_pred)
        ),
        confusion_matrix=cm,
        classification_report=report,
    )
@dataclass
class BinaryEvaluationResult:
    """Métriques complètes pour la cible binaire du Dataset D."""

    accuracy: float
    balanced_accuracy: float
    precision_positive: float
    recall_positive: float
    f1_positive: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    mcc: float
    roc_auc: float | None
    average_precision: float | None
    confusion_matrix: np.ndarray
    classification_report: dict


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> BinaryEvaluationResult:
    """
    Évalue une classification binaire.

    La classe positive est la classe 1 :
    apparition d'une dégradation future.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true et y_pred doivent avoir la même forme."
        )

    if y_true.size == 0:
        raise ValueError(
            "Les vecteurs d'évaluation sont vides."
        )

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError(
            "y_true doit contenir uniquement 0 et 1."
        )

    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError(
            "y_pred doit contenir uniquement 0 et 1."
        )

    roc_auc = None
    average_precision = None

    if probabilities is not None:
        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if probabilities.ndim == 2:
            if probabilities.shape[1] != 2:
                raise ValueError(
                    "Les probabilités doivent avoir deux colonnes."
                )

            positive_scores = probabilities[:, 1]

        elif probabilities.ndim == 1:
            positive_scores = probabilities

        else:
            raise ValueError(
                "Format des probabilités invalide."
            )

        if len(positive_scores) != len(y_true):
            raise ValueError(
                "Nombre de probabilités incohérent."
            )

        if np.unique(y_true).size == 2:
            roc_auc = float(
                roc_auc_score(
                    y_true,
                    positive_scores,
                )
            )

            average_precision = float(
                average_precision_score(
                    y_true,
                    positive_scores,
                )
            )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "NO_DEGRADATION",
            "FUTURE_DEGRADATION",
        ],
        output_dict=True,
        zero_division=0,
    )

    return BinaryEvaluationResult(
        accuracy=float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        precision_positive=float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        recall_positive=float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        f1_positive=float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        precision_macro=float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        recall_macro=float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        f1_macro=float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        mcc=float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        roc_auc=roc_auc,
        average_precision=average_precision,
        confusion_matrix=cm,
        classification_report=report,
    )
