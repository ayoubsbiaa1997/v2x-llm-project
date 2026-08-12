#!/usr/bin/env python3
"""
Benchmark D1 sur un hold-out strict par runs complets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.dataset_d_preprocessing import (
    prepare_dataset_d,
)
from src.ml.dataset_d_split import (
    build_group_holdout_d1,
)
from src.ml.evaluator import (
    evaluate_binary_predictions,
)
from src.ml.model_factory import (
    build_model,
    requires_scaling,
)
from src.ml.trainer import train_and_predict


DATASET_FILE = Path(
    "datasets/processed/dynamic/"
    "prediction_dataset_d.csv"
)

RESULT_DIR = Path(
    "results/ml/dataset_d/d1"
)

MODELS = [
    "decision_tree",
    "random_forest",
    "xgboost",
    "mlp",
]

TEST_RUNS = [
    "dyn_u8_d100_s101",
    "dyn_u8_d200_s202",
    "dyn_u8_d300_s303",
    "dyn_u12_d100_s202",
    "dyn_u12_d200_s303",
    "dyn_u12_d300_s404",
    "dyn_u16_d100_s303",
    "dyn_u16_d200_s404",
    "dyn_u16_d300_s101",
]


def main() -> None:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset D introuvable : {DATASET_FILE}"
        )

    dataset = pd.read_csv(
        DATASET_FILE
    )

    split = build_group_holdout_d1(
        df=dataset,
        test_runs=TEST_RUNS,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("=" * 80)
    print("BENCHMARK DATASET D — PROTOCOLE D1")
    print("=" * 80)

    print("Train runs :", len(split.train_runs))
    print("Test runs  :", len(split.test_runs))
    print("Train rows :", len(split.train_df))
    print("Test rows  :", len(split.test_df))

    for model_name in MODELS:
        print("\n" + "=" * 80)
        print("Modèle :", model_name)
        print("=" * 80)

        scale = requires_scaling(
            model_name
        )

        prepared = prepare_dataset_d(
            train_df=split.train_df,
            test_df=split.test_df,
            scale=scale,
        )

        model = build_model(
            model_name
        )

        training = train_and_predict(
            model=model,
            X_train=prepared.X_train,
            X_test=prepared.X_test,
            y_train=prepared.y_train,
            y_test=prepared.y_test,
        )

        evaluation = evaluate_binary_predictions(
            y_true=prepared.y_test,
            y_pred=training.predictions,
            probabilities=training.probabilities,
        )

        model_dir = (
            RESULT_DIR / model_name
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions = pd.DataFrame({
            "run_id": split.test_df[
                "run_id"
            ].to_numpy(),
            "history_end_time_s": (
                split.test_df[
                    "history_end_time_s"
                ].to_numpy()
            ),
            "y_true": prepared.y_test,
            "y_pred": training.predictions,
        })

        if training.probabilities is not None:
            predictions[
                "probability_positive"
            ] = training.probabilities[:, 1]

        predictions.to_csv(
            model_dir / "predictions.csv",
            index=False,
        )

        metrics = {
            "protocol": "D1",
            "model": model_name,
            "train_runs": split.train_runs,
            "test_runs": split.test_runs,
            "n_train_samples": (
                training.n_train_samples
            ),
            "n_test_samples": (
                training.n_test_samples
            ),
            "n_features": len(
                prepared.feature_names
            ),
            "scaling": scale,
            "accuracy": evaluation.accuracy,
            "balanced_accuracy": (
                evaluation.balanced_accuracy
            ),
            "precision_positive": (
                evaluation.precision_positive
            ),
            "recall_positive": (
                evaluation.recall_positive
            ),
            "f1_positive": (
                evaluation.f1_positive
            ),
            "precision_macro": (
                evaluation.precision_macro
            ),
            "recall_macro": (
                evaluation.recall_macro
            ),
            "f1_macro": evaluation.f1_macro,
            "mcc": evaluation.mcc,
            "roc_auc": evaluation.roc_auc,
            "average_precision": (
                evaluation.average_precision
            ),
            "training_time_s": (
                training.training_time_s
            ),
            "prediction_time_s": (
                training.prediction_time_s
            ),
            "confusion_matrix": (
                evaluation.confusion_matrix
                .tolist()
            ),
        }

        with open(
            model_dir / "metrics.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=4,
                ensure_ascii=False,
            )

        rows.append({
            "model": model_name,
            "accuracy": evaluation.accuracy,
            "balanced_accuracy": (
                evaluation.balanced_accuracy
            ),
            "precision_positive": (
                evaluation.precision_positive
            ),
            "recall_positive": (
                evaluation.recall_positive
            ),
            "f1_positive": (
                evaluation.f1_positive
            ),
            "f1_macro": evaluation.f1_macro,
            "mcc": evaluation.mcc,
            "roc_auc": evaluation.roc_auc,
            "average_precision": (
                evaluation.average_precision
            ),
            "training_time_s": (
                training.training_time_s
            ),
            "prediction_time_s": (
                training.prediction_time_s
            ),
        })

        print(
            "Accuracy          :",
            round(evaluation.accuracy, 4),
        )
        print(
            "Balanced accuracy :",
            round(
                evaluation.balanced_accuracy,
                4,
            ),
        )
        print(
            "Precision positive:",
            round(
                evaluation.precision_positive,
                4,
            ),
        )
        print(
            "Recall positive   :",
            round(
                evaluation.recall_positive,
                4,
            ),
        )
        print(
            "F1 positive       :",
            round(
                evaluation.f1_positive,
                4,
            ),
        )
        print(
            "F1 macro          :",
            round(evaluation.f1_macro, 4),
        )
        print(
            "MCC               :",
            round(evaluation.mcc, 4),
        )
        print(
            "ROC-AUC           :",
            None
            if evaluation.roc_auc is None
            else round(
                evaluation.roc_auc,
                4,
            ),
        )
        print(
            "Average precision :",
            None
            if evaluation.average_precision
            is None
            else round(
                evaluation.average_precision,
                4,
            ),
        )
        print("Matrice de confusion :")
        print(evaluation.confusion_matrix)

    comparison = (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "f1_positive",
                "f1_macro",
                "balanced_accuracy",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    comparison.to_csv(
        RESULT_DIR / "comparison.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("COMPARAISON D1")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nRésultats sauvegardés dans :",
        RESULT_DIR,
    )


if __name__ == "__main__":
    main()
