#!/usr/bin/env python3

from pathlib import Path
import json
import time

import pandas as pd

from src.ml.dataset_loader import load_dataset
from src.ml.evaluator import evaluate_predictions
from src.ml.model_factory import (
    build_model,
    list_models,
    requires_scaling,
)
from src.ml.temporal_preprocessing import (
    prepare_temporal_train_test,
)
from src.ml.temporal_split import build_group_holdout
from src.ml.trainer import train_and_predict


TEST_SCENARIOS = [
    "qos_D100_STABLE",
    "qos_D250_MEDIUM",
    "qos_D400_HIGH",
]

MODELS_TO_RUN = [
    "decision_tree",
    "random_forest",
    "xgboost",
    "mlp",
]

RESULT_DIR = Path("results/ml/temporal_c1")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    dataset = load_dataset("C")

    split = build_group_holdout(
        dataset,
        test_scenarios=TEST_SCENARIOS,
    )

    all_results = []

    for model_name in MODELS_TO_RUN:
        print("\n" + "=" * 70)
        print("Modèle :", model_name)
        print("=" * 70)

        scale = requires_scaling(model_name)

        prepared = prepare_temporal_train_test(
            train_df=split.train_df,
            test_df=split.test_df,
            scale=scale,
        )

        model = build_model(model_name)

        training_result = train_and_predict(
            model=model,
            X_train=prepared.X_train,
            X_test=prepared.X_test,
            y_train=prepared.y_train,
            y_test=prepared.y_test,
        )

        evaluation = evaluate_predictions(
            y_true=prepared.y_test,
            y_pred=training_result.predictions,
            class_names=prepared.label_encoder.classes_.tolist(),
        )

        model_dir = RESULT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        prediction_df = pd.DataFrame({
            "y_true": prepared.y_test,
            "y_pred": training_result.predictions,
            "true_label": prepared.label_encoder.inverse_transform(
                prepared.y_test
            ),
            "predicted_label": prepared.label_encoder.inverse_transform(
                training_result.predictions
            ),
        })

        prediction_df.to_csv(
            model_dir / "predictions.csv",
            index=False,
        )

        metrics = {
            "protocol": "C1",
            "model": model_name,
            "test_scenarios": TEST_SCENARIOS,
            "n_train_samples": training_result.n_train_samples,
            "n_test_samples": training_result.n_test_samples,
            "n_features": len(prepared.feature_names),
            "scaling": scale,
            "accuracy": evaluation.accuracy,
            "balanced_accuracy": evaluation.balanced_accuracy,
            "precision_macro": evaluation.precision_macro,
            "recall_macro": evaluation.recall_macro,
            "f1_macro": evaluation.f1_macro,
            "mcc": evaluation.mcc,
            "training_time_s": training_result.training_time_s,
            "prediction_time_s": training_result.prediction_time_s,
            "confusion_matrix": evaluation.confusion_matrix.tolist(),
            "class_names": prepared.label_encoder.classes_.tolist(),
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

        all_results.append({
            "model": model_name,
            "accuracy": evaluation.accuracy,
            "balanced_accuracy": evaluation.balanced_accuracy,
            "precision_macro": evaluation.precision_macro,
            "recall_macro": evaluation.recall_macro,
            "f1_macro": evaluation.f1_macro,
            "mcc": evaluation.mcc,
            "training_time_s": training_result.training_time_s,
            "prediction_time_s": training_result.prediction_time_s,
            "scaling": scale,
        })

        print("Accuracy          :", round(evaluation.accuracy, 4))
        print(
            "Balanced accuracy :",
            round(evaluation.balanced_accuracy, 4),
        )
        print(
            "Precision macro   :",
            round(evaluation.precision_macro, 4),
        )
        print(
            "Recall macro      :",
            round(evaluation.recall_macro, 4),
        )
        print(
            "F1 macro          :",
            round(evaluation.f1_macro, 4),
        )
        print("MCC               :", round(evaluation.mcc, 4))
        print(
            "Training time     :",
            round(training_result.training_time_s, 4),
            "s",
        )
        print(
            "Prediction time   :",
            round(training_result.prediction_time_s, 4),
            "s",
        )
        print("Confusion matrix:")
        print(evaluation.confusion_matrix)

    comparison = pd.DataFrame(all_results).sort_values(
        by="f1_macro",
        ascending=False,
    )

    comparison.to_csv(
        RESULT_DIR / "comparison.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("COMPARAISON C1")
    print("=" * 70)
    print(comparison.to_string(index=False))

    print(
        "\nRésultats sauvegardés dans :",
        RESULT_DIR,
    )


if __name__ == "__main__":
    main()
