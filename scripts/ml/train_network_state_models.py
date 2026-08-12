#!/usr/bin/env python3

"""
Entraînement de modèles ML pour prédire l'état du réseau V2X.

Entrées :
    paramètres connus avant ou pendant la configuration du scénario ns-3.

Sorties prédites :
    - QoSScore
    - CongestionScore
    - HORiskScore

Les niveaux qualitatifs et les décisions ne sont pas utilisés comme
variables d'entrée afin d'éviter toute fuite de données.

Méthode d'évaluation :
    validation croisée répétée avec prédictions out-of-fold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    RepeatedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================================
# Chemins
# ============================================================================

PROJECT_DIR = Path.home() / "v2x-llm-project"

INPUT_FILE = (
    PROJECT_DIR
    / "datasets"
    / "processed"
    / "final_dataset_v3.csv"
)

MODEL_DIR = PROJECT_DIR / "models" / "network_state_v1"

REPORT_DIR = PROJECT_DIR / "results" / "network_state_v1"

METRICS_FILE = REPORT_DIR / "model_metrics.csv"
PREDICTIONS_FILE = REPORT_DIR / "out_of_fold_predictions.csv"
SUMMARY_FILE = REPORT_DIR / "training_summary.txt"
FEATURES_FILE = REPORT_DIR / "selected_features.json"


# ============================================================================
# Configuration
# ============================================================================

RANDOM_STATE = 42

TARGETS = [
    "QoSScore",
    "CongestionScore",
    "HORiskScore",
]

# Variables candidates connues avant la sortie de la simulation.
#
# Le script ne conserve que celles réellement présentes dans le CSV.
CANDIDATE_FEATURES = [
    "gNbNum",
    "gnbNum",
    "ueNumPerGnb",
    "ueNumPergNb",
    "totalUE",
    "bsDistance",
    "scenarioLength",
    "ueSpeed",
    "lambdaULL",
    "lambdaBe",
    "packetULL",
    "packetBE",
]

# Variables explicitement interdites comme entrées.
#
# Elles sont produites après simulation ou directement dérivées des labels.
FORBIDDEN_FEATURES = {
    "Throughput",
    "Delay",
    "Jitter",
    "PacketLoss",
    "PDR",
    "Congestion",
    "HOAttempts",
    "HOSuccess",
    "HOFailure",
    "HOSuccessRate",
    "HOFailureRate",
    "HODelay",
    "QoSScore",
    "CongestionScore",
    "HORiskScore",
    "NetworkSeverityScore",
    "QoSLevel",
    "CongestionLevel",
    "HORisk",
    "Decision",
    "PolicyRule",
    "PolicyReason",
    "Explanation",
}


# ============================================================================
# Utilitaires
# ============================================================================

def require_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    """Vérifie que toutes les colonnes demandées existent."""

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Colonnes absentes : "
            + ", ".join(missing)
        )


def select_features(df: pd.DataFrame) -> list[str]:
    """Sélectionne uniquement les paramètres de scénario disponibles."""

    selected: list[str] = []

    for column in CANDIDATE_FEATURES:
        if (
            column in df.columns
            and column not in FORBIDDEN_FEATURES
            and column not in selected
        ):
            selected.append(column)

    if not selected:
        raise ValueError(
            "Aucune variable de scénario reconnue dans le dataset."
        )

    return selected


def coerce_numeric_when_possible(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Convertit en numérique les colonnes compatibles."""

    result = df.copy()

    for column in columns:
        converted = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        # On considère la colonne numérique lorsqu'au moins 90 %
        # de ses valeurs peuvent être converties.
        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.90:
            result[column] = converted

    return result


def remove_constant_features(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Élimine les variables sans variation."""

    constant_features = [
        column
        for column in features
        if df[column].nunique(dropna=False) <= 1
    ]

    retained_features = [
        column
        for column in features
        if column not in constant_features
    ]

    if not retained_features:
        raise ValueError(
            "Toutes les variables candidates sont constantes."
        )

    return (
        df,
        retained_features,
        constant_features,
    )


def split_feature_types(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[list[str], list[str]]:
    """Sépare les variables numériques et catégorielles."""

    numeric_features = [
        column
        for column in features
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    categorical_features = [
        column
        for column in features
        if column not in numeric_features
    ]

    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Construit le prétraitement sans fuite entre les folds."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_models() -> dict[str, Any]:
    """Définit plusieurs modèles adaptés à un petit dataset tabulaire."""

    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def compute_prediction_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calcule les métriques à partir des prédictions out-of-fold."""

    predictions = np.clip(
        np.asarray(y_pred, dtype=float),
        0.0,
        1.0,
    )

    rmse = mean_squared_error(
        y_true,
        predictions,
    ) ** 0.5

    return {
        "MAE": mean_absolute_error(
            y_true,
            predictions,
        ),
        "RMSE": rmse,
        "R2": r2_score(
            y_true,
            predictions,
        ),
    }


# ============================================================================
# Entraînement
# ============================================================================

def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {INPUT_FILE}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_FILE)

    require_columns(
        df,
        TARGETS,
    )

    feature_columns = select_features(df)

    df = coerce_numeric_when_possible(
        df,
        feature_columns,
    )

    (
        df,
        feature_columns,
        constant_features,
    ) = remove_constant_features(
        df,
        feature_columns,
    )

    numeric_features, categorical_features = split_feature_types(
        df,
        feature_columns,
    )

    X = df[feature_columns].copy()

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    models = build_models()

    # 5 folds répétés 5 fois : 25 évaluations par modèle.
    repeated_cv = RepeatedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )

    # Une seule validation 5-fold est utilisée pour produire exactement
    # une prédiction out-of-fold par observation.
    prediction_cv = RepeatedKFold(
        n_splits=5,
        n_repeats=1,
        random_state=RANDOM_STATE,
    )

    metrics_records: list[dict[str, Any]] = []

    best_models: dict[str, dict[str, Any]] = {}

    predictions_output = df.copy()

    for target in TARGETS:

        print("\n" + "=" * 72)
        print(f"CIBLE : {target}")
        print("=" * 72)

        y = pd.to_numeric(
            df[target],
            errors="raise",
        )

        target_results: list[dict[str, Any]] = []

        for model_name, estimator in models.items():

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        clone(preprocessor),
                    ),
                    (
                        "regressor",
                        clone(estimator),
                    ),
                ]
            )

            scoring = {
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            }

            cv_result = cross_validate(
                pipeline,
                X,
                y,
                cv=repeated_cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False,
            )

            mean_mae = -cv_result["test_mae"].mean()
            std_mae = cv_result["test_mae"].std()

            mean_rmse = -cv_result["test_rmse"].mean()
            std_rmse = cv_result["test_rmse"].std()

            mean_r2 = cv_result["test_r2"].mean()
            std_r2 = cv_result["test_r2"].std()

            out_of_fold = cross_val_predict(
                pipeline,
                X,
                y,
                cv=prediction_cv,
                n_jobs=-1,
                method="predict",
            )

            out_of_fold = np.clip(
                out_of_fold,
                0.0,
                1.0,
            )

            oof_metrics = compute_prediction_metrics(
                y,
                out_of_fold,
            )

            record = {
                "Target": target,
                "Model": model_name,
                "CV_MAE_Mean": mean_mae,
                "CV_MAE_Std": std_mae,
                "CV_RMSE_Mean": mean_rmse,
                "CV_RMSE_Std": std_rmse,
                "CV_R2_Mean": mean_r2,
                "CV_R2_Std": std_r2,
                "OOF_MAE": oof_metrics["MAE"],
                "OOF_RMSE": oof_metrics["RMSE"],
                "OOF_R2": oof_metrics["R2"],
            }

            target_results.append(record)
            metrics_records.append(record)

            print(
                f"{model_name:<22} "
                f"MAE={mean_mae:.4f} ± {std_mae:.4f} | "
                f"RMSE={mean_rmse:.4f} ± {std_rmse:.4f} | "
                f"R²={mean_r2:.4f} ± {std_r2:.4f}"
            )

        # Sélection fondée d'abord sur le MAE moyen.
        best_result = min(
            target_results,
            key=lambda item: item["CV_MAE_Mean"],
        )

        best_model_name = str(best_result["Model"])
        best_estimator = models[best_model_name]

        best_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "regressor",
                    clone(best_estimator),
                ),
            ]
        )

        best_oof_predictions = cross_val_predict(
            best_pipeline,
            X,
            y,
            cv=prediction_cv,
            n_jobs=-1,
            method="predict",
        )

        best_oof_predictions = np.clip(
            best_oof_predictions,
            0.0,
            1.0,
        )

        predictions_output[
            f"{target}_PredictedOOF"
        ] = best_oof_predictions

        predictions_output[
            f"{target}_AbsoluteError"
        ] = np.abs(
            y.to_numpy()
            - best_oof_predictions
        )

        # Réentraînement final sur les 190 observations.
        best_pipeline.fit(
            X,
            y,
        )

        model_file = MODEL_DIR / f"{target}_{best_model_name}.joblib"

        joblib.dump(
            best_pipeline,
            model_file,
        )

        best_models[target] = {
            "model_name": best_model_name,
            "model_file": str(model_file),
            "cv_mae": float(best_result["CV_MAE_Mean"]),
            "cv_rmse": float(best_result["CV_RMSE_Mean"]),
            "cv_r2": float(best_result["CV_R2_Mean"]),
        }

        print(
            f"\nMeilleur modèle pour {target} : "
            f"{best_model_name}"
        )

    metrics_df = pd.DataFrame(metrics_records)

    metrics_df = metrics_df.sort_values(
        ["Target", "CV_MAE_Mean"],
        ascending=[True, True],
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    predictions_output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    feature_metadata = {
        "input_file": str(INPUT_FILE),
        "observation_count": int(len(df)),
        "selected_features": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "removed_constant_features": constant_features,
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "targets": TARGETS,
        "best_models": best_models,
    }

    with FEATURES_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write("ENTRAÎNEMENT DES MODÈLES D'ÉTAT RÉSEAU V2X\n")
        file.write("=" * 72 + "\n\n")

        file.write(f"Observations : {len(df)}\n")
        file.write(
            "Variables utilisées : "
            + ", ".join(feature_columns)
            + "\n"
        )

        if constant_features:
            file.write(
                "Variables constantes supprimées : "
                + ", ".join(constant_features)
                + "\n"
            )

        file.write(
            "Variables numériques : "
            + ", ".join(numeric_features)
            + "\n"
        )

        file.write(
            "Variables catégorielles : "
            + (
                ", ".join(categorical_features)
                if categorical_features
                else "aucune"
            )
            + "\n\n"
        )

        file.write(
            "Validation : RepeatedKFold, "
            "5 folds × 5 répétitions.\n\n"
        )

        for target, information in best_models.items():
            file.write(f"Cible : {target}\n")
            file.write(
                f"  Meilleur modèle : "
                f"{information['model_name']}\n"
            )
            file.write(
                f"  MAE CV : "
                f"{information['cv_mae']:.6f}\n"
            )
            file.write(
                f"  RMSE CV : "
                f"{information['cv_rmse']:.6f}\n"
            )
            file.write(
                f"  R² CV : "
                f"{information['cv_r2']:.6f}\n\n"
            )

        file.write("PRÉCAUTIONS MÉTHODOLOGIQUES\n")
        file.write("-" * 72 + "\n")
        file.write(
            "Les métriques mesurées après simulation, les niveaux "
            "qualitatifs et les décisions ont été exclus des entrées.\n"
        )
        file.write(
            "Cette exclusion limite les risques de fuite de données et "
            "évite d'entraîner un modèle qui reproduirait directement "
            "la politique de pseudo-étiquetage V3.\n"
        )
        file.write(
            "Les résultats restent exploratoires, car le dataset ne "
            "contient que 190 observations issues de simulations.\n"
        )
        file.write(
            "Le risque de handover est particulièrement limité par "
            "l'absence d'échec de handover dans les observations valides.\n"
        )

    print("\n" + "=" * 72)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 72)
    print(f"Métriques       : {METRICS_FILE}")
    print(f"Prédictions OOF : {PREDICTIONS_FILE}")
    print(f"Résumé          : {SUMMARY_FILE}")
    print(f"Modèles         : {MODEL_DIR}")
    print(f"Configuration   : {FEATURES_FILE}")


if __name__ == "__main__":
    main()
