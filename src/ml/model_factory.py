"""
Création centralisée des modèles de classification du projet V2X.
"""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "decision_tree": {
        "class": DecisionTreeClassifier,
        "requires_scaling": False,
        "params": {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
    },

    "random_forest": {
        "class": RandomForestClassifier,
        "requires_scaling": False,
        "params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        },
    },

    "xgboost": {
        "class": XGBClassifier,
        "requires_scaling": False,
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },

    "mlp": {
        "class": MLPClassifier,
        "requires_scaling": True,
        "params": {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 25,
            "random_state": RANDOM_STATE,
        },
    },
}


def list_models() -> list[str]:
    """Retourne les noms des modèles disponibles."""

    return sorted(MODEL_CONFIGS.keys())


def _validate_model_name(model_name: str) -> str:
    """Normalise et valide le nom du modèle."""

    normalized_name = model_name.lower()

    if normalized_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Modèle inconnu : {normalized_name}. "
            f"Modèles disponibles : {list_models()}"
        )

    return normalized_name


def requires_scaling(model_name: str) -> bool:
    """Indique si le modèle exige une standardisation."""

    model_name = _validate_model_name(model_name)

    return bool(
        MODEL_CONFIGS[model_name]["requires_scaling"]
    )


def build_model(model_name: str, **overrides: Any):
    """
    Construit un modèle avec sa configuration par défaut.

    Les paramètres dans overrides remplacent les valeurs par défaut.
    """

    model_name = _validate_model_name(model_name)

    config = MODEL_CONFIGS[model_name]
    model_class = config["class"]

    params = config["params"].copy()
    params.update(overrides)

    return model_class(**params)


def get_model_parameters(
    model_name: str,
) -> dict[str, Any]:
    """Retourne les paramètres configurés pour un modèle."""

    model_name = _validate_model_name(model_name)

    return MODEL_CONFIGS[model_name]["params"].copy()
