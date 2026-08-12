"""
Chargement des datasets ML du projet V2X.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATASETS = {
   "C": {
         "data": PROJECT_ROOT
         / "datasets/processed/mobility/"
         / "temporal_prediction_dataset_c.csv",
    },
    "P1": {
        "train": PROJECT_ROOT / "datasets/ml/dataset_a/dataset_a_train.csv",
        "test": PROJECT_ROOT / "datasets/ml/dataset_a/dataset_a_test.csv",
    },

    "P2": {
        "train": PROJECT_ROOT / "datasets/ml/dataset_a_grouped/dataset_a_grouped_train.csv",
        "test": PROJECT_ROOT / "datasets/ml/dataset_a_grouped/dataset_a_grouped_test.csv",
    },

    "HANDOVER": {
        "data": PROJECT_ROOT / "datasets/processed/handover_dataset_v6.csv",
    },
}


def list_protocols():
    """
    Retourne les protocoles disponibles.
    """
    return list(DATASETS.keys())


def load_dataset(protocol: str = "P1"):
    """
    Charge automatiquement le dataset demandé.

    Parameters
    ----------
    protocol : str

        P1
        P2
        HANDOVER
    """

    protocol = protocol.upper()

    if protocol not in DATASETS:
        raise ValueError(
            f"Protocole inconnu : {protocol}"
        )

    files = DATASETS[protocol]

    if protocol in {"HANDOVER", "C"}:

        df = pd.read_csv(files["data"])

        return df

    train = pd.read_csv(files["train"])

    test = pd.read_csv(files["test"])

    return train, test
