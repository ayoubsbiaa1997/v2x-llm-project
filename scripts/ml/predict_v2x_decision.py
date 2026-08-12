#!/usr/bin/env python3

"""
Pipeline d'inférence V2X :

1. Lecture des paramètres d'un nouveau scénario.
2. Prédiction de QoSScore, CongestionScore et HORiskScore.
3. Conversion des scores en niveaux qualitatifs.
4. Application de la politique décisionnelle V3.
5. Génération d'une sortie JSON et d'un prompt pour un LLM.

Ce script n'utilise pas les métriques observées après simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path.home() / "v2x-llm-project"

MODEL_DIR = PROJECT_DIR / "models" / "network_state_v1"

CONFIG_FILE = (
    PROJECT_DIR
    / "results"
    / "network_state_v1"
    / "selected_features.json"
)

OUTPUT_DIR = PROJECT_DIR / "results" / "inference"


# ============================================================================
# Conversion des scores en niveaux
# ============================================================================

def qos_level(score: float) -> str:
    if score <= 0.25:
        return "GOOD"

    if score <= 0.50:
        return "ACCEPTABLE"

    if score <= 0.75:
        return "DEGRADED"

    return "CRITICAL"


def congestion_level(score: float) -> str:
    if score <= 0.25:
        return "LOW"

    if score <= 0.50:
        return "MODERATE"

    if score <= 0.75:
        return "HIGH"

    return "CRITICAL"


def ho_risk_level(score: float) -> str:
    if score <= 0.25:
        return "LOW"

    if score <= 0.50:
        return "MEDIUM"

    if score <= 0.75:
        return "HIGH"

    return "CRITICAL"


# ============================================================================
# Politique décisionnelle V3
# ============================================================================

def decide_network_action(
    qos: str,
    congestion: str,
    ho_risk: str,
    qos_score: float,
    congestion_score: float,
    ho_score: float,
) -> tuple[str, str, str]:

    severity = (
        0.40 * qos_score
        + 0.35 * congestion_score
        + 0.25 * ho_score
    )

    if ho_risk == "CRITICAL" and qos == "CRITICAL":
        return (
            "HANDOVER_NOW",
            "R01",
            (
                "Le risque de handover et la dégradation QoS sont "
                "simultanément critiques."
            ),
        )

    if ho_risk == "CRITICAL" and qos == "DEGRADED":
        return (
            "HANDOVER_NOW",
            "R02",
            (
                "Le risque de mobilité est critique et la QoS est "
                "déjà dégradée."
            ),
        )

    if ho_risk == "CRITICAL":
        return (
            "PREPARE_HANDOVER",
            "R03",
            (
                "Le risque de handover est critique, mais la QoS "
                "reste encore exploitable."
            ),
        )

    if qos == "CRITICAL" and ho_risk == "HIGH":
        return (
            "HANDOVER_NOW",
            "R04",
            (
                "La QoS est critique et le risque de handover est élevé."
            ),
        )

    if (
        congestion == "CRITICAL"
        and ho_risk in {"LOW", "MEDIUM"}
    ):
        return (
            "REDUCE_TRAFFIC_LOAD",
            "R05",
            (
                "La congestion est critique alors que le risque de "
                "handover reste limité."
            ),
        )

    if congestion == "CRITICAL" and ho_risk == "HIGH":

        if ho_score > congestion_score + 0.10:
            return (
                "PREPARE_HANDOVER",
                "R06A",
                (
                    "Le risque de handover devient dominant malgré "
                    "une congestion critique."
                ),
            )

        return (
            "REDUCE_TRAFFIC_LOAD",
            "R06B",
            (
                "La congestion critique reste la cause dominante."
            ),
        )

    if (
        congestion == "HIGH"
        and ho_risk in {"LOW", "MEDIUM"}
        and qos in {"DEGRADED", "CRITICAL"}
    ):
        return (
            "REDUCE_TRAFFIC_LOAD",
            "R07",
            (
                "La QoS est dégradée par une congestion élevée, sans "
                "risque de mobilité dominant."
            ),
        )

    if ho_risk == "HIGH":
        return (
            "PREPARE_HANDOVER",
            "R08",
            (
                "Le risque de handover est élevé. La cellule cible et "
                "les ressources doivent être préparées."
            ),
        )

    if qos == "CRITICAL":

        if congestion in {"HIGH", "CRITICAL"}:
            return (
                "REDUCE_TRAFFIC_LOAD",
                "R09A",
                (
                    "La QoS est critique et la congestion est élevée."
                ),
            )

        return (
            "MONITOR_NETWORK",
            "R09B",
            (
                "La QoS est critique, mais aucune cause dominante "
                "n'est clairement identifiée."
            ),
        )

    if (
        qos == "DEGRADED"
        and congestion in {"HIGH", "CRITICAL"}
    ):
        return (
            "REDUCE_TRAFFIC_LOAD",
            "R10",
            (
                "La QoS est dégradée et la congestion est élevée."
            ),
        )

    if qos == "DEGRADED":
        return (
            "MONITOR_NETWORK",
            "R11",
            (
                "Une dégradation QoS est détectée sans situation "
                "critique de mobilité ou de congestion."
            ),
        )

    if (
        qos == "ACCEPTABLE"
        or congestion in {"MODERATE", "HIGH"}
        or ho_risk == "MEDIUM"
        or severity > 0.40
    ):
        return (
            "MONITOR_NETWORK",
            "R12",
            (
                "L'état du réseau est intermédiaire et nécessite "
                "une surveillance continue."
            ),
        )

    return (
        "KEEP_CURRENT_CONNECTION",
        "R13",
        (
            "La QoS est bonne, la congestion est faible et le risque "
            "de handover est limité."
        ),
    )


# ============================================================================
# Chargement des modèles
# ============================================================================

def load_configuration() -> dict[str, Any]:

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {CONFIG_FILE}"
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_models(
    configuration: dict[str, Any],
) -> dict[str, Any]:

    models: dict[str, Any] = {}

    best_models = configuration.get("best_models", {})

    for target in [
        "QoSScore",
        "CongestionScore",
        "HORiskScore",
    ]:
        if target not in best_models:
            raise KeyError(
                f"Modèle absent dans la configuration : {target}"
            )

        model_file = Path(
            best_models[target]["model_file"]
        )

        if not model_file.exists():
            raise FileNotFoundError(
                f"Fichier modèle introuvable : {model_file}"
            )

        models[target] = joblib.load(model_file)

    return models


# ============================================================================
# Préparation du scénario
# ============================================================================

def build_scenario(
    args: argparse.Namespace,
    selected_features: list[str],
) -> pd.DataFrame:

    supplied_values = {
        "gNbNum": args.gnb_num,
        "gnbNum": args.gnb_num,
        "ueNumPerGnb": args.ue_per_gnb,
        "ueNumPergNb": args.ue_per_gnb,
        "totalUE": args.total_ue,
        "bsDistance": args.bs_distance,
        "scenarioLength": args.scenario_length,
        "ueSpeed": args.ue_speed,
        "lambdaULL": args.lambda_ull,
        "lambdaBe": args.lambda_be,
        "packetULL": args.packet_ull,
        "packetBE": args.packet_be,
    }

    scenario: dict[str, float] = {}

    for feature in selected_features:
        if feature not in supplied_values:
            raise ValueError(
                f"Aucune valeur disponible pour la variable {feature}"
            )

        scenario[feature] = supplied_values[feature]

    return pd.DataFrame([scenario])


# ============================================================================
# Prédiction
# ============================================================================

def predict_scores(
    scenario_df: pd.DataFrame,
    models: dict[str, Any],
) -> dict[str, float]:

    predictions: dict[str, float] = {}

    for target, model in models.items():
        predicted_value = float(
            model.predict(scenario_df)[0]
        )

        predictions[target] = float(
            np.clip(predicted_value, 0.0, 1.0)
        )

    return predictions


# ============================================================================
# Prompt LLM
# ============================================================================

def build_llm_prompt(
    scenario: dict[str, Any],
    analysis: dict[str, Any],
) -> str:

    return f"""
Tu es un expert en communications V2X sur réseaux 5G/6G.

Tu dois analyser l'état prédit du réseau et prendre une décision autonome.

PARAMÈTRES DU SCÉNARIO
- Nombre de gNB : {scenario['gNbNum']}
- Nombre d'UE par gNB : {scenario['ueNumPerGnb']}
- Nombre total d'UE : {scenario['totalUE']}
- Distance entre gNB : {scenario['bsDistance']} m
- Vitesse des véhicules : {scenario['ueSpeed']} m/s
- Charge URLLC : {scenario['lambdaULL']} paquets/s
- Charge Best Effort : {scenario['lambdaBe']} paquets/s

ÉTAT PRÉDIT PAR LES MODÈLES ML
- QoSScore : {analysis['scores']['QoSScore']:.4f}
- Niveau QoS : {analysis['levels']['QoSLevel']}
- CongestionScore : {analysis['scores']['CongestionScore']:.4f}
- Niveau de congestion : {analysis['levels']['CongestionLevel']}
- HORiskScore : {analysis['scores']['HORiskScore']:.4f}
- Niveau de risque de handover : {analysis['levels']['HORisk']}
- Sévérité globale : {analysis['scores']['NetworkSeverityScore']:.4f}

Choisis exactement une décision parmi :
- KEEP_CURRENT_CONNECTION
- MONITOR_NETWORK
- REDUCE_TRAFFIC_LOAD
- PREPARE_HANDOVER
- HANDOVER_NOW

Produis uniquement un objet JSON valide avec exactement ces champs :

{{
  "decision": "UNE_DECISION_AUTORISEE",
  "confidence": 0.0,
  "reasoning": "Justification technique concise.",
  "dominant_factor": "QOS",
  "recommended_follow_up": "Action de suivi concise.",
  "limitations": "Limites de l'analyse."
}}

Le champ dominant_factor doit être exactement l'une des valeurs suivantes :
QOS, CONGESTION, HANDOVER_RISK ou BALANCED.

N'ajoute aucun texte avant ou après le JSON.
""".strip()


# ============================================================================
# Arguments CLI
# ============================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Prédit l'état réseau et la décision V2X pour un nouveau scénario."
        )
    )

    parser.add_argument(
        "--gnb-num",
        type=int,
        required=True,
        help="Nombre de gNB.",
    )

    parser.add_argument(
        "--ue-per-gnb",
        type=int,
        required=True,
        help="Nombre d'UE par gNB.",
    )

    parser.add_argument(
        "--total-ue",
        type=int,
        default=None,
        help=(
            "Nombre total d'UE. Par défaut : gNbNum × ueNumPerGnb."
        ),
    )

    parser.add_argument(
        "--bs-distance",
        type=float,
        required=True,
        help="Distance entre les gNB en mètres.",
    )

    parser.add_argument(
        "--scenario-length",
        type=float,
        default=600.0,
        help="Longueur du scénario en mètres.",
    )

    parser.add_argument(
        "--ue-speed",
        type=float,
        required=True,
        help="Vitesse des véhicules en m/s.",
    )

    parser.add_argument(
        "--lambda-ull",
        type=float,
        required=True,
        help="Charge URLLC en paquets/s.",
    )

    parser.add_argument(
        "--lambda-be",
        type=float,
        required=True,
        help="Charge Best Effort en paquets/s.",
    )

    parser.add_argument(
        "--packet-ull",
        type=float,
        default=128.0,
        help="Taille des paquets URLLC en octets.",
    )

    parser.add_argument(
        "--packet-be",
        type=float,
        default=1024.0,
        help="Taille des paquets Best Effort en octets.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="new_scenario",
        help="Nom du scénario.",
    )

    return parser.parse_args()


# ============================================================================
# Programme principal
# ============================================================================

def main() -> None:

    args = parse_arguments()

    if args.total_ue is None:
        args.total_ue = args.gnb_num * args.ue_per_gnb

    configuration = load_configuration()

    selected_features = configuration["selected_features"]

    models = load_models(configuration)

    scenario_df = build_scenario(
        args,
        selected_features,
    )

    predictions = predict_scores(
        scenario_df,
        models,
    )

    qos_score = predictions["QoSScore"]
    congestion_score = predictions["CongestionScore"]
    ho_score = predictions["HORiskScore"]

    network_severity = (
        0.40 * qos_score
        + 0.35 * congestion_score
        + 0.25 * ho_score
    )

    levels = {
        "QoSLevel": qos_level(qos_score),
        "CongestionLevel": congestion_level(
            congestion_score
        ),
        "HORisk": ho_risk_level(ho_score),
    }

    decision, rule, reason = decide_network_action(
        qos=levels["QoSLevel"],
        congestion=levels["CongestionLevel"],
        ho_risk=levels["HORisk"],
        qos_score=qos_score,
        congestion_score=congestion_score,
        ho_score=ho_score,
    )

    scores = {
        **predictions,
        "NetworkSeverityScore": float(network_severity),
    }

    scenario_output = {
        "scenarioName": args.name,
        "gNbNum": args.gnb_num,
        "ueNumPerGnb": args.ue_per_gnb,
        "totalUE": args.total_ue,
        "bsDistance": args.bs_distance,
        "scenarioLength": args.scenario_length,
        "ueSpeed": args.ue_speed,
        "lambdaULL": args.lambda_ull,
        "lambdaBe": args.lambda_be,
        "packetULL": args.packet_ull,
        "packetBE": args.packet_be,
    }

    analysis = {
        "scores": {
            key: round(value, 6)
            for key, value in scores.items()
        },
        "levels": levels,
        "decision": decision,
        "policyRule": rule,
        "policyReason": reason,
    }

    result = {
        "scenario": scenario_output,
        "analysis": analysis,
    }

    prompt = build_llm_prompt(
        scenario_output,
        analysis,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = (
        args.name
        .replace("/", "_")
        .replace(" ", "_")
    )

    json_file = OUTPUT_DIR / f"{safe_name}_prediction.json"
    prompt_file = OUTPUT_DIR / f"{safe_name}_llm_prompt.txt"

    with json_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with prompt_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(prompt)
        file.write("\n")

    print("=" * 72)
    print("INFÉRENCE V2X TERMINÉE")
    print("=" * 72)

    print("\nScénario :")
    print(
        json.dumps(
            scenario_output,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nAnalyse :")
    print(
        json.dumps(
            analysis,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nFichiers générés :")
    print(f"- Prédiction : {json_file}")
    print(f"- Prompt LLM : {prompt_file}")


if __name__ == "__main__":
    main()
