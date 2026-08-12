#!/usr/bin/env python3

"""
Construction du dataset décisionnel V3 pour les communications V2X 5G/6G.

Méthode :
1. Lecture des métriques produites par ns-3/5G-LENA.
2. Transformation des métriques en rangs percentiles.
3. Calcul de scores continus :
   - QoSScore
   - CongestionScore
   - HORiskScore
   - NetworkSeverityScore
4. Attribution de niveaux qualitatifs par seuils explicites.
5. Application d'une politique de décision V2X fondée sur des règles métier.
6. Export CSV, JSONL et rapport statistique.

Les décisions restent des pseudo-labels générés par une politique explicite.
Elles ne remplacent pas une validation par un opérateur réseau.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ===========================================================================
# Chemins
# ===========================================================================

PROJECT_DIR = Path.home() / "v2x-llm-project"
INPUT_FILE = Path.home() / "ns-3-dev" / "dataset_metrics.csv"

PROCESSED_DIR = PROJECT_DIR / "datasets" / "processed"
LLM_DIR = PROJECT_DIR / "datasets" / "llm"

OUTPUT_CSV = PROCESSED_DIR / "final_dataset_v3.csv"
OUTPUT_JSONL = LLM_DIR / "dataset_llm_v3.jsonl"
OUTPUT_SUMMARY = PROCESSED_DIR / "dataset_summary_v3.txt"
OUTPUT_POLICY = PROCESSED_DIR / "decision_policy_v3.csv"


# ===========================================================================
# Validation et fonctions utilitaires
# ===========================================================================

def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Vérifie la présence des colonnes indispensables."""

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            + ", ".join(missing)
        )


def numeric_series(series: pd.Series) -> pd.Series:
    """Convertit une série en valeurs numériques sans NaN."""

    values = pd.to_numeric(series, errors="coerce")

    if values.isna().all():
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
            dtype=float,
        )

    return values.fillna(values.median())


def percentile_rank(
    series: pd.Series,
    inverse: bool = False,
) -> pd.Series:
    """
    Transforme une métrique en rang percentile compris entre 0 et 1.

    inverse=False :
        une valeur élevée représente une dégradation plus importante.

    inverse=True :
        une valeur faible représente une dégradation plus importante.
    """

    values = numeric_series(series)
    ranks = values.rank(method="average", pct=True)

    if inverse:
        ranks = 1.0 - ranks

    return ranks.clip(0.0, 1.0)


def safe_float(
    row: pd.Series,
    column: str,
    default: float = 0.0,
) -> float:
    """Extrait une valeur numérique sûre depuis une ligne."""

    try:
        value = float(row[column])

        if np.isfinite(value):
            return value

    except (KeyError, TypeError, ValueError):
        pass

    return default


def safe_int(
    row: pd.Series,
    columns: list[str],
    default: int = 0,
) -> int:
    """Recherche une valeur entière parmi plusieurs noms possibles."""

    for column in columns:
        if column in row.index:
            return int(round(safe_float(row, column, default)))

    return default


def safe_text(
    row: pd.Series,
    columns: list[str],
    default: str = "",
) -> str:
    """Recherche une valeur textuelle parmi plusieurs noms possibles."""

    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return str(row[column])

    return default


# ===========================================================================
# Calcul des scores
# ===========================================================================

def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les scores de dégradation réseau."""

    result = df.copy()

    # Une valeur élevée représente ici une dégradation importante.
    delay_bad = percentile_rank(result["Delay"])
    jitter_bad = percentile_rank(result["Jitter"])
    loss_bad = percentile_rank(result["PacketLoss"])

    # Pour le débit et le PDR, une valeur faible représente une dégradation.
    throughput_bad = percentile_rank(
        result["Throughput"],
        inverse=True,
    )

    pdr_bad = percentile_rank(
        result["PDR"],
        inverse=True,
    )

    # -----------------------------------------------------------------------
    # Score QoS
    # -----------------------------------------------------------------------
    # Les délais et pertes sont fortement pondérés en contexte V2X.
    result["QoSScore"] = (
        0.30 * delay_bad
        + 0.15 * jitter_bad
        + 0.25 * loss_bad
        + 0.15 * throughput_bad
        + 0.15 * pdr_bad
    ).clip(0.0, 1.0)

    # -----------------------------------------------------------------------
    # Score de congestion
    # -----------------------------------------------------------------------
    congestion_bad = percentile_rank(result["Congestion"])

    result["CongestionScore"] = (
        0.45 * congestion_bad
        + 0.20 * delay_bad
        + 0.20 * loss_bad
        + 0.10 * throughput_bad
        + 0.05 * jitter_bad
    ).clip(0.0, 1.0)

    # -----------------------------------------------------------------------
    # Score de risque de handover
    # -----------------------------------------------------------------------
    attempts_bad = percentile_rank(result["HOAttempts"])
    ho_delay_bad = percentile_rank(result["HODelay"])
    failure_rate_bad = percentile_rank(result["HOFailureRate"])

    success_rate_bad = percentile_rank(
        result["HOSuccessRate"],
        inverse=True,
    )

    result["HORiskScore"] = (
        0.20 * attempts_bad
        + 0.20 * ho_delay_bad
        + 0.35 * failure_rate_bad
        + 0.25 * success_rate_bad
    ).clip(0.0, 1.0)

    # -----------------------------------------------------------------------
    # Sévérité globale
    # -----------------------------------------------------------------------
    result["NetworkSeverityScore"] = (
        0.40 * result["QoSScore"]
        + 0.35 * result["CongestionScore"]
        + 0.25 * result["HORiskScore"]
    ).clip(0.0, 1.0)

    return result


# ===========================================================================
# Attribution des niveaux qualitatifs
# ===========================================================================

def qos_level(score: float) -> str:
    """Convertit le score QoS en niveau qualitatif."""

    if score <= 0.25:
        return "GOOD"

    if score <= 0.50:
        return "ACCEPTABLE"

    if score <= 0.75:
        return "DEGRADED"

    return "CRITICAL"


def congestion_level(score: float) -> str:
    """Convertit le score de congestion en niveau qualitatif."""

    if score <= 0.25:
        return "LOW"

    if score <= 0.50:
        return "MODERATE"

    if score <= 0.75:
        return "HIGH"

    return "CRITICAL"


def ho_risk_level(score: float) -> str:
    """Convertit le score de risque HO en niveau qualitatif."""

    if score <= 0.25:
        return "LOW"

    if score <= 0.50:
        return "MEDIUM"

    if score <= 0.75:
        return "HIGH"

    return "CRITICAL"


def assign_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les niveaux qualitatifs au dataset."""

    result = df.copy()

    result["QoSLevel"] = result["QoSScore"].apply(qos_level)

    result["CongestionLevel"] = (
        result["CongestionScore"]
        .apply(congestion_level)
    )

    result["HORisk"] = (
        result["HORiskScore"]
        .apply(ho_risk_level)
    )

    return result


# ===========================================================================
# Moteur de décision V2X
# ===========================================================================

def decide_network_action(
    row: pd.Series,
) -> tuple[str, str, str]:
    """
    Retourne :
    - décision ;
    - identifiant de la règle ;
    - justification métier.

    Les règles sont évaluées par ordre de priorité.
    """

    qos = str(row["QoSLevel"])
    congestion = str(row["CongestionLevel"])
    ho_risk = str(row["HORisk"])

    qos_score = safe_float(row, "QoSScore")
    congestion_score = safe_float(row, "CongestionScore")
    ho_score = safe_float(row, "HORiskScore")
    severity = safe_float(row, "NetworkSeverityScore")

    # -----------------------------------------------------------------------
    # R01 : situation critique de mobilité et QoS critique
    # -----------------------------------------------------------------------
    if ho_risk == "CRITICAL" and qos == "CRITICAL":
        return (
            "HANDOVER_NOW",
            "R01",
            (
                "Le risque de handover et la dégradation QoS sont tous "
                "deux critiques. Le maintien de la cellule actuelle "
                "présente un risque élevé d'interruption ou de perte "
                "de performance."
            ),
        )

    # -----------------------------------------------------------------------
    # R02 : risque critique et QoS déjà dégradée
    # -----------------------------------------------------------------------
    if ho_risk == "CRITICAL" and qos == "DEGRADED":
        return (
            "HANDOVER_NOW",
            "R02",
            (
                "Le risque de mobilité est critique et la QoS est déjà "
                "dégradée. Un handover immédiat est recommandé avant "
                "une détérioration supplémentaire."
            ),
        )

    # -----------------------------------------------------------------------
    # R03 : risque critique mais QoS encore acceptable
    # -----------------------------------------------------------------------
    if ho_risk == "CRITICAL":
        return (
            "PREPARE_HANDOVER",
            "R03",
            (
                "Le risque de handover est critique, mais la QoS reste "
                "encore exploitable. Le changement de cellule doit être "
                "préparé afin d'éviter une interruption prématurée."
            ),
        )

    # -----------------------------------------------------------------------
    # R04 : QoS critique et risque HO élevé
    # -----------------------------------------------------------------------
    if qos == "CRITICAL" and ho_risk == "HIGH":
        return (
            "HANDOVER_NOW",
            "R04",
            (
                "La QoS est critique et le risque de handover est élevé. "
                "La dégradation ne peut plus être traitée par une simple "
                "surveillance."
            ),
        )

    # -----------------------------------------------------------------------
    # R05 : forte congestion sans risque HO dominant
    # -----------------------------------------------------------------------
    if (
        congestion == "CRITICAL"
        and ho_risk in {"LOW", "MEDIUM"}
    ):
        return (
            "REDUCE_TRAFFIC_LOAD",
            "R05",
            (
                "La congestion est critique tandis que le risque de "
                "handover reste limité. La surcharge de trafic est la "
                "cause dominante ; la charge doit être réduite ou "
                "redistribuée."
            ),
        )

    # -----------------------------------------------------------------------
    # R06 : congestion critique avec risque élevé
    # -----------------------------------------------------------------------
    if congestion == "CRITICAL" and ho_risk == "HIGH":
        if ho_score > congestion_score + 0.10:
            return (
                "PREPARE_HANDOVER",
                "R06A",
                (
                    "La congestion est critique, mais le risque de "
                    "handover devient dominant. Un changement de cellule "
                    "doit être préparé."
                ),
            )

        return (
            "REDUCE_TRAFFIC_LOAD",
            "R06B",
            (
                "La congestion critique demeure la cause dominante de "
                "la dégradation. Une réduction de charge est prioritaire "
                "avant le déclenchement d'un handover."
            ),
        )

    # -----------------------------------------------------------------------
    # R07 : QoS dégradée par une congestion élevée
    # -----------------------------------------------------------------------
    if (
        congestion == "HIGH"
        and ho_risk in {"LOW", "MEDIUM"}
        and qos in {"DEGRADED", "CRITICAL"}
    ):
        return (
            "REDUCE_TRAFFIC_LOAD",
            "R07",
            (
                "La QoS est dégradée en présence d'une congestion élevée, "
                "mais le risque de handover n'est pas dominant. Une action "
                "sur la charge réseau est recommandée."
            ),
        )

    # -----------------------------------------------------------------------
    # R08 : risque HO élevé
    # -----------------------------------------------------------------------
    if ho_risk == "HIGH":
        return (
            "PREPARE_HANDOVER",
            "R08",
            (
                "Le risque de handover est élevé. La cellule cible, les "
                "ressources radio et la continuité de session doivent être "
                "préparées avant toute dégradation critique."
            ),
        )

    # -----------------------------------------------------------------------
    # R09 : QoS critique sans forte indication de mobilité
    # -----------------------------------------------------------------------
    if qos == "CRITICAL":
        if congestion in {"HIGH", "CRITICAL"}:
            return (
                "REDUCE_TRAFFIC_LOAD",
                "R09A",
                (
                    "La QoS est critique et la congestion est élevée, "
                    "sans risque de handover suffisamment important. "
                    "La réduction de charge est prioritaire."
                ),
            )

        return (
            "MONITOR_NETWORK",
            "R09B",
            (
                "La QoS est critique, mais ni la congestion ni le risque "
                "de handover ne permettent d'identifier une cause dominante. "
                "Une surveillance renforcée est nécessaire."
            ),
        )

    # -----------------------------------------------------------------------
    # R10 : QoS dégradée avec congestion élevée
    # -----------------------------------------------------------------------
    if (
        qos == "DEGRADED"
        and congestion in {"HIGH", "CRITICAL"}
    ):
        return (
            "REDUCE_TRAFFIC_LOAD",
            "R10",
            (
                "La QoS est dégradée et la congestion est élevée. La "
                "réduction ou la redistribution du trafic doit être "
                "privilégiée."
            ),
        )

    # -----------------------------------------------------------------------
    # R11 : QoS dégradée sans cause critique
    # -----------------------------------------------------------------------
    if qos == "DEGRADED":
        return (
            "MONITOR_NETWORK",
            "R11",
            (
                "Une dégradation QoS est observée, mais aucune condition "
                "critique de congestion ou de mobilité n'est détectée. "
                "Une surveillance renforcée est recommandée."
            ),
        )

    # -----------------------------------------------------------------------
    # R12 : situation intermédiaire
    # -----------------------------------------------------------------------
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
                "Les métriques indiquent une situation intermédiaire. "
                "Aucune action immédiate n'est nécessaire, mais une "
                "surveillance continue est recommandée."
            ),
        )

    # -----------------------------------------------------------------------
    # R13 : fonctionnement stable
    # -----------------------------------------------------------------------
    return (
        "KEEP_CURRENT_CONNECTION",
        "R13",
        (
            "La QoS est bonne, la congestion est faible et le risque de "
            "handover est limité. La connexion actuelle peut être maintenue."
        ),
    )


def assign_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le moteur de décision à toutes les observations."""

    result = df.copy()

    policy_results = result.apply(
        decide_network_action,
        axis=1,
        result_type="expand",
    )

    policy_results.columns = [
        "Decision",
        "PolicyRule",
        "PolicyReason",
    ]

    result[
        ["Decision", "PolicyRule", "PolicyReason"]
    ] = policy_results

    return result


# ===========================================================================
# Explication finale
# ===========================================================================

def build_explanation(row: pd.Series) -> str:
    """Construit une explication détaillée et traçable."""

    return (
        f"QoS={row['QoSLevel']} "
        f"(score={row['QoSScore']:.3f}), "
        f"congestion={row['CongestionLevel']} "
        f"(score={row['CongestionScore']:.3f}), "
        f"risque de handover={row['HORisk']} "
        f"(score={row['HORiskScore']:.3f}). "
        f"Les mesures observées sont : "
        f"débit={safe_float(row, 'Throughput'):.3f} Mbit/s, "
        f"délai={safe_float(row, 'Delay'):.3f} ms, "
        f"jitter={safe_float(row, 'Jitter'):.3f} ms, "
        f"pertes={safe_float(row, 'PacketLoss'):.3f} %, "
        f"PDR={safe_float(row, 'PDR'):.3f} %, "
        f"succès de handover="
        f"{safe_float(row, 'HOSuccessRate'):.3f} %. "
        f"Décision={row['Decision']}. "
        f"Règle appliquée={row['PolicyRule']}. "
        f"{row['PolicyReason']}"
    )


# ===========================================================================
# Export JSONL
# ===========================================================================

def export_jsonl(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    """Exporte les observations au format instruction/input/output."""

    with output_file.open("w", encoding="utf-8") as file:

        for _, row in df.iterrows():

            input_payload: dict[str, Any] = {
                "scenario": safe_text(
                    row,
                    ["Scenario", "scenario_id"],
                ),
                "gNbNum": safe_int(
                    row,
                    ["gNbNum", "gnbNum"],
                ),
                "ueNumPerGnb": safe_int(
                    row,
                    ["ueNumPerGnb", "ueNumPergNb"],
                ),
                "totalUE": safe_int(
                    row,
                    ["totalUE"],
                ),
                "bsDistanceM": safe_float(
                    row,
                    "bsDistance",
                ),
                "scenarioLengthM": safe_float(
                    row,
                    "scenarioLength",
                ),
                "ueSpeedMps": safe_float(
                    row,
                    "ueSpeed",
                ),
                "lambdaULL": safe_float(
                    row,
                    "lambdaULL",
                ),
                "lambdaBe": safe_float(
                    row,
                    "lambdaBe",
                ),
                "packetULLBytes": safe_float(
                    row,
                    "packetULL",
                ),
                "packetBEBytes": safe_float(
                    row,
                    "packetBE",
                ),
                "throughputMbps": safe_float(
                    row,
                    "Throughput",
                ),
                "delayMs": safe_float(
                    row,
                    "Delay",
                ),
                "jitterMs": safe_float(
                    row,
                    "Jitter",
                ),
                "packetLossPercent": safe_float(
                    row,
                    "PacketLoss",
                ),
                "pdrPercent": safe_float(
                    row,
                    "PDR",
                ),
                "congestionIndex": safe_float(
                    row,
                    "Congestion",
                ),
                "handoverAttempts": safe_float(
                    row,
                    "HOAttempts",
                ),
                "handoverSuccess": safe_float(
                    row,
                    "HOSuccess",
                ),
                "handoverFailure": safe_float(
                    row,
                    "HOFailure",
                ),
                "handoverSuccessRate": safe_float(
                    row,
                    "HOSuccessRate",
                ),
                "handoverFailureRate": safe_float(
                    row,
                    "HOFailureRate",
                ),
                "handoverDelayMs": safe_float(
                    row,
                    "HODelay",
                ),
            }

            output_payload = {
                "qosScore": round(
                    safe_float(row, "QoSScore"),
                    6,
                ),
                "qosLevel": row["QoSLevel"],
                "congestionScore": round(
                    safe_float(row, "CongestionScore"),
                    6,
                ),
                "congestionLevel": row["CongestionLevel"],
                "handoverRiskScore": round(
                    safe_float(row, "HORiskScore"),
                    6,
                ),
                "handoverRisk": row["HORisk"],
                "networkSeverityScore": round(
                    safe_float(row, "NetworkSeverityScore"),
                    6,
                ),
                "decision": row["Decision"],
                "policyRule": row["PolicyRule"],
                "explanation": row["Explanation"],
            }

            record = {
                "instruction": (
                    "Analyse les métriques d'un scénario V2X sur réseau "
                    "5G/6G. Évalue la QoS, la congestion et le risque de "
                    "handover, puis propose une décision réseau structurée "
                    "et explicable."
                ),
                "input": input_payload,
                "output": output_payload,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ===========================================================================
# Export de la politique
# ===========================================================================

def export_policy_table(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    """Exporte les règles réellement utilisées et leur fréquence."""

    policy = (
        df.groupby(
            ["PolicyRule", "Decision", "PolicyReason"],
            dropna=False,
        )
        .size()
        .reset_index(name="ObservationCount")
        .sort_values(
            ["PolicyRule", "Decision"],
        )
    )

    policy.to_csv(
        output_file,
        index=False,
    )


# ===========================================================================
# Rapport statistique
# ===========================================================================

def write_distribution(
    file: Any,
    df: pd.DataFrame,
    column: str,
) -> None:
    """Écrit les effectifs et pourcentages d'une variable catégorielle."""

    counts = df[column].value_counts()
    percentages = (
        df[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    table = pd.DataFrame(
        {
            "Nombre": counts,
            "Pourcentage": percentages,
        }
    )

    file.write(f"Distribution {column} :\n")
    file.write(table.to_string())
    file.write("\n\n")


def write_summary(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    """Produit le rapport final du dataset V3."""

    numeric_columns = [
        "Throughput",
        "Delay",
        "Jitter",
        "PacketLoss",
        "PDR",
        "Congestion",
        "HOAttempts",
        "HOSuccessRate",
        "HOFailureRate",
        "HODelay",
        "QoSScore",
        "CongestionScore",
        "HORiskScore",
        "NetworkSeverityScore",
    ]

    available = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    with output_file.open("w", encoding="utf-8") as file:

        file.write("RAPPORT DU DATASET DÉCISIONNEL V3\n")
        file.write("=" * 65 + "\n")
        file.write(f"Nombre d'observations : {len(df)}\n")
        file.write(f"Nombre de colonnes : {len(df.columns)}\n")
        file.write(
            f"Valeurs manquantes : "
            f"{int(df.isna().sum().sum())}\n"
        )
        file.write(
            f"Doublons complets : "
            f"{int(df.duplicated().sum())}\n\n"
        )

        for column in [
            "QoSLevel",
            "CongestionLevel",
            "HORisk",
            "Decision",
            "PolicyRule",
        ]:
            write_distribution(
                file,
                df,
                column,
            )

        file.write("Statistiques numériques principales :\n")
        file.write(
            df[available]
            .describe()
            .T
            .to_string()
        )

        file.write("\n\n")
        file.write("MÉTHODOLOGIE\n")
        file.write("-" * 65 + "\n")
        file.write(
            "Les scores continus sont calculés à partir des rangs "
            "percentiles des métriques de la campagne expérimentale.\n"
        )
        file.write(
            "Les niveaux qualitatifs sont obtenus à partir de seuils "
            "explicites appliqués aux scores normalisés.\n"
        )
        file.write(
            "Les décisions sont générées par un moteur de règles métier "
            "V2X et non par une répartition forcée des classes.\n"
        )
        file.write(
            "Chaque observation contient l'identifiant de la règle "
            "appliquée afin d'assurer la traçabilité de la décision.\n\n"
        )

        file.write("AVERTISSEMENT SCIENTIFIQUE\n")
        file.write("-" * 65 + "\n")
        file.write(
            "Les décisions et niveaux générés sont des pseudo-labels. "
            "La politique proposée est une politique expérimentale "
            "explicable ; elle ne constitue ni une recommandation 3GPP "
            "officielle ni une décision validée par un opérateur réel.\n"
        )


# ===========================================================================
# Contrôles finaux
# ===========================================================================

def validate_output(df: pd.DataFrame) -> None:
    """Vérifie la cohérence du dataset final."""

    if df.empty:
        raise ValueError("Le dataset final est vide.")

    required_outputs = [
        "QoSScore",
        "CongestionScore",
        "HORiskScore",
        "NetworkSeverityScore",
        "QoSLevel",
        "CongestionLevel",
        "HORisk",
        "Decision",
        "PolicyRule",
        "Explanation",
    ]

    require_columns(
        df,
        required_outputs,
    )

    score_columns = [
        "QoSScore",
        "CongestionScore",
        "HORiskScore",
        "NetworkSeverityScore",
    ]

    for column in score_columns:

        invalid = (
            (~np.isfinite(df[column]))
            | (df[column] < 0)
            | (df[column] > 1)
        )

        if invalid.any():
            raise ValueError(
                f"Valeurs invalides dans {column} : "
                f"{int(invalid.sum())}"
            )

    if df["Decision"].eq("").any():
        raise ValueError(
            "Certaines observations ne possèdent aucune décision."
        )


# ===========================================================================
# Programme principal
# ===========================================================================

def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset brut introuvable : {INPUT_FILE}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LLM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "Throughput",
        "Delay",
        "Jitter",
        "PacketLoss",
        "PDR",
        "Congestion",
        "HOAttempts",
        "HOSuccessRate",
        "HOFailureRate",
        "HODelay",
    ]

    require_columns(
        df,
        required_columns,
    )

    if df.empty:
        raise ValueError(
            "Le fichier dataset_metrics.csv est vide."
        )

    # Construction du dataset.
    df = build_scores(df)
    df = assign_levels(df)
    df = assign_decisions(df)

    df["Explanation"] = df.apply(
        build_explanation,
        axis=1,
    )

    validate_output(df)

    # Export.
    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    export_jsonl(
        df,
        OUTPUT_JSONL,
    )

    export_policy_table(
        df,
        OUTPUT_POLICY,
    )

    write_summary(
        df,
        OUTPUT_SUMMARY,
    )

    print("=" * 65)
    print("DATASET DÉCISIONNEL V3 GÉNÉRÉ")
    print("=" * 65)
    print(f"Observations : {len(df)}")
    print(f"Colonnes     : {len(df.columns)}")
    print(f"CSV          : {OUTPUT_CSV}")
    print(f"JSONL        : {OUTPUT_JSONL}")
    print(f"Rapport      : {OUTPUT_SUMMARY}")
    print(f"Politique    : {OUTPUT_POLICY}")

    print("\nDistribution des décisions :")
    print(df["Decision"].value_counts())

    print("\nDistribution des règles :")
    print(df["PolicyRule"].value_counts().sort_index())


if __name__ == "__main__":
    main()
