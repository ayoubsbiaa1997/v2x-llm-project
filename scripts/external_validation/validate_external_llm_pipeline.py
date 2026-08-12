#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path.home() / "v2x-llm-project"
DEFAULT_INPUT = ROOT / "results/external_validation/ngsim/adaptive_ml/ml_predictions_ngsim.csv"
DEFAULT_OUTPUT = ROOT / "results/external_validation/ngsim/llm_validation"

LABELS = [
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "REDUCE_TRAFFIC_LOAD",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
]

REQUIRED_COLUMNS = [
    "Scenario",
    "gNbNum",
    "ueNumPerGnb",
    "totalUE",
    "bsDistance",
    "ueSpeed",
    "lambdaULL",
    "lambdaBe",
    "QoSScore_PredictedNGSIM",
    "CongestionScore_PredictedNGSIM",
    "HORiskScore_PredictedNGSIM",
    "QoSLevel_PredictedNGSIM",
    "CongestionLevel_PredictedNGSIM",
    "HORiskLevel_PredictedNGSIM",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default="qwen2.5:3b")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")[:80]


def require_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Colonnes requises absentes : " + ", ".join(missing))


def select_sample(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    stratify_column = (
        "adaptive_scenario_class"
        if "adaptive_scenario_class" in df.columns
        else "CongestionLevel_PredictedNGSIM"
    )

    groups = [group for _, group in df.groupby(stratify_column, dropna=False)]
    base, remainder = divmod(min(limit, len(df)), max(1, len(groups)))

    parts: list[pd.DataFrame] = []
    used: set[int] = set()

    for index, group in enumerate(groups):
        n = min(base + (1 if index < remainder else 0), len(group))
        if n:
            part = group.sample(n=n, random_state=seed + index)
            parts.append(part)
            used.update(part.index.tolist())

    selected = pd.concat(parts) if parts else df.iloc[0:0]
    missing = min(limit, len(df)) - len(selected)

    if missing > 0:
        remaining = df.loc[~df.index.isin(used)]
        if not remaining.empty:
            selected = pd.concat([
                selected,
                remaining.sample(
                    n=min(missing, len(remaining)),
                    random_state=seed + 100,
                ),
            ])

    return selected.sample(frac=1, random_state=seed + 200).reset_index(drop=True)


def derive_reference_policy(row: pd.Series) -> tuple[str, str]:
    qos = float(row["QoSScore_PredictedNGSIM"])
    congestion = float(row["CongestionScore_PredictedNGSIM"])
    ho_risk = float(row["HORiskScore_PredictedNGSIM"])

    if ho_risk >= 0.75 or (ho_risk >= 0.50 and qos >= 0.75):
        return (
            "HANDOVER_NOW",
            "Risque de handover élevé associé à une QoS critique.",
        )

    if ho_risk >= 0.50:
        return (
            "PREPARE_HANDOVER",
            "Le risque de handover est élevé sans urgence immédiate.",
        )

    if congestion >= 0.75:
        return (
            "REDUCE_TRAFFIC_LOAD",
            "La congestion est critique avec un risque de handover limité.",
        )

    if qos >= 0.50 or congestion >= 0.50 or ho_risk >= 0.25:
        return (
            "MONITOR_NETWORK",
            "Au moins un indicateur est dégradé et nécessite une surveillance.",
        )

    return (
        "KEEP_CURRENT_CONNECTION",
        "Les indicateurs restent compatibles avec la connexion actuelle.",
    )


def build_prompt(row: pd.Series) -> str:
    mobility_context = {
        "vehicle_density_per_km": row.get("vehicle_density_per_km"),
        "mean_speed_mps": row.get("mean_speed_mps", row.get("ueSpeed")),
        "speed_std_mps": row.get("speed_std_mps"),
        "lane_change_rate_per_vehicle": row.get(
            "lane_change_rate_per_vehicle"
        ),
        "mean_space_headway_m": row.get("mean_space_headway_m"),
        "mobility_congestion_index": row.get("mobility_congestion_index"),
    }

    context_json = json.dumps(
        mobility_context,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    scenario_class = str(row.get("adaptive_scenario_class", "UNSPECIFIED"))

    return f"""You are a V2X network decision assistant.

Select exactly one action from:
KEEP_CURRENT_CONNECTION
MONITOR_NETWORK
REDUCE_TRAFFIC_LOAD
PREPARE_HANDOVER
HANDOVER_NOW

Use only the supplied values. Do not invent measurements and do not mention
any hidden reference policy.

SCENARIO
scenario_id: {row["Scenario"]}
adaptive_scenario_class: {scenario_class}
gNbNum: {int(row["gNbNum"])}
ueNumPerGnb: {int(row["ueNumPerGnb"])}
totalUE: {int(row["totalUE"])}
bsDistance_m: {float(row["bsDistance"]):.3f}
ueSpeed_mps: {float(row["ueSpeed"]):.6f}
lambdaULL: {float(row["lambdaULL"]):.3f}
lambdaBE: {float(row["lambdaBe"]):.3f}

PREDICTED NETWORK STATE
QoSScore: {float(row["QoSScore_PredictedNGSIM"]):.6f}
QoSLevel: {row["QoSLevel_PredictedNGSIM"]}
CongestionScore: {float(row["CongestionScore_PredictedNGSIM"]):.6f}
CongestionLevel: {row["CongestionLevel_PredictedNGSIM"]}
HORiskScore: {float(row["HORiskScore_PredictedNGSIM"]):.6f}
HORiskLevel: {row["HORiskLevel_PredictedNGSIM"]}

REAL MOBILITY CONTEXT FROM NGSIM
{context_json}

Return ONLY valid JSON:
{{
  "decision": "ONE_AUTHORIZED_ACTION",
  "explanation": "Brief technical explanation grounded in the values."
}}
"""


def run_ollama(model: str, prompt: str, timeout: int) -> tuple[str, float]:
    start = time.perf_counter()
    completed = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start

    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"Code de retour {completed.returncode}"
        )

    return completed.stdout.strip(), elapsed


def escape_control_characters_in_strings(text: str) -> str:
    """
    Échappe les retours à la ligne et tabulations placés directement
    dans les chaînes JSON par certains LLM.
    """
    output: list[str] = []
    inside_string = False
    escaped = False

    for character in text:
        if escaped:
            output.append(character)
            escaped = False
            continue

        if character == "\\":
            output.append(character)

            if inside_string:
                escaped = True

            continue

        if character == '"':
            inside_string = not inside_string
            output.append(character)
            continue

        if inside_string:
            if character == "\n":
                output.append(r"\n")
                continue

            if character == "\r":
                output.append(r"\r")
                continue

            if character == "\t":
                output.append(r"\t")
                continue

        output.append(character)

    return "".join(output)


def extract_json_object(response: str) -> dict[str, Any]:
    candidates = [response.strip()]

    candidates.extend(
        re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )

    first = response.find("{")
    last = response.rfind("}")

    if first >= 0 and last > first:
        candidates.append(
            response[first:last + 1]
        )

    errors: list[str] = []

    for candidate in candidates:
        attempts = [
            candidate,
            escape_control_characters_in_strings(
                candidate
            ),
        ]

        for attempt in attempts:
            try:
                data = json.loads(attempt, strict=False)
            except Exception as exc:
                errors.append(str(exc))
                continue

            if isinstance(data, dict):
                return data

    # Repli pour les modèles qui répondent en texte libre
    # au lieu de produire un objet JSON strict.
    cleaned_response = response.strip()

    explicit_patterns = [
        r"(?i)\bdecision\s*[:=\-]\s*([A-Z_]+)",
        r"(?i)\baction\s*[:=\-]\s*([A-Z_]+)",
        r"(?i)\brecommended\s+action\s*[:=\-]\s*([A-Z_]+)",
        r"(?i)\bnetwork\s+action\s*[:=\-]\s*([A-Z_]+)",
    ]

    detected_decision = None

    for pattern in explicit_patterns:
        match = re.search(pattern, cleaned_response)

        if match:
            candidate = match.group(1).strip().upper()

            aliases = {
                "KEEP": "KEEP_CURRENT_CONNECTION",
                "KEEP_CONNECTION": "KEEP_CURRENT_CONNECTION",
                "MONITOR": "MONITOR_NETWORK",
                "REDUCE_LOAD": "REDUCE_TRAFFIC_LOAD",
                "PREPARE_HO": "PREPARE_HANDOVER",
                "HANDOVER": "HANDOVER_NOW",
                "HO_NOW": "HANDOVER_NOW",
            }

            candidate = aliases.get(candidate, candidate)

            if candidate in LABELS:
                detected_decision = candidate
                break

    # Si aucune ligne explicite n'est trouvée, rechercher les actions
    # autorisées présentes dans la réponse.
    if detected_decision is None:
        detected_labels = [
            label
            for label in LABELS
            if re.search(
                rf"(?<![A-Z_]){re.escape(label)}(?![A-Z_])",
                cleaned_response.upper(),
            )
        ]

        if len(detected_labels) == 1:
            detected_decision = detected_labels[0]
        elif len(detected_labels) > 1:
            # Rechercher d'abord une formulation décisionnelle explicite.
            priority_patterns = [
                r"(?i)(?:choose|select|recommend|decision is|action is)"
                r"[^A-Z_]*(KEEP_CURRENT_CONNECTION|MONITOR_NETWORK|"
                r"REDUCE_TRAFFIC_LOAD|PREPARE_HANDOVER|HANDOVER_NOW)",
                r"(?i)\b(KEEP_CURRENT_CONNECTION|MONITOR_NETWORK|"
                r"REDUCE_TRAFFIC_LOAD|PREPARE_HANDOVER|HANDOVER_NOW)"
                r"\b\s*(?:is selected|is recommended)",
            ]

            for priority_pattern in priority_patterns:
                priority_match = re.search(
                    priority_pattern,
                    cleaned_response,
                )

                if priority_match:
                    detected_decision = (
                        priority_match.group(1).upper()
                    )
                    break

    if detected_decision is not None:
        explanation = cleaned_response

        explanation = re.sub(
            r"(?i)^\s*(decision|action|recommended action|network action)"
            r"\s*[:=\-]\s*[A-Z_]+\s*",
            "",
            explanation,
            count=1,
        ).strip()

        explanation = re.sub(
            r"(?i)^\s*explanation\s*[:=\-]\s*",
            "",
            explanation,
            count=1,
        ).strip()

        return {
            "decision": detected_decision,
            "explanation": explanation,
        }

    raise ValueError(
        "Aucun objet JSON ni aucune décision textuelle valide détectée. "
        + " | ".join(errors[:4])
    )


def normalize_decision(value: Any) -> str:
    decision = str(value).strip().upper()
    aliases = {
        "KEEP": "KEEP_CURRENT_CONNECTION",
        "KEEP_CONNECTION": "KEEP_CURRENT_CONNECTION",
        "MONITOR": "MONITOR_NETWORK",
        "REDUCE_LOAD": "REDUCE_TRAFFIC_LOAD",
        "PREPARE_HO": "PREPARE_HANDOVER",
        "HANDOVER": "HANDOVER_NOW",
        "HO_NOW": "HANDOVER_NOW",
    }
    decision = aliases.get(decision, decision)

    if decision not in LABELS:
        raise ValueError(f"Décision non autorisée : {decision}")

    return decision


def evaluate_model(
    model: str,
    sample: pd.DataFrame,
    output_dir: Path,
    timeout: int,
    resume: bool,
) -> pd.DataFrame:
    model_dir = output_dir / slug(model.replace(":", "_"))
    response_dir = model_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)

    results_path = model_dir / "external_llm_results.csv"
    previous: dict[str, dict[str, Any]] = {}

    if resume and results_path.exists():
        old = pd.read_csv(results_path)
        previous = {
            str(row["Scenario"]): row.to_dict()
            for _, row in old.iterrows()
            if str(row.get("status", "")) == "SUCCESS"
        }

    records: list[dict[str, Any]] = []

    print()
    print("=" * 76)
    print(f"MODÈLE : {model}")
    print("=" * 76)

    for index, row in sample.iterrows():
        scenario = str(row["Scenario"])
        reference, reference_reason = derive_reference_policy(row)

        if scenario in previous:
            records.append(previous[scenario])
            print(f"[{index + 1}/{len(sample)}] {scenario} -> repris")
            continue

        prompt = build_prompt(row)
        base = response_dir / slug(scenario)
        base.with_name(base.name + "_prompt.txt").write_text(
            prompt,
            encoding="utf-8",
        )

        record: dict[str, Any] = {
            "Scenario": scenario,
            "model": model,
            "reference_policy": reference,
            "reference_reason": reference_reason,
            "llm_decision": "",
            "llm_explanation": "",
            "agreement": False,
            "valid_json": False,
            "inference_seconds": np.nan,
            "status": "FAILED",
            "error": "",
        }

        try:
            raw_response, elapsed = run_ollama(model, prompt, timeout)
            base.with_name(base.name + "_raw.txt").write_text(
                raw_response + "\n",
                encoding="utf-8",
            )

            data = extract_json_object(raw_response)
            decision = normalize_decision(data.get("decision", ""))
            explanation = str(data.get("explanation", "")).strip()

            base.with_name(base.name + "_response.json").write_text(
                json.dumps(
                    {
                        "decision": decision,
                        "explanation": explanation,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            record.update({
                "llm_decision": decision,
                "llm_explanation": explanation,
                "agreement": decision == reference,
                "valid_json": True,
                "inference_seconds": elapsed,
                "status": "SUCCESS",
            })

            print(
                f"[{index + 1}/{len(sample)}] {scenario} "
                f"| ref={reference} | LLM={decision} "
                f"| accord={'OUI' if decision == reference else 'NON'} "
                f"| {elapsed:.2f}s"
            )

        except Exception as exc:
            record["error"] = str(exc)
            print(
                f"[{index + 1}/{len(sample)}] "
                f"{scenario} | ÉCHEC : {exc}"
            )

        records.append(record)
        pd.DataFrame(records).to_csv(results_path, index=False)

    results = pd.DataFrame(records)
    results.to_csv(results_path, index=False)
    return results


def create_metrics(model: str, results: pd.DataFrame) -> dict[str, Any]:
    successful = results[results["status"] == "SUCCESS"].copy()

    metrics: dict[str, Any] = {
        "model": model,
        "requested": int(len(results)),
        "successful": int(len(successful)),
        "failed": int(len(results) - len(successful)),
        "valid_json_rate": float(
            results["valid_json"].fillna(False).astype(bool).mean()
        ),
    }

    if successful.empty:
        return metrics

    y_true = successful["reference_policy"].astype(str).tolist()
    y_pred = successful["llm_decision"].astype(str).tolist()
    active = [label for label in LABELS if label in set(y_true)]

    metrics.update({
        "agreement_rate": float(accuracy_score(y_true, y_pred)),
        "balanced_agreement": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "active_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=active,
                average="macro",
                zero_division=0,
            )
        ) if active else float("nan"),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "mean_inference_seconds": float(
            successful["inference_seconds"].mean()
        ),
        "median_inference_seconds": float(
            successful["inference_seconds"].median()
        ),
        "p95_inference_seconds": float(
            successful["inference_seconds"].quantile(0.95)
        ),
        "active_reference_labels": active,
        "reference_distribution": {
            str(key): int(value)
            for key, value
            in successful["reference_policy"].value_counts().items()
        },
        "llm_distribution": {
            str(key): int(value)
            for key, value
            in successful["llm_decision"].value_counts().items()
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        ).tolist(),
        "methodological_note": (
            "Les métriques mesurent l'accord avec une politique dérivée "
            "des états ML transférés, et non une accuracy radio externe."
        ),
    })

    return metrics


def main() -> int:
    args = parse_args()
    input_file = args.input_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_file.exists():
        print(f"Fichier introuvable : {input_file}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_file)
    require_columns(df)
    sample = select_sample(df, args.limit, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(
        output_dir / "selected_ngsim_scenarios.csv",
        index=False,
    )

    reference_rows = []
    for _, row in sample.iterrows():
        decision, reason = derive_reference_policy(row)
        reference_rows.append({
            "Scenario": row["Scenario"],
            "reference_policy": decision,
            "reference_reason": reason,
        })

    pd.DataFrame(reference_rows).to_csv(
        output_dir / "transferred_reference_policy.csv",
        index=False,
    )

    models = [
        model.strip()
        for model in args.models.split(",")
        if model.strip()
    ]
    if not models:
        raise ValueError("Aucun modèle fourni.")

    print("=" * 76)
    print("VALIDATION EXTERNE DU PIPELINE LLM SUR NGSIM")
    print("=" * 76)
    print(f"Entrée    : {input_file}")
    print(f"Scénarios : {len(sample)}")
    print(f"Modèles   : {', '.join(models)}")
    print(f"Sortie    : {output_dir}")

    comparison_rows = []

    for model in models:
        results = evaluate_model(
            model,
            sample,
            output_dir,
            args.timeout,
            args.resume,
        )
        metrics = create_metrics(model, results)

        model_dir = output_dir / slug(model.replace(":", "_"))
        (model_dir / "external_llm_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        comparison_rows.append({
            key: value
            for key, value in metrics.items()
            if key not in {
                "classification_report",
                "confusion_matrix",
                "reference_distribution",
                "llm_distribution",
                "active_reference_labels",
                "methodological_note",
            }
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        output_dir / "external_llm_model_comparison.csv",
        index=False,
    )

    print()
    print("=" * 76)
    print("VALIDATION EXTERNE TERMINÉE")
    print("=" * 76)

    display = [
        column
        for column in [
            "model",
            "successful",
            "valid_json_rate",
            "agreement_rate",
            "balanced_agreement",
            "macro_f1",
            "active_macro_f1",
            "mean_inference_seconds",
        ]
        if column in comparison.columns
    ]

    if display:
        print(comparison[display].to_string(index=False))

    print()
    print(f"Sortie : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
