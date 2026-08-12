#!/usr/bin/env python3

"""
Comparaison entre :
- la décision de la politique experte V3 ;
- la décision produite par le LLM.

Le script accepte différentes structures JSON et recherche
récursivement les champs de décision les plus probables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path.home() / "v2x-llm-project"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "inference"


ALLOWED_ACTIONS = {
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
    "REDUCE_TRAFFIC_LOAD",
}


EXPERT_DECISION_KEYS = (
    "expert_decision",
    "policy_decision",
    "recommended_action",
    "network_action",
    "action",
    "decision",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparer la politique experte et la décision LLM."
    )

    parser.add_argument(
        "--prediction-file",
        required=True,
        type=Path,
        help="Fichier JSON généré par le pipeline de prédiction.",
    )

    parser.add_argument(
        "--llm-file",
        required=True,
        type=Path,
        help="Fichier JSON produit par run_llm_decision.py.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Fichier JSON de comparaison.",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Le fichier n'est pas un objet JSON : {path}")

    return data


def normalize_action(value: Any) -> str | None:
    if value is None:
        return None

    action = str(value).strip().upper()
    action = action.replace("-", "_").replace(" ", "_")

    if action in ALLOWED_ACTIONS:
        return action

    return None


def find_action_recursively(
    data: Any,
    preferred_keys: tuple[str, ...],
) -> tuple[str | None, str | None]:
    """
    Retourne :
    - l'action trouvée ;
    - le chemin JSON correspondant.
    """

    def visit(
        node: Any,
        path: str,
    ) -> tuple[str | None, str | None]:
        if isinstance(node, dict):

            for key in preferred_keys:
                if key in node:
                    action = normalize_action(node[key])

                    if action:
                        return action, f"{path}.{key}"

            for key, value in node.items():
                result, result_path = visit(
                    value,
                    f"{path}.{key}",
                )

                if result:
                    return result, result_path

        elif isinstance(node, list):
            for index, value in enumerate(node):
                result, result_path = visit(
                    value,
                    f"{path}[{index}]",
                )

                if result:
                    return result, result_path

        return None, None

    return visit(data, "$")


def extract_llm_decision(
    llm_data: dict[str, Any],
) -> tuple[str | None, str | None]:
    llm_section = llm_data.get("llm_decision")

    if isinstance(llm_section, dict):
        action = normalize_action(llm_section.get("decision"))

        if action:
            return action, "$.llm_decision.decision"

    return find_action_recursively(
        llm_data,
        ("decision", "action", "recommended_action"),
    )


def determine_relation(
    expert_action: str,
    llm_action: str,
) -> str:
    if expert_action == llm_action:
        return "EXACT_AGREEMENT"

    close_actions = {
        frozenset({
            "MONITOR_NETWORK",
            "KEEP_CURRENT_CONNECTION",
        }),
        frozenset({
            "PREPARE_HANDOVER",
            "HANDOVER_NOW",
        }),
        frozenset({
            "MONITOR_NETWORK",
            "PREPARE_HANDOVER",
        }),
    }

    if frozenset({expert_action, llm_action}) in close_actions:
        return "PARTIAL_AGREEMENT"

    return "DISAGREEMENT"


def main() -> int:
    args = parse_arguments()

    prediction_file = args.prediction_file.expanduser().resolve()
    llm_file = args.llm_file.expanduser().resolve()

    try:
        prediction_data = load_json(prediction_file)
        llm_data = load_json(llm_file)

        expert_action, expert_path = find_action_recursively(
            prediction_data,
            EXPERT_DECISION_KEYS,
        )

        llm_action, llm_path = extract_llm_decision(llm_data)

        if expert_action is None:
            raise ValueError(
                "Impossible d'identifier la décision experte dans "
                f"{prediction_file}."
            )

        if llm_action is None:
            raise ValueError(
                "Impossible d'identifier la décision LLM dans "
                f"{llm_file}."
            )

        relation = determine_relation(
            expert_action=expert_action,
            llm_action=llm_action,
        )

        exact_agreement = expert_action == llm_action

        llm_section = llm_data.get("llm_decision", {})

        comparison = {
            "prediction_file": str(prediction_file),
            "llm_file": str(llm_file),
            "expert": {
                "decision": expert_action,
                "json_path": expert_path,
            },
            "llm": {
                "decision": llm_action,
                "json_path": llm_path,
                "confidence": (
                    llm_section.get("confidence")
                    if isinstance(llm_section, dict)
                    else None
                ),
                "reasoning": (
                    llm_section.get("reasoning")
                    if isinstance(llm_section, dict)
                    else None
                ),
            },
            "comparison": {
                "exact_agreement": exact_agreement,
                "relation": relation,
            },
        }

        if args.output_file:
            output_file = args.output_file.expanduser().resolve()
        else:
            output_file = (
                DEFAULT_OUTPUT_DIR
                / f"{prediction_file.stem}_policy_llm_comparison.json"
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(
                comparison,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print("\nComparaison politique experte / LLM")
        print("-----------------------------------")
        print(f"Décision experte : {expert_action}")
        print(f"Décision LLM     : {llm_action}")
        print(f"Relation         : {relation}")
        print(f"Accord exact     : {exact_agreement}")
        print(f"\nRésultat : {output_file}")

        return 0

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
