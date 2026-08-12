#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ALLOWED_ACTIONS = {
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
    "REDUCE_TRAFFIC_LOAD",
}
VALID_FACTORS = {"QOS", "CONGESTION", "HANDOVER_RISK", "BALANCED"}
DEFAULT_OUTPUT_DIR = Path("results/inference")

SYSTEM_INSTRUCTION = """
Tu es un contrôleur intelligent de mobilité V2X sur réseau 5G/6G.

Analyse uniquement les paramètres et les scores présents dans le scénario.

Choisis exactement une décision parmi :
- KEEP_CURRENT_CONNECTION
- MONITOR_NETWORK
- PREPARE_HANDOVER
- HANDOVER_NOW
- REDUCE_TRAFFIC_LOAD

Règles générales d'interprétation :
- KEEP_CURRENT_CONNECTION : état stable, QoS bonne, congestion faible
  et risque de handover faible.
- MONITOR_NETWORK : dégradation légère ou état intermédiaire sans
  nécessité d'action immédiate.
- REDUCE_TRAFFIC_LOAD : congestion dominante ou surcharge du réseau.
- PREPARE_HANDOVER : risque de handover élevé, mais situation encore
  suffisamment stable pour préparer la transition.
- HANDOVER_NOW : QoS critique ou risque de handover critique nécessitant
  une action immédiate.

Retourne uniquement un objet JSON contenant exactement les champs :
- decision
- confidence
- reasoning
- dominant_factor
- recommended_follow_up
- limitations

Contraintes de sortie :
- decision doit être une chaîne appartenant aux cinq décisions autorisées ;
- confidence doit être un nombre compris entre 0 et 1 ;
- dominant_factor doit être exactement QOS, CONGESTION,
  HANDOVER_RISK ou BALANCED ;
- reasoning doit citer les niveaux ou scores qui justifient la décision ;
- n'utilise aucune valeur d'exemple ;
- ne crée aucun objet imbriqué dans decision ;
- aucun texte ne doit apparaître avant ou après le JSON.
""".strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path)
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def read_json_file(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON introuvable : {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Le fichier de prédiction doit contenir un objet JSON.")
    return data


def http_get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Impossible de joindre Ollama : {exc}") from exc


def http_post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Erreur HTTP Ollama {exc.code} : {details}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Impossible de joindre Ollama : {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Réponse Ollama invalide.")
    return data


def check_ollama_server(base_url: str, model: str, timeout: float) -> None:
    tags = http_get_json(
        f"{base_url.rstrip('/')}/api/tags",
        min(timeout, 15.0),
    )
    names = {
        str(item.get("name", ""))
        for item in tags.get("models", [])
        if isinstance(item, dict)
    }
    if model not in names:
        raise RuntimeError(
            f"Modèle '{model}' absent. Exécute : ollama pull {model}"
        )


def normalize_action(value: Any) -> str:
    return (
        str(value)
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )


def first_valid_action(*candidates: Any) -> str:
    for candidate in candidates:
        values = candidate if isinstance(candidate, list) else [candidate]
        for value in values:
            normalized = normalize_action(value)
            if normalized in ALLOWED_ACTIONS:
                return normalized
    return ""


def find_expert_decision(data: Any) -> str | None:
    if isinstance(data, dict):
        for key in (
            "expert_decision",
            "policy_decision",
            "recommended_decision",
            "decision_expert",
            "expert_action",
        ):
            value = data.get(key)
            if isinstance(value, str):
                action = normalize_action(value)
                if action in ALLOWED_ACTIONS:
                    return action
        for value in data.values():
            found = find_expert_decision(value)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_expert_decision(value)
            if found:
                return found
    return None


def build_full_prompt(
    scenario_prompt: str,
    prediction_data: dict[str, Any] | None,
) -> str:
    result = (
        SYSTEM_INSTRUCTION
        + "\n\nSCÉNARIO À ANALYSER :\n"
        + scenario_prompt.strip()
    )
    
    return result


def extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Aucun objet JSON valide trouvé dans la réponse du LLM.")


def normalize_llm_response(
    response: dict[str, Any],
    expert_decision: str | None = None,
) -> dict[str, Any]:
    decision_node = response.get("decision")
    nested = decision_node if isinstance(decision_node, dict) else {}

    if isinstance(decision_node, str):
        decision = first_valid_action(decision_node)
    elif isinstance(decision_node, dict):
        decision = first_valid_action(
            nested.get("decision"),
            nested.get("action"),
            nested.get("recommended_action"),
            nested.get("recommended_actions"),
        )
    else:
        decision = first_valid_action(
            response.get("action"),
            response.get("recommended_action"),
            response.get("recommended_actions"),
        )

    if not decision:
        raise ValueError(
            f"Décision LLM invalide. Actions autorisées : {sorted(ALLOWED_ACTIONS)}"
        )

    raw_confidence = response.get(
        "confidence",
        nested.get(
            "confidence",
            response.get("confidence_score", response.get("certainty", 0.5)),
        ),
    )
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(
        response.get(
            "reasoning",
            nested.get(
                "reasoning",
                response.get(
                    "technical_explanation",
                    response.get(
                        "explanation",
                        response.get("justification", ""),
                    ),
                ),
            ),
        )
    ).strip()

    dominant_factor = str(
        response.get(
            "dominant_factor",
            nested.get(
                "dominant_factor",
                response.get("main_factor", "BALANCED"),
            ),
        )
    ).strip().upper()

    aliases = {
        "QOS_SCORE": "QOS",
        "QUALITY_OF_SERVICE": "QOS",
        "CONGESTION_SCORE": "CONGESTION",
        "HANDOVER": "HANDOVER_RISK",
        "HO_RISK": "HANDOVER_RISK",
        "MOBILITY": "HANDOVER_RISK",
    }
    dominant_factor = aliases.get(dominant_factor, dominant_factor)
    if "|" in dominant_factor or dominant_factor not in VALID_FACTORS:
        dominant_factor = "BALANCED"

    if expert_decision is not None:
        agreement = decision == normalize_action(expert_decision)
    else:
        agreement = response.get(
            "agreement_with_expert",
            nested.get("agreement_with_expert"),
        )
        if isinstance(agreement, str):
            agreement = agreement.strip().lower() in {
                "true",
                "yes",
                "oui",
                "1",
            }
        elif not isinstance(agreement, bool):
            agreement = None

    follow_up = response.get(
        "recommended_follow_up",
        nested.get(
            "recommended_follow_up",
            response.get("follow_up", nested.get("follow_up", "")),
        ),
    )
    if not follow_up:
        actions = nested.get(
            "recommended_actions",
            response.get("recommended_actions", []),
        )
        if isinstance(actions, list):
            follow_up = ", ".join(str(item) for item in actions)
        elif actions:
            follow_up = str(actions)

    limitations = str(
        response.get("limitations", nested.get("limitations", ""))
    ).strip()

    placeholders = {
        "",
        "...",
        "Justification technique concise.",
        "Action de suivi concise.",
    }
    if reasoning in placeholders:
        reasoning = ""
    if str(follow_up).strip() in placeholders:
        follow_up = ""
    if limitations in placeholders:
        limitations = ""

    return {
        "decision": decision,
        "confidence": confidence,
        "reasoning": reasoning,
        "dominant_factor": dominant_factor,
        "agreement_with_expert": agreement,
        "recommended_follow_up": str(follow_up).strip(),
        "limitations": limitations,
    }


def main() -> int:
    args = parse_arguments()
    prompt_file = args.prompt_file.expanduser().resolve()

    if not prompt_file.exists():
        print(f"Erreur : prompt introuvable : {prompt_file}", file=sys.stderr)
        return 1

    prediction_file = (
        args.prediction_file.expanduser().resolve()
        if args.prediction_file
        else None
    )

    try:
        check_ollama_server(args.ollama_url, args.model, args.timeout)

        scenario_prompt = prompt_file.read_text(encoding="utf-8")
        prediction_data = read_json_file(prediction_file)
        expert_decision = find_expert_decision(prediction_data)
        response_schema = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "KEEP_CURRENT_CONNECTION",
                "MONITOR_NETWORK",
                "PREPARE_HANDOVER",
                "HANDOVER_NOW",
                "REDUCE_TRAFFIC_LOAD",
            ],
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "reasoning": {
            "type": "string",
        },
        "dominant_factor": {
            "type": "string",
            "enum": [
                "QOS",
                "CONGESTION",
                "HANDOVER_RISK",
                "BALANCED",
            ],
        },
        "recommended_follow_up": {
            "type": "string",
        },
        "limitations": {
            "type": "string",
        },
    },
    "required": [
        "decision",
        "confidence",
        "reasoning",
        "dominant_factor",
        "recommended_follow_up",
        "limitations",
    ],
    "additionalProperties": False,
}
        payload = {
            "model": args.model,
            "prompt": build_full_prompt(
                 scenario_prompt,
                 prediction_data=None,
             ),
            "stream": False,
            "format": response_schema,
            "options": {
                "temperature": args.temperature,
                "seed": 42,
            },
        }

        print(f"Modèle      : {args.model}")
        print(f"Prompt      : {prompt_file}")
        print(f"Serveur     : {args.ollama_url}")
        print("Inférence en cours...")

        start = time.perf_counter()
        ollama_response = http_post_json(
            f"{args.ollama_url.rstrip('/')}/api/generate",
            payload,
            args.timeout,
        )
        elapsed = time.perf_counter() - start

        raw_text = str(ollama_response.get("response", "")).strip()
        print("\n===== RÉPONSE BRUTE DU LLM =====")
        print(raw_text)
        print("================================\n")

        parsed_response = extract_json_object(raw_text)
        normalized = normalize_llm_response(
            parsed_response,
            expert_decision=expert_decision,
        )

        output_file = (
            args.output_file.expanduser().resolve()
            if args.output_file
            else (
                DEFAULT_OUTPUT_DIR
                / f"{prompt_file.stem}_llm_response.json"
            ).resolve()
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        final_result = {
            "model": args.model,
            "prompt_file": str(prompt_file),
            "prediction_file": (
                str(prediction_file) if prediction_file else None
            ),
            "expert_decision": expert_decision,
            "elapsed_seconds": round(elapsed, 4),
            "llm_decision": normalized,
            "raw_response": parsed_response,
            "ollama_metadata": {
                key: value
                for key, value in ollama_response.items()
                if key != "response"
            },
        }
        output_file.write_text(
            json.dumps(final_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"Décision     : {normalized['decision']}")
        print(f"Confiance    : {normalized['confidence']:.2f}")
        print(f"Facteur      : {normalized['dominant_factor']}")
        print(f"Accord expert: {normalized['agreement_with_expert']}")
        print(f"Durée        : {elapsed:.2f} s")
        print(f"Résultat     : {output_file}")
        return 0

    except (
        ConnectionError,
        FileNotFoundError,
        json.JSONDecodeError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterruption utilisateur.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
