#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path.home() / "v2x-llm-project"
DATASET = ROOT / "datasets/processed/final_dataset_v3.csv"
PREDICT = ROOT / "scripts/predict_v2x_decision.py"
RUN_LLM = ROOT / "scripts/run_llm_decision.py"
INFERENCE = ROOT / "results/inference"
OUT_ROOT = ROOT / "results/llm_evaluation"

LABELS = [
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "REDUCE_TRAFFIC_LOAD",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
]


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3.2:1b")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")[:70]


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Objet JSON attendu : {path}")
    return data


def select_sample(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    available = [
        label for label in LABELS if label in set(df["Decision"].astype(str))
    ]
    if not available:
        raise ValueError("Aucune classe de décision reconnue.")

    base, remainder = divmod(min(limit, len(df)), len(available))
    parts = []
    used: set[int] = set()

    for i, label in enumerate(available):
        group = df[df["Decision"] == label]
        n = min(base + (1 if i < remainder else 0), len(group))
        if n:
            part = group.sample(n=n, random_state=seed + i)
            parts.append(part)
            used.update(part.index.tolist())

    selected = pd.concat(parts) if parts else df.iloc[0:0]
    missing = min(limit, len(df)) - len(selected)

    if missing > 0:
        remaining = df.loc[~df.index.isin(used)]
        if not remaining.empty:
            selected = pd.concat(
                [
                    selected,
                    remaining.sample(
                        n=min(missing, len(remaining)),
                        random_state=seed + 100,
                    ),
                ]
            )

    return selected.sample(frac=1, random_state=seed + 200).reset_index(drop=True)


def save_confusion(y_true: list[str], y_pred: list[str], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Décision LLM",
        ylabel="Politique V3",
        title="Matrice de confusion — Politique V3 vs LLM",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = args_parser()

    for required in (DATASET, PREDICT, RUN_LLM):
        if not required.exists():
            print(f"Fichier introuvable : {required}", file=sys.stderr)
            return 1

    df = pd.read_csv(DATASET)
    sample = select_sample(df, args.limit, args.seed)

    model_name = slug(args.model.replace(":", "_"))
    out_dir = OUT_ROOT / f"{model_name}_pilot_{len(sample)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_dir / "selected_scenarios.csv", index=False)

    records: list[dict[str, Any]] = []

    print("=" * 72)
    print("ÉVALUATION PILOTE ZERO-SHOT V2X")
    print("=" * 72)
    print(f"Modèle    : {args.model}")
    print(f"Scénarios : {len(sample)}")
    print(f"Sortie    : {out_dir}\n")

    for idx, row in sample.iterrows():
        scenario = str(row.get("Scenario", f"row_{idx}"))
        name = f"pilot_{idx+1:03d}_{slug(scenario)}_{model_name}"
        pred_file = INFERENCE / f"{name}_prediction.json"
        prompt_file = INFERENCE / f"{name}_llm_prompt.txt"
        llm_file = out_dir / f"{name}_llm_response.json"

        print(
            f"[{idx+1}/{len(sample)}] {scenario} "
            f"| dataset_policy={row['Decision']}"
        )
        record: dict[str, Any] = {
            "scenario": scenario,
            "run_name": name,
            "dataset_policy_decision": str(row["Decision"]),
            "status": "FAILED",
            "error": "",
        }

        try:
            if not (
                args.resume and pred_file.exists() and prompt_file.exists()
            ):
                pred_cmd = [
                    sys.executable,
                    str(PREDICT),
                    "--name", name,
                    "--gnb-num", str(row["gNbNum"]),
                    "--ue-per-gnb", str(row["ueNumPerGnb"]),
                    "--bs-distance", str(row["bsDistance"]),
                    "--ue-speed", str(row["ueSpeed"]),
                    "--lambda-ull", str(row["lambdaULL"]),
                    "--lambda-be", str(row["lambdaBe"]),
                ]
                completed = run(pred_cmd, 120)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip())

            if not pred_file.exists() or not prompt_file.exists():
                raise FileNotFoundError("Prédiction ou prompt absent.")

            prompt = prompt_file.read_text(encoding="utf-8")
            leaked = [
                term
                for term in (
                    "POLITIQUE DE RÉFÉRENCE",
                    "Décision recommandée",
                    "Règle appliquée",
                    "agreement_with_expert",
                )
                if term in prompt
            ]
            if leaked:
                raise ValueError("Fuite de label : " + ", ".join(leaked))

            if not (args.resume and llm_file.exists()):
                llm_cmd = [
                    sys.executable,
                    str(RUN_LLM),
                    "--prompt-file", str(prompt_file),
                    "--prediction-file", str(pred_file),
                    "--model", args.model,
                    "--output-file", str(llm_file),
                    "--timeout", str(args.timeout),
                ]
                completed = run(llm_cmd, args.timeout + 30)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip())

            prediction = load_json(pred_file)
            llm = load_json(llm_file)

            expert = str(prediction["analysis"]["decision"])
            llm_decision = str(llm["llm_decision"]["decision"])
            agreement = expert == llm_decision

            record.update(
                {
                    "status": "SUCCESS",
                    "predicted_state_policy_decision": expert,
                    "llm_decision": llm_decision,
                    "exact_agreement": agreement,
                    "confidence": llm["llm_decision"].get("confidence"),
                    "dominant_factor": llm["llm_decision"].get(
                        "dominant_factor"
                    ),
                    "reasoning": llm["llm_decision"].get("reasoning"),
                    "elapsed_seconds": llm.get("elapsed_seconds"),
                    "prediction_file": str(pred_file),
                    "prompt_file": str(prompt_file),
                    "llm_file": str(llm_file),
                }
            )
            print(
              f"  -> predicted_policy={expert} "
              f"| LLM={llm_decision} "
              f"| accord={'OUI' if agreement else 'NON'}"
            )

        except Exception as exc:
            record["error"] = str(exc)
            print(f"  -> ÉCHEC : {exc}")

        records.append(record)
        pd.DataFrame(records).to_csv(
            out_dir / "pilot_results_partial.csv",
            index=False,
        )

    results = pd.DataFrame(records)
    results.to_csv(out_dir / "pilot_results.csv", index=False)

    success = results[results["status"] == "SUCCESS"].copy()
    failures = results[results["status"] != "SUCCESS"].copy()

    metrics: dict[str, Any] = {
        "model": args.model,
        "requested": args.limit,
        "selected": len(sample),
        "successful": len(success),
        "failed": len(failures),
        "valid_json_rate": len(success) / len(results) if len(results) else 0.0,
    }

    if not success.empty:
        y_true = success[
             "predicted_state_policy_decision"
        ].tolist()
        y_pred = success["llm_decision"].tolist()

        metrics.update(
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_precision": precision_score(
                    y_true, y_pred, labels=LABELS,
                    average="macro", zero_division=0
                ),
                "macro_recall": recall_score(
                    y_true, y_pred, labels=LABELS,
                    average="macro", zero_division=0
                ),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=LABELS,
                    average="macro", zero_division=0
                ),
                "mean_inference_seconds": pd.to_numeric(
                    success["elapsed_seconds"], errors="coerce"
                ).mean(),
                "median_inference_seconds": pd.to_numeric(
                    success["elapsed_seconds"], errors="coerce"
                ).median(),
            }
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(
            out_dir / "classification_report.csv"
        )
        save_confusion(
            y_true,
            y_pred,
            out_dir / "confusion_matrix.png",
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("ÉVALUATION TERMINÉE")
    print("=" * 72)
    print(f"Succès      : {len(success)}/{len(results)}")
    print(f"Échecs      : {len(failures)}")
    if "accuracy" in metrics:
        print(f"Accuracy    : {metrics['accuracy']:.4f}")
        print(f"Macro-F1    : {metrics['macro_f1']:.4f}")
        print(f"Temps moyen : {metrics['mean_inference_seconds']:.2f} s")
    print(f"Résultats   : {out_dir / 'pilot_results.csv'}")
    print(f"Métriques   : {out_dir / 'metrics.json'}")
    print(f"Matrice     : {out_dir / 'confusion_matrix.png'}")

    return 0 if failures.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
