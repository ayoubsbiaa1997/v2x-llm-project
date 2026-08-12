#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.home() / "v2x-llm-project"
DEFAULT_INPUT = ROOT / "results/llm_evaluation/final_benchmark_190/llm_final_comparison.csv"
DEFAULT_OUTPUT = ROOT / "results/llm_evaluation/benchmark_sensitivity_190"

PROFILES = {
    "Performance": {
        "Accuracy": 0.20, "MacroF1": 0.45, "MacroPrecision": 0.10,
        "MacroRecall": 0.15, "JSONValidRate": 0.05, "LatencyScore": 0.05,
    },
    "Equilibre": {
        "Accuracy": 0.25, "MacroF1": 0.30, "MacroPrecision": 0.10,
        "MacroRecall": 0.10, "JSONValidRate": 0.05, "LatencyScore": 0.20,
    },
    "Temps_reel": {
        "Accuracy": 0.15, "MacroF1": 0.20, "MacroPrecision": 0.10,
        "MacroRecall": 0.10, "JSONValidRate": 0.05, "LatencyScore": 0.40,
    },
    "Ultra_temps_reel": {
        "Accuracy": 0.10, "MacroF1": 0.10, "MacroPrecision": 0.05,
        "MacroRecall": 0.05, "JSONValidRate": 0.05, "LatencyScore": 0.65,
    },
    "Recherche": {
        "Accuracy": 0.15, "MacroF1": 0.50, "MacroPrecision": 0.10,
        "MacroRecall": 0.15, "JSONValidRate": 0.05, "LatencyScore": 0.05,
    },
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--latency-min", type=float, default=0.0)
    p.add_argument("--latency-max", type=float, default=0.80)
    p.add_argument("--latency-step", type=float, default=0.01)
    return p.parse_args()

def normalize(series, higher=True):
    s = series.astype(float)
    lo, hi = float(s.min()), float(s.max())
    if math.isclose(lo, hi):
        z = pd.Series(np.ones(len(s)), index=s.index)
    else:
        z = (s - lo) / (hi - lo)
    return z if higher else 1.0 - z

def prepare(df):
    required = {
        "Model", "Accuracy", "MacroF1", "MacroPrecision", "MacroRecall",
        "JSONValidRate", "MeanInferenceSeconds"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    out = df.copy()
    out["AccuracyNorm"] = normalize(out["Accuracy"])
    out["MacroF1Norm"] = normalize(out["MacroF1"])
    out["MacroPrecisionNorm"] = normalize(out["MacroPrecision"])
    out["MacroRecallNorm"] = normalize(out["MacroRecall"])
    out["JSONValidRateNorm"] = normalize(out["JSONValidRate"])
    out["LatencyScore"] = normalize(out["MeanInferenceSeconds"], higher=False)
    return out

def col(name):
    return {
        "Accuracy": "AccuracyNorm",
        "MacroF1": "MacroF1Norm",
        "MacroPrecision": "MacroPrecisionNorm",
        "MacroRecall": "MacroRecallNorm",
        "JSONValidRate": "JSONValidRateNorm",
        "LatencyScore": "LatencyScore",
    }[name]

def score(df, profile, weights):
    out = df.copy()
    out["Profile"] = profile
    out["Score"] = 0.0
    for criterion, weight in weights.items():
        out["Score"] += weight * out[col(criterion)]
    out["Rank"] = out["Score"].rank(method="min", ascending=False).astype(int)
    return out.sort_values(["Rank", "Model"]).reset_index(drop=True)

def profile_results(df):
    return pd.concat(
        [score(df, name, weights) for name, weights in PROFILES.items()],
        ignore_index=True
    )

def base_weights():
    base = {
        "Accuracy": 0.25, "MacroF1": 0.40, "MacroPrecision": 0.10,
        "MacroRecall": 0.15, "JSONValidRate": 0.10,
    }
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}

def continuous(df, weights_grid):
    rows = []
    base = base_weights()
    for lw in weights_grid:
        remaining = 1.0 - lw
        weights = {k: v * remaining for k, v in base.items()}
        weights["LatencyScore"] = float(lw)
        scored = score(df, f"latency_{lw:.2f}", weights)
        for _, row in scored.iterrows():
            rows.append({
                "LatencyWeight": float(lw),
                "Model": row["Model"],
                "Score": float(row["Score"]),
                "Rank": int(row["Rank"]),
                "Winner": bool(row["Rank"] == 1),
                **{f"Weight_{k}": float(v) for k, v in weights.items()},
            })
    return pd.DataFrame(rows)

def stability(df):
    rows = []
    for model, group in df.groupby("Model"):
        rows.append({
            "Model": model,
            "WinnerCount": int(group["Winner"].sum()),
            "WinnerRate": float(group["Winner"].mean()),
            "MeanRank": float(group["Rank"].mean()),
            "RankStd": float(group["Rank"].std(ddof=0)),
            "MeanScore": float(group["Score"].mean()),
            "ScoreStd": float(group["Score"].std(ddof=0)),
            "BestRank": int(group["Rank"].min()),
            "WorstRank": int(group["Rank"].max()),
        })
    return pd.DataFrame(rows).sort_values(["MeanRank", "WinnerRate"], ascending=[True, False])

def intervals(df):
    winners = (
        df[df["Winner"]]
        .sort_values("LatencyWeight")
        .drop_duplicates("LatencyWeight")
        [["LatencyWeight", "Model"]]
        .reset_index(drop=True)
    )
    if winners.empty:
        return pd.DataFrame(columns=["StartLatencyWeight", "EndLatencyWeight", "WinnerModel"])
    rows = []
    start = prev = float(winners.iloc[0]["LatencyWeight"])
    current = str(winners.iloc[0]["Model"])
    for _, row in winners.iloc[1:].iterrows():
        weight = float(row["LatencyWeight"])
        model = str(row["Model"])
        if model != current:
            rows.append({
                "StartLatencyWeight": start,
                "EndLatencyWeight": prev,
                "WinnerModel": current,
            })
            start = weight
            current = model
        prev = weight
    rows.append({
        "StartLatencyWeight": start,
        "EndLatencyWeight": prev,
        "WinnerModel": current,
    })
    return pd.DataFrame(rows)

def grouped_plot(profile_df, output):
    pivot = profile_df.pivot(index="Profile", columns="Model", values="Score").reindex(PROFILES.keys())
    x = np.arange(len(pivot))
    width = 0.8 / len(pivot.columns)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[model], width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score multicritère")
    ax.set_title("Scores selon les profils de pondération")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

def heatmap(profile_df, output):
    pivot = profile_df.pivot(index="Model", columns="Profile", values="Rank").reindex(columns=PROFILES.keys())
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Rang des modèles selon le profil")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(int(pivot.iloc[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Rang")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

def score_curve(df, output):
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, group in df.groupby("Model"):
        group = group.sort_values("LatencyWeight")
        ax.plot(group["LatencyWeight"], group["Score"], linewidth=2, label=model)
    ax.set_xlabel("Poids attribué à la latence")
    ax.set_ylabel("Score multicritère")
    ax.set_title("Sensibilité du score au poids de la latence")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

def rank_curve(df, output):
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, group in df.groupby("Model"):
        group = group.sort_values("LatencyWeight")
        ax.step(group["LatencyWeight"], group["Rank"], where="mid", linewidth=2, label=model)
    ax.set_xlabel("Poids attribué à la latence")
    ax.set_ylabel("Rang")
    ax.set_yticks([1, 2, 3])
    ax.invert_yaxis()
    ax.set_title("Évolution du classement selon le poids de la latence")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

def winner_map(df, output):
    winners = df[df["Winner"]].sort_values("LatencyWeight").drop_duplicates("LatencyWeight")
    models = list(df["Model"].drop_duplicates())
    mapping = {m: i for i, m in enumerate(models)}
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(winners["LatencyWeight"], winners["Model"].map(mapping), s=35)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Poids attribué à la latence")
    ax.set_title("Modèle classé premier selon le poids de la latence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

def latex_escape(value):
    text = str(value)
    for old, new in {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}.items():
        text = text.replace(old, new)
    return text

def write_profile_tex(df, path):
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Classement multicritère des LLM selon cinq profils V2X.}",
        r"\label{tab:llm_sensitivity_profiles}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Profil} & \textbf{Modèle} & \textbf{Score} & \textbf{Rang} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['Profile'])} & {latex_escape(row['Model'])} & "
            f"{row['Score']:.4f} & {int(row['Rank'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def write_intervals_tex(df, path):
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Intervalles de dominance selon le poids de la latence.}",
        r"\label{tab:llm_latency_weight_intervals}",
        r"\begin{tabular}{rrl}",
        r"\toprule",
        r"\textbf{Poids min.} & \textbf{Poids max.} & \textbf{Modèle dominant} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['StartLatencyWeight']:.2f} & {row['EndLatencyWeight']:.2f} & "
            f"{latex_escape(row['WinnerModel'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def summary_text(profile_df, stability_df, intervals_df):
    lines = [
        "ANALYSE DE SENSIBILITÉ MULTICRITÈRE DES LLM V2X",
        "=" * 76, "", "CLASSEMENT PAR PROFIL", "-" * 76,
    ]
    for profile in PROFILES:
        lines.append(f"\n{profile}")
        subset = profile_df[profile_df["Profile"] == profile].sort_values("Rank")
        for _, row in subset.iterrows():
            lines.append(f"  {int(row['Rank'])}. {row['Model']} (score={row['Score']:.4f})")
    lines += ["", "STABILITÉ", "-" * 76]
    for _, row in stability_df.iterrows():
        lines.append(
            f"{row['Model']} | victoire={row['WinnerRate']:.3f} | "
            f"rang moyen={row['MeanRank']:.3f} | "
            f"rang min={int(row['BestRank'])} | rang max={int(row['WorstRank'])}"
        )
    lines += ["", "INTERVALLES DE DOMINANCE", "-" * 76]
    for _, row in intervals_df.iterrows():
        lines.append(
            f"Poids latence [{row['StartLatencyWeight']:.2f}, "
            f"{row['EndLatencyWeight']:.2f}] : {row['WinnerModel']}"
        )
    lines += [
        "", "PRÉCAUTIONS", "-" * 76,
        "La normalisation min-max est relative aux trois modèles comparés.",
        "Le classement dépend des poids choisis et ne remplace pas les contraintes réelles.",
        "La latence correspond à l'environnement local d'inférence.",
    ]
    return "\n".join(lines) + "\n"

def main():
    args = parse_args()
    for name, weights in PROFILES.items():
        total = sum(weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"Profil {name}: somme={total}")

    input_file = args.input_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    figures = output_dir / "figures"
    tables = output_dir / "tables"

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(input_file)
    normalized = prepare(comparison)
    profiles = profile_results(normalized)

    grid = np.round(
        np.arange(
            args.latency_min,
            args.latency_max + args.latency_step / 2,
            args.latency_step,
        ),
        10,
    )
    cont = continuous(normalized, grid)
    stable = stability(cont)
    dominant = intervals(cont)

    profiles.to_csv(output_dir / "profile_rankings.csv", index=False)
    cont.to_csv(output_dir / "continuous_latency_sensitivity.csv", index=False)
    stable.to_csv(output_dir / "ranking_stability.csv", index=False)
    dominant.to_csv(output_dir / "winner_intervals.csv", index=False)
    pd.DataFrame([{"Profile": name, **weights} for name, weights in PROFILES.items()]).to_csv(
        output_dir / "profile_weights.csv", index=False
    )

    grouped_plot(profiles, figures / "profile_scores.png")
    heatmap(profiles, figures / "profile_ranking_heatmap.png")
    score_curve(cont, figures / "continuous_latency_score.png")
    rank_curve(cont, figures / "continuous_latency_rank.png")
    winner_map(cont, figures / "latency_weight_winner_map.png")

    write_profile_tex(profiles, tables / "llm_sensitivity_profiles.tex")
    write_intervals_tex(dominant, tables / "llm_latency_weight_intervals.tex")

    summary = summary_text(profiles, stable, dominant)
    (output_dir / "sensitivity_summary.txt").write_text(summary, encoding="utf-8")

    manifest = {
        "input_file": str(input_file),
        "profiles": PROFILES,
        "latency_min": args.latency_min,
        "latency_max": args.latency_max,
        "latency_step": args.latency_step,
        "continuous_base_weights_without_latency": base_weights(),
    }
    (output_dir / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("ANALYSE DE SENSIBILITÉ LLM TERMINÉE")
    print("=" * 76)
    print("\nClassement par profil :")
    print(profiles[["Profile", "Model", "Score", "Rank"]].to_string(index=False))
    print("\nStabilité :")
    print(stable.to_string(index=False))
    print("\nIntervalles de dominance :")
    print(dominant.to_string(index=False))
    print(f"\nSortie  : {output_dir}")
    print(f"Résumé  : {output_dir / 'sensitivity_summary.txt'}")
    print(f"Figures : {figures}")
    print(f"Tables  : {tables}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
