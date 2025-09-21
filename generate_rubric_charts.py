#!/usr/bin/env python3
"""Generate charts summarizing rubric scores for Codex runs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

MODEL_ORDER = ["gpt-4o", "o4-mini", "o3", "gpt-5"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create charts from rubric_score.json files")
    parser.add_argument(
        "--runs-root",
        default="codex_runs",
        help="Root directory containing Codex run outputs (default: codex_runs)",
    )
    parser.add_argument(
        "--charts-dir",
        default="charts",
        help="Directory where charts will be saved (default: charts)",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format for charts (default: png)",
    )
    return parser.parse_args()


def collect_rubric_scores(root: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for rubric_file in root.rglob("rubric_score.json"):
        try:
            data = json.loads(rubric_file.read_text())
        except json.JSONDecodeError:
            continue

        parts = rubric_file.parts
        # expect codex_runs/<scenario>/<model>/<timestamp>/...
        try:
            scenario = parts[-4]
            model = parts[-3]
            timestamp = parts[-2]
        except IndexError:
            scenario = rubric_file.parent.name
            model = "unknown"
            timestamp = "unknown"

        total_points = data.get("total_points")
        records.append(
            {
                "scenario": scenario,
                "model": model,
                "timestamp": timestamp,
                "total_points": total_points,
                "rubric_path": rubric_file,
            }
        )

        for criterion in data.get("criteria", []):
            records.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "timestamp": timestamp,
                    "criterion": criterion.get("description"),
                    "max_points": criterion.get("max_points"),
                    "awarded_points": criterion.get("awarded_points"),
                    "rubric_path": rubric_file,
                }
            )

    if not records:
        raise RuntimeError(f"No rubric_score.json files found under {root}")

    df = pd.DataFrame(records)
    return df


def ensure_charts_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in MODEL_ORDER if c in df.columns]
    columns.extend([c for c in df.columns if c not in columns])
    return df.reindex(columns=columns)


def chart_total_scores(df: pd.DataFrame, charts_dir: Path, fmt: str) -> None:
    scores = df.dropna(subset=["total_points"]).copy()
    if scores.empty:
        return

    pivot = scores.pivot_table(
        index="scenario",
        columns="model",
        values="total_points",
        aggfunc="mean",
    )
    pivot = sort_columns(pivot)

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar")
    plt.title("Average Rubric Scores per Scenario")
    plt.ylabel("Total Points")
    plt.xlabel("Scenario")
    plt.ylim(0, pivot.max().max() + 1)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    output_path = charts_dir / f"rubric_scores_total.{fmt}"
    plt.savefig(output_path)
    plt.close()


def chart_criteria_heatmap(df: pd.DataFrame, charts_dir: Path, fmt: str) -> None:
    criteria = df.dropna(subset=["criterion"]).copy()
    if criteria.empty:
        return

    criteria["criterion_short"] = criteria["criterion"].str.slice(0, 60)
    pivot = criteria.pivot_table(
        index="criterion_short",
        columns="model",
        values="awarded_points",
        aggfunc="mean",
    )
    pivot = sort_columns(pivot)

    plt.figure(figsize=(12, max(4, len(pivot) * 0.4)))
    plt.imshow(pivot, aspect="auto", cmap="Blues")
    plt.colorbar(label="Average Points")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.title("Average Criterion Scores by Model")
    plt.tight_layout()
    output_path = charts_dir / f"rubric_scores_criteria_heatmap.{fmt}"
    plt.savefig(output_path)
    plt.close()


def chart_model_distribution(df: pd.DataFrame, charts_dir: Path, fmt: str) -> None:
    scores = df.dropna(subset=["total_points"]).copy()
    if scores.empty:
        return

    plt.figure(figsize=(10, 6))
    for model, group in scores.groupby("model"):
        plt.hist(group["total_points"], bins=range(0, int(group["total_points"].max()) + 2), alpha=0.5, label=model)
    plt.title("Distribution of Total Scores by Model")
    plt.xlabel("Total Points")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    output_path = charts_dir / f"rubric_scores_distribution.{fmt}"
    plt.savefig(output_path)
    plt.close()


def chart_model_summary(df: pd.DataFrame, charts_dir: Path, fmt: str) -> None:
    scores = df.dropna(subset=["total_points"]).copy()
    if scores.empty:
        return

    summary = (
        scores.groupby("model")["total_points"].agg(["mean", "count"]).rename(columns={"mean": "average", "count": "count"})
    )
    summary = summary.reindex([m for m in MODEL_ORDER if m in summary.index])
    summary_path = charts_dir / "rubric_scores_model_summary.csv"
    summary.to_csv(summary_path)

    plt.figure(figsize=(8, 5))
    summary["average"].plot(kind="bar")
    plt.title("Average Rubric Score by Model")
    plt.ylabel("Average Total Points")
    plt.xlabel("Model")
    plt.ylim(0, max(summary["average"]) + 1)
    for idx, (avg, cnt) in enumerate(zip(summary["average"], summary["count"])):
        plt.text(idx, avg + 0.15, f"count={int(cnt)}", ha="center", va="bottom", fontsize=8)
        plt.text(idx, avg / 2, f"{avg:.2f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    plt.tight_layout()
    output_path = charts_dir / f"rubric_scores_model_summary.{fmt}"
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    charts_dir = Path(args.charts_dir).resolve()

    df = collect_rubric_scores(runs_root)
    ensure_charts_dir(charts_dir)

    chart_total_scores(df, charts_dir, args.format)
    chart_criteria_heatmap(df, charts_dir, args.format)
    chart_model_distribution(df, charts_dir, args.format)
    chart_model_summary(df, charts_dir, args.format)

    print(f"Charts written to {charts_dir} using format {args.format}")


if __name__ == "__main__":
    main()
