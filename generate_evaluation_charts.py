#!/usr/bin/env python3
"""Generate charts from evaluation_report.json files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

MODEL_ORDER = ["gpt-4o", "o4-mini", "o3", "gpt-5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create charts from evaluation_report.json files")
    parser.add_argument(
        "--runs-root",
        default="codex_runs",
        help="Root directory containing evaluation reports (default: codex_runs)",
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


def collect_evaluation_reports(root: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for report_path in root.rglob("evaluation_report.json"):
        try:
            data = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            continue

        parts = report_path.parts
        try:
            scenario = parts[-4]
            model = parts[-3]
            timestamp = parts[-2]
        except IndexError:
            scenario = report_path.parent.name
            model = "unknown"
            timestamp = "unknown"

        aggregate = data.get("aggregate", {})
        records.append(
            {
                "scenario": scenario,
                "model": model,
                "timestamp": timestamp,
                "average_score": aggregate.get("average_score"),
                "pass_rate": aggregate.get("pass_rate"),
                "max_score": aggregate.get("max_score"),
                "min_score": aggregate.get("min_score"),
                "runs": aggregate.get("runs"),
                "report_path": report_path,
            }
        )

        for run in data.get("runs", []):
            evaluation = run.get("evaluation", {})
            plan = run.get("plan", {})
            records.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "timestamp": timestamp,
                    "plan": plan,
                    "score": evaluation.get("score"),
                    "passed": evaluation.get("passed"),
                    "reasoning": evaluation.get("reasoning"),
                    "report_path": report_path,
                    "unique_run_id": hash(json.dumps(plan, sort_keys=True)),
                }
            )

    if not records:
        raise RuntimeError(f"No evaluation_report.json files found under {root}")

    return pd.DataFrame(records)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in MODEL_ORDER if c in df.columns]
    columns.extend([c for c in df.columns if c not in columns])
    return df.reindex(columns=columns)


def chart_eval_average_scores(df: pd.DataFrame, charts_dir: Path, fmt: str) -> None:
    agg = df.dropna(subset=["average_score"]).copy()
    agg = agg[agg["average_score"] >= 0]
    if agg.empty:
        return

    pivot = agg.pivot_table(index="scenario", columns="model", values="average_score", aggfunc="mean")
    pivot = sort_columns(pivot)

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar")
    plt.title("Average Evaluation Score per Scenario")
    plt.ylabel("Average Score")
    plt.xlabel("Scenario")
    plt.ylim(0, min(1.0, pivot.max().max() + 0.1))
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(charts_dir / f"eval_average_scores.{fmt}")
    plt.close()


def chart_eval_pass_rates(df: pd.DataFrame, charts_dir: Path, fmt: str) -> None:
    agg = df.dropna(subset=["pass_rate"]).copy()
    if agg.empty:
        return

    pivot = agg.pivot_table(index="scenario", columns="model", values="pass_rate", aggfunc="mean")
    pivot = sort_columns(pivot)

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar")
    plt.title("Pass Rate per Scenario")
    plt.ylabel("Pass Rate")
    plt.xlabel("Scenario")
    plt.ylim(0, min(1.0, pivot.max().max() + 0.1))
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(charts_dir / f"eval_pass_rates.{fmt}")
    plt.close()


def chart_eval_model_summary(df: pd.DataFrame, charts_dir: Path, fmt: str) -> None:
    agg = df.dropna(subset=["average_score"]).copy()
    agg = agg[agg["average_score"] >= 0]
    if agg.empty:
        return

    pivot = agg.pivot_table(index="scenario", columns="model", values="average_score", aggfunc="mean")
    pivot = sort_columns(pivot)
    summary = pd.DataFrame({
        "average": pivot.mean(axis=0),
        "count": pivot.count(axis=0),
    })
    summary_path = charts_dir / "eval_model_summary.csv"
    summary.to_csv(summary_path)

    plt.figure(figsize=(8, 5))
    summary["average"].plot(kind="bar", color="skyblue")
    plt.title("Average Evaluation Score by Model")
    plt.ylabel("Average Score")
    plt.xlabel("Model")
    plt.ylim(0, min(1.0, max(summary["average"]) + 0.1))
    for idx, (avg, cnt) in enumerate(zip(summary["average"], summary["count"])):
        plt.text(idx, avg + 0.05, f"count={int(cnt)}", ha="center", va="bottom", fontsize=8)
        plt.text(idx, avg / 2, f"{avg:.2f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(charts_dir / f"eval_model_summary.{fmt}")
    plt.close()


def chart_eval_run_scores(df: pd.DataFrame, charts_dir: Path, fmt: str) -> None:
    runs = df.dropna(subset=["score"]).copy()
    runs = runs[runs["score"] >= 0]
    if runs.empty:
        return

    runs["unique_run_id"] = runs["unique_run_id"].astype(str)
    pivot = runs.pivot_table(index="unique_run_id", columns="model", values="score", aggfunc="mean")
    pivot = sort_columns(pivot)

    plt.figure(figsize=(10, 6))
    pivot.boxplot()
    plt.title("Distribution of Evaluation Scores by Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(charts_dir / f"eval_score_distribution.{fmt}")
    plt.close()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    charts_dir = Path(args.charts_dir).resolve()

    df = collect_evaluation_reports(runs_root)
    ensure_dir(charts_dir)

    chart_eval_average_scores(df, charts_dir, args.format)
    chart_eval_pass_rates(df, charts_dir, args.format)
    chart_eval_model_summary(df, charts_dir, args.format)
    chart_eval_run_scores(df, charts_dir, args.format)

    print(f"Evaluation charts written to {charts_dir} using format {args.format}")


if __name__ == "__main__":
    main()
