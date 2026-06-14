"""
recap_experiment.py — Aggregate experiment results into summary Excel/CSV files.

Walks experiments/{run_name}/ and collects metrics.json from every completed
seed directory into two DataFrames:
  - recap_all  : one row per (model, dataset, prep_mode, augmentation, seed)
  - per_class  : same index + one column per (class, metric) triple

Outputs written to analysis/{run_name}/recap/:
  recap_all.xlsx        — all experiments, aggregate metrics
  aggregated_all.xlsx   — mean ± std across seeds per configuration
  aggregated_all.csv    — same as above in CSV
  recap_ieee.xlsx       — per-class metrics for IEEE dataset
  recap_kaggle.xlsx     — per-class metrics for Kaggle dataset
  recap_tweet.xlsx      — per-class metrics for TweetEval dataset

Column notes
------------
prep_mode    : int  — 0 = none, 1 = full, 2 = partial
preprocessing: bool — True iff prep_mode == 1 (backward compat with compute_kruskal.py)
augmentation : bool — True iff aug1

Usage
-----
    python notebooks/recap_experiment.py                      # defaults to grid-search
    python notebooks/recap_experiment.py --run_name grid-search
    python notebooks/recap_experiment.py --run_name supcon-grid
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics into recap Excel files.")
    parser.add_argument(
        "--run_name",
        type=str,
        default="grid-search",
        help="Experiment run name (subfolder under experiments/). Default: grid-search",
    )
    return parser.parse_args()


def parse_scenario(scenario: str) -> tuple[int, bool]:
    """Parse 'prep{N}_aug{N}' into (prep_mode: int, augmentation: bool)."""
    parts = scenario.split("_")
    prep_mode = int(parts[0].replace("prep", ""))
    augmentation = parts[1] == "aug1"
    return prep_mode, augmentation


def collect_results(experiments_dir: Path) -> tuple[list, list]:
    all_rows = []
    per_class_rows = []

    for model in sorted(os.listdir(experiments_dir)):
        model_path = experiments_dir / model
        if not model_path.is_dir():
            continue
        for dataset in sorted(os.listdir(model_path)):
            dataset_path = model_path / dataset
            if not dataset_path.is_dir():
                continue
            for scenario in sorted(os.listdir(dataset_path)):
                scenario_path = dataset_path / scenario
                if not scenario_path.is_dir():
                    continue

                try:
                    prep_mode, augmentation = parse_scenario(scenario)
                except (IndexError, ValueError):
                    print(f"  [SKIP] Unrecognised scenario folder: {scenario_path}")
                    continue

                # prep_mode==1 is the only "full preprocessing" scenario;
                # kept as a boolean for backward compatibility with compute_kruskal.py.
                preprocessing = prep_mode == 1

                for seed_folder in sorted(os.listdir(scenario_path)):
                    seed_path = scenario_path / seed_folder
                    if not seed_path.is_dir() or not seed_folder.startswith("seed_"):
                        continue
                    seed_num = int(seed_folder.split("_")[1])
                    metrics_file = seed_path / "metrics.json"
                    if not metrics_file.exists():
                        continue

                    with open(metrics_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)

                    all_rows.append({
                        "model":         model,
                        "dataset":       dataset,
                        "prep_mode":     prep_mode,
                        "preprocessing": preprocessing,
                        "augmentation":  augmentation,
                        "seed":          seed_num,
                        "accuracy":         metrics["accuracy"],
                        "macro_precision":  metrics["macro_avg"]["precision"],
                        "macro_recall":     metrics["macro_avg"]["recall"],
                        "macro_f1":         metrics["macro_avg"]["f1-score"],
                    })

                    pc = metrics.get("per_class", {})
                    pc_row = {
                        "model":         model,
                        "dataset":       dataset,
                        "prep_mode":     prep_mode,
                        "preprocessing": preprocessing,
                        "augmentation":  augmentation,
                        "seed":          seed_num,
                    }
                    for cls_name, cls_metrics in pc.items():
                        pc_row[f"{cls_name}_precision"] = cls_metrics["precision"]
                        pc_row[f"{cls_name}_recall"]    = cls_metrics["recall"]
                        pc_row[f"{cls_name}_f1"]        = cls_metrics["f1-score"]
                    per_class_rows.append(pc_row)

    return all_rows, per_class_rows


def main():
    args = parse_args()

    experiments_dir = BASE_DIR / "experiments" / args.run_name
    output_dir = BASE_DIR / "analysis" / args.run_name / "recap"

    if not experiments_dir.exists():
        raise SystemExit(f"Experiments directory not found: {experiments_dir}")

    print(f"Scanning : {experiments_dir}")
    print(f"Output   : {output_dir}")
    print("-" * 60)

    all_rows, per_class_rows = collect_results(experiments_dir)

    recap_all_df = pd.DataFrame(all_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    print(f"Total experiments loaded : {len(recap_all_df)}")
    print(f"Models     : {sorted(recap_all_df['model'].unique().tolist())}")
    print(f"Datasets   : {sorted(recap_all_df['dataset'].unique().tolist())}")
    print(f"Prep modes : {sorted(recap_all_df['prep_mode'].unique().tolist())}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- recap_all.xlsx ---
    recap_all_df.to_excel(output_dir / "recap_all.xlsx", index=False)
    print(f"Saved recap_all.xlsx ({len(recap_all_df)} rows)")

    # --- Per-dataset per-class files ---
    dataset_filename_map = {
        "ieee":      "recap_ieee",
        "kaggle":    "recap_kaggle",
        "tweeteval": "recap_tweet",
    }
    for dataset_key, filename in dataset_filename_map.items():
        df = per_class_df[per_class_df["dataset"] == dataset_key].copy()
        df = df.drop(columns=["dataset"])
        df = df.dropna(axis=1, how="all")
        df.to_excel(output_dir / f"{filename}.xlsx", index=False)
        metric_cols = [c for c in df.columns
                       if c not in ("model", "prep_mode", "preprocessing", "augmentation", "seed")]
        print(f"Saved {filename}.xlsx ({len(df)} rows, {len(metric_cols)} metric columns)")

    # --- aggregated_all (mean ± std across seeds) ---
    agg_df = (
        recap_all_df
        .groupby(["model", "dataset", "prep_mode", "preprocessing", "augmentation"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_precision_mean=("macro_precision", "mean"),
            macro_precision_std=("macro_precision", "std"),
            macro_recall_mean=("macro_recall", "mean"),
            macro_recall_std=("macro_recall", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index()
        .round(4)
    )
    agg_df.to_excel(output_dir / "aggregated_all.xlsx", index=False)
    agg_df.to_csv(output_dir / "aggregated_all.csv", index=False)
    print(f"Saved aggregated_all.xlsx / .csv ({len(agg_df)} configurations)")

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
