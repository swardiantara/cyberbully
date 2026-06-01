"""
Statistical comparison of two preprocessing/augmentation configurations
using the Wilcoxon signed-rank test across seeds.

NOTE: Despite the filename, we use the Wilcoxon signed-rank test (paired,
non-parametric) because we only have aggregate metrics across 10 seeds —
not per-sample predictions required for a true McNemar's test.

Heatmap: 8 models × 3 datasets
  - Cell value : mean weighted-F1 difference (treatment − control)
  - Annotation : *** p<0.001 | ** p<0.01 | * p<0.05 | (blank) not significant
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, rankdata, wilcoxon

# ---------------------------------------------------------------------------
# CONFIGURATION — change these lines to compare a different pair / metric
# ---------------------------------------------------------------------------
CONTROL   = "prep0_aug0"   # baseline config
TREATMENT = "prep1_aug0"   # config under investigation
# For oversampling effect: CONTROL="prep0_aug0", TREATMENT="prep0_aug1"

METRIC = "macro_avg"       # metric key in metrics.json: "macro_avg" |
                           # "weighted_avg" | "micro_avg"
# ---------------------------------------------------------------------------

EXPERIMENT_ROOT = Path(__file__).parent.parent / "experiments" / "grid-search"
OUTPUT_DIR      = Path(__file__).parent.parent / "analysis" / "statistical-test"
RELIABILITY_DIR = Path(__file__).parent.parent / "analysis" / "reliability"

DATASETS = ["ieee", "kaggle", "tweeteval"]

group = {
    "general":  ["bert-base-cased", "roberta-base", "bert-base-uncased", "xlnet-base-cased", "gpt2"],
    "domain":   ["GroNLP/hateBERT", "vinai/bertweet-base", "Twitter/twhin-bert-base", "sarkerlab/SocBERT-base"],
    "sentence": ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2", "all-distilroberta-v1"],
    "small":    ["albert/albert-base-v2", "distilbert-base-uncased", "google/mobilebert-uncased", "distilbert-base-cased"],
    "modern": ["chandar-lab/NeoBERT", "answerdotai/ModernBERT-base"],
}

# Selected models (as they appear in the experiment folder, using underscores)
# MODELS_DISPLAY = [
#     "bert-base-cased",
#     "roberta-base",
#     "vinai/bertweet-base",
#     "Twitter/twhin-bert-base",
#     "all-distilroberta-v1",
#     "all-mpnet-base-v2",
#     "albert/albert-base-v2",
#     "distilbert-base-cased",
# ]

MODELS_DISPLAY = [model for _, models in group.items() for model in models]
MODELS_MAP = {
    "bert-base-cased": "BERT-cased",
    "bert-base-uncased": "BERT-uncased",
    "roberta-base": "RoBERTa",
    "xlnet-base-cased": "XLNet",
    "gpt2": "GPT-2",
    "GroNLP/hateBERT": "HateBERT",
    "vinai/bertweet-base": "BERTweet",
    "Twitter/twhin-bert-base": "TWHIN-BERT",
    "all-distilroberta-v1": "MiniLM-RoBERTa",
    "all-mpnet-base-v2": "MiniLM-MPNet",
    "all-MiniLM-L6-v2": "MiniLM-L6",
    "all-MiniLM-L12-v2": "MiniLM-L12",
    "albert/albert-base-v2": "ALBERT",
    "distilbert-base-cased": "DistilBERT-cased",
    "distilbert-base-uncased": "DistilBERT-uncased",
    "google/mobilebert-uncased": "MobileBERT",
    "sarkerlab/SocBERT-base": "SocBERT",
    "chandar-lab/NeoBERT": "NeoBERT",
    "answerdotai/ModernBERT-base": "ModernBERT",
}
DATASET_MAP = {
    "ieee": "IEEE",
    "kaggle": "Kaggle",
    "tweeteval": "TweetEval",
}

def model_to_folder(model_name: str) -> str:
    """Convert HuggingFace model name to filesystem folder name."""
    return model_name.replace("/", "_")


def get_all_model_folders() -> list[str]:
    """
    Return all model folder names found directly under EXPERIMENT_ROOT,
    sorted alphabetically. These can be passed straight to load_f1_scores
    because model_to_folder() is a no-op when there is no '/' in the name.
    """
    return sorted(d.name for d in EXPERIMENT_ROOT.iterdir() if d.is_dir())


def load_f1_scores(model: str, dataset: str, config: str) -> list[float]:
    """
    Load weighted-average F1 scores across all seeds for a given
    (model, dataset, config) combination.

    Returns a sorted list of F1 values (one per seed).
    Raises FileNotFoundError if no seed directories are found.
    """
    config_path = EXPERIMENT_ROOT / model_to_folder(model) / dataset / config
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    scores = []
    for seed_dir in sorted(config_path.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        metrics_file = seed_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        scores.append(metrics[METRIC]["f1-score"])

    if not scores:
        raise ValueError(f"No seed metrics found at: {config_path}")
    return scores


def significance_mark(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def run_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    For each (model, dataset) pair, compare CONTROL vs TREATMENT using the
    Wilcoxon signed-rank test across seeds (local claims).

    Returns:
      - mean_diff  : DataFrame of mean F1 differences (treatment - control)
      - std_diff   : DataFrame of sample std of per-seed F1 differences (ddof=1)
      - p_values   : DataFrame of Wilcoxon p-values
      - marks      : DataFrame of significance annotation strings
      - records_df : long-form DataFrame with one row per (model, dataset)
      - raw_diffs  : dict keyed by (model, dataset) -> np.ndarray of per-seed diffs
    """
    mean_diff = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=float)
    std_diff  = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=float)
    p_values  = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=float)
    marks     = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=object)

    records   = []
    raw_diffs = {}   # (model, dataset) -> np.ndarray of per-seed diffs

    for model in MODELS_DISPLAY:
        for dataset in DATASETS:
            ctrl_scores = trt_scores = None
            try:
                ctrl_scores = load_f1_scores(model, dataset, CONTROL)
                trt_scores  = load_f1_scores(model, dataset, TREATMENT)

                # Align lengths (take the minimum in case of missing seeds)
                n = min(len(ctrl_scores), len(trt_scores))
                ctrl_arr = np.array(ctrl_scores[:n])
                trt_arr  = np.array(trt_scores[:n])

                diff   = trt_arr - ctrl_arr
                mean_d = diff.mean()
                std_d  = diff.std(ddof=1)   # sample std across seeds

                raw_diffs[(model, dataset)] = diff

                # Wilcoxon requires non-zero differences; handle the edge case
                if np.all(diff == 0):
                    p = 1.0
                else:
                    _, p = wilcoxon(diff, alternative="two-sided")

                mark = significance_mark(p)

            except (FileNotFoundError, ValueError) as e:
                print(f"  [SKIP] {model} / {dataset}: {e}")
                mean_d, std_d, p, mark = np.nan, np.nan, np.nan, ""

            mean_diff.loc[model, dataset] = mean_d
            std_diff.loc[model, dataset]  = std_d
            p_values.loc[model, dataset]  = p
            marks.loc[model, dataset]     = mark

            records.append({
                "model":        model,
                "dataset":      dataset,
                "control":      CONTROL,
                "treatment":    TREATMENT,
                "ctrl_mean_f1": np.mean(ctrl_scores) if ctrl_scores else np.nan,
                "trt_mean_f1":  np.mean(trt_scores)  if trt_scores  else np.nan,
                "mean_diff":    mean_d,
                "std_diff":     std_d,
                "p_value":      p,
                "significant":  mark != "",
                "mark":         mark,
            })

    records_df = pd.DataFrame(records)
    return (mean_diff.astype(float), std_diff.astype(float),
            p_values.astype(float), marks, records_df, raw_diffs)


# ---------------------------------------------------------------------------
# Global analysis helpers
# ---------------------------------------------------------------------------

def _rank_biserial(diffs: np.ndarray) -> float:
    """
    Rank-biserial correlation as effect size for the Wilcoxon signed-rank test.
    Range [-1, 1]: positive means treatment tends to be better than control.
    """
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0
    abs_ranks = rankdata(np.abs(nonzero))
    W_pos = float(abs_ranks[nonzero > 0].sum())
    W_neg = float(abs_ranks[nonzero < 0].sum())
    return (W_pos - W_neg) / (n * (n + 1) / 2)


def _full_stats(arr: np.ndarray) -> dict:
    """Compute descriptive + Wilcoxon + binomial stats for one array of differences."""
    n_pos  = int((arr > 0).sum())
    n_neg  = int((arr < 0).sum())
    n_zero = int((arr == 0).sum())

    # Wilcoxon signed-rank test
    nonzero = arr[arr != 0]
    n_nz    = len(nonzero)
    if n_nz == 0:
        w_stat, w_p, w_r = np.nan, 1.0, 0.0
    else:
        w_stat, w_p = wilcoxon(nonzero, alternative="two-sided")
        w_r         = _rank_biserial(arr)

    # Binomial sign test  H0: P(improvement) = 0.5  (two-sided)
    n_nonzero_signed = n_pos + n_neg
    binom_p = float(
        binomtest(n_pos, n_nonzero_signed, p=0.5, alternative="two-sided").pvalue
    ) if n_nonzero_signed > 0 else 1.0

    return {
        "n":               len(arr),
        "mean":            float(arr.mean()),
        "median":          float(np.median(arr)),
        "std":             float(arr.std()),
        "n_positive":      n_pos,
        "n_negative":      n_neg,
        "n_zero":          n_zero,
        "wilcoxon_W":      w_stat,
        "wilcoxon_p":      w_p,
        "wilcoxon_mark":   significance_mark(w_p),
        "effect_size_r":   w_r,
        "binom_n_pos":     n_pos,
        "binom_n_nonzero": n_nonzero_signed,
        "binom_p":         binom_p,
        "binom_mark":      significance_mark(binom_p),
    }


def run_global_analysis(raw_diffs: dict) -> dict:
    """
    Perform global statistical analysis across all (model, dataset) cells.

    Two levels:
      agg (n=24) : one mean difference per (model, dataset) cell.
      raw (n=240): all per-seed differences pooled.

    Both levels get the same three analyses:
      - Descriptive statistics
      - Wilcoxon signed-rank test
      - Binomial sign test

    Returns a dict with keys "agg" and "raw", each holding identical fields.
    """
    # --- Build arrays ---
    agg_list = []
    raw_list = []
    for key in sorted(raw_diffs):
        d = raw_diffs[key]
        agg_list.append(float(d.mean()))
        raw_list.extend(d.tolist())

    agg = np.array(agg_list)
    raw = np.array(raw_list)

    return {"agg": _full_stats(agg), "raw": _full_stats(raw)}


# ---------------------------------------------------------------------------
# Multi-metric analysis (F1 + ECE + Confidence Gap + Stability + Entropy)
# ---------------------------------------------------------------------------

def load_per_seed_agg_diffs(csv_path: Path, value_col: str) -> np.ndarray:
    """
    Load a per-seed reliability CSV and return aggregated differences
    (treatment - control), one value per (Model, Dataset) cell.

    Each cell value is: mean_treatment_seeds - mean_control_seeds.
    Only cells present in both CONTROL and TREATMENT are included.
    """
    df = pd.read_csv(csv_path)
    ctrl_means = (df[df["Config"] == CONTROL]
                  .groupby(["Model", "Dataset"])[value_col].mean())
    trt_means  = (df[df["Config"] == TREATMENT]
                  .groupby(["Model", "Dataset"])[value_col].mean())
    common = ctrl_means.index.intersection(trt_means.index)
    return (trt_means.loc[common] - ctrl_means.loc[common]).values


def load_stability_agg_diffs(csv_path: Path) -> np.ndarray:
    """
    Load the pre-aggregated stability CSV and return differences
    (treatment - control), one per (Model, Dataset) cell.

    Only cells present in both CONTROL and TREATMENT are included.
    """
    df = pd.read_csv(csv_path)
    ctrl = df[df["Config"] == CONTROL].set_index(["Model", "Dataset"])["Stability"]
    trt  = df[df["Config"] == TREATMENT].set_index(["Model", "Dataset"])["Stability"]
    common = ctrl.index.intersection(trt.index)
    return (trt.loc[common] - ctrl.loc[common]).values


def load_entropy_agg_diffs(csv_path: Path) -> np.ndarray:
    """
    Load the pre-aggregated stability/entropy CSV and return entropy differences
    (treatment - control), one per (Model, Dataset) cell.

    Only cells present in both CONTROL and TREATMENT are included.
    """
    df = pd.read_csv(csv_path)
    ctrl = df[df["Config"] == CONTROL].set_index(["Model", "Dataset"])["Entropy"]
    trt  = df[df["Config"] == TREATMENT].set_index(["Model", "Dataset"])["Entropy"]
    common = ctrl.index.intersection(trt.index)
    return (trt.loc[common] - ctrl.loc[common]).values


def run_multi_metric_analysis(f1_raw_diffs: dict) -> dict:
    """
    Run Wilcoxon + binomial sign test on aggregated differences for 5 metrics:
    Macro-avg F1, ECE, Confidence Gap, Stability, and Entropy.

    All differences are at the aggregated level: one mean per (Model, Dataset) cell.
    Direction is always (treatment - control).
      - F1       : positive diff = improvement
      - ECE      : negative diff = improvement (lower calibration error = better)
      - ConfGap  : negative diff = improvement (lower gap = better)
      - Stability: negative diff = improvement (lower variation = more stable)
      - Entropy  : negative diff = improvement (lower entropy = more consistent)

    Parameters
    ----------
    f1_raw_diffs : dict returned by run_comparison() — {(model, dataset): np.ndarray}

    Returns
    -------
    dict : {metric_display_name: stats_dict}  where stats_dict has the same
           fields as returned by _full_stats().
    """
    f1_agg = np.array([float(d.mean()) for d in f1_raw_diffs.values()])

    ece_agg = load_per_seed_agg_diffs(
        RELIABILITY_DIR / "ece_per_seed.csv", "ECE"
    )
    conf_gap_agg = load_per_seed_agg_diffs(
        RELIABILITY_DIR / "confidence_gap_per_seed.csv", "ConfidenceGap"
    )
    stability_agg = load_stability_agg_diffs(
        RELIABILITY_DIR / "stability_per_config.csv"
    )
    entropy_agg = load_entropy_agg_diffs(
        RELIABILITY_DIR / "stability_per_config.csv"
    )

    print(f"  F1 agg diffs          : n={len(f1_agg)}")
    print(f"  ECE agg diffs         : n={len(ece_agg)}")
    print(f"  Confidence Gap diffs  : n={len(conf_gap_agg)}")
    print(f"  Stability diffs       : n={len(stability_agg)}")
    print(f"  Entropy diffs         : n={len(entropy_agg)}")

    return {
        "Macro-avg F1":   _full_stats(f1_agg),
        "ECE":            _full_stats(ece_agg),
        "Confidence Gap": _full_stats(conf_gap_agg),
        "Stability":      _full_stats(stability_agg),
        "Entropy":        _full_stats(entropy_agg),
    }


def save_multi_metric_excel(multi_stats: dict, comparison_tag: str) -> None:
    """
    Export multi-metric statistical analysis to an Excel file.

    Sheets
    ------
    summary      : all metrics as columns, all statistics as rows —
                   the primary paper-reporting table.
    <MetricName> : one sheet per metric with a vertical detail table.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUTPUT_DIR / f"multi_metric_stats_{comparison_tag}.xlsx"

    stat_labels = {
        "n":               "N (aggregated cells)",
        "mean":            "Mean difference",
        "median":          "Median difference",
        "std":             "Std deviation",
        "n_positive":      "# Positive (trt > ctrl)",
        "n_negative":      "# Negative (trt < ctrl)",
        "n_zero":          "# Zero",
        "wilcoxon_W":      "Wilcoxon W statistic",
        "wilcoxon_p":      "Wilcoxon p-value",
        "wilcoxon_mark":   "Wilcoxon significance",
        "effect_size_r":   "Effect size r (rank-biserial)",
        "binom_n_pos":     "Binomial: # positive",
        "binom_n_nonzero": "Binomial: # non-zero",
        "binom_p":         "Binomial p-value",
        "binom_mark":      "Binomial significance",
    }
    stat_keys = list(stat_labels.keys())

    # --- Summary DataFrame: rows = statistics, columns = metrics ---
    summary = pd.DataFrame(
        index=[stat_labels[k] for k in stat_keys],
        columns=list(multi_stats.keys()),
        dtype=object,
    )
    for metric, stats in multi_stats.items():
        for key, label in stat_labels.items():
            summary.loc[label, metric] = stats.get(key, np.nan)
    summary.index.name = "Statistic"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Primary summary sheet
        summary.to_excel(writer, sheet_name="summary")

        # Per-metric detail sheets
        for metric, stats in multi_stats.items():
            sheet_name = metric[:31]   # Excel sheet name: max 31 chars
            detail = pd.DataFrame({
                "Statistic": [stat_labels[k] for k in stat_keys],
                "Value":     [stats.get(k, np.nan) for k in stat_keys],
            })
            detail.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved multi-metric Excel : {xlsx_path}")


# ---------------------------------------------------------------------------
# Reliability metric heatmaps (ECE, Confidence Gap, Stability)
# ---------------------------------------------------------------------------

def run_per_seed_metric_comparison(
    csv_path: Path, value_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-(model, dataset) mean differences and Wilcoxon significance
    marks for a per-seed reliability metric (ECE or Confidence Gap).

    Seeds are paired by name (sorted), matching the approach used for F1.

    Returns
    -------
    mean_diff : DataFrame[MODELS_DISPLAY × DATASETS], float
    marks     : DataFrame[MODELS_DISPLAY × DATASETS], str
    """
    df = pd.read_csv(csv_path)

    mean_diff = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=float)
    marks     = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=object)

    for model in MODELS_DISPLAY:
        for dataset in DATASETS:
            ctrl_vals = (df[(df["Model"] == model) & (df["Dataset"] == dataset) &
                            (df["Config"] == CONTROL)]
                         .sort_values("Seed")[value_col].values)
            trt_vals  = (df[(df["Model"] == model) & (df["Dataset"] == dataset) &
                            (df["Config"] == TREATMENT)]
                         .sort_values("Seed")[value_col].values)

            if len(ctrl_vals) == 0 or len(trt_vals) == 0:
                mean_diff.loc[model, dataset] = np.nan
                marks.loc[model, dataset]     = ""
                continue

            n    = min(len(ctrl_vals), len(trt_vals))
            diff = trt_vals[:n] - ctrl_vals[:n]
            mean_diff.loc[model, dataset] = diff.mean()

            if np.all(diff == 0):
                p = 1.0
            else:
                _, p = wilcoxon(diff, alternative="two-sided")
            marks.loc[model, dataset] = significance_mark(p)

    return mean_diff.astype(float), marks


def run_stability_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-(model, dataset) differences for the pre-aggregated Stability
    metric. No significance testing is possible (one value per cell).

    Returns
    -------
    mean_diff : DataFrame[MODELS_DISPLAY × DATASETS], float
    marks     : DataFrame[MODELS_DISPLAY × DATASETS], all empty strings
    """
    df = pd.read_csv(RELIABILITY_DIR / "stability_per_config.csv")

    mean_diff = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=float)
    marks     = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=object)

    for model in MODELS_DISPLAY:
        for dataset in DATASETS:
            ctrl_row = df[(df["Model"] == model) & (df["Dataset"] == dataset) &
                          (df["Config"] == CONTROL)]["Stability"].values
            trt_row  = df[(df["Model"] == model) & (df["Dataset"] == dataset) &
                          (df["Config"] == TREATMENT)]["Stability"].values

            if len(ctrl_row) == 0 or len(trt_row) == 0:
                mean_diff.loc[model, dataset] = np.nan
            else:
                mean_diff.loc[model, dataset] = float(trt_row[0]) - float(ctrl_row[0])
            marks.loc[model, dataset] = ""

    return mean_diff.astype(float), marks


def run_entropy_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-(model, dataset) differences for the pre-aggregated Entropy
    metric. No significance testing is possible (one value per cell).

    Returns
    -------
    mean_diff : DataFrame[MODELS_DISPLAY × DATASETS], float
    marks     : DataFrame[MODELS_DISPLAY × DATASETS], all empty strings
    """
    df = pd.read_csv(RELIABILITY_DIR / "stability_per_config.csv")

    mean_diff = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=float)
    marks     = pd.DataFrame(index=MODELS_DISPLAY, columns=DATASETS, dtype=object)

    for model in MODELS_DISPLAY:
        for dataset in DATASETS:
            ctrl_row = df[(df["Model"] == model) & (df["Dataset"] == dataset) &
                          (df["Config"] == CONTROL)]["Entropy"].values
            trt_row  = df[(df["Model"] == model) & (df["Dataset"] == dataset) &
                          (df["Config"] == TREATMENT)]["Entropy"].values

            if len(ctrl_row) == 0 or len(trt_row) == 0:
                mean_diff.loc[model, dataset] = np.nan
            else:
                mean_diff.loc[model, dataset] = float(trt_row[0]) - float(ctrl_row[0])
            marks.loc[model, dataset] = ""

    return mean_diff.astype(float), marks


def save_reliability_heatmaps(comparison_tag: str) -> None:
    """
    Generate and save heatmaps for ECE, Confidence Gap, Stability, and Entropy
    using the same visual format as the F1 heatmap.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_cfg = [
        {
            "name":        "ECE",
            "loader":      lambda: run_per_seed_metric_comparison(
                               RELIABILITY_DIR / "ece_per_seed.csv", "ECE"),
            "cbar_label":  "Mean ECE Difference (treatment - control)",
            "tag":         "ece",
            "flip_colors": True,   # lower ECE = better calibration
        },
        {
            "name":        "Confidence Gap",
            "loader":      lambda: run_per_seed_metric_comparison(
                               RELIABILITY_DIR / "confidence_gap_per_seed.csv",
                               "ConfidenceGap"),
            "cbar_label":  "Mean Confidence Gap Difference (treatment - control)",
            "tag":         "confidence_gap",
            "flip_colors": False,  # higher gap difference interpretation same as F1
        },
        {
            "name":        "Stability",
            "loader":      lambda: run_stability_comparison(),
            "cbar_label":  "Stability Difference (treatment - control)",
            "tag":         "stability",
            "flip_colors": True,   # lower stability score = more stable
        },
        {
            "name":        "Entropy",
            "loader":      lambda: run_entropy_comparison(),
            "cbar_label":  "Mean Entropy Difference (treatment - control)",
            "tag":         "entropy",
            "flip_colors": True,   # lower entropy = more consistent predictions
        },
    ]

    for cfg in metrics_cfg:
        print(f"  Generating heatmap: {cfg['name']} ...")
        mean_diff, marks = cfg["loader"]()
        fig = plot_heatmap(mean_diff, marks, metric_label=cfg["cbar_label"],
                           flip_colors=cfg["flip_colors"])
        out_path = OUTPUT_DIR / f"heatmap_{cfg['tag']}_{comparison_tag}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved : {out_path}")


def print_global_analysis(stats: dict) -> None:
    """Pretty-print the global analysis results symmetrically for agg and raw."""
    sep  = "-" * 60
    sep2 = "=" * 60
    labels = {
        "agg": "AGGREGATED DIFFS  (n=24, one mean per model x dataset cell)",
        "raw": f"RAW DIFFS         (n={stats['raw']['n']}, all per-seed diffs pooled)",
    }

    print(f"\n{sep2}")
    print("GLOBAL ANALYSIS")
    print(sep2)

    for level in ("agg", "raw"):
        s = stats[level]
        print(f"\n{sep2}")
        print(labels[level])
        print(sep2)

        print(f"\n{sep}")
        print("Descriptive Statistics")
        print(sep)
        print(f"  Mean   improvement : {s['mean']:+.6f}")
        print(f"  Median improvement : {s['median']:+.6f}")
        print(f"  Std    deviation   : {s['std']:.6f}")
        print(f"  # positive (trt > ctrl) : {s['n_positive']} / {s['n']}")
        print(f"  # negative (trt < ctrl) : {s['n_negative']} / {s['n']}")
        print(f"  # zero                  : {s['n_zero']} / {s['n']}")

        print(f"\n{sep}")
        print("Wilcoxon Signed-Rank Test  (two-sided)")
        print(sep)
        print(f"  W statistic    : {s['wilcoxon_W']:.4f}")
        print(f"  p-value        : {s['wilcoxon_p']:.6f}  {s['wilcoxon_mark']}")
        print(f"  Effect size r  : {s['effect_size_r']:+.4f}  (rank-biserial correlation)")

        print(f"\n{sep}")
        print("Binomial Sign Test  (H0: P(improvement) = 0.5, two-sided)")
        print(sep)
        print(f"  # positive / # non-zero : {s['binom_n_pos']} / {s['binom_n_nonzero']}")
        print(f"  p-value                 : {s['binom_p']:.6f}  {s['binom_mark']}")


def save_global_outputs(stats: dict, comparison_tag: str) -> None:
    """Save global analysis results as a structured text report and CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    level_labels = {
        "agg": "Aggregated Diffs (n=24, one mean per model x dataset cell)",
        "raw": f"Raw Diffs (n={stats['raw']['n']}, all per-seed diffs pooled)",
    }

    # --- Plain-text report ---
    report_path = OUTPUT_DIR / f"global_{comparison_tag}.txt"
    lines = [
        f"Global Analysis: {CONTROL} vs {TREATMENT}",
        "=" * 60,
        "",
    ]
    for level in ("agg", "raw"):
        s = stats[level]
        lines += [
            level_labels[level],
            "=" * 60,
            "",
            "Descriptive Statistics",
            "-" * 60,
            f"Mean   improvement : {s['mean']:+.6f}",
            f"Median improvement : {s['median']:+.6f}",
            f"Std    deviation   : {s['std']:.6f}",
            f"# positive (trt > ctrl) : {s['n_positive']} / {s['n']}",
            f"# negative (trt < ctrl) : {s['n_negative']} / {s['n']}",
            f"# zero                  : {s['n_zero']} / {s['n']}",
            "",
            "Wilcoxon Signed-Rank Test (two-sided)",
            "-" * 60,
            f"W statistic    : {s['wilcoxon_W']:.4f}",
            f"p-value        : {s['wilcoxon_p']:.6f}  {s['wilcoxon_mark']}",
            f"Effect size r  : {s['effect_size_r']:+.4f}  (rank-biserial correlation)",
            "",
            "Binomial Sign Test (H0: P(improvement) = 0.5, two-sided)",
            "-" * 60,
            f"# positive / # non-zero : {s['binom_n_pos']} / {s['binom_n_nonzero']}",
            f"p-value                 : {s['binom_p']:.6f}  {s['binom_mark']}",
            "",
        ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved global report : {report_path}")

    # --- Flat CSV (one row, prefixed columns for agg_ and raw_) ---
    flat = {"control": CONTROL, "treatment": TREATMENT}
    for level in ("agg", "raw"):
        s = stats[level]
        p = level + "_"
        flat.update({
            p + "n":               s["n"],
            p + "mean":            s["mean"],
            p + "median":          s["median"],
            p + "std":             s["std"],
            p + "n_positive":      s["n_positive"],
            p + "n_negative":      s["n_negative"],
            p + "n_zero":          s["n_zero"],
            p + "wilcoxon_W":      s["wilcoxon_W"],
            p + "wilcoxon_p":      s["wilcoxon_p"],
            p + "wilcoxon_mark":   s["wilcoxon_mark"],
            p + "effect_size_r":   s["effect_size_r"],
            p + "binom_n_pos":     s["binom_n_pos"],
            p + "binom_n_nonzero": s["binom_n_nonzero"],
            p + "binom_p":         s["binom_p"],
            p + "binom_mark":      s["binom_mark"],
        })
    pd.DataFrame([flat]).to_csv(
        OUTPUT_DIR / f"global_{comparison_tag}.csv", index=False
    )
    print(f"Saved global CSV    : {OUTPUT_DIR / f'global_{comparison_tag}.csv'}")


def plot_heatmap(
    mean_diff: pd.DataFrame,
    marks: pd.DataFrame,
    metric_label: str = None,
    flip_colors: bool = False,
) -> plt.Figure:
    """
    Generate a heatmap of mean difference with significance annotations.

    Color : diverging palette centred at 0, raw scale (treatment - control).
    Cell  : fixed 9-char monospace string, center-aligned.
              Format: "{sign}{X.XXX}{mark:<3}"
              = 1 (sign) + 5 (value) + 3 (marker padded with trailing spaces)
              All cells are exactly the same width, so columns align
              character-for-character regardless of significance level.

    Parameters
    ----------
    mean_diff    : DataFrame of mean differences (treatment - control)
    marks        : DataFrame of significance annotation strings
    metric_label : label for the colorbar axis; defaults to macro-avg F1 label
    flip_colors  : if True, reverse the diverging palette so that negative
                   differences are green (use for lower-is-better metrics
                   such as ECE and Stability)
    """
    n_rows = len(mean_diff)
    n_cols = len(mean_diff.columns)
    # Cell dimensions: narrow width keeps text tight; height scales with row count
    cell_w  = 0.7    # inches per column
    cell_h  = 0.2   # inches per row
    label_w = 2.2    # left margin for y-axis model names
    cbar_w  = 1.0    # right margin for colorbar
    title_h = 0.9    # top margin for title
    xlab_h  = 0.6    # bottom margin for x-axis label
    fig_w = cell_w * n_cols + label_w + cbar_w
    fig_h = cell_h * n_rows + title_h + xlab_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Build fixed-width annotation matrix.
    # mark is padded to 3 chars with trailing spaces so every string is 9 chars.
    annot = pd.DataFrame(index=mean_diff.index, columns=mean_diff.columns, dtype=str)
    for model in mean_diff.index:
        for dataset in mean_diff.columns:
            m    = mean_diff.loc[model, dataset]
            mark = marks.loc[model, dataset]
            if pd.isna(m):
                annot.loc[model, dataset] = f"{'N/A':<9}"
            else:
                annot.loc[model, dataset] = f"{m:+.3f}{mark:<3}"

    # Symmetric colour scale (raw 0–1 units)
    abs_max = mean_diff.abs().max().max()
    if pd.isna(abs_max) or abs_max == 0:
        abs_max = 0.01
    vmax = abs_max * 1.1

    # Default: red (negative) → white → green (positive)  [higher-is-better]
    # Flipped: green (negative) → white → red (positive)  [lower-is-better]
    cmap = sns.diverging_palette(130, 10, as_cmap=True) if flip_colors \
        else sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(
        mean_diff,
        ax=ax,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 8.5, "fontfamily": "monospace"},
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": metric_label or f"Mean {METRIC.replace('_', ' ').title()} F1 Difference (treatment - control)", "shrink": 0.8},
    )

    # Seaborn centres text by default; be explicit and restore default position.
    for text in ax.texts:
        text.set_ha("center")

    # Apply clean display names from MODELS_MAP (fall back to raw name if missing)
    display_labels = [MODELS_MAP.get(m, m) for m in mean_diff.index]
    display_dataset = [DATASET_MAP.get(d, d) for d in mean_diff.columns]
    ax.set_xticklabels(display_dataset)
    ax.set_yticklabels(display_labels)

    comparison_label = f"{TREATMENT} vs {CONTROL}"
    # ax.set_title(
    #     # f"Statistical Comparison: {comparison_label}\n"
    #     f"Wilcoxon signed-rank test\n"
    #     f"* p<0.05, ** p<0.01, *** p<0.001)",
    #     fontsize=10,
    #     pad=10,
    # )
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8.5, rotation=0)

    plt.tight_layout()
    return fig


def save_outputs(mean_diff: pd.DataFrame, std_diff: pd.DataFrame,
                 p_values: pd.DataFrame, marks: pd.DataFrame,
                 records_df: pd.DataFrame) -> None:
    """Save heatmap figure and detailed CSV/Excel to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_tag = f"{CONTROL}_vs_{TREATMENT}"

    # --- Heatmap ---
    fig = plot_heatmap(mean_diff, marks)
    fig_path = OUTPUT_DIR / f"heatmap_{comparison_tag}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap : {fig_path}")

    # --- Detailed records (CSV) ---
    csv_path = OUTPUT_DIR / f"results_{comparison_tag}.csv"
    records_df.to_csv(csv_path, index=False)
    print(f"Saved CSV     : {csv_path}")

    # --- Summary tables (Excel, one sheet per table) ---
    xlsx_path = OUTPUT_DIR / f"summary_{comparison_tag}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        mean_diff.to_excel(writer, sheet_name="mean_diff")
        std_diff.to_excel(writer, sheet_name="std_diff")
        p_values.to_excel(writer, sheet_name="p_values")
        marks.to_excel(writer, sheet_name="significance_marks")
        records_df.to_excel(writer, sheet_name="full_records", index=False)
    print(f"Saved Excel   : {xlsx_path}")


# ---------------------------------------------------------------------------
# Distribution boxplots (both comparisons together)
# ---------------------------------------------------------------------------

def collect_comparison_diffs(
    control: str,
    treatment: str,
    models: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect aggregated and raw differences for a given (control, treatment) pair.

    Parameters
    ----------
    control, treatment : config folder names (e.g. "prep0_aug0")
    models : list of model names/folder-names to iterate over.
             Defaults to MODELS_DISPLAY when None.

    Returns
    -------
    agg_diffs : mean diff per (model, dataset) cell
    raw_diffs : all per-seed diffs pooled
    """
    if models is None:
        models = MODELS_DISPLAY
    agg_list = []
    raw_list = []
    for model in models:
        for dataset in DATASETS:
            try:
                ctrl_scores = load_f1_scores(model, dataset, control)
                trt_scores  = load_f1_scores(model, dataset, treatment)
                n = min(len(ctrl_scores), len(trt_scores))
                diff = np.array(trt_scores[:n]) - np.array(ctrl_scores[:n])
                agg_list.append(float(diff.mean()))
                raw_list.extend(diff.tolist())
            except (FileNotFoundError, ValueError):
                pass
    return np.array(agg_list), np.array(raw_list)


def plot_distribution_boxplots() -> plt.Figure:
    """
    Single horizontal boxplot showing the distribution of macro-averaged F1
    differences (TREATMENT - CONTROL) across ALL model folders and datasets.

    Y-axis : two levels — Raw (all per-seed diffs) and Aggregated (cell means)
    X-axis : macro-averaged F1 difference
    A vertical reference line at x = 0 marks no change.

    Returns a single Figure.
    """
    all_models = get_all_model_folders()
    agg, raw   = collect_comparison_diffs(CONTROL, TREATMENT, models=all_models)

    n_agg = len(agg)
    n_raw = len(raw)

    # Y-axis order: Aggregated on top (index 2), Raw on bottom (index 1)
    y_labels = [f"Raw\n(n={n_raw})", f"Aggr.\n(n={n_agg})"]
    data     = [raw, agg]

    fig, ax = plt.subplots(figsize=(5, 2))

    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=y_labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, linestyle="none",
                        markerfacecolor="grey", alpha=0.5),
        widths=0.45,
    )

    colours = ["#DD8452", "#4C72B0"]   # orange = Raw, blue = Aggregated
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    # Reference line at x = 0
    ax.axvline(x=0, color="crimson", linestyle="--", linewidth=1.2, label="x = 0", ymax=1)

    comparison_label = f"{TREATMENT} vs {CONTROL}"
    ax.set_xlabel(
        f"Macro-averaged F1 Difference (treatment - control)",
        fontsize=10,
    )
    ax.set_ylabel("Distribution", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(axis="both", labelsize=10)
    plt.yticks(rotation=90)

    plt.tight_layout()
    return fig


def main():
    print(f"Comparing  CONTROL={CONTROL}  vs  TREATMENT={TREATMENT}")
    print(f"Experiment root : {EXPERIMENT_ROOT}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print("-" * 60)

    # --- Local analysis (one test per model x dataset cell) ---
    mean_diff, std_diff, p_values, marks, records_df, raw_diffs = run_comparison()

    print("\n--- Mean F1 Difference (treatment - control) ---")
    print(mean_diff.to_string())
    print("\n--- Std of F1 Differences ---")
    print(std_diff.to_string())
    print("\n--- P-values ---")
    print(p_values.to_string())
    print("\n--- Significance marks ---")
    print(marks.to_string())

    save_outputs(mean_diff, std_diff, p_values, marks, records_df)

    # --- Global analysis (aggregated across all cells) ---
    global_stats = run_global_analysis(raw_diffs)
    print_global_analysis(global_stats)

    comparison_tag = f"{CONTROL}_vs_{TREATMENT}"
    save_global_outputs(global_stats, comparison_tag)

    # --- Multi-metric analysis (F1 + ECE + Confidence Gap + Stability) ---
    print("\n--- Multi-metric analysis (aggregated diffs) ---")
    multi_stats = run_multi_metric_analysis(raw_diffs)
    save_multi_metric_excel(multi_stats, comparison_tag)

    # --- Reliability metric heatmaps (ECE, Confidence Gap, Stability) ---
    print("\n--- Reliability metric heatmaps ---")
    save_reliability_heatmaps(comparison_tag)

    # --- Distribution boxplot (Agg + Raw, horizontal, all models) ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_tag = f"{CONTROL}_vs_{TREATMENT}"
    fig_box = plot_distribution_boxplots()
    box_path = OUTPUT_DIR / f"boxplot_{comparison_tag}.pdf"
    fig_box.savefig(box_path, bbox_inches="tight")
    plt.close(fig_box)
    print(f"Saved boxplot : {box_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()