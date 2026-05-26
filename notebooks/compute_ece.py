"""
compute_ece.py
==============
Compute Expected Calibration Error (ECE), confidence-score distributions,
and prediction stability for each model × dataset × preprocessing-config
combination.

Output per combination (stored in analysis/reliability/{dataset}_{config}/):
  - reliability_diagram.pdf     : 5×2 calibration plots
  - confidence_distribution.pdf : 5×2 KDE density plots
  - scores.xlsx                 : ECE + stability scores table

"Combined" dataset = ieee + kaggle + tweeteval predictions pooled together.
Predictions across all 10 seeds are concatenated before computing ECE.
Stability = average number of unique predicted labels per sample across seeds.
"""

import glob
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path("experiments/grid-search")
OUTPUT_BASE = Path("analysis/reliability")

DATASETS = ["ieee", "kaggle", "tweeteval"]
CONFIGS = ["prep0_aug0", "prep1_aug0"] # "prep0_aug1", "prep1_aug1" 
N_BINS = 10


def discover_models(base_dir: Path = BASE_DIR) -> list:
    """
    Discover all model names from subfolder names under base_dir.

    Folder names encode the HuggingFace model path with '/' replaced by '_'
    (only the first underscore is the separator; flat model names have no '_').

    Examples:
        vinai_bertweet-base  -> vinai/bertweet-base
        chandar-lab_NeoBERT  -> chandar-lab/NeoBERT
        bert-base-cased      -> bert-base-cased   (no underscore)
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Grid-search directory not found: {base_dir}")
    folders = sorted(p.name for p in base_dir.iterdir() if p.is_dir())
    return [name.replace("_", "/", 1) if "_" in name else name for name in folders]


MODELS = discover_models()

# Plot aesthetics
CORRECT_COLOR = "#2ca02c"    # green
INCORRECT_COLOR = "#d62728"  # red
MODEL_BAR_COLOR = "#4878cf"  # steel blue
GAP_COLOR = "#f4a896"        # salmon
PERFECT_LINE_STYLE = dict(color="crimson", linestyle="--", linewidth=1.5)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def model_to_folder(model: str) -> str:
    return model.replace("/", "_")


def display_name(model: str) -> str:
    """Short name shown in subplot titles."""
    return model.split("/")[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_seed_files(folder: str, dataset: str, config: str) -> dict:
    """
    Return {seed_key: [tagged_prediction, ...]} for one model/dataset/config.

    Each prediction is tagged with two keys:
      _sample_key : '<dataset>_<id>'          — sequential integer position key
      _text_key   : '<dataset>__<original_text>' — content-based key

    For 'prep0' configs every seed evaluates on the SAME test set in the SAME
    order, so the integer id is a reliable sample identifier.
    For 'prep1' configs the test set is SHUFFLED differently per seed, so the
    same integer id refers to a DIFFERENT sample in each seed.  Using the raw
    original_text as the key is the only reliable way to match the same sample
    across seeds in that case.
    _text_key is what compute_stability() uses.
    """
    pattern = str(BASE_DIR / folder / dataset / config / "seed_*" / "predictions.json")
    seed_preds = {}
    for fpath in sorted(glob.glob(pattern)):
        seed = Path(fpath).parent.name          # e.g. "seed_14298463"
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        tagged = []
        for p in raw:
            p2 = dict(p)
            p2["_sample_key"] = f"{dataset}_{p['id']}"
            p2["_text_key"] = f"{dataset}__{p['original_text']}"
            tagged.append(p2)
        seed_preds[f"{dataset}_{seed}"] = tagged
    return seed_preds


def load_predictions(model: str, dataset: str, config: str) -> tuple:
    """
    Load predictions for a single dataset.
    Returns (all_preds_flat, seed_preds_dict).
    """
    folder = model_to_folder(model)
    seed_preds = _load_seed_files(folder, dataset, config)
    all_preds = [p for preds in seed_preds.values() for p in preds]
    return all_preds, seed_preds


def load_predictions_combined(model: str, config: str) -> tuple:
    """
    Load predictions from ALL datasets, tagging each prediction with its
    dataset of origin so sample keys remain unique across datasets.
    Returns (all_preds_flat, seed_preds_dict).
    """
    folder = model_to_folder(model)
    all_preds = []
    seed_preds = {}
    for ds in DATASETS:
        sp = _load_seed_files(folder, ds, config)
        seed_preds.update(sp)
        all_preds.extend(p for preds in sp.values() for p in preds)
    return all_preds, seed_preds


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(predictions: list, n_bins: int = N_BINS) -> tuple:
    """
    Expected Calibration Error over equal-width confidence bins.

    Returns
    -------
    ece        : float
    bin_data   : list of dicts with keys lo, hi, mid, accuracy, confidence, count
    confidences: np.ndarray  — confidence of the predicted class
    corrects   : np.ndarray  — 1.0 = correct, 0.0 = incorrect
    """
    confs, corrs = [], []
    for p in predictions:
        conf = p["probabilities"][p["predicted_label"]]
        confs.append(conf)
        corrs.append(1.0 if p["correct"] else 0.0)

    confs = np.array(confs, dtype=float)
    corrs = np.array(corrs, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # First bin is closed on both sides; the rest are (lo, hi]
        mask = (confs >= lo) & (confs <= hi) if i == 0 else (confs > lo) & (confs <= hi)
        n = int(mask.sum())
        mid = (lo + hi) / 2.0

        if n > 0:
            acc = float(corrs[mask].mean())
            conf_mean = float(confs[mask].mean())
            ece += (n / len(confs)) * abs(acc - conf_mean)
        else:
            acc, conf_mean = 0.0, mid

        bin_data.append({"lo": lo, "hi": hi, "mid": mid,
                         "accuracy": acc, "confidence": conf_mean, "count": n})

    return float(ece), bin_data, confs, corrs


def compute_stability(seed_preds: dict) -> float:
    """
    Average number of unique predicted labels per sample across all seeds.

    Uses '_text_key' (dataset + original_text) to identify samples.
    This is robust to per-seed test-set shuffling (prep1 configs), where the
    sequential integer id refers to a different sample in each seed.
    Falls back to '_sample_key' then raw 'id' for backward compatibility.
    """
    sample_labels = {}
    for preds in seed_preds.values():
        for p in preds:
            key = p.get("_text_key") or p.get("_sample_key") or p["id"]
            sample_labels.setdefault(key, []).append(p["predicted_label"])

    if not sample_labels:
        return float("nan")
    return float(np.mean([len(set(lbls)) for lbls in sample_labels.values()]))


def compute_entropy(seed_preds: dict) -> float:
    """
    Average prediction entropy (bits) across samples.

    For each sample the empirical label distribution across all seeds is built
    from raw label counts, and its Shannon entropy H = -Σ p·log₂(p) is
    computed.  The per-sample entropies are then averaged over all samples.

    A value of 0 means every seed agreed on the same label for every sample.
    Higher values indicate greater predictive uncertainty induced by stochastic
    training.  The theoretical maximum is log₂(#classes) bits.
    """
    sample_labels: dict = {}
    for preds in seed_preds.values():
        for p in preds:
            key = p.get("_text_key") or p.get("_sample_key") or p["id"]
            sample_labels.setdefault(key, []).append(p["predicted_label"])

    if not sample_labels:
        return float("nan")

    entropies = []
    for lbls in sample_labels.values():
        counts = np.array(list(Counter(lbls).values()), dtype=float)
        probs = counts / counts.sum()
        # All counts > 0 by construction, so log2 is safe
        entropies.append(-float(np.sum(probs * np.log2(probs))))

    return float(np.mean(entropies))


def compute_confidence_gap(predictions: list) -> float:
    """
    Confidence gap = mean confidence of correct predictions
                   - mean confidence of incorrect predictions.

    A larger positive value means the model assigns clearly higher confidence
    to its correct decisions than to its wrong ones.
    Returns NaN if either group is empty.
    """
    correct_confs, incorrect_confs = [], []
    for p in predictions:
        conf = p["probabilities"][p["predicted_label"]]
        if p["correct"]:
            correct_confs.append(conf)
        else:
            incorrect_confs.append(conf)

    if not correct_confs or not incorrect_confs:
        return float("nan")
    return float(np.mean(correct_confs) - np.mean(incorrect_confs))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — reliability diagrams
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagrams(model_results: dict, title_suffix: str,
                               out_path: Path) -> None:
    """
    N×2 grid of reliability (calibration) diagrams, where N = ceil(#models / 2).

    Subplot title = "<display_name>  ECE = <value>".
    Shared x-axis and y-axis (both [0, 1]).
    Legend appears once at the bottom centre.
    """
    ncols = 2
    nrows = math.ceil(len(model_results) / ncols)
    bar_w = 0.9 / N_BINS

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, nrows * 3.2),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.45, "wspace": 0.08},
    )
    axes_flat = axes.flatten()

    for idx, (model, res) in enumerate(model_results.items()):
        ax = axes_flat[idx]
        bd = res["bin_data"]

        mids = [b["mid"] for b in bd]
        accs = [b["accuracy"] for b in bd]

        # Model accuracy bars
        ax.bar(mids, accs, width=bar_w, color=MODEL_BAR_COLOR,
               alpha=0.85, zorder=2)

        # Gap shading between model and perfect calibration
        for b in bd:
            if b["count"] > 0:
                lo_y = min(b["accuracy"], b["mid"])
                hi_y = max(b["accuracy"], b["mid"])
                ax.bar(b["mid"], hi_y - lo_y, bottom=lo_y, width=bar_w,
                       color=GAP_COLOR, alpha=0.6, zorder=3)

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], zorder=4, **PERFECT_LINE_STYLE)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{display_name(model)}\nECE = {res['ece']:.4f}")
        ax.set_aspect("equal", adjustable="box")

    # Hide unused axes
    for idx in range(len(model_results), nrows * ncols):
        axes_flat[idx].set_visible(False)

    # Shared axis labels
    fig.text(0.5, 0.01, "Confidence", ha="center", fontsize=11)
    fig.text(0.01, 0.5, "Accuracy", va="center", rotation="vertical", fontsize=11)

    # Single legend
    legend_handles = [
        mpatches.Patch(color=MODEL_BAR_COLOR, alpha=0.85, label="Model"),
        mpatches.Patch(color=GAP_COLOR, alpha=0.6, label="Gap"),
        Line2D([0], [0], label="Perfect calibration", **PERFECT_LINE_STYLE),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(f"Reliability Diagrams - {title_suffix}", fontsize=12, y=1.005)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — confidence score distributions
# ─────────────────────────────────────────────────────────────────────────────

def _kde_fill(ax, data: np.ndarray, color: str, x: np.ndarray,
              lw: float = 1.5):
    """Fit a KDE and draw a filled density curve; return the fitted KDE object."""
    if len(data) < 2:
        return None
    kde = gaussian_kde(data, bw_method="scott")
    y = kde(x)
    ax.plot(x, y, color=color, linewidth=lw)
    ax.fill_between(x, y, alpha=0.25, color=color)
    return kde


def plot_confidence_distributions(model_results: dict, title_suffix: str,
                                   out_path: Path) -> None:
    """
    Nx2 KDE density plots of prediction-confidence scores split by correctness,
    where N = ceil(#models / 2).

    Each subplot:
      • Main plot  : full range [0, 1]
      • Inset      : zoomed view [0.80, 1.00] placed at top-right
      • Annotation : μ and σ for correct (green) and incorrect (red) at top-left
    Shared x-axis [0, 1]; shared y-axis (density). Single legend at bottom.
    """
    ncols = 2
    nrows = math.ceil(len(model_results) / ncols)
    x_full = np.linspace(0.0, 1.0, 500)
    x_zoom = np.linspace(0.80, 1.00, 300)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, nrows * 3.6),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.40, "wspace": 0.08},
    )
    axes_flat = axes.flatten()

    for idx, (model, res) in enumerate(model_results.items()):
        ax = axes_flat[idx]
        confs = res["confidences"]
        corrs = res["corrects"]

        c_conf = confs[corrs == 1.0]
        i_conf = confs[corrs == 0.0]

        kde_c = _kde_fill(ax, c_conf, CORRECT_COLOR, x_full)
        kde_i = _kde_fill(ax, i_conf, INCORRECT_COLOR, x_full)

        ax.set_xlim(0, 1)
        ax.set_title(display_name(model))

        # ── Stats annotation at top-left ──────────────────────────────────
        def _fmt(arr: np.ndarray) -> str:
            if len(arr) == 0:
                return "μ=–, σ=–"
            return f"μ={arr.mean():.3f}, σ={arr.std():.3f}"

        ann_kw = dict(transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.65, ec="none"))
        ax.text(0.03, 0.97, f"Correct:   {_fmt(c_conf)}",
                color=CORRECT_COLOR, **ann_kw)
        ax.text(0.03, 0.87, f"Incorrect: {_fmt(i_conf)}",
                color=INCORRECT_COLOR, **ann_kw)

        # ── Zoomed inset at top-right [0.80 – 1.00] ──────────────────────
        ax_ins = ax.inset_axes([0.53, 0.50, 0.45, 0.46])
        if kde_c is not None:
            y_z = kde_c(x_zoom)
            ax_ins.plot(x_zoom, y_z, color=CORRECT_COLOR, linewidth=1.2)
            ax_ins.fill_between(x_zoom, y_z, alpha=0.25, color=CORRECT_COLOR)
        if kde_i is not None:
            y_z = kde_i(x_zoom)
            ax_ins.plot(x_zoom, y_z, color=INCORRECT_COLOR, linewidth=1.2)
            ax_ins.fill_between(x_zoom, y_z, alpha=0.25, color=INCORRECT_COLOR)
        ax_ins.set_xlim(0.80, 1.00)
        ax_ins.set_xticks([0.80, 0.90, 1.00])
        ax_ins.tick_params(labelsize=6)
        ax_ins.set_title("Zoom [0.80-1.00]", fontsize=6.5)

    # Hide unused axes
    for idx in range(len(model_results), nrows * ncols):
        axes_flat[idx].set_visible(False)

    # Shared axis labels
    fig.text(0.5, 0.01, "Confidence", ha="center", fontsize=11)
    fig.text(0.01, 0.5, "Density", va="center", rotation="vertical", fontsize=11)

    # Single legend
    legend_handles = [
        Line2D([0], [0], color=CORRECT_COLOR, linewidth=2, label="Correct"),
        Line2D([0], [0], color=INCORRECT_COLOR, linewidth=2, label="Incorrect"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(f"Confidence Score Distributions - {title_suffix}", fontsize=12, y=1.005)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Excel export
# ─────────────────────────────────────────────────────────────────────────────

def export_scores(model_results: dict, out_path: Path) -> None:
    """
    Write ECE and stability scores to a single-sheet Excel file.
    If the target file is locked (e.g. open in Excel), falls back to a
    timestamped filename in the same directory so the run is never aborted.
    """
    from datetime import datetime

    rows = [
        {
            "Model": model,
            "ECE": round(res["ece"], 6),
            "Stability (avg unique labels/sample)": round(res["stability"], 4),
            "Entropy (avg bits/sample)": round(res["entropy"], 6),
        }
        for model, res in model_results.items()
    ]
    df = pd.DataFrame(rows)

    try:
        df.to_excel(out_path, index=False, sheet_name="Scores")
        print(f"    Saved: {out_path}")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = out_path.with_name(f"scores_{ts}.xlsx")
        df.to_excel(fallback, index=False, sheet_name="Scores")
        print(f"    [WARN] {out_path.name} is locked — saved to: {fallback.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_combination(dataset_key: str, config: str) -> None:
    """Process one (dataset, config) combination — metrics, plots, export."""
    out_dir = OUTPUT_BASE / f"{dataset_key}_{config}"
    out_dir.mkdir(parents=True, exist_ok=True)

    label = f"{dataset_key} / {config}"
    print(f"\n{'-' * 60}")
    print(f"  Dataset : {dataset_key}   Config : {config}")
    print(f"  Output  : {out_dir}")
    print(f"{'-' * 60}")

    model_results = {}

    for model in MODELS:
        try:
            if dataset_key == "combined":
                all_preds, seed_preds = load_predictions_combined(model, config)
            else:
                all_preds, seed_preds = load_predictions(model, dataset_key, config)

            if not all_preds:
                print(f"  [WARN] No predictions found for {model} — skipping.")
                continue

            ece, bin_data, confs, corrs = compute_ece(all_preds)
            stability = compute_stability(seed_preds)
            entropy = compute_entropy(seed_preds)

            model_results[model] = {
                "ece": ece,
                "bin_data": bin_data,
                "confidences": confs,
                "corrects": corrs,
                "stability": stability,
                "entropy": entropy,
            }
            print(f"  {display_name(model):30s}  ECE={ece:.4f}  Stability={stability:.4f}  Entropy={entropy:.4f}")

        except Exception as exc:
            print(f"  [ERROR] {model}: {exc}")

    if not model_results:
        print("  No results — skipping plots.")
        return

    plot_reliability_diagrams(model_results, label,
                               out_dir / "reliability_diagram.pdf")
    plot_confidence_distributions(model_results, label,
                                   out_dir / "confidence_distribution.pdf")
    export_scores(model_results, out_dir / "scores.xlsx")


def collect_and_export_csv_metrics() -> None:
    """
    Collect per-seed ECE and confidence gap, and per-config stability,
    for all real datasets (ieee / kaggle / tweeteval — NOT combined).

    Exports three CSV files to OUTPUT_BASE:
      ece_per_seed.csv            — Model, Dataset, Config, Seed, ECE
      confidence_gap_per_seed.csv — Model, Dataset, Config, Seed, ConfidenceGap
      stability_per_config.csv    — Model, Dataset, Config, Stability
    """
    print("\n" + "=" * 60)
    print("Collecting per-seed metrics for CSV export ...")
    print("=" * 60)

    ece_rows: list = []
    gap_rows: list = []
    stab_rows: list = []

    for config in CONFIGS:
        for ds in DATASETS:
            for model in MODELS:
                try:
                    _, seed_preds = load_predictions(model, ds, config)
                    if not seed_preds:
                        continue

                    # ── Per-seed: ECE and confidence gap ──────────────────
                    for seed_key, preds in sorted(seed_preds.items()):
                        # seed_key format: "{dataset}_seed_{number}"
                        seed_label = seed_key.split("_", 1)[1]   # "seed_14298463"

                        ece_val, *_ = compute_ece(preds)
                        gap_val = compute_confidence_gap(preds)

                        base = {"Model": model, "Dataset": ds,
                                "Config": config, "Seed": seed_label}
                        ece_rows.append({**base, "ECE": round(ece_val, 6)})
                        gap_rows.append({**base, "ConfidenceGap": round(gap_val, 6)})

                    # ── Per-config: stability + entropy (all seeds together) ─
                    stab_val = compute_stability(seed_preds)
                    entropy_val = compute_entropy(seed_preds)
                    stab_rows.append({
                        "Model": model, "Dataset": ds, "Config": config,
                        "Stability": round(stab_val, 6),
                        "Entropy": round(entropy_val, 6),
                    })

                    print(f"  {display_name(model):30s}  {ds:10s}  {config}")

                except Exception as exc:
                    print(f"  [ERROR] {model} / {ds} / {config}: {exc}")

    # ── Write CSVs ─────────────────────────────────────────────────────────
    ece_path  = OUTPUT_BASE / "ece_per_seed.csv"
    gap_path  = OUTPUT_BASE / "confidence_gap_per_seed.csv"
    stab_path = OUTPUT_BASE / "stability_per_config.csv"

    pd.DataFrame(ece_rows).to_csv(ece_path, index=False)
    print(f"\n  Saved: {ece_path}")

    pd.DataFrame(gap_rows).to_csv(gap_path, index=False)
    print(f"  Saved: {gap_path}")

    pd.DataFrame(stab_rows).to_csv(stab_path, index=False)
    print(f"  Saved: {stab_path}")


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    for config in CONFIGS:
        for ds in DATASETS + ["combined"]:
            run_combination(ds, config)

    collect_and_export_csv_metrics()

    print("\nAll combinations done.")


if __name__ == "__main__":
    main()