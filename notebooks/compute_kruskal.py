# Kruskal-Wallis H-test
# Research Question: Which model families are most robust to the absence of preprocessing?
#
# Approach:
#   For each (model, dataset, augmentation, seed) pair, compute:
#     delta = score(prep1) - score(prep0)
#   where delta ~ 0 means the model is insensitive to preprocessing.
#   Group deltas by model family, then test whether families differ in their
#   preprocessing sensitivity using the Kruskal-Wallis H-test. Dunn's post-hoc
#   test (Benjamini-Hochberg correction) identifies which pairs differ.

import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── Model Groups ──────────────────────────────────────────────────────────────
MODEL_GROUP = {
    "general":  ["bert-base-cased", "roberta-base", "bert-base-uncased", "xlnet-base-cased", "gpt2"],
    "domain":   ["GroNLP/hateBERT", "vinai/bertweet-base", "Twitter/twhin-bert-base", "sarkerlab/SocBERT-base"],
    "sentence": ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2", "all-distilroberta-v1"],
    "small":    ["albert/albert-base-v2", "distilbert-base-uncased", "google/mobilebert-uncased", "distilbert-base-cased"],
    "modern":   ["chandar-lab/NeoBERT", "answerdotai/ModernBERT-base"],
}

FAMILY_ORDER = list(MODEL_GROUP.keys())
DATASETS = ['ieee', 'kaggle', 'tweeteval']
ALPHA = 0.05

# Reverse lookup: model → family
# F1 data uses underscores instead of slashes in model names
MODEL_TO_FAMILY_SLASH = {}   # for ECE data (original slash format)
MODEL_TO_FAMILY_UNDER = {}   # for F1 data (slash replaced by underscore)
for family, models in MODEL_GROUP.items():
    for m in models:
        MODEL_TO_FAMILY_SLASH[m] = family
        MODEL_TO_FAMILY_UNDER[m.replace('/', '_')] = family

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
F1_PATH  = BASE_DIR / "analysis/grid-search/recap/recap_all.xlsx"
ECE_PATH = BASE_DIR / "analysis/reliability/ece_per_seed.csv"
OUT_DIR  = BASE_DIR / "analysis/statistical-test"
OUT_DIR.mkdir(exist_ok=True)

# ─── Helper Functions ──────────────────────────────────────────────────────────

def interpret_effect_size(eta_sq):
    """Interpret eta-squared: negligible / small / medium / large."""
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


def compute_eta_squared(h_stat, n_groups, n_total):
    """Epsilon-squared (rank-based) effect size for Kruskal-Wallis."""
    return (h_stat - n_groups + 1) / (n_total - n_groups)


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def compute_deltas(df_prep0, df_prep1, value_col, merge_keys):
    """Merge prep0 and prep1 rows and compute delta = value(prep1) - value(prep0)."""
    merged = df_prep0.merge(df_prep1, on=merge_keys, suffixes=('_prep0', '_prep1'))
    merged['delta'] = merged[f'{value_col}_prep1'] - merged[f'{value_col}_prep0']
    return merged


def run_kruskal(groups_dict, family_order):
    """Run Kruskal-Wallis across groups; return H, p, eta-squared, families used.

    Returns (nan, nan, nan, families_used) if the test cannot be computed
    (e.g. all R values are identical across all groups, which can happen when
    R = max(0, delta) collapses to all zeros within a dataset).
    """
    families_used = [f for f in family_order if f in groups_dict and len(groups_dict[f]) > 0]
    if len(families_used) < 2:
        return float('nan'), float('nan'), float('nan'), families_used
    arrays = [groups_dict[f] for f in families_used]
    try:
        h_stat, p_val = stats.kruskal(*arrays)
    except ValueError:
        # All values identical — test is undefined; treat as no difference
        return float('nan'), float('nan'), float('nan'), families_used
    n_total = sum(len(a) for a in arrays)
    eta_sq = compute_eta_squared(h_stat, len(families_used), n_total)
    return h_stat, p_val, eta_sq, families_used


def build_dunn_long(dunn_matrix):
    """Convert Dunn p-value matrix to a long-format DataFrame."""
    families = dunn_matrix.index.tolist()
    rows = []
    for i, f1 in enumerate(families):
        for j, f2 in enumerate(families):
            if j > i:
                p = dunn_matrix.loc[f1, f2]
                rows.append({
                    'Family 1': f1,
                    'Family 2': f2,
                    'Adjusted p-value (BH)': round(p, 6),
                    'Significance': sig_stars(p),
                    'Significant (a=0.05)': 'Yes' if p < ALPHA else 'No',
                    'Interpretation': (
                        f"Significant difference in preprocessing sensitivity between '{f1}' and '{f2}'."
                        if p < ALPHA else
                        f"No significant difference between '{f1}' and '{f2}'."
                    )
                })
    return pd.DataFrame(rows)


def autofit_columns(ws):
    """Auto-fit column widths in an openpyxl worksheet."""
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                cell_len = len(str(cell.value)) if cell.value is not None else 0
                max_len = max(max_len, cell_len)
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + 4, 80)


# ─── Load Data ─────────────────────────────────────────────────────────────────

print("Loading F1 and ECE data...")
df_f1  = pd.read_excel(F1_PATH)
df_ece = pd.read_csv(ECE_PATH)

# Assign family to each row
df_f1['family']  = df_f1['model'].map(MODEL_TO_FAMILY_UNDER)
df_ece['family'] = df_ece['Model'].map(MODEL_TO_FAMILY_SLASH)

# Warn about any unmapped models
unmapped_f1  = df_f1[df_f1['family'].isna()]['model'].unique()
unmapped_ece = df_ece[df_ece['family'].isna()]['Model'].unique()
if len(unmapped_f1)  > 0: print(f"Warning: Unmapped F1 models:  {unmapped_f1}")
if len(unmapped_ece) > 0: print(f"Warning: Unmapped ECE models: {unmapped_ece}")

# ─── Compute F1 Deltas ─────────────────────────────────────────────────────────
# Only aug0 is used to isolate the preprocessing effect without confounding
# from augmentation. Configs compared: prep0_aug0 vs prep1_aug0.
# Δ_F1 = F1(prep1_aug0) − F1(prep0_aug0)
# Positive → preprocessing improved F1; near 0 → family is robust to its absence.

print("Computing F1 deltas (prep1_aug0 - prep0_aug0, aug0 only)...")
f1_aug0 = df_f1[df_f1['augmentation'] == False]
merge_keys_f1 = ['model', 'dataset', 'seed', 'family']
f1_delta = compute_deltas(
    f1_aug0[f1_aug0['preprocessing'] == False],
    f1_aug0[f1_aug0['preprocessing'] == True],
    'macro_f1',
    merge_keys_f1
)
# Rename for consistent downstream usage
f1_delta = f1_delta.rename(columns={'dataset': 'Dataset'})

# ─── Compute ECE Deltas ────────────────────────────────────────────────────────
# Only aug0 is used. Configs compared: prep0_aug0 vs prep1_aug0.
# Δ_ECE = ECE(prep1_aug0) − ECE(prep0_aug0)
# Negative → preprocessing reduced ECE (improved calibration); near 0 → robust.

print("Computing ECE deltas (prep1_aug0 - prep0_aug0, aug0 only)...")
df_ece['seed_num'] = df_ece['Seed'].str.replace('seed_', '').astype(int)

merge_keys_ece = ['Model', 'Dataset', 'seed_num', 'family']
ece_delta = compute_deltas(
    df_ece[df_ece['Config'] == 'prep0_aug0'],
    df_ece[df_ece['Config'] == 'prep1_aug0'],
    'ECE',
    merge_keys_ece
)

# ─── Full Analysis Function ────────────────────────────────────────────────────

def run_full_analysis(delta_df, metric_name, metric_col, r_transform,
                      r_transform_note, out_filename, direction_note,
                      robustness_note, data_source):
    """
    Run Kruskal-Wallis + Dunn's post-hoc analysis on a robustness score R
    derived from raw Δ = score(prep1) − score(prep0), and export to Excel.

    The K-W test and Dunn's post-hoc operate on R (not on raw Δ), where R is
    a one-sided penalty that is zero when a family is already robust and
    positive when it depends on preprocessing.

    Parameters
    ----------
    delta_df        : DataFrame with 'family', 'Dataset', and metric_col columns.
    metric_name     : Human-readable metric name.
    metric_col      : Column name for the raw delta values.
    r_transform     : Callable (array → array) converting Δ to R.
                      F1  → max(0,  Δ)   penalise when preprocessing improves F1
                      ECE → max(0, −Δ)   penalise when preprocessing reduces ECE
    r_transform_note: One-sentence description of the transform (for the sheet).
    out_filename    : Output Excel filename.
    direction_note  : Explanation of raw Δ direction (shown in Overview sheet).
    robustness_note : Explanation of what robustness means for this metric.
    data_source     : Path string for provenance tracking.
    """
    print(f"\n=== {metric_name} Analysis ===")

    # ── Compute R per observation ─────────────────────────────────────────────
    delta_df = delta_df.copy()
    delta_df['R'] = r_transform(delta_df[metric_col].values)

    # ── Descriptive statistics per family (global, all datasets pooled) ────────
    desc_rows = []
    for fam in FAMILY_ORDER:
        sub  = delta_df[delta_df['family'] == fam]
        dv   = sub[metric_col].values
        rv   = sub['R'].values
        if len(dv) == 0:
            continue
        desc_rows.append({
            'Family':             fam,
            'N (observations)':   len(dv),
            # Raw delta — context only, NOT used in test
            'Mean Δ (raw)':       round(float(np.mean(dv)),            6),
            'Median Δ (raw)':     round(float(np.median(dv)),          6),
            'Std Δ (raw)':        round(float(np.std(dv,  ddof=1)),    6),
            '% Δ < 0 (robust)':   round(float(np.mean(dv < 0) * 100), 2),
            # Robustness score R — used in K-W test
            'Mean R':             round(float(np.mean(rv)),            6),
            'Median R':           round(float(np.median(rv)),          6),
            'Std R':              round(float(np.std(rv,   ddof=1)),   6),
            '% R = 0 (no loss)':  round(float(np.mean(rv == 0) * 100),2),
            'Robustness Rank':    None,   # filled after sorting
        })
    desc_df = pd.DataFrame(desc_rows).sort_values('Mean R').reset_index(drop=True)
    desc_df['Robustness Rank'] = range(1, len(desc_df) + 1)

    # ── Global Kruskal-Wallis (on R) ──────────────────────────────────────────
    groups_global = {
        f: delta_df[delta_df['family'] == f]['R'].values
        for f in FAMILY_ORDER
    }
    h_glob, p_glob, eta_glob, _ = run_kruskal(groups_global, FAMILY_ORDER)
    sig_glob = p_glob < ALPHA

    print(f"  Kruskal-Wallis (global, on R): H={h_glob:.4f}, p={p_glob:.6f}, eps2={eta_glob:.4f}")

    # ── Per-dataset Kruskal-Wallis (on R) ─────────────────────────────────────
    per_ds_rows = []
    for ds in DATASETS:
        sub = delta_df[delta_df['Dataset'] == ds]
        groups_ds = {f: sub[sub['family'] == f]['R'].values for f in FAMILY_ORDER}
        h_ds, p_ds, eta_ds, _ = run_kruskal(groups_ds, FAMILY_ORDER)
        if np.isnan(h_ds):
            per_ds_rows.append({
                'Dataset':              ds,
                'H-statistic':          'N/A',
                'p-value':              'N/A',
                'Significance':         'N/A',
                'Epsilon-squared (e2)': 'N/A',
                'Effect Size':          'N/A',
                'Significant (a=0.05)': 'N/A',
                'Note': 'All R values identical (all zeros) — test undefined for this dataset',
            })
        else:
            per_ds_rows.append({
                'Dataset':              ds,
                'H-statistic':          round(h_ds,  4),
                'p-value':              round(p_ds,  6),
                'Significance':         sig_stars(p_ds),
                'Epsilon-squared (e2)': round(eta_ds, 4),
                'Effect Size':          interpret_effect_size(eta_ds),
                'Significant (a=0.05)': 'Yes' if p_ds < ALPHA else 'No',
                'Note': '',
            })
    per_ds_df = pd.DataFrame(per_ds_rows)

    # ── Dunn's post-hoc (global, on R) ────────────────────────────────────────
    dunn_input = delta_df[['family', 'R']].rename(columns={'R': 'r_score'})
    dunn_matrix = sp.posthoc_dunn(
        dunn_input, val_col='r_score', group_col='family', p_adjust='fdr_bh'
    )
    dunn_long = build_dunn_long(dunn_matrix)

    # ── Build Excel ────────────────────────────────────────────────────────────
    out_path     = OUT_DIR / out_filename
    most_robust  = desc_df.iloc[0]
    least_robust = desc_df.iloc[-1]
    sig_pairs    = dunn_long[dunn_long['Significant (a=0.05)'] == 'Yes']

    def kw_interpretation(h, p, eta, sig):
        if sig:
            return (
                f"The Kruskal-Wallis H-test on robustness scores R reveals a statistically significant "
                f"difference among model families in their dependence on preprocessing "
                f"(H = {h:.4f}, p = {p:.6f}, eps2 = {eta:.4f}, {interpret_effect_size(eta)} effect). "
                f"This indicates that not all families are equally robust to the absence of preprocessing. "
                f"Dunn's post-hoc test (sheet 'Dunn Post-hoc') identifies which specific pairs differ."
            )
        else:
            return (
                f"The Kruskal-Wallis H-test on robustness scores R finds no statistically significant "
                f"difference among model families in their dependence on preprocessing "
                f"(H = {h:.4f}, p = {p:.6f}, eps2 = {eta:.4f}). "
                f"All families appear equally robust (or sensitive) to the absence of preprocessing."
            )

    conclusion = (
        f"The analysis of robustness scores R for {metric_name} shows that model families "
        f"differ {'significantly' if sig_glob else 'non-significantly'} in how much they depend "
        f"on preprocessing. "
        f"The '{most_robust['Family']}' family (Rank 1, Mean R = {most_robust['Mean R']:.4f}) is the "
        f"most robust — it loses the least (or gains) when preprocessing is removed. "
        f"The '{least_robust['Family']}' family (Rank {int(least_robust['Robustness Rank'])}, "
        f"Mean R = {least_robust['Mean R']:.4f}) is most dependent on preprocessing. "
        + (
            f"Significant pairwise differences were found between: "
            + '; '.join(
                f"{r['Family 1']} vs {r['Family 2']} ({r['Significance']})"
                for _, r in sig_pairs.iterrows()
            ) + '.'
            if len(sig_pairs) > 0 else
            "No significant pairwise differences were detected after BH correction."
        )
    )

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:

        # ── Sheet 1: Overview ─────────────────────────────────────────────────
        overview = pd.DataFrame({
            'Item': [
                'Research Question',
                'Metric',
                'Raw Delta Definition',
                'Direction Note',
                '',
                'Robustness Score R',
                'R Transform Note',
                'Robustness Definition',
                '',
                'Statistical Test',
                'Input to Test',
                'Post-hoc Test',
                'Multiple-comparison Correction',
                'Significance Level (a)',
                '',
                'F1 Data Source',
                'ECE Data Source',
                'Total Observations',
                'Datasets Included',
                'Augmentation Condition',
                'Seeds per Model-Dataset Combo',
                '',
                'Model Families',
            ],
            'Detail': [
                'Which model families are most robust to the absence of preprocessing?',
                metric_name,
                'Delta = score(prep1_aug0) - score(prep0_aug0) for each (model, dataset, seed)',
                direction_note,
                '',
                r_transform_note,
                (
                    'R = 0 when the family does not lose from preprocessing absence '
                    '(delta <= 0 for F1; delta >= 0 for ECE). '
                    'R > 0 only when preprocessing provides a net benefit.'
                ),
                robustness_note,
                '',
                'Kruskal-Wallis H-test — non-parametric, compares R distributions across families',
                'Robustness score R (not raw delta)',
                "Dunn's test — pairwise comparisons between all family pairs",
                'Benjamini-Hochberg (BH) — controls false discovery rate (FDR)',
                str(ALPHA),
                '',
                str(F1_PATH),
                str(ECE_PATH),
                str(len(delta_df)),
                ', '.join(DATASETS),
                'aug0 only — held fixed to avoid confounding the preprocessing effect',
                '10',
                '',
                '\n'.join(
                    f"  {k}: {', '.join(v)}"
                    for k, v in MODEL_GROUP.items()
                ),
            ]
        })
        overview.to_excel(writer, sheet_name='Overview', index=False)

        # ── Sheet 2: Kruskal-Wallis ───────────────────────────────────────────
        kw_global_rows = [
            ('Global Kruskal-Wallis H-test on R (All Datasets Pooled)', ''),
            ('H-statistic',                    round(h_glob,  4)),
            ('Degrees of freedom (k-1)',        len(FAMILY_ORDER) - 1),
            ('p-value',                         round(p_glob,  6)),
            ('Significance',                    sig_stars(p_glob)),
            ('Epsilon-squared (eps2)',          round(eta_glob, 4)),
            ('Effect Size Interpretation',      interpret_effect_size(eta_glob)),
            ('Significant (a=0.05)',            'Yes' if sig_glob else 'No'),
            ('', ''),
            ('Interpretation', kw_interpretation(h_glob, p_glob, eta_glob, sig_glob)),
        ]
        kw_global_df = pd.DataFrame(kw_global_rows, columns=['Metric', 'Value'])
        kw_global_df.to_excel(writer, sheet_name='Kruskal-Wallis', index=False)

        ws_kw  = writer.sheets['Kruskal-Wallis']
        start  = len(kw_global_df) + 3
        ws_kw.cell(row=start, column=1, value='Per-Dataset Kruskal-Wallis H-test Results (on R)')
        per_ds_df.to_excel(writer, sheet_name='Kruskal-Wallis', index=False, startrow=start)

        # ── Sheet 3: Dunn Post-hoc ────────────────────────────────────────────
        dunn_long.to_excel(writer, sheet_name='Dunn Post-hoc', index=False)

        ws_dunn       = writer.sheets['Dunn Post-hoc']
        matrix_start  = len(dunn_long) + 3
        ws_dunn.cell(row=matrix_start, column=1,
                     value="Dunn's Test Adjusted p-value Matrix (BH correction, input = R)")
        dunn_matrix.round(6).to_excel(
            writer, sheet_name='Dunn Post-hoc', startrow=matrix_start
        )

        # ── Sheet 4: Family Statistics ────────────────────────────────────────
        desc_display = desc_df.copy()
        desc_display.insert(
            desc_display.columns.get_loc('Robustness Rank') + 1,
            'Robustness Interpretation',
            desc_display['Robustness Rank'].apply(
                lambda r: (
                    'Most robust — lowest dependence on preprocessing'
                    if r == 1 else
                    f'Rank {int(r)} — {"moderately" if r < len(desc_df) else "least"} robust to preprocessing absence'
                )
            )
        )
        desc_display.to_excel(writer, sheet_name='Family Statistics', index=False)

        # ── Sheet 5: Interpretation ───────────────────────────────────────────
        interp = pd.DataFrame({
            'Section': [
                'Research Question',
                'Test Used',
                'Input to Test',
                'Global Result',
                'Effect Size',
                'Robustness Ranking (1 = most robust, by Mean R)',
                'Significant Pairwise Differences',
                'Conclusion',
            ],
            'Content': [
                'Which model families are most robust to the absence of preprocessing?',
                "Kruskal-Wallis H-test + Dunn's post-hoc (BH correction)",
                f"Robustness score R = {r_transform_note}",
                (
                    f"H = {h_glob:.4f}, p = {p_glob:.6f} -> "
                    f"{'Significant' if sig_glob else 'Not significant'} at a = {ALPHA}"
                ),
                f"eps2 = {eta_glob:.4f} ({interpret_effect_size(eta_glob)} effect)",
                ' > '.join(
                    f"{row['Family']} (Mean R={row['Mean R']:.4f})"
                    for _, row in desc_df.iterrows()
                ),
                (
                    '; '.join(
                        f"{r['Family 1']} vs {r['Family 2']} "
                        f"(p={r['Adjusted p-value (BH)']:.4f}, {r['Significance']})"
                        for _, r in sig_pairs.iterrows()
                    )
                    if len(sig_pairs) > 0 else 'None after BH correction'
                ),
                conclusion,
            ]
        })
        interp.to_excel(writer, sheet_name='Interpretation', index=False)

        # ── Auto-fit all sheets ───────────────────────────────────────────────
        for sheet_name in writer.sheets:
            autofit_columns(writer.sheets[sheet_name])

    print(f"  Saved -> {out_path}")
    return desc_df, dunn_long, h_glob, p_glob, eta_glob


# ─── Run F1 Analysis ───────────────────────────────────────────────────────────

f1_desc, f1_dunn, f1_h, f1_p, f1_eta = run_full_analysis(
    delta_df         = f1_delta,
    metric_name      = 'Macro F1',
    metric_col       = 'delta',
    # F1 is higher-is-better: penalise only when preprocessing improves F1
    # (delta > 0). Negative delta means raw text is at least as good → robust.
    r_transform      = lambda d: np.maximum(0.0, d),
    r_transform_note = (
        'R = max(0, delta_F1) where delta_F1 = F1(prep1) - F1(prep0). '
        'R = 0 when raw text matches or exceeds preprocessed performance (robust). '
        'R > 0 only when preprocessing improves F1 (the family depends on it).'
    ),
    out_filename  = 'kruskal_preprocessing_f1.xlsx',
    direction_note = (
        'Configs compared: prep1_aug0 vs prep0_aug0 (aug0 held fixed). '
        'delta = F1(prep1_aug0) - F1(prep0_aug0). '
        'Positive delta -> preprocessing improved F1. '
        'Negative delta -> raw text is at least as good (robust case).'
    ),
    robustness_note = (
        'A family is robust if it does not lose F1 when preprocessing is absent '
        '(delta <= 0). Families with smaller Mean R depend less on preprocessing.'
    ),
    data_source = str(F1_PATH),
)

# ─── Run ECE Analysis ─────────────────────────────────────────────────────────

ece_desc, ece_dunn, ece_h, ece_p, ece_eta = run_full_analysis(
    delta_df         = ece_delta,
    metric_name      = 'ECE (Expected Calibration Error)',
    metric_col       = 'delta',
    # ECE is lower-is-better: the direction is reversed relative to F1.
    # delta = ECE(prep1) - ECE(prep0).
    # delta < 0 -> preprocessing REDUCED ECE -> family NEEDS preprocessing
    #              for calibration -> penalise -> R = -delta > 0.
    # delta >= 0 -> raw text is at least as well calibrated -> robust -> R = 0.
    r_transform      = lambda d: np.maximum(0.0, -d),
    r_transform_note = (
        'R = max(0, -delta_ECE) where delta_ECE = ECE(prep1) - ECE(prep0). '
        'ECE is lower-is-better, so the penalty is applied when preprocessing '
        'reduces ECE (delta < 0), meaning the family needs preprocessing for '
        'good calibration. R = 0 when raw-text ECE is no worse than preprocessed.'
    ),
    out_filename  = 'kruskal_preprocessing_ece.xlsx',
    direction_note = (
        'Configs compared: prep1_aug0 vs prep0_aug0 (aug0 held fixed). '
        'delta = ECE(prep1_aug0) - ECE(prep0_aug0). ECE is lower-is-better. '
        'Negative delta -> preprocessing improved calibration (family depends on it). '
        'Positive delta -> raw text is at least as well calibrated (robust case).'
    ),
    robustness_note = (
        'A family is robust if its ECE does not worsen when preprocessing is absent '
        '(delta >= 0, i.e. ECE on raw text is no higher than on preprocessed text). '
        'Families with smaller Mean R depend less on preprocessing for calibration.'
    ),
    data_source = str(ECE_PATH),
)

# ─── Console Summary ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nMacro F1 -- Kruskal-Wallis on R=max(0,delta): H={f1_h:.4f}, p={f1_p:.6f}, eps2={f1_eta:.4f}")
print(f"  {'Significant' if f1_p < ALPHA else 'Not significant'} at a={ALPHA}")
print("  Robustness Ranking on R=max(0,delta) (1=most robust, smaller R=less dependent on preprocessing):")
print(f1_desc[['Family', 'Mean Δ (raw)', '% Δ < 0 (robust)', 'Mean R', 'Robustness Rank']].to_string(index=False))

print(f"\nECE — Kruskal-Wallis on R=max(0,-delta): H={ece_h:.4f}, p={ece_p:.6f}, eps2={ece_eta:.4f}")
print(f"  {'Significant' if ece_p < ALPHA else 'Not significant'} at a={ALPHA}")
print("  Robustness Ranking on R=max(0,-delta) (1=most robust, smaller R=less dependent on preprocessing):")
print(ece_desc[['Family', 'Mean Δ (raw)', '% Δ < 0 (robust)', 'Mean R', 'Robustness Rank']].to_string(index=False))