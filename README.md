# Towards Forensically Sound Automated Cyberbullying Detection

<!-- > **Swardiantara Silalahi, Jiangbin Zheng, Tohari Ahmad, Hudan Studiawan**
> Submitted to *IEEE Transactions on Information Forensics and Security (T-IFS)* -->

---

## Paper Summary

Common cyberbullying detection pipelines apply extensive text preprocessing (lowercasing, stop-word removal, stemming, emoji stripping, etc.) to social media posts before feeding them into a classifier. While this is standard NLP practice, it is problematic when the detection output must serve as evidence in legal or disciplinary proceedings: modifying the original artifact before analysis breaks the **evidence integrity** and the **chain of custody** as required by established digital forensics standards (ACPO Principle 1 & 3).

This paper addresses that gap from three angles:

1. **Taxonomy.** A formal, three-grade reversibility taxonomy is proposed to classify 24 common NLP preprocessing operations:
   - **Grade A – Strictly Forensically Reversible:** bijective transformations (e.g., subword tokenization, character encoding normalization) that can be exactly inverted without any auxiliary data.
   - **Grade B – Conditionally Forensically Reversible:** non-bijective but reconstructable transformations (e.g., whitespace normalization, offset mapping) *if and only if* a cryptographically authenticated auxiliary record is preserved alongside the evidence.
   - **Grade C – Forensically Irreversible:** transformations that permanently destroy information (e.g., lowercasing, stop-word removal, stemming, lemmatization, emoji stripping) and constitute *algorithmic destructive alteration* of digital evidence.

2. **Empirical Evaluation.** A systematic experiment fine-tunes **19 pre-trained transformer-based models** from **five architectural families** under two conditions: *raw* (Grade A only) vs. *preprocessed* (full pipeline from Philipo et al. [7]), on **three cyberbullying benchmarks**, each repeated **10 times** with different random seeds. Performance (macro-F1), reliability (ECE), and stability (per-sample prediction variance) are all measured.

3. **Forensic Implications.** Two considerations regarding post-hoc attribution on transformed text are formalized, along with actionable recommendations for forensically admissible deployment.

---

## Key Findings

### Finding 1 — Preprocessing consistently degrades classification performance
A Linear Mixed-Effects Model (LMEM) fitted on all 19 models × 3 datasets × 10 runs reveals a **highly significant negative effect** of preprocessing on macro-F1 ($\beta$ = −0.0200, $p$ < 0.001). Of 57 model–dataset configurations, only **1** (DistilBERT-cased on the IEEE DataPort dataset) was not harmed. Wilcoxon and binomial sign tests confirm the effect (both $p$ < 0.001, effect size $r$ = −0.999).

### Finding 2 — Preprocessing degrades model reliability (calibration)
Expected Calibration Error (ECE) increases significantly under preprocessing ($\beta$ = +0.0085, $p$ < 0.001 via LMEM). **45 of 57** model–dataset pairs show worse calibration after preprocessing, indicating that the model's confidence scores are less trustworthy when cleansed text is used.

### Finding 3 — Preprocessing reduces prediction stability
Preprocessing significantly increases both the mean number of unique per-sample predicted labels across runs ($\hat{Y}_{\mu}$: 52/57 pairs degraded, *p* < 0.001) and per-sample prediction entropy ($H$: 53/57 pairs degraded, *p* < 0.001), reducing the reproducibility of the analysis, which is a critical requirement for forensic admissibility.

### Finding 4 — Preprocessing sensitivity differs across model families
A Kruskal–Wallis test reveals a statistically significant difference in preprocessing sensitivity across model families ($p$ < 0.001 on F1, $p$ = 0.034 on ECE). The sensitivity ranking from least to most affected is:

> **Recent > General > Domain ≈ Sentence ≈ Small**

**NeoBERT** and **ModernBERT** (the *Recent* group) achieve zero observed performance gain from preprocessing across all configurations ($\bar{R}_{F1}$ = 0, 100% of cases show non-positive $\Delta$), making them the best candidates for forensically sound deployment. This robustness is attributed to their larger and more up-to-date pre-training corpora that capture evolving social media language patterns.

### Finding 5 — Two XAI attribution considerations for forensic pipelines
1. **Non-Injective Attribution Transfer Problem (Grade B/C):** When non-bijective preprocessing is applied, post-hoc attribution scores computed on transformed tokens cannot be deterministically projected back onto the original raw text without introducing an arbitrary heuristic weighting function, making the resulting attribution legally challengeable.
2. **Aggregation Vulnerability in Bijective Tokenization (Grade A):** Even with fully reversible BPE/WordPiece tokenization, attributions are computed at the *subword* level. Aggregating them to raw words (via sum, mean, or max) introduces a methodological choice that operates outside the model's computational graph, threatening evidentiary integrity unless a standardized aggregation protocol is defined prior to analysis.

---

## Reproducing the Experiments

### Requirements

| Component | Version |
|---|---|
| OS | Ubuntu LTS 20.04 |
| Python | 3.12.4 |
| PyTorch | 2.3.1 |
| GPU | NVIDIA RTX 3080 Ti (12 GB VRAM) or equivalent |

Install dependencies:

```bash
pip install torch==2.3.1 transformers datasets scikit-learn scipy numpy pandas
```
<!-- pip install pingouin statsmodels  # for statistical tests (Python-side) -->

> **Note:** The LMEM analysis (Table V in the paper) uses the `lme4` package in R with Satterthwaite degree-of-freedom approximation. Install R and run: `install.packages(c("lme4", "lmerTest"))`.

---

### Step 1 — Obtain the Datasets

Three datasets are used. Download and place them under `data/`:

| Dataset | Source | Encoding |
|---|---|---|
| Kaggle Cyberbullying | [Wang et al., IEEE BigData 2020](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) | UTF-8 |
| IEEE DataPort Cyberbullying Types | [Ananthi, 2021](https://dx.doi.org/10.21227/bsdy-zw62) | ISO-8859-1 |
| TweetEval | [Barbieri et al., EMNLP Findings 2020](https://github.com/cardiffnlp/tweeteval) | ISO-8859-1 |

```
dataset/
├── ieee/
├── kaggle/
└── tweeteval/
```

---

### Step 2 — Split the Datasets

All splits are performed **before any preprocessing** using random state `2048` to ensure reproducibility. The same split is used for both experimental conditions (raw and preprocessed).

```python
from sklearn.model_selection import train_test_split

RANDOM_STATE = 2048
# stratified train/val/test split: 64% / 16% / 20%
X_train_val, X_test, y_train_val, y_test = train_test_split(
    texts, labels, test_size=0.20, stratify=labels, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, stratify=y_train_val, random_state=RANDOM_STATE)
```

After splitting, verify split integrity by computing the SHA-256 hash of each split file against the reference hashes in `dataset/split_hashes.json`. This confirms you are running on exactly the same data used in the paper.

---

### Step 3 — Configure Experimental Conditions

Two preprocessing conditions are evaluated:

**Condition A — Baseline (Raw):** Only Grade A transformations are applied:
1. Character encoding normalization (ISO-8859-1 → Unicode for IEEE DataPort and TweetEval; UTF-8 for Kaggle).
2. Bijective subword tokenization by each model's built-in HuggingFace tokenizer.

**Condition B — Preprocessed:** The full pipeline from Philipo et al. [7] is additionally applied on top of Condition A. This includes Grade B/C operations: lowercasing, URL/mention/hashtag removal, emoji stripping, contraction expansion, number removal, short-word removal, elongated-word normalization, repeated-punctuation collapsing, punctuation stripping, stop-word removal, stemming, and lemmatization.

> Preprocessing operations that reduce the number of test samples (e.g., non-English filtering, short-document filtering) are **excluded from the test set** to ensure identical test samples across both conditions and a fair comparison.

---

### Step 4 — Select Models

The 19 models evaluated, grouped by family, are listed below. All are loaded via HuggingFace `transformers`.

| Family | HuggingFace Model ID | Params |
|---|---|---|
| **General** | `bert-base-cased` | 108M |
| | `bert-base-uncased` | 108M |
| | `roberta-base` | 125M |
| | `xlnet-base-cased` | 117M |
| | `gpt2` | 124M |
| **Domain** | `Twitter/twhin-bert-base` | 279M |
| | `vinai/bertweet-base` | 135M |
| | `sarkerlab/SocBERT-base` | 143M |
| | `GroNLP/hateBERT` | 109M |
| **Sentence** | `sentence-transformers/all-mpnet-base-v2` | 109M |
| | `sentence-transformers/all-distilroberta-v1` | 82M |
| | `sentence-transformers/all-MiniLM-L12-v2` | 33M |
| | `sentence-transformers/all-MiniLM-L6-v2` | 23M |
| **Small** | `distilbert-base-cased` | 66M |
| | `distilbert-base-uncased` | 66M |
| | `google/mobilebert-uncased` | 25M |
| | `albert-base-v2` | 12M |
| **Recent** | `answerdotai/ModernBERT-base` | 150M |
| | `liu-nlp/NeoBERT` | 245M |

---

### Step 5 — Training Configuration

All models are fine-tuned using the following hyperparameters, identical to the reference study [7]:

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Learning rate | 5e-5 |
| Batch size | 32 |
| Optimizer | AdamW |
| Checkpoint selection | Best validation macro-F1 (no early stopping) |
| Runs per configuration | 10 (with different random seeds) |

Use the **same set of 10 seeds** across all model–dataset configurations to ensure fair comparison and reproducible variance estimates.

---

### Step 6 — Run Training

```bash
bash scripts/run_all.sh
```

This will run all 19 models × 3 datasets × 2 conditions = **114 configurations**, each executed 10 times (1,140 total runs).

---

### Step 7 — Evaluate and Compute Metrics

For each run, the experimental result is stored under the `experiments/grid-search` directory, with a structure of `{model_name}/{dataset}/prep{1/0}_aug{1/0}/seed_{seed}`. Each run directory contains several files:

```
~/
├── config.json                 # The run's configs
├── confusion_matrix.json       # Confusion matrix in JSON
├── confusion_matrix.pdf        # Confusion matrix in PDF
├── metrics.json                # Accuracy, precision, recall, and F1
├── predictions.json            # Raw predictions with per-class probability for analysis
└── run.log                     # Run log for execution debugging
```

The main metrics used and reported in the paper are:
- **Macro-averaged Accuracy, Precision, Recall, F1**
- **Expected Calibration Error (ECE)** with *M* = 10 equal-width confidence bins (Eq. 4 in the paper)
- **Prediction stability $\hat{Y}_{\mu}$** — mean number of unique predicted labels per sample across 10 runs (Eq. 5)
- **Per-sample entropy $H$** — mean empirical entropy of predictions across 10 runs (Eq. 6)

We report mean $\pm$ std across the 10 runs for each metric.

<!-- ```bash
python evaluate.py \
  --results_dir results/ \
  --output metrics_summary.csv
``` -->

---

### Step 8 — Statistical Testing

Reproduce the statistical analyses from the paper:

**LMEM on F1, ECE, Stability, and Entropy (R):**
```R
Rscript notebooks/compute_lmm.r
```

This command will produce excel files (`lmm_results_F1.xlsx`, `lmm_results_ECE.xlsx`, `lmm_results_Stability.xlsx`, and `lmm_results_Entropy.xlsx`) storing the LMM fitting results on each metric with a model of:
$$
\text{Metric} \sim  \text{Prep.} + (1|\text{Data}) + (1|\text{Model}) + (1|\text{Model}:\text{Data})
$$
The fitting results are used to construct Table IV and Table V.

**Wilcoxon & Binomial Sign Tests (Python):**
```bash
python notebooks/compute_wilcoxon_test.py     # Table VI
python notebooks/compute_ece.py               # Fig. 2-5
```

**Kruskal–Wallis + Dunn post-hoc with Benjamini–Hochberg correction (Python):**
```bash
python notebooks/compute_kruskal.py
```

---

### Expected Results

The key quantitative results from the paper are summarized below for verification:

| Metric | Baseline (Raw) | + Preprocessing | Effect |
|---|---|---|---|
| Combined F1 ($\uparrow$) | 0.914 $\pm$ 0.067 | 0.894 $\pm$ 0.064 | $\beta$ = −0.0200*** |
| Combined ECE ($\downarrow$) | 0.059 $\pm$ 0.043 | 0.067 $\pm$ 0.041 | $\beta$ = +0.0085*** |
| Combined $\hat{Y}_{\mu}$ ($\downarrow$) | 1.174 $\pm$ 0.163 | 1.208 $\pm$ 0.167 | 52/57 degraded*** |
| Combined $H$ ($\downarrow$) | 0.112 $\pm$ 0.102 | 0.134 $\pm$ 0.104 | 53/57 degraded*** |

`***` $p$ < 0.001. The raw baseline consistently outperforms the preprocessed condition on all four forensic metrics.

---

## Citation

If you use this codebase or find the taxonomy and findings useful in your research, please cite:

```bibtex
@article{silalahi2026forensic,
  author    = {Silalahi, Swardiantara and Zheng, Jiangbin and Ahmad, Tohari and Studiawan, Hudan},
  title     = {Towards Forensically Sound Automated Cyberbullying Detection},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.