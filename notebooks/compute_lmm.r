# =============================================================================
# Linear Mixed-Effects Model (LMM) — Effect of Preprocessing
# =============================================================================
# Metrics  : macro-F1, ECE, Stability, Entropy
# Configs  : prep0_aug0 (Preprocessing=0) and prep1_aug0 (Preprocessing=1) only
#
# Model formula (one LMM per metric):
#   Metric ~ Preprocessing + (1|Dataset) + (1|Model) + (1|Model:Dataset)
#
# Fixed effect : Preprocessing (0 = no preprocessing, 1 = preprocessing applied)
# Random effects: Random intercept for Dataset, Model, Model × Dataset
# Estimation   : REML
# P-values     : Satterthwaite degrees-of-freedom approximation (lmerTest)
#
# Metric definitions
#   F1          : macro-averaged F1 from metrics.json  (per seed)
#   ECE         : Expected Calibration Error           (per seed, from predictions.json)
#   Stability   : avg. unique predicted labels per sample across seeds
#                 (per config, from predictions.json; lower = more stable)
#   Entropy     : avg. Shannon entropy (bits) of per-sample label distributions
#                 across seeds (per config; lower = more consistent predictions)
#
# Output: one Excel file per metric → analysis/lmm-test/lmm_results_{metric}.xlsx
#         Each file has 6 sheets: Fixed_Effects, Random_Effects,
#         RE_Interpretation, Model_Summary, FE_Interpretation, Descriptive_Stats
# =============================================================================

.libPaths(c(file.path(Sys.getenv("USERPROFILE"), "R", "library"), .libPaths()))

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
  library(writexl)
})

cat("lme4     :", as.character(packageVersion("lme4")),     "\n")
cat("lmerTest :", as.character(packageVersion("lmerTest")), "\n\n")

# =============================================================================
# Configuration
# =============================================================================
MODELS <- list(
  general  = c("bert-base-cased", "roberta-base", "bert-base-uncased",
               "xlnet-base-cased", "gpt2"),
  domain   = c("GroNLP/hateBERT", "vinai/bertweet-base",
               "Twitter/twhin-bert-base", "sarkerlab/SocBERT-base"),
  sentence = c("all-MiniLM-L6-v2", "all-MiniLM-L12-v2",
               "all-mpnet-base-v2", "all-distilroberta-v1"),
  small    = c("albert/albert-base-v2", "distilbert-base-uncased",
               "google/mobilebert-uncased", "distilbert-base-cased"),
  modern   = c("chandar-lab/NeoBERT", "answerdotai/ModernBERT-base")
)

DATASETS <- c("ieee", "kaggle", "tweeteval")
CONFIGS  <- c("prep0_aug0", "prep1_aug0")   # Preprocessing only; aug=0 fixed
N_BINS   <- 10L

args <- commandArgs(trailingOnly = FALSE)
script_flag <- grep("^--file=", args, value = TRUE)
if (length(script_flag) > 0) {
  script_path <- sub("^--file=", "", script_flag[1])
  ROOT <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
} else {
  ROOT <- normalizePath(file.path(getwd(), ".."), mustWork = FALSE)
}

EXPR_PATH   <- file.path(ROOT, "experiments", "grid-search")
OUTPUT_PATH <- file.path(ROOT, "analysis", "lmm-test")
dir.create(OUTPUT_PATH, recursive = TRUE, showWarnings = FALSE)

cat("Root     :", ROOT, "\n")
cat("Output   :", OUTPUT_PATH, "\n\n")

# =============================================================================
# Metric computation helpers (mirrors compute_ece.py logic)
# =============================================================================

#' Expected Calibration Error (per-seed, equal-width bins)
compute_ece <- function(preds, n_bins = N_BINS) {
  confs <- vapply(preds, function(p)
    as.numeric(p$probabilities[[p$predicted_label]]), numeric(1))
  corrs <- vapply(preds, function(p)
    if (isTRUE(p$correct)) 1.0 else 0.0, numeric(1))

  edges   <- seq(0, 1, length.out = n_bins + 1L)
  n_total <- length(confs)
  ece     <- 0.0

  for (i in seq_len(n_bins)) {
    lo <- edges[i]; hi <- edges[i + 1L]
    mask <- if (i == 1L) (confs >= lo & confs <= hi) else (confs > lo & confs <= hi)
    n <- sum(mask)
    if (n > 0L)
      ece <- ece + (n / n_total) * abs(mean(corrs[mask]) - mean(confs[mask]))
  }
  ece
}

#' Stability = mean unique predicted labels per sample across all seeds
#' `all_seed_preds` is a named list of per-seed prediction lists.
compute_stability <- function(all_seed_preds) {
  sample_labels <- list()
  for (seed_preds in all_seed_preds) {
    for (p in seed_preds) {
      key <- as.character(p$id)
      sample_labels[[key]] <- c(sample_labels[[key]], p$predicted_label)
    }
  }
  if (length(sample_labels) == 0L) return(NA_real_)
  mean(vapply(sample_labels, function(lbls) length(unique(lbls)), numeric(1)))
}

#' Average Shannon entropy (bits) of per-sample label distributions across seeds
#' `all_seed_preds` is a named list of per-seed prediction lists.
compute_entropy <- function(all_seed_preds) {
  sample_labels <- list()
  for (seed_preds in all_seed_preds) {
    for (p in seed_preds) {
      key <- as.character(p$id)
      sample_labels[[key]] <- c(sample_labels[[key]], p$predicted_label)
    }
  }
  if (length(sample_labels) == 0L) return(NA_real_)
  entropies <- vapply(sample_labels, function(lbls) {
    counts <- as.numeric(table(lbls))   # all > 0 by construction
    probs  <- counts / sum(counts)
    -sum(probs * log2(probs))
  }, numeric(1))
  mean(entropies)
}

# =============================================================================
# Data Loading
# =============================================================================
load_all_data <- function() {
  cat("Loading data ...\n")
  rows_f1      <- list()
  rows_ece     <- list()
  rows_stab    <- list()
  rows_entropy <- list()
  missing      <- character(0)

  for (grp_name in names(MODELS)) {
    for (model in MODELS[[grp_name]]) {
      model_dir <- gsub("/", "_", model)

      for (dataset in DATASETS) {
        for (config in CONFIGS) {
          prep <- as.numeric(substr(config, 5L, 5L))  # "prep0" -> 0, "prep1" -> 1

          config_path <- file.path(EXPR_PATH, model_dir, dataset, config)
          if (!dir.exists(config_path)) {
            missing <- c(missing, paste(model_dir, dataset, config, sep = "/"))
            next
          }

          seed_dirs <- sort(list.dirs(config_path, recursive = FALSE, full.names = FALSE))
          seed_dirs <- seed_dirs[grepl("^seed_", seed_dirs)]

          all_seed_preds <- list()   # collect for stability + entropy

          for (seed_name in seed_dirs) {
            base_row <- list(Group = grp_name, Model = model, Dataset = dataset,
                             Preprocessing = prep, Config = config, Seed = seed_name)

            # ── F1 from metrics.json ─────────────────────────────────────
            mf <- file.path(EXPR_PATH, model_dir, dataset, config, seed_name, "metrics.json")
            if (file.exists(mf)) {
              m  <- fromJSON(mf)
              rows_f1 <- c(rows_f1, list(c(base_row, list(Value = m$macro_avg$`f1-score`))))
            } else {
              missing <- c(missing, mf)
            }

            # ── ECE from predictions.json (per seed) ─────────────────────
            pf <- file.path(EXPR_PATH, model_dir, dataset, config, seed_name, "predictions.json")
            if (file.exists(pf)) {
              preds <- fromJSON(pf, simplifyVector = FALSE)
              rows_ece <- c(rows_ece,
                list(c(base_row, list(Value = compute_ece(preds)))))
              all_seed_preds[[seed_name]] <- preds
            } else {
              missing <- c(missing, pf)
            }
          }

          # ── Stability + Entropy across all seeds (per config) ─────────
          if (length(all_seed_preds) > 0L) {
            config_row <- list(Group = grp_name, Model = model, Dataset = dataset,
                               Preprocessing = prep, Config = config)
            rows_stab <- c(rows_stab,
              list(c(config_row, list(Value = compute_stability(all_seed_preds)))))
            rows_entropy <- c(rows_entropy,
              list(c(config_row, list(Value = compute_entropy(all_seed_preds)))))
          }
        }
      }
    }
  }

  make_df <- function(rows) {
    df <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
    df$Dataset       <- as.factor(df$Dataset)
    df$Model         <- as.factor(df$Model)
    df$Model_Dataset <- as.factor(paste0(
      gsub("/", "_", df$Model), ":", df$Dataset))
    df$Preprocessing <- as.numeric(df$Preprocessing)
    df
  }

  list(
    F1        = make_df(rows_f1),
    ECE       = make_df(rows_ece),
    Stability = make_df(rows_stab),
    Entropy   = make_df(rows_entropy)
  )
}

# =============================================================================
# LMM Fitting
# =============================================================================
fit_lmm <- function(df, metric_name) {
  cat(sprintf("\nFitting LMM for %s (N = %d, REML) ...\n", metric_name, nrow(df)))

  fit <- lmerTest::lmer(
    Value ~ Preprocessing + (1 | Dataset) + (1 | Model) + (1 | Model:Dataset),
    data    = df,
    REML    = TRUE,
    control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  )

  msgs <- fit@optinfo$conv$lme4$messages
  if (!is.null(msgs) && length(msgs) > 0)
    cat("  [WARN]", paste(msgs, collapse = "; "), "\n")
  else
    cat("  Converged OK\n")

  cat("\n--- lme4 summary ---\n")
  print(summary(fit))
  fit
}

# =============================================================================
# Result extraction helpers
# =============================================================================
sig_stars <- function(p) {
  ifelse(p < 0.001, "***",
  ifelse(p < 0.01,  "**",
  ifelse(p < 0.05,  "*",
  ifelse(p < 0.10,  ".", "ns"))))
}

extract_fixed_effects <- function(fit) {
  ct  <- as.data.frame(coef(summary(fit)))
  colnames(ct) <- c("Estimate", "Std.Error", "df", "t.value", "p.value")
  z95 <- qnorm(0.975)

  rename <- c("(Intercept)"   = "Intercept",
              "Preprocessing" = "Preprocessing")
  effects <- ifelse(rownames(ct) %in% names(rename),
                    rename[rownames(ct)], rownames(ct))

  data.frame(
    Effect       = effects,
    Estimate     = round(ct$Estimate,  6),
    Std.Error    = round(ct$Std.Error, 6),
    df           = round(ct$df,         2),
    t.value      = round(ct$t.value,    4),
    p.value      = round(ct$p.value,    6),
    CI.Lower.95  = round(ct$Estimate - z95 * ct$Std.Error, 6),
    CI.Upper.95  = round(ct$Estimate + z95 * ct$Std.Error, 6),
    Significance = sig_stars(ct$p.value),
    row.names    = NULL,
    stringsAsFactors = FALSE
  )
}

extract_random_effects <- function(fit) {
  vc <- as.data.frame(VarCorr(fit))
  desc_map <- c(
    "Dataset"       = "Between-dataset variability in baseline metric value",
    "Model"         = "Between-model variability in baseline metric value",
    "Model:Dataset" = "Model-by-dataset interaction variability beyond model and dataset main effects",
    "Residual"      = "Within-cell (seed-to-seed) variability"
  )
  data.frame(
    Random.Effect = vc$grp,
    Variance      = round(vc$vcov,  8),
    Std.Dev.      = round(vc$sdcor, 8),
    Description   = ifelse(vc$grp %in% names(desc_map), desc_map[vc$grp], ""),
    row.names     = NULL,
    stringsAsFactors = FALSE
  )
}

extract_model_summary <- function(fit, df, metric_name) {
  msgs      <- fit@optinfo$conv$lme4$messages
  converged <- is.null(msgs) || length(msgs) == 0

  has_seed <- "Seed" %in% colnames(df)
  n_seeds  <- if (has_seed) length(unique(df$Seed)) else NA

  data.frame(
    Statistic = c(
      "Metric",
      "Formula",
      "Estimation method",
      "Optimizer",
      "P-value method",
      "Converged",
      "N observations",
      "N models",
      "N datasets",
      "N model x dataset",
      "N configs (Preprocessing levels)",
      if (has_seed) "N seeds" else NULL,
      "Log-likelihood (REML)",
      "AIC",
      "BIC",
      "Significance codes"
    ),
    Value = c(
      metric_name,
      "Metric ~ Preprocessing + (1|Dataset) + (1|Model) + (1|Model:Dataset)",
      "REML",
      "BOBYQA",
      "Satterthwaite degrees of freedom (lmerTest)",
      as.character(converged),
      as.character(nrow(df)),
      as.character(nlevels(df$Model)),
      as.character(nlevels(df$Dataset)),
      as.character(nlevels(df$Model_Dataset)),
      as.character(length(CONFIGS)),
      if (has_seed) as.character(n_seeds) else NULL,
      as.character(round(logLik(fit)[1], 4)),
      as.character(round(AIC(fit),       4)),
      as.character(round(BIC(fit),       4)),
      "*** p<0.001  ** p<0.01  * p<0.05  . p<0.10  ns p>=0.10"
    ),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Interpretation
# =============================================================================

# Direction guidance: is a positive effect of Preprocessing "good" or "bad"?
metric_direction <- list(
  F1        = list(good = "higher", positive_is = "beneficial",
                   unit = "macro-F1 score"),
  ECE       = list(good = "lower",  positive_is = "harmful (worse calibration)",
                   unit = "ECE (lower = better calibrated)"),
  Stability = list(good = "lower",  positive_is = "harmful (less stable predictions)",
                   unit = "avg. unique labels/sample (lower = more stable)"),
  Entropy   = list(good = "lower",  positive_is = "harmful (less consistent predictions)",
                   unit = "avg. entropy in bits/sample (lower = more consistent)")
)

build_fe_interpretation <- function(fe_df, metric_name) {
  emap <- setNames(
    lapply(seq_len(nrow(fe_df)), function(i) as.list(fe_df[i, ])),
    fe_df$Effect
  )
  dir_info <- metric_direction[[metric_name]]
  rows <- list()

  # Intercept
  r <- emap[["Intercept"]]
  rows[[1]] <- list(
    Effect         = "Intercept",
    Estimate       = r$Estimate,
    Significance   = r$Significance,
    Interpretation = sprintf(
      "The expected %s when Preprocessing = 0 (no preprocessing), averaged over models and datasets, is %.6f.",
      dir_info$unit, r$Estimate
    )
  )

  # Preprocessing
  r   <- emap[["Preprocessing"]]
  sig <- r$Significance
  dir <- if (r$Estimate > 0) "increases" else "decreases"
  good_bad <- if ((r$Estimate > 0 && dir_info$positive_is == "beneficial") ||
                  (r$Estimate < 0 && dir_info$positive_is != "beneficial"))
    "This is beneficial." else "This is detrimental."

  rows[[2]] <- list(
    Effect         = "Preprocessing",
    Estimate       = r$Estimate,
    Significance   = sig,
    Interpretation = paste0(
      sprintf(
        "Applying text preprocessing %s the %s by %.6f (t(%.1f) = %.3f, p = %.4f). ",
        dir, dir_info$unit, abs(r$Estimate), r$df, r$t.value, r$p.value
      ),
      if (sig %in% c("ns", "."))
        sprintf(
          "This effect is not statistically significant (p = %.4f), suggesting preprocessing has no reliable impact on %s.",
          r$p.value, dir_info$unit
        )
      else
        sprintf(
          "This effect is statistically significant (%s). %s",
          sig, good_bad
        )
    )
  )

  do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
}

build_re_interpretation <- function(re_df) {
  total_var <- sum(re_df$Variance)
  desc_map  <- c(
    "Model"         = "Reflects how much baseline metric value differs across models.",
    "Dataset"       = "Reflects systematic differences across datasets — some datasets yield inherently higher/lower metric values for all models.",
    "Model:Dataset" = "Reflects model-specific responses to datasets beyond what model and dataset main effects predict.",
    "Residual"      = "Within-cell variability across seeds (stochastic variation from random initialisation)."
  )
  pct <- round(100 * re_df$Variance / total_var, 2)
  data.frame(
    Random.Effect  = re_df$Random.Effect,
    Variance       = re_df$Variance,
    Std.Dev.       = re_df$Std.Dev.,
    Pct.of.Total   = pct,
    Interpretation = ifelse(
      re_df$Random.Effect %in% names(desc_map),
      paste0(sprintf("Accounts for %.1f%% of total variance. ", pct), desc_map[re_df$Random.Effect]),
      ""
    ),
    stringsAsFactors = FALSE
  )
}

build_descriptives <- function(df, metric_name) {
  cond_labels <- list(
    c(0, "No Preprocessing (prep0_aug0)"),
    c(1, "Preprocessing     (prep1_aug0)")
  )
  rows <- list()
  all_datasets <- c("ieee", "kaggle", "tweeteval", "ALL")

  for (ds in all_datasets) {
    sub <- if (ds == "ALL") df else df[as.character(df$Dataset) == ds, ]
    for (cond in cond_labels) {
      prep <- as.numeric(cond[1])
      cell <- sub[sub$Preprocessing == prep, "Value"]
      rows <- c(rows, list(data.frame(
        Dataset       = ds,
        Condition     = cond[2],
        Preprocessing = prep,
        N             = length(cell),
        Mean          = round(mean(cell, na.rm = TRUE), 6),
        SD            = round(sd(cell,   na.rm = TRUE), 6),
        Min           = round(min(cell,  na.rm = TRUE), 6),
        Max           = round(max(cell,  na.rm = TRUE), 6),
        stringsAsFactors = FALSE
      )))
    }
  }
  result <- do.call(rbind, rows)
  colnames(result)[5:8] <- paste0(c("Mean.", "SD.", "Min.", "Max."), metric_name)
  result
}

# =============================================================================
# Excel export (one file per metric)
# =============================================================================
save_metric_excel <- function(metric_name, fe_df, re_df, re_interp,
                               summary_df, fe_interp, desc_df) {
  filename <- sprintf("lmm_results_%s.xlsx", metric_name)
  out_file <- file.path(OUTPUT_PATH, filename)
  write_xlsx(
    list(
      Fixed_Effects     = fe_df,
      Random_Effects    = re_df,
      RE_Interpretation = re_interp,
      Model_Summary     = summary_df,
      FE_Interpretation = fe_interp,
      Descriptive_Stats = desc_df
    ),
    path = out_file
  )
  cat(sprintf("  Saved: %s\n", out_file))
}

# =============================================================================
# Run pipeline for one metric
# =============================================================================
run_metric <- function(df, metric_name) {
  cat(sprintf("\n%s\n", strrep("=", 60)))
  cat(sprintf(" Metric: %s\n", metric_name))
  cat(sprintf("%s\n", strrep("=", 60)))

  fit        <- fit_lmm(df, metric_name)
  fe_df      <- extract_fixed_effects(fit)
  re_df      <- extract_random_effects(fit)
  re_interp  <- build_re_interpretation(re_df)
  summary_df <- extract_model_summary(fit, df, metric_name)
  fe_interp  <- build_fe_interpretation(fe_df, metric_name)
  desc_df    <- build_descriptives(df, metric_name)

  cat(sprintf("\n--- Fixed Effects (%s) ---\n", metric_name))
  print(fe_df[, c("Effect", "Estimate", "Std.Error", "df", "t.value", "p.value", "Significance")])

  cat(sprintf("\n--- Random Effects (%s) ---\n", metric_name))
  print(re_df[, c("Random.Effect", "Variance", "Std.Dev.")])

  save_metric_excel(metric_name, fe_df, re_df, re_interp, summary_df, fe_interp, desc_df)
}

# =============================================================================
# Main
# =============================================================================
cat(strrep("=", 60), "\n")
cat(" LMM Analysis — Effect of Preprocessing\n")
cat(strrep("=", 60), "\n\n")

data_list <- load_all_data()

cat(sprintf("\nLoaded F1        : %d observations\n", nrow(data_list$F1)))
cat(sprintf("Loaded ECE       : %d observations\n",  nrow(data_list$ECE)))
cat(sprintf("Loaded Stability : %d observations\n",  nrow(data_list$Stability)))
cat(sprintf("Loaded Entropy   : %d observations\n",  nrow(data_list$Entropy)))

run_metric(data_list$F1,        "F1")
run_metric(data_list$ECE,       "ECE")
run_metric(data_list$Stability, "Stability")
run_metric(data_list$Entropy,   "Entropy")

cat("\n", strrep("=", 60), "\n")
cat(" Done. All results saved to:", OUTPUT_PATH, "\n")
cat(strrep("=", 60), "\n")