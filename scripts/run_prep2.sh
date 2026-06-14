#!/usr/bin/env bash
#
# run_prep2.sh — Partial-preprocessing (prep2) experiment suite
#
# Runs the same models x datasets x seeds as run_all.sh, but with
# --prep_mode 2 (transformer-friendly preprocessing: removes noise such as
# URLs, mentions, and emojis, but preserves morphological features —
# no stopword removal, no punctuation removal, no lemmatization, no
# number removal, and no short-word filtering).
#
# Augmentation is held at 0 so that results are directly comparable to the
# prep0_aug0 and prep1_aug0 configs produced by run_all.sh.
#
# Output path pattern (same experiment tree as run_all.sh):
#   experiments/grid-search/{model}/{dataset}/prep2_aug0/seed_{N}/
#
# Statistical analysis after runs:
#   - Wilcoxon test  : set CONTROL="prep0_aug0" TREATMENT="prep2_aug0"
#                      in notebooks/wilcoxon_test.py, then run it.
#   - Kruskal test   : regenerate analysis/grid-search/recap/recap_all.xlsx
#                      to include prep2 rows, then run notebooks/compute_kruskal.py.
#
# Total runs: 20 models x 3 datasets x 10 seeds = 600
#
# Usage:
#   chmod +x scripts/run_prep2.sh
#   bash scripts/run_prep2.sh
#

set -euo pipefail

# --- Configuration ---
# Set OVERWRITE=1 to rerun all scenarios and overwrite existing results.
# Set OVERWRITE=0 (default) to skip scenarios that already have metrics.json
# and predictions.json (bash-level check) and let main.py skip them too
# (python-level check via --overwrite flag).
OVERWRITE=0

SEEDS=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )

MODELS=(
    "answerdotai/ModernBERT-base"
    "google/mobilebert-uncased"
    "chandar-lab/NeoBERT"
    "all-distilroberta-v1"
    "roberta-base"
    "Twitter/twhin-bert-base"
    "albert/albert-base-v2"
    "bert-base-cased"
    "xlnet-base-cased"
    "bert-base-uncased"
    "distilbert-base-cased"
    "distilbert-base-uncased"
    "gpt2"
    "vinai/bertweet-base"
    "sarkerlab/SocBERT-base"
    "GroNLP/hateBERT"
    "all-mpnet-base-v2"
    "all-MiniLM-L6-v2"
    "all-MiniLM-L12-v2"
)

DATASETS=(
    "ieee"
    "tweeteval"
    "kaggle"
)

# Training hyperparameters — identical to run_all.sh for fair comparison
EPOCHS=10
BATCH_SIZE=32
LR=5e-5
MAX_LENGTH=128

# Directories
OUTPUT_DIR="grid-search"
DATA_DIR="dataset"

# --- Run experiments ---
total=0
completed=0
failed=0

# Count total runs
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            total=$((total + 1))
        done
    done
done

echo "============================================================"
echo "Cyberbullying Detection — Partial Preprocessing (prep2) Suite"
echo "============================================================"
echo "Preprocessing : --prep_mode 2 (transformer-friendly, aug0 only)"
echo "Total runs    : ${total}"
echo "Models        : ${MODELS[*]}"
echo "Datasets      : ${DATASETS[*]}"
echo "Seeds         : ${SEEDS[*]}"
echo "============================================================"
echo ""

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_id=$((completed + failed + 1))

            overwrite_flag=""
            if [ "${OVERWRITE}" -eq 1 ]; then
                overwrite_flag="--overwrite"
            fi

            model_short="${model//\//_}"
            OUTDIR="experiments/${OUTPUT_DIR}/${model_short}/${dataset}/prep2_aug0/seed_${seed}"

            # Bash-level skip: only active when OVERWRITE=0
            if [ "${OVERWRITE}" -eq 0 ] && [ -f "${OUTDIR}/metrics.json" ] && [ -f "${OUTDIR}/predictions.json" ]; then
                echo "[${run_id}/${total}] SKIP (complete): ${model} / ${dataset} / prep2_aug0 / seed_${seed}"
                completed=$((completed + 1))
                continue
            fi

            echo "------------------------------------------------------------"
            echo "[${run_id}/${total}] model=${model} dataset=${dataset} prep=2 aug=0 seed=${seed}"
            echo "------------------------------------------------------------"

            if python src/main.py \
                --model "${model}" \
                --dataset "${dataset}" \
                --prep_mode 2 \
                ${overwrite_flag} \
                --seed "${seed}" \
                --epochs "${EPOCHS}" \
                --batch_size "${BATCH_SIZE}" \
                --lr "${LR}" \
                --max_length "${MAX_LENGTH}" \
                --output_dir "${OUTPUT_DIR}" \
                --data_dir "${DATA_DIR}"; then
                completed=$((completed + 1))
                echo "[${run_id}/${total}] COMPLETED"
            else
                failed=$((failed + 1))
                echo "[${run_id}/${total}] FAILED"
            fi

            echo ""
        done
    done
done

echo "============================================================"
echo "Partial preprocessing suite finished."
echo "  Completed : ${completed}"
echo "  Failed    : ${failed}"
echo "  Total     : ${total}"
echo "============================================================"
echo ""
echo "Next steps for statistical analysis:"
echo "  1. Wilcoxon test (prep2 vs prep0):"
echo "       Edit notebooks/wilcoxon_test.py:"
echo "         CONTROL   = \"prep0_aug0\""
echo "         TREATMENT = \"prep2_aug0\""
echo "       Then: python notebooks/wilcoxon_test.py"
echo ""
echo "  2. Wilcoxon test (prep2 vs prep1):"
echo "       Edit notebooks/wilcoxon_test.py:"
echo "         CONTROL   = \"prep1_aug0\""
echo "         TREATMENT = \"prep2_aug0\""
echo "       Then: python notebooks/wilcoxon_test.py"
echo ""
echo "  3. Kruskal-Wallis test:"
echo "       Regenerate analysis/grid-search/recap/recap_all.xlsx to include"
echo "       prep2_aug0 rows, then run: python notebooks/compute_kruskal.py"
echo "============================================================"
