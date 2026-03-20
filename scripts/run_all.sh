#!/usr/bin/env bash
#
# run_all.sh — Baseline (CE-only) experiment suite
#
# Loops over: models × datasets × preprocess (on/off) × augment (on/off) × seeds
# Total: 14 models × 3 datasets × 2 × 2 × 10 seeds = 1680 runs
#
# No SupCon — use run_supcon.sh for the SupCon auxiliary-loss experiments.
#
# Usage:
#   chmod +x scripts/run_all.sh
#   bash scripts/run_all.sh
#

set -euo pipefail

# --- Configuration ---
SEEDS=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )

MODELS=(
    "bert-base-cased"           # cased variant — investigate case sensitivity
    "distilbert-base-cased"     # cased variant — investigate case sensitivity
    "roberta-base"
    "vinai/bertweet-base"
    "google/mobilebert-uncased"
    "bert-base-uncased"
    "Twitter/twhin-bert-base"
    "all-MiniLM-L6-v2"
    "xlnet-base-cased"
    "sarkerlab/SocBERT-base"
    "distilbert-base-uncased"
    "gpt2"
    "all-MiniLM-L12-v2"
    "all-mpnet-base-v2"
)

DATASETS=(
    "ieee"
    "kaggle"
    "tweeteval"
)

# Training hyperparameters
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
        for preprocess in 0 1; do
            for augment in 0 1; do
                for seed in "${SEEDS[@]}"; do
                    total=$((total + 1))
                done
            done
        done
    done
done

echo "============================================================"
echo "Cyberbullying Detection — Full Experiment Suite"
echo "============================================================"
echo "Total experiment runs: ${total}"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "============================================================"
echo ""

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for preprocess in 0 1; do
            for augment in 0 1; do
                for seed in "${SEEDS[@]}"; do
                    run_id=$((completed + failed + 1))

                    # Build flag arguments
                    prep_flag=""
                    if [ "${preprocess}" -eq 1 ]; then
                        prep_flag="--preprocess"
                    fi

                    aug_flag=""
                    if [ "${augment}" -eq 1 ]; then
                        aug_flag="--augment"
                    fi

                    # Skip if already completed
                    model_short="${model//\//_}"
                    OUTDIR="experiments/${OUTPUT_DIR}/${model_short}/${dataset}/prep${preprocess}_aug${augment}/seed_${seed}"
                    if [ -f "${OUTDIR}/metrics.json" ]; then
                        echo "[${run_id}/${total}] SKIP (complete): ${model} / ${dataset} / prep${preprocess}_aug${augment} / seed_${seed}"
                        continue
                    fi

                    echo "------------------------------------------------------------"
                    echo "[${run_id}/${total}] model=${model} dataset=${dataset} prep=${preprocess} aug=${augment} seed=${seed}"
                    echo "------------------------------------------------------------"

                    if python src/main.py \
                        --model "${model}" \
                        --dataset "${dataset}" \
                        ${prep_flag} \
                        ${aug_flag} \
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
    done
done

echo "============================================================"
echo "Experiment suite finished."
echo "  Completed: ${completed}"
echo "  Failed:    ${failed}"
echo "  Total:     ${total}"
echo "============================================================"
