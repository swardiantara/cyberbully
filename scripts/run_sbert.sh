#!/usr/bin/env bash
#
# run_all.sh — Automate all experiment configurations
#
# Loops over: models × datasets × preprocess (on/off) × augment (on/off) × seeds
# Total: 5 models × 3 datasets × 2 × 2 × 10 seeds = 600 runs
#
# Usage:
#   chmod +x scripts/run_all.sh
#   bash scripts/run_all.sh
#

set -euo pipefail

# --- Configuration ---
SEEDS=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )

MODELS=(
    "all-MiniLM-L6-v2"
    "all-MiniLM-L12-v2"
    "all-mpnet-base-v2"
    "swardiantara/ieee-all-MiniLM-L6-v2"
    "swardiantara/kaggle-all-MiniLM-L6-v2"
    "swardiantara/tweeteval-all-MiniLM-L6-v2"
    "swardiantara/ieee-all-MiniLM-L12-v2"
    "swardiantara/kaggle-all-MiniLM-L12-v2"
    "swardiantara/tweeteval-all-MiniLM-L12-v2"
    "swardiantara/ieee-all-mpnet-base-v2"
    "swardiantara/kaggle-all-mpnet-base-v2"
    "swardiantara/tweeteval-all-mpnet-base-v2"
)

DATASETS=(
    "ieee"
    "kaggle"
    "tweeteval"
)

# Training hyperparameters
EPOCHS=5
BATCH_SIZE=16
LR=2e-5
MAX_LENGTH=128

# Directories
OUTPUT_DIR="sbert-model"
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

                    echo "------------------------------------------------------------"
                    echo "[${run_id}/${total}] model=${model} dataset=${dataset} prep=${preprocess} aug=${augment} seed=${seed}"
                    echo "------------------------------------------------------------"

                    if python src/main.py \
                        --model "${model}" \
                        --dataset "${dataset}" \
                        --sbert \
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
