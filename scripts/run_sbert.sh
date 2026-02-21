#!/usr/bin/env bash
#
# run_sbert.sh — SBERT experiment suite (base + contrastive fine-tuned models)
#
# Valid model-dataset pairs:
#   - 3 base SBERT models × 3 datasets          =  9 pairs
#   - 9 custom fine-tuned models × 1 dataset each =  9 pairs  (enforced below)
#   - Total pairs: 18 × 4 scenarios × 10 seeds  = 720 runs
#
# Usage:
#   chmod +x scripts/run_sbert.sh
#   bash scripts/run_sbert.sh
#

set -euo pipefail

# --- Configuration ---
SEEDS=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )

MODELS=(
    # "all-MiniLM-L6-v2"
    # "all-MiniLM-L12-v2"
    # "all-mpnet-base-v2"
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

# --- Helper: return the matched dataset for a custom model, or empty string ---
get_custom_dataset() {
    local model="$1"
    if [[ "${model}" == swardiantara/* ]]; then
        local repo="${model#swardiantara/}"
        for ds in "ieee" "kaggle" "tweeteval"; do
            if [[ "${repo}" == ${ds}-* ]]; then
                echo "${ds}"
                return
            fi
        done
    fi
    echo ""
}

# --- Count valid runs ---
total=0
completed=0
failed=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        custom_ds=$(get_custom_dataset "${model}")
        if [[ -n "${custom_ds}" && "${custom_ds}" != "${dataset}" ]]; then
            continue  # Custom model must match its fine-tuning dataset
        fi
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
echo "Cyberbullying Detection — SBERT Experiment Suite"
echo "============================================================"
echo "Total experiment runs: ${total}"
echo "  (18 model-dataset pairs × 4 scenarios × 10 seeds)"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "============================================================"
echo ""

# --- Run experiments ---
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        custom_ds=$(get_custom_dataset "${model}")
        if [[ -n "${custom_ds}" && "${custom_ds}" != "${dataset}" ]]; then
            continue  # Skip: custom model does not match this dataset
        fi
        for preprocess in 0 1; do
            for augment in 0 1; do
                for seed in "${SEEDS[@]}"; do
                    run_id=$((completed + failed + 1))

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
