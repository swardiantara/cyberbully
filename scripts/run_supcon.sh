#!/usr/bin/env bash
#
# run_supcon.sh — SupCon auxiliary loss experiments
#
# Runs two scenarios back-to-back over the same models/datasets/seeds:
#
#   baseline  — standard CE loss only, no projection head
#               output: experiments/supcon-baseline/...
#
#   supcon    — CE + SupCon auxiliary loss (Khosla et al., 2020)
#               projection head trained alongside the classifier head
#               output: experiments/supcon-grid/...
#
# Toggle RUN_BASELINE / RUN_SUPCON below to run only one scenario.
#
# SBERT models are intentionally excluded — SupConClassifier requires a
# standard AutoModel backbone (not SentenceTransformer).
#
# Loops: N_models × N_datasets × prep(0/1) × aug(0/1) × 10 seeds
#
# Usage:
#   chmod +x scripts/run_supcon.sh
#   bash scripts/run_supcon.sh
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Toggle scenarios
# ---------------------------------------------------------------------------
RUN_BASELINE=0   # set to 0 to skip the CE-only baseline
RUN_SUPCON=1     # set to 0 to skip the SupCon scenario

# ---------------------------------------------------------------------------
# Models (non-SBERT standard transformers only)
# ---------------------------------------------------------------------------
MODELS=(
    "roberta-base"
    # "bert-base-uncased"
    # "distilbert-base-uncased"
    # "google/mobilebert-uncased"
    # "vinai/bertweet-base"
)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS=(
    "ieee"
    "kaggle"
    "tweeteval"
    # Planned — uncomment as datasets become available:
    # "youtube"
    # "reddit"
    # "wikipedia"
    # "instagram"
)

# ---------------------------------------------------------------------------
# Seeds (shared with run_all.sh for cross-script comparability)
# ---------------------------------------------------------------------------
SEEDS=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )

# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------
EPOCHS=10
LR=5e-5
MAX_LENGTH=128
DATA_DIR="dataset"

# Baseline: no contrastive loss, standard batch
BASELINE_BATCH_SIZE=32
BASELINE_GRAD_ACCUM=1
BASELINE_OUTPUT_DIR="supcon-baseline"

# SupCon: larger effective batch for pair diversity (batch × accum = 256)
SUPCON_BATCH_SIZE=32
SUPCON_GRAD_ACCUM=1
SUPCON_OUTPUT_DIR="supcon-grid"
SUPCON_WEIGHT=0.1
PROJ_DIM=128

# ---------------------------------------------------------------------------
# Count total runs per scenario
# ---------------------------------------------------------------------------
runs_per_scenario=0
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for preprocess in 0 1; do
            for augment in 0 1; do
                for seed in "${SEEDS[@]}"; do
                    runs_per_scenario=$((runs_per_scenario + 1))
                done
            done
        done
    done
done

total_scenarios=0
[ "${RUN_BASELINE}" -eq 1 ] && total_scenarios=$((total_scenarios + 1))
[ "${RUN_SUPCON}"   -eq 1 ] && total_scenarios=$((total_scenarios + 1))
total=$((runs_per_scenario * total_scenarios))

echo "============================================================"
echo "Cyberbullying Detection — SupCon Experiment Suite"
echo "============================================================"
echo "Models:            ${MODELS[*]}"
echo "Datasets:          ${DATASETS[*]}"
echo "Runs per scenario: ${runs_per_scenario}"
echo "Active scenarios:  ${total_scenarios}  (baseline=${RUN_BASELINE}, supcon=${RUN_SUPCON})"
echo "Total runs:        ${total}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper: run one scenario loop
#
# Arguments (positional):
#   $1  scenario label (for display)
#   $2  OUTPUT_DIR
#   $3  BATCH_SIZE
#   $4  GRAD_ACCUM_STEPS
#   $5  extra flags to append verbatim (e.g. "--supcon --supcon_weight 0.1 --proj_dim 128")
# ---------------------------------------------------------------------------
run_scenario() {
    local label="$1"
    local output_dir="$2"
    local batch_size="$3"
    local grad_accum="$4"
    local extra_flags="$5"

    local completed=0
    local failed=0
    local run_id=0

    echo "============================================================"
    echo "SCENARIO: ${label}"
    echo "  output_dir=${output_dir}"
    echo "  batch_size=${batch_size}  grad_accum=${grad_accum}"
    echo "  effective_batch=$((batch_size * grad_accum))"
    [ -n "${extra_flags}" ] && echo "  extra: ${extra_flags}"
    echo "============================================================"
    echo ""

    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for preprocess in 0; do
                for augment in 0; do
                    for seed in "${SEEDS[@]}"; do
                        run_id=$((run_id + 1))

                        prep_flag=""
                        [ "${preprocess}" -eq 1 ] && prep_flag="--preprocess"

                        aug_flag=""
                        [ "${augment}" -eq 1 ] && aug_flag="--augment"

                        # Skip if already completed
                        model_short="${model//\//_}"
                        OUTDIR="experiments/${output_dir}/${model_short}/${dataset}/prep${preprocess}_aug${augment}/seed_${seed}"
                        if [ -f "${OUTDIR}/metrics.json" ]; then
                            echo "[${label}][${run_id}/${runs_per_scenario}] SKIP (complete): ${model} / ${dataset} / prep${preprocess}_aug${augment} / seed_${seed}"
                            completed=$((completed + 1))
                            continue
                        fi

                        echo "------------------------------------------------------------"
                        echo "[${label}][${run_id}/${runs_per_scenario}] model=${model} dataset=${dataset} prep=${preprocess} aug=${augment} seed=${seed}"
                        echo "------------------------------------------------------------"

                        # shellcheck disable=SC2086
                        if python src/main.py \
                            --model "${model}" \
                            --dataset "${dataset}" \
                            ${prep_flag} \
                            ${aug_flag} \
                            --seed "${seed}" \
                            --epochs "${EPOCHS}" \
                            --batch_size "${batch_size}" \
                            --grad_accum_steps "${grad_accum}" \
                            --lr "${LR}" \
                            --max_length "${MAX_LENGTH}" \
                            --output_dir "${output_dir}" \
                            --data_dir "${DATA_DIR}" \
                            --preprocess \
                            ${extra_flags}; then
                            completed=$((completed + 1))
                            echo "[${label}][${run_id}/${runs_per_scenario}] COMPLETED"
                        else
                            failed=$((failed + 1))
                            echo "[${label}][${run_id}/${runs_per_scenario}] FAILED"
                        fi

                        echo ""
                    done
                done
            done
        done
    done

    echo "============================================================"
    echo "SCENARIO ${label} finished."
    echo "  Completed: ${completed}"
    echo "  Failed:    ${failed}"
    echo "  Total:     ${runs_per_scenario}"
    echo "============================================================"
    echo ""
}

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

if [ "${RUN_BASELINE}" -eq 1 ]; then
    run_scenario \
        "baseline" \
        "${BASELINE_OUTPUT_DIR}" \
        "${BASELINE_BATCH_SIZE}" \
        "${BASELINE_GRAD_ACCUM}" \
        ""
fi

if [ "${RUN_SUPCON}" -eq 1 ]; then
    run_scenario \
        "supcon" \
        "${SUPCON_OUTPUT_DIR}" \
        "${SUPCON_BATCH_SIZE}" \
        "${SUPCON_GRAD_ACCUM}" \
        "--supcon --supcon_weight ${SUPCON_WEIGHT} --proj_dim ${PROJ_DIM}"
fi

echo "All requested scenarios complete."
