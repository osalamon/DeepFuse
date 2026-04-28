#!/bin/bash

# Get a unique timestamp for this entire run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting 4-Fold Cross-Validation Pipeline at $TIMESTAMP"
echo "============================================================"

# Array of experiments: dataset_name fold_num
experiments=(
    "BF-C2DL-HSC 1"
    "BF-C2DL-HSC 2"
    "BF-C2DL-MuSC 1"
    "BF-C2DL-MuSC 2"
)

results_file="results_${TIMESTAMP}.txt"
echo "Results will be saved to: ${results_file}"
echo ""

# ---------------------------------------------------------
# Training Phase
# ---------------------------------------------------------
echo "============================================================"
echo "PHASE 1: TRAINING ALL 4 MODELS"
echo "============================================================"

for exp in "${experiments[@]}"; do
    dataset=$(echo $exp | cut -d' ' -f1)
    fold=$(echo $exp | cut -d' ' -f2)
    
    echo ""
    echo "---------------------------------------------------"
    echo "Training ${dataset} Fold ${fold}"
    echo "---------------------------------------------------"
    
    python train_DeepFuse.py \
        --dataset "${dataset}" \
        --fold "${fold}" \
        --timestamp "${TIMESTAMP}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for ${dataset} fold ${fold}"
        exit 1
    fi
done

# ---------------------------------------------------------
# Evaluation Phase
# ---------------------------------------------------------
echo ""
echo "============================================================"
echo "PHASE 2: EVALUATING ALL 4 MODELS ON TEST SETS"
echo "============================================================"

# Clear results file
echo "DeepFuse Cross-Validation Results - ${TIMESTAMP}" > "${results_file}"
echo "================================================" >> "${results_file}"
echo "" >> "${results_file}"

for exp in "${experiments[@]}"; do
    dataset=$(echo $exp | cut -d' ' -f1)
    fold=$(echo $exp | cut -d' ' -f2)
    
    model_dir="trained_on_${dataset}_fold${fold}_${TIMESTAMP}"
    model_path="${model_dir}/model_*.h5"
    
    # Find the actual model file
    model_file=$(ls ${model_path} 2>/dev/null | head -n 1)
    
    if [ -z "${model_file}" ]; then
        echo "ERROR: No model file found in ${model_dir}"
        continue
    fi
    
    echo ""
    echo "---------------------------------------------------"
    echo "Evaluating ${dataset} Fold ${fold}"
    echo "Model: ${model_file}"
    echo "---------------------------------------------------"
    
    # Run evaluation and capture output
    eval_output=$(python evaluate_DeepFuse.py \
        --dataset "${dataset}" \
        --fold "${fold}" \
        --model "${model_file}" 2>&1)
    
    echo "${eval_output}"
    
    # Extract F1 and Jaccard scores from output
    f1_score=$(echo "${eval_output}" | grep "TEST" | awk '{print $4}')
    jaccard_score=$(echo "${eval_output}" | grep "TEST" | awk '{print $6}')
    
    # Save to results file
    echo "${dataset} Fold ${fold}:" >> "${results_file}"
    echo "  F1 (Dice): ${f1_score}" >> "${results_file}"
    echo "  Jaccard:   ${jaccard_score}" >> "${results_file}"
    echo "" >> "${results_file}"
done

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Results saved to: ${results_file}"
echo ""
cat "${results_file}"
