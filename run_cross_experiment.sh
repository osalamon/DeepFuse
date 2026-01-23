#!/bin/bash

# Get a unique timestamp for this entire run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting Cross-Validation Pipeline at $TIMESTAMP"

# ---------------------------------------------------------
# STEP 1: Train on Sequence 01
# ---------------------------------------------------------
echo "---------------------------------------------------"
echo "Step 1: Training on Batch 01"
echo "---------------------------------------------------"
python train_DeepFuse.py --seq 01 --timestamp "$TIMESTAMP"

# Capture the model path (assuming the naming convention in the python script)
# We know the folder is trained_on_01_$TIMESTAMP
MODEL_01_PATH="./trained_on_01_${TIMESTAMP}/model_5x5_4e-04_100_16_trained_on_01.h5"


# ---------------------------------------------------------
# STEP 2: Train on Sequence 02
# ---------------------------------------------------------
echo "---------------------------------------------------"
echo "Step 2: Training on Batch 02"
echo "---------------------------------------------------"
python train_DeepFuse.py --seq 02 --timestamp "$TIMESTAMP"

# Capture the model path
MODEL_02_PATH="./trained_on_02_${TIMESTAMP}/model_5x5_4e-04_100_16_trained_on_02.h5"


# ---------------------------------------------------------
# STEP 3: Evaluate Model 01 on Data 02
# ---------------------------------------------------------
echo "---------------------------------------------------"
echo "Step 3: Evaluating Model 01 (Trained on 01, Testing on 02)"
echo "---------------------------------------------------"

# Note: Using 'tee' to print to screen AND save to file
# I assume your evaluate script accepts arguments. If it doesn't, 
# you need to modify evaluate_DeepFuse.py similar to train_DeepFuse.py
# Or let me know and I can help modify it.

# Hypothetical command - adjust arguments as needed for your specific evaluate script
python evaluate_DeepFuse.py \
    --model "$MODEL_01_PATH" \
    --seq 02 \
    | tee "trained_on_01_evaluated_for_02.txt"


# ---------------------------------------------------------
# STEP 4: Evaluate Model 02 on Data 01
# ---------------------------------------------------------
echo "---------------------------------------------------"
echo "Step 4: Evaluating Model 02 (Trained on 02, Testing on 01)"
echo "---------------------------------------------------"

python evaluate_DeepFuse.py \
    --model "$MODEL_02_PATH" \
    --seq 01 \
    | tee "trained_on_02_evaluated_for_01.txt"

echo "Pipeline finished."