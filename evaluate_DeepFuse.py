import os
import glob
import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from create_mask_images import create_multi_input_data_from_parquet
from tensorflow.keras.layers import Input, concatenate, Conv2D, add


# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Evaluate DeepFuse on specific dataset fold')
parser.add_argument('--dataset', type=str, required=True, choices=['BF-C2DL-HSC', 'BF-C2DL-MuSC'], help='Dataset name')
parser.add_argument('--fold', type=int, required=True, choices=[1, 2], help='Cross-validation fold (1 or 2)')
parser.add_argument('--model', type=str, required=True, help='Path to .h5 model file to evaluate')
args = parser.parse_args()

dataset_name = args.dataset
fold_num = args.fold
model_path = args.model


print(f"--- EVALUATING CONFIGURATION ---")
print(f"Dataset: {dataset_name}")
print(f"Fold: {fold_num}")
print(f"Evaluating Model: {model_path}")
# ------------------------


# --- Configuration ---
data_root = Path("data/synchronized_data") / dataset_name
parquet_path = Path("data/dataframes") / dataset_name / "whole_image" / f"{dataset_name}_split_fold-{fold_num}.parquet"

# Competitor input paths (5 raters)
# Fold 1: Train/Val on seq 01, Test on seq 02
# Fold 2: Train/Val on seq 02, Test on seq 01
if fold_num == 1:
    # Fold 1 tests on sequence 02
    input_paths = [
        str(data_root / f"CALT-US/02_RES/"),
        str(data_root / f"DREX-US/02_RES/"),
        str(data_root / f"KIT-Sch-GE/02_RES/"),
        str(data_root / f"KTH-SE (5)/02_RES/"),
        str(data_root / f"MU-Lux-CZ/02_RES/"),
    ]
    gt_path = str(data_root / "02_GT/SEG/")
else:
    # Fold 2 tests on sequence 01
    input_paths = [
        str(data_root / f"CALT-US/01_RES/"),
        str(data_root / f"DREX-US/01_RES/"),
        str(data_root / f"KIT-Sch-GE/01_RES/"),
        str(data_root / f"KTH-SE (5)/01_RES/"),
        str(data_root / f"MU-Lux-CZ/01_RES/"),
    ]
    gt_path = str(data_root / "01_GT/SEG/")

# Determine output directory from model path so log is saved there
output_dir = os.path.dirname(os.path.abspath(model_path)) 

# Redirect print output to both terminal and a file in the output_dir
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_filename = os.path.join(output_dir, f"evaluation_results_{dataset_name}_fold{fold_num}.txt")
sys.stdout = DualLogger(log_filename)
print(f"Saving evaluation log to: {log_filename}")

im_len = 101
im_wid = 101
num_of_filters = 16
batch_size = 1 # Use batch size 1 for safer inference memory-wise

# --- Metrics ---
smooth = 1e-15

def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def jaccard_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# --- Model Builder (Must match training structure exactly) ---
def build_model():
    K.set_image_data_format('channels_last')  # TF dimension ordering in this code
    learning_rate = 4*1e-4
    smooth = 1e-16
    num_of_filters = 16
    im_len = 101
    im_wid = 101

    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)

    # define input image size for each input
    input1 = Input(shape=(im_len, im_wid, 1))
    input2 = Input(shape=(im_len, im_wid, 1))
    input3 = Input(shape=(im_len, im_wid, 1))
    input4 = Input(shape=(im_len, im_wid, 1))
    input5 = Input(shape=(im_len, im_wid, 1))
    # the first branch operates on the first input
    x1 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input1)
    x1 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x1)
    x1 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x1)
    x1 = Model(inputs=input1, outputs=x1)
    # the second branch operates on the second input
    x2 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input2)
    x2 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x2)
    x2 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x2)
    x2 = Model(inputs=input2, outputs=x2)
    # 
    x3 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input3)
    x3 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x3)
    x3 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x3)
    x3 = Model(inputs=input3, outputs=x3)
    # 
    x4 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input4)
    x4 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x4)
    x4 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x4)
    x4 = Model(inputs=input4, outputs=x4)
    # 
    x5 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input5)
    x5 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x5)
    x5 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x5)
    x5 = Model(inputs=input5, outputs=x5)
    # combine the output of all branches
    combined = concatenate([x1.output, x2.output, x3.output, x4.output, x5.output])

    # feed the combined output to a non-linear activation function
    output = Conv2D(1, (1,1), activation='sigmoid')(combined)

    model = Model(inputs=[x1.input, x2.input, x3.input, x4.input, x5.input], outputs=[output])

    model.compile(loss=dice_coef_loss, optimizer=Adam(lr=learning_rate), metrics=[dice_coef])
    model.summary()
    return model

# --- Main Evaluation Loop ---
def main():
    print("--- Loading Test Data ---")

    # Load test split from parquet
    test_inputs, test_gts = create_multi_input_data_from_parquet(
        str(parquet_path), 'test', input_paths, gt_path
    )
    
    total_samples = test_gts.shape[0]
    print(f"Test samples loaded: {total_samples}")

    # Find model files
    model_files = glob.glob(os.path.join(model_path), recursive=True)
    if not model_files:
        print("No .h5 files found!")
        return

    print(f"\nFound {len(model_files)} models to evaluate.")
    
    # Build model architecture once (weights will be reloaded)
    model = build_model()

    print(f"\n{'Model Path':<80} | {'Set':<10} | {'F1 (Dice)':<10} | {'Jaccard':<10}")
    print("-" * 120)

    for m_path in model_files:
        try:
            # Load weights
            model.load_weights(m_path)
            print(f"Evaluating Model: {m_path}")
            
            # --- Evaluate on Test Set ---
            preds_test = model.predict(test_inputs, batch_size=batch_size, verbose=0)
            # Threshold predictions (binary classification)
            preds_test_bin = (preds_test > 0.5).astype(np.float32)
            
            f1_test = dice_coef_np(test_gts, preds_test_bin)
            jac_test = jaccard_coef_np(test_gts, preds_test_bin)
            
            print(f"{os.path.relpath(m_path, output_dir):<80} | {'TEST':<10} | {f1_test:.4f}     | {jac_test:.4f}")
            print("-" * 120)

        except Exception as e:
            print(f"Error evaluating {m_path}: {e}")

if __name__ == "__main__":
    main()
