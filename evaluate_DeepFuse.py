import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from create_mask_images import create_train_data, create_gt_data

# --- Configuration ---
# Hardcoded paths from the training script
input_paths = [
    '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/CALT-US/01_RES/',
    '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/DREX-US/01_RES/',
    '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/KIT-Sch-GE/01_RES/',
    '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/KTH-SE (5)/01_RES/',
    '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/MU-Lux-CZ/01_RES/'
]
gt_path = '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/01_GT/SEG/'

output_dir = "./output" # Point this to your output folder
im_len = 1010
im_wid = 1010
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
    input_layers = []
    processed_branches = []
    
    for _ in range(5):
        inp = Input(shape=(im_len, im_wid, 1))
        input_layers.append(inp)
        
        x = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(inp)
        x = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x)
        x = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x)
        # Note: In training we created intermediate Models, but logically we just need the tensor flow
        processed_branches.append(x)

    merged = add(processed_branches)
    output = Conv2D(1, (1,1), activation='sigmoid')(merged)
    
    model = Model(inputs=input_layers, outputs=[output])
    return model

# --- Main Evaluation Loop ---
def main():
    print("--- Loading Data ---")
    X_data = []
    for p in input_paths:
        print(f"Loading from: {p}")
        data = create_train_data(p, gt_path)
        X_data.append(data)
    
    print(f"Loading GT from: {gt_path}")
    Y_data = create_gt_data(gt_path)
    
    # Calculate split point for validation (last 20%)
    total_samples = Y_data.shape[0]
    split_idx = int(total_samples * 0.8)
    
    # Prepare lists for inputs
    X_full = X_data
    X_val = [x[split_idx:] for x in X_data]
    Y_full = Y_data
    Y_val = Y_data[split_idx:]
    
    print(f"Data Loaded. Total samples: {total_samples}, Validation samples: {len(Y_val)}")

    # Find model files
    model_files = glob.glob(os.path.join(output_dir, "**/*.h5"), recursive=True)
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
            
            # --- Evaluate on Full Set ---
            preds_full = model.predict(X_full, batch_size=batch_size, verbose=0)
            # Threshold predictions (binary classification)
            preds_full_bin = (preds_full > 0.5).astype(np.float32)
            
            f1_full = dice_coef_np(Y_full, preds_full_bin)
            jac_full = jaccard_coef_np(Y_full, preds_full_bin)
            
            print(f"{os.path.relpath(m_path, output_dir):<80} | {'FULL':<10} | {f1_full:.4f}     | {jac_full:.4f}")

            # --- Evaluate on Validation Set ---
            preds_val = model.predict(X_val, batch_size=batch_size, verbose=0)
            preds_val_bin = (preds_val > 0.5).astype(np.float32)
            
            f1_val = dice_coef_np(Y_val, preds_val_bin)
            jac_val = jaccard_coef_np(Y_val, preds_val_bin)
            
            print(f"{'':<80} | {'VAL (20%)':<10} | {f1_val:.4f}     | {jac_val:.4f}")
            print("-" * 120)

        except Exception as e:
            print(f"Error evaluating {m_path}: {e}")

if __name__ == "__main__":
    main()

