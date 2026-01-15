import os
import glob
from turtle import mode
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from create_mask_images import create_train_data
from tensorflow.keras.layers import Input, concatenate, Conv2D, add

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

output_dir = "." # Point this to your output folder
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
    # input6 = Input(shape=(im_len, im_wid, 1))
    # input7 = Input(shape=(im_len, im_wid, 1))
    # input8 = Input(shape=(im_len, im_wid, 1))
    # input9 = Input(shape=(im_len, im_wid, 1))
    # input10 = Input(shape=(im_len, im_wid, 1))
    # input11 = Input(shape=(im_len, im_wid, 1))
    # input12 = Input(shape=(im_len, im_wid, 1))
    # input13 = Input(shape=(im_len, im_wid, 1))
    # input14 = Input(shape=(im_len, im_wid, 1))
    # input15 = Input(shape=(im_len, im_wid, 1))
    # input16 = Input(shape=(im_len, im_wid, 1))
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
    # # 
    # x6 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input6)
    # x6 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x6)
    # x6 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x6)
    # x6 = Model(inputs=input6, outputs=x6)
    # # 
    # x7 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input7)
    # x7 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x7)
    # x7 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x7)
    # x7 = Model(inputs=input7, outputs=x7)
    # # 
    # x8 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input8)
    # x8 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x8)
    # x8 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x8)
    # x8 = Model(inputs=input8, outputs=x8)
    # # 
    # x9 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input9)
    # x9 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x9)
    # x9 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x9)
    # x9 = Model(inputs=input9, outputs=x9)
    # # 
    # x10 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input10)
    # x10 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x10)
    # x10 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x10)
    # x10 = Model(inputs=input10, outputs=x10)
    # # 
    # x11 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input11)
    # x11 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x11)
    # x11 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x11)
    # x11 = Model(inputs=input11, outputs=x11)
    # # 
    # x12 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input12)
    # x12 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x12)
    # x12 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x12)
    # x12 = Model(inputs=input12, outputs=x12)
    # #
    # x13 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input13)
    # x13 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x13)
    # x13 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x13)
    # x13 = Model(inputs=input13, outputs=x13)
    # # 
    # x14 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input14)
    # x14 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x14)
    # x14 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x14)
    # x14 = Model(inputs=input14, outputs=x14)
    # # 
    # x15 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input15)
    # x15 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x15)
    # x15 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x15)
    # x15 = Model(inputs=input15, outputs=x15)
    # # 
    # x16 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input16)
    # x16 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x16)
    # x16 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x16)
    # x16 = Model(inputs=input16, outputs=x16)
    # combine the output of all branches
    # combined = concatenate([x1.output, x2.output, x3.output, x4.output, x5.output, x6.output, x7.output, x8.output, x9.output, x10.output, x11.output, x12.output, x13.output, x14.output, x15.output, x16.output])
    # combined = concatenate([x1.output, x2.output, x3.output, x4.output])
    # print(f"DEBUG: x1={x1}, x2={x2}, x3={x3}, x4={x4}")
    # merged = add([x1.output, x2.output, x3.output, x4.output, x5.output])
    combined = concatenate([x1.output, x2.output, x3.output, x4.output, x5.output])

    # feed the combined output to a non-linear activation function
    # z = Conv2D(1, (1, 1), activation='sigmoid')(combined)
    output = Conv2D(1, (1,1), activation='sigmoid')(combined)

    # our model will accept the inputs of 16 branches and then output a single value
    # model = Model(inputs=[x1.input, x2.input, x3.input, x4.input, x5.input, x6.input, x7.input, x8.input, x9.input, x10.input, x11.input, x12.input, x13.input, x14.input, x15.input, x16.input], outputs=z)
    # model = Model(inputs=[x1.input, x2.input, x3.input, x4.input], outputs=z)
    # model = Model(inputs=[input1, input2, input3, input4], outputs=[output])

    # Check that all lengths match



    model = Model(inputs=[x1.input, x2.input, x3.input, x4.input, x5.input], outputs=[output])

    model.compile(loss=dice_coef_loss, optimizer=Adam(lr=learning_rate), metrics=[dice_coef])
    model.summary()
    return model

# --- Main Evaluation Loop ---
def main():
    print("--- Loading Data ---")

    # FIX: Logic to handle multiple inputs vs single target
    X_data_list = []
    Y_ground_truth = None

    for i, p in enumerate(input_paths):
        print(f"Loading input branch {i+1} from: {p}")
        data, gt = create_train_data(p, gt_path)

        X_data_list.append(data)    
    # print(f"Loading GT from: {gt_path}")
    # Y_data = create_gt_data(gt_path)

        if i == 0:
            Y_ground_truth = gt
        else:
            # Optional: Safety check to ensure sample counts match
            if gt.shape[0] != Y_ground_truth.shape[0]:
                raise ValueError("Mismatch in number of samples between input folders!")


    # Calculate split point for validation (last 20%)
    total_samples = Y_ground_truth.shape[0]
    split_idx = int(total_samples * 0.8)
    
    # Prepare lists for inputs
    X_full = X_data_list
    X_val = [x[split_idx:] for x in X_data_list]

    Y_full = Y_ground_truth
    Y_val = Y_ground_truth[split_idx:]
    
    print(f"Data Loaded. Total samples: {total_samples}, Validation samples: {len(Y_val)}")

    # Find model files
    model_files = glob.glob(os.path.join("*.h5"), recursive=True)
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

