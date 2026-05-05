import numpy as np
import argparse
import sys
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, add
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from create_mask_images import create_multi_input_data_from_parquet
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Train DeepFuse on HSC or MuSC dataset using parquet splits')
parser.add_argument('--dataset', type=str, required=True, choices=['BF-C2DL-HSC', 'BF-C2DL-MuSC'], help='Dataset name')
parser.add_argument('--fold', type=int, required=True, choices=[1, 2], help='Cross-validation fold (1 or 2)')
parser.add_argument('--timestamp', type=str, default='debug', help='Timestamp for unique folder generation')
args = parser.parse_args()

dataset_name = args.dataset
fold_num = args.fold
timestamp = args.timestamp

# Create specific output folder: e.g., "trained_on_BF-C2DL-HSC_fold1_20260123"
output_folder = f"trained_on_{dataset_name}_fold{fold_num}_{timestamp}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print(f"--- TRAINING CONFIGURATION ---")
print(f"Dataset: {dataset_name}")
print(f"Fold: {fold_num}")
print(f"Output Folder: {output_folder}")
# ------------------------

# --- MODEL DEFINITION ---
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
learning_rate = 4*1e-4
smooth = 1e-16
num_of_epochs = 100
num_of_filters = 16
im_len = 101
im_wid = 101

# Relative paths from repo root
data_root = Path("data/synchronized_data") / dataset_name
parquet_path = Path("data/dataframes") / dataset_name / "whole_image" / f"{dataset_name}_split_fold-{fold_num}.parquet"

# Competitor input paths (5 raters)
input_paths = [
    str(data_root / f"CALT-US/01_RES/"),
    str(data_root / f"DREX-US/01_RES/"),
    str(data_root / f"KIT-Sch-GE/01_RES/"),
    str(data_root / f"KTH-SE (5)/01_RES/"),
    str(data_root / f"MU-Lux-CZ/01_RES/"),
]
gt_path = str(data_root / "01_GT/SEG/")

# For fold 2, we might need to use sequence 02 paths instead
# The parquet file handles which frames to use, but we need to point to the right RES folders
# For simplicity, we use 01_RES for all competitors since the parquet file filters by frame_id
# If fold 2 uses sequence 02, we should adjust paths accordingly
if fold_num == 2:
    input_paths = [
        str(data_root / f"CALT-US/02_RES/"),
        str(data_root / f"DREX-US/02_RES/"),
        str(data_root / f"KIT-Sch-GE/02_RES/"),
        str(data_root / f"KTH-SE (5)/02_RES/"),
        str(data_root / f"MU-Lux-CZ/02_RES/"),
    ]
    gt_path = str(data_root / "02_GT/SEG/")

# --- HYPERPARAMETER SUMMARY ---
print("\n" + "="*60)
print("HYPERPARAMETER SUMMARY")
print("="*60)
print(f"  Architecture:        DeepFuse (5-branch CNN fusion)")
print(f"  Kernel size:         (5, 5)")
print(f"  Filters per branch:  {num_of_filters}")
print(f"  Branch depth:        3 Conv2D layers")
print(f"  Input crop size:     {im_len} x {im_wid}")
print(f"  Final layer:         Conv2D(1, (1,1)) with sigmoid")
print(f"  Optimizer:           Adam")
print(f"  Learning rate:       {learning_rate}")
print(f"  Loss function:       1 - Dice (smooth={smooth})")
print(f"  Epochs:              {num_of_epochs}")
print(f"  Batch size:          1")
print(f"  Validation split:    20% (from train/val sequence)")
print(f"  Checkpoint:          save_best_only=True, monitor=val_loss")
print(f"  Total params:        ~66,321 (~259 KB)")
print(f"  Dataset:             {dataset_name}")
print(f"  Fold:                {fold_num}")
print(f"  Parquet file:        {parquet_path}")
print(f"  Competitors:         5 (CALT-US, DREX-US, KIT-Sch-GE, KTH-SE, MU-Lux-CZ)")
print("="*60 + "\n")

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

x3 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input3)
x3 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x3)
x3 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x3)
x3 = Model(inputs=input3, outputs=x3)

x4 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input4)
x4 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x4)
x4 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x4)
x4 = Model(inputs=input4, outputs=x4)

x5 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input5)
x5 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x5)
x5 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x5)
x5 = Model(inputs=input5, outputs=x5)

combined = concatenate([x1.output, x2.output, x3.output, x4.output, x5.output])

output = Conv2D(1, (1,1), activation='sigmoid')(combined)

model = Model(inputs=[x1.input, x2.input, x3.input, x4.input, x5.input], outputs=[output])

model.compile(loss=dice_coef_loss, optimizer=Adam(lr=learning_rate), metrics=[dice_coef])
model.summary()

## Create training data ------------------------------------------------------
print(f"\n--- Loading TRAIN split from {parquet_path} ---")
train_inputs, train_gts = create_multi_input_data_from_parquet(
    str(parquet_path), 'train', input_paths, gt_path
)

print(f"\n--- Loading VAL split from {parquet_path} ---")
val_inputs, val_gts = create_multi_input_data_from_parquet(
    str(parquet_path), 'val', input_paths, gt_path
)

# Verify shapes
print(f"\nShapes consistency check:")
for i, inp in enumerate(train_inputs):
    print(f"Train Input {i+1}: {inp.shape}")
print(f"Train GT: {train_gts.shape}")

# Check all inputs have same number of samples
train_n = train_inputs[0].shape[0]
for i, inp in enumerate(train_inputs):
    if inp.shape[0] != train_n:
        raise ValueError(f"Mismatch in train samples! Input {i+1} has {inp.shape[0]}, expected {train_n}")
if train_gts.shape[0] != train_n:
    raise ValueError(f"Mismatch in train GT samples! GT has {train_gts.shape[0]}, expected {train_n}")

val_n = val_inputs[0].shape[0]
for i, inp in enumerate(val_inputs):
    if inp.shape[0] != val_n:
        raise ValueError(f"Mismatch in val samples! Input {i+1} has {inp.shape[0]}, expected {val_n}")
if val_gts.shape[0] != val_n:
    raise ValueError(f"Mismatch in val GT samples! GT has {val_gts.shape[0]}, expected {val_n}")

# Train the model
model_filename = f"model_5x5_{learning_rate:.0e}_{num_of_epochs}_{num_of_filters}_{dataset_name}_fold{fold_num}.h5"
model_save_path = os.path.join(output_folder, model_filename)

mcp_save = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')

print(f"\n--- Training ---")
print(f"Train samples: {train_n}, Val samples: {val_n}")
print(f"Model will be saved to: {model_save_path}")

model.fit(
    x=train_inputs,
    y=train_gts,
    batch_size=1,
    epochs=num_of_epochs,
    verbose=2,
    shuffle=True,
    callbacks=[mcp_save],
    validation_data=(val_inputs, val_gts)
)

# ==========================================
# VISUALIZATION BLOCK
# ==========================================
import matplotlib.pyplot as plt

print("\nStarting visualization...")

# Pick a random image index from validation set
test_idx = min(48, val_n - 1)
print(f"Visualizing validation image index: {test_idx}")

sample_inputs = [
    val_inputs[0][test_idx : test_idx+1],
    val_inputs[1][test_idx : test_idx+1],
    val_inputs[2][test_idx : test_idx+1],
    val_inputs[3][test_idx : test_idx+1],
    val_inputs[4][test_idx : test_idx+1]
]

prediction = model.predict(sample_inputs)
prediction_binary = (prediction > 0.5).astype(float)

fig, axes = plt.subplots(1, 7, figsize=(25, 5))

def to_img(tensor):
    return tensor.squeeze()

axes[0].imshow(to_img(val_inputs[0][test_idx]), cmap='gray'); axes[0].set_title("Input 1 (CALT)")
axes[1].imshow(to_img(val_inputs[1][test_idx]), cmap='gray'); axes[1].set_title("Input 2 (DREX)")
axes[2].imshow(to_img(val_inputs[2][test_idx]), cmap='gray'); axes[2].set_title("Input 3 (KIT)")
axes[3].imshow(to_img(val_inputs[3][test_idx]), cmap='gray'); axes[3].set_title("Input 4 (KTH)")
axes[4].imshow(to_img(val_inputs[4][test_idx]), cmap='gray'); axes[4].set_title("Input 5 (MU)")

axes[5].imshow(to_img(prediction_binary[0]), cmap='jet'); axes[5].set_title("DeepFuse Result")
axes[6].imshow(to_img(val_gts[test_idx]), cmap='gray'); axes[6].set_title("Gold Truth")

for ax in axes:
    ax.axis('off')

viz_filename = f"vizualization_result_{dataset_name}_fold{fold_num}.png"
viz_save_path = os.path.join(output_folder, viz_filename)

plt.savefig(viz_save_path, dpi=150)
print(f"Saved visualization to {viz_save_path}")
