import numpy as np
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, concatenate, Conv2D
from tensorflow.keras.layers import Input, concatenate, Conv2D, add
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from create_mask_images import create_train_data
from tensorflow.keras.callbacks import ModelCheckpoint
import os

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
learning_rate = 4*1e-4
smooth = 1e-16
num_of_epochs = 100
num_of_filters = 16
im_len = 101
im_wid = 101


input_path1 = '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/CALT-US/01_RES/'
input_path2 = '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/DREX-US/01_RES/'
input_path3 = '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/KIT-Sch-GE/01_RES/'
input_path4 = '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/KTH-SE (5)/01_RES/'
input_path5 = '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/MU-Lux-CZ/01_RES/'
gt_path = '/home/osalamon/silver-truth/data/synchronized_data/BF-C2DL-HSC/01_GT/SEG/'

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

## Create training data ------------------------------------------------------
# We unpack the tuple (Input, GroundTruth) returned by the function.
print("--- Loading Input 1 ---")
in1, target = create_train_data(input_path1, gt_path) 

print("--- Loading Input 2 ---")
# We use '_' to ignore the second return value because 'target' is already loaded
in2, _ = create_train_data(input_path2, gt_path)

print("--- Loading Input 3 ---")
in3, _ = create_train_data(input_path3, gt_path)

print("--- Loading Input 4 ---")
in4, _ = create_train_data(input_path4, gt_path)

print("--- Loading Input 5 ---")
in5, _ = create_train_data(input_path5, gt_path)

# Verify that all datasets have the same number of samples
print(f"Shapes consistency check:")
print(f"In1: {in1.shape}, In2: {in2.shape}, Target: {target.shape}")
# in6 = create_train_data(input_path6, gt_path)
# in7 = create_train_data(input_path7, gt_path)
# in8 = create_train_data(input_path8, gt_path)
# in9 = create_train_data(input_path9, gt_path)
# in10 = create_train_data(input_path10, gt_path)
# in11 = create_train_data(input_path11, gt_path)
# in12 = create_train_data(input_path12, gt_path)
# in13 = create_train_data(input_path13, gt_path)
# in14 = create_train_data(input_path14, gt_path)
# in15 = create_train_data(input_path15, gt_path)
# in16 = create_train_data(input_path16, gt_path)
# target = create_train_data(gt_path)

if not (in1.shape[0] == in2.shape[0] == target.shape[0]):
    raise ValueError("Mismatch in number of samples! Check if all folders have the exact same files.")

# Train the model
mcp_save = ModelCheckpoint('model_in16_5x5_' + format(learning_rate, '.0e') + '_' + str(num_of_epochs) + '_' + str(num_of_filters) + '.h5', save_best_only=True, monitor='val_loss', mode='min')
# model.fit(x=[in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16], y=target, batch_size=1, epochs=num_of_epochs, verbose=2, shuffle=True, callbacks=[mcp_save], validation_split=0.2)

model.fit(x=[in1, in2, in3, in4, in5], y=target, batch_size=1, epochs=num_of_epochs, verbose=2, shuffle=True, callbacks=[mcp_save], validation_split=0.2)

# ==========================================
# VISUALIZATION BLOCK (Paste at end of script)
# ==========================================
import matplotlib.pyplot as plt

print("Starting visualization...")

# 1. Pick a random image index to test (e.g., the 10th image in the dataset)
# Ensure we don't pick an index larger than we have
test_idx = 48
if test_idx >= in1.shape[0]:
    test_idx = 0

print(f"Visualizing image index: {test_idx}")

# 2. Prepare inputs for prediction
# We use [test_idx : test_idx+1] to keep the 4th dimension.
# In R, this is like preventing 'drop=TRUE'.
# Shape becomes (1, 1010, 1010, 1)
sample_inputs = [
    in1[test_idx : test_idx+1],
    in2[test_idx : test_idx+1],
    in3[test_idx : test_idx+1],
    in4[test_idx : test_idx+1],
    in5[test_idx : test_idx+1]
]

# 3. Run the prediction
# Returns a probability map (values 0.0 to 1.0)
prediction = model.predict(sample_inputs)

# 4. Threshold the output (DeepFuse outputs probabilities)
# Anything > 0.5 is a cell, anything < 0.5 is background
prediction_binary = (prediction > 0.5).astype(float)

# 5. Create a plot with 7 columns: 5 Inputs + 1 Prediction + 1 Ground Truth
fig, axes = plt.subplots(1, 7, figsize=(25, 5))

# Helper to remove single dims for plotting (1010, 1010, 1) -> (1010, 1010)
# In R, this is basically 'as.matrix()'
def to_img(tensor):
    return tensor.squeeze()

# Plot Inputs
axes[0].imshow(to_img(in1[test_idx]), cmap='gray'); axes[0].set_title("Input 1 (CALT)")
axes[1].imshow(to_img(in2[test_idx]), cmap='gray'); axes[1].set_title("Input 2 (KIT)")
axes[2].imshow(to_img(in3[test_idx]), cmap='gray'); axes[2].set_title("Input 3 (KTH)")
axes[3].imshow(to_img(in4[test_idx]), cmap='gray'); axes[3].set_title("Input 4 (MU)")
axes[4].imshow(to_img(in5[test_idx]), cmap='gray'); axes[4].set_title("Input 5 (New)")

# Plot Result
axes[5].imshow(to_img(prediction_binary[0]), cmap='jet'); axes[5].set_title("DeepFuse Result")

# Plot Ground Truth
axes[6].imshow(to_img(target[test_idx]), cmap='gray'); axes[6].set_title("Gold Truth")

# Remove axes ticks for cleanliness
for ax in axes:
    ax.axis('off')

# 6. Save to disk
output_filename = "visualization_result.png"
plt.savefig(output_filename, dpi=150)
print(f"Saved visualization to {os.getcwd()}/{output_filename}")
