import numpy as np
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, concatenate, Conv2D
from tensorflow.keras.layers import Input, concatenate, Conv2D, add
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from create_mask_images import create_train_data, create_gt_data
import tensorflow.compat.v1 as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
learning_rate = 4*1e-4
smooth = 1e-16
num_of_epochs = 100
num_of_filters = 16
im_len = 1036
im_wid = 1070

# Define input paths
input_path1 = '/mnt/proj1/eu-25-40/innovaite/synchronized_data/BF-C2DL-MuSC/CALT-US/01_RES/'
input_path2 = '/mnt/proj1/eu-25-40/innovaite/synchronized_data/BF-C2DL-MuSC/KIT-Sch-GE/01_RES/'
input_path3 = '/mnt/proj1/eu-25-40/innovaite/synchronized_data/BF-C2DL-MuSC/KTH-SE (3)/01_RES/'
input_path4 = '/mnt/proj1/eu-25-40/innovaite/synchronized_data/BF-C2DL-MuSC/MU-Lux-CZ/01_RES/'

gt_path = '/mnt/proj1/eu-25-40/innovaite/synchronized_data/BF-C2DL-MuSC/01_GT/SEG/'

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
# input5 = Input(shape=(im_len, im_wid, 1))
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
# x5 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(input5)
# x5 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x5)
# x5 = Conv2D(num_of_filters, (5, 5), activation="relu", padding='same')(x5)
# x5 = Model(inputs=input5, outputs=x5)
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
merged = add([x1.output, x2.output, x3.output, x4.output])
# feed the combined output to a non-linear activation function
# z = Conv2D(1, (1, 1), activation='sigmoid')(combined)
output = Conv2D(1, (1,1), activation='sigmoid')(merged)

# our model will accept the inputs of 16 branches and then output a single value
# model = Model(inputs=[x1.input, x2.input, x3.input, x4.input, x5.input, x6.input, x7.input, x8.input, x9.input, x10.input, x11.input, x12.input, x13.input, x14.input, x15.input, x16.input], outputs=z)
# model = Model(inputs=[x1.input, x2.input, x3.input, x4.input], outputs=z)
# model = Model(inputs=[input1, input2, input3, input4], outputs=[output])

# Check that all lengths match



model = Model(inputs=[x1.input, x2.input, x3.input, x4.input], outputs=[output])

model.compile(loss=dice_coef_loss, optimizer=Adam(lr=learning_rate), metrics=[dice_coef])
model.summary()

## Create training data
in1 = create_train_data(input_path1, gt_path)	# input_path1 should contain segmentation masks from the first source
in2 = create_train_data(input_path2, gt_path)	# gt_path should contain tracking/detection markers
in3 = create_train_data(input_path3, gt_path)	# only masks which have a matching marker will be loaded.
in4 = create_train_data(input_path4, gt_path)
# in5 = create_train_data(input_path5, gt_path)
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
target = create_gt_data(gt_path)
# Train the model
mcp_save = ModelCheckpoint('model_in16_5x5_' + format(learning_rate, '.0e') + '_' + str(num_of_epochs) + '_' + str(num_of_filters) + '.h5', save_best_only=True, monitor='val_loss', mode='min')
# model.fit(x=[in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16], y=target, batch_size=1, epochs=num_of_epochs, verbose=2, shuffle=True, callbacks=[mcp_save], validation_split=0.2)

model.fit(x=[in1, in2, in3, in4], y=target, batch_size=1, epochs=num_of_epochs, verbose=2, shuffle=True, callbacks=[mcp_save], validation_split=0.2)
