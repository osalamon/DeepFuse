import tifffile as tiff
import numpy as np
import os

# creates single-mask images on-the-fly and returns them in a single array!

def create_train_data(input_path, gt_path):
    # AUTHORS's original code
    # mask_imgs = np.zeros((144, 1024, 1024))  
	# first dimension is the number of available GT single-mask images (Gold segmentation annotations)
	# if mask images contain multiple masks, they should be split into single-mask images beforehand.
	# second&third dim. is the dimension of input images

    # NEW CODE dynamically assigning number of golden truth images -----------------
    # 1. Filter for valid TIF files first
    valid_files = [f for f in os.listdir(gt_path) if f.endswith(".tif")]
    num_files = len(valid_files)

    if num_files == 0:
        raise ValueError(f"No .tif files found in {gt_path}")

    # 2. Dynamically allocate memory based on what you actually have
    mask_imgs = np.zeros((num_files, 1070, 1036), dtype='float32')

    ii = 0
    print('Loading masks from ', input_path)
    for image_name in os.listdir(gt_path):  # Go through all the .tif files in the folder
        if image_name.endswith(".tif"):
            participant_image_name = 'mask' + image_name.split('man_seg')[1][:3] + '.tif'
            im_gt = tiff.imread(os.path.join(gt_path, image_name))
            im_allmasks = tiff.imread(os.path.join(input_path, participant_image_name))
            # mask_im = np.zeros((im_allmasks.shape[0], im_allmasks.shape[1]), dtype='float32')
            mask_im = np.zeros((im_allmasks.shape[0], im_allmasks.shape[1]), dtype='float32')
            cur_label = np.unique(im_gt)[-1]
            mask_im[im_allmasks == cur_label] = 1.0	# to produce a binary image for each label since DeepFuse is a 2-class classifier
            mask_imgs[ii, :, :] = mask_im
            ii = ii+1

    mask_imgs = mask_imgs[..., np.newaxis]
    return mask_imgs

# creates single-mask GT images on-the-fly and returns them in a single array!

def create_gt_data(gt_path):
    # AUTHORS's original code
    # mask_imgs = np.zeros((144, 1024, 1024))  
	# first dimension is the number of available GT single-mask images (Gold segmentation annotations)
	# if mask images contain multiple masks, they should be split into single-mask images beforehand.
	# second&third dim. is the dimension of input images
    
    # NEW CODE dynamically assigning number of golden truth images -----------------
    # 1. Filter for valid TIF files first
    valid_files = [f for f in os.listdir(gt_path) if f.endswith(".tif")]
    num_files = len(valid_files)

    if num_files == 0:
        raise ValueError(f"No .tif files found in {gt_path}")

    # 2. Dynamically allocate memory based on what you actually have
    
    gt_mask_imgs = np.zeros((num_files, 1070, 1036), dtype='float32')
    ii = 0
    for image_name in os.listdir(gt_path):  # Go through all the .tif files in the folder
        if image_name.endswith(".tif"):
            print('Reading image', image_name)
            im_allmasks = tiff.imread(os.path.join(gt_path, image_name))
            im_allmasks [im_allmasks > 0.5] = 1.0 # to produce a binary image
            im_allmasks [im_allmasks < 0.5] = 0.0
            gt_mask_imgs[ii, :, :] = im_allmasks
            ii = ii+1

    gt_mask_imgs = gt_mask_imgs[..., np.newaxis]
    return gt_mask_imgs
