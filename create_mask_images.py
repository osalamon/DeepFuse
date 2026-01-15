import tifffile as tf
import numpy as np
import os

# creates single-mask images on-the-fly and returns them in a single array!

def create_train_data(input_path, gt_path):
    # AUTHORS's original code
    # mask_imgs = np.zeros((144, 1024, 1024))  
	# first dimension is the number of available GT single-mask images (Gold segmentation annotations)
	# if mask images contain multiple masks, they should be split into single-mask images beforehand.
	# second&third dim. is the dimension of input images
    
    # 1. Filter AND SORT valid TIF files
    # CRITICAL: 'sorted' ensures that image 001 in Input1 corresponds to image 001 in Input2
    valid_files = sorted([f for f in os.listdir(gt_path) if f.endswith(".tif")])
    num_files = len(valid_files)

    if num_files == 0:
        raise ValueError(f"No .tif files found in {gt_path}")
    
    # Read the first image to get the TRUE shape (for debug print)
    first_image_path = os.path.join(gt_path, valid_files[0])
    sample_img = tf.imread(first_image_path)
    true_shape = sample_img.shape 
    print(f"Detected image shape: {true_shape}")

    # 2. init lists
    objects_subimages = []
    objects_gts = []

    # 3. Define crop size
    crop_h, crop_w = 101, 101 

    # 4. Outer loop: process each file/image
    ## ii = 0
    print('Loading masks from ', input_path)
    for image_name in valid_files:  # Go through all the .tif files in the folder
        if image_name.endswith(".tif"):
            # Extract the number safely (removes '.tif' first so we don't accidentally slice it)
            image_id = image_name.split('man_seg')[1].split('.')[0]
            # Construct the filename. 
            # I need 4 digits, i.e. 'mask0090.tif', use .zfill(4)
            participant_image_name = 'mask' + image_id.zfill(4) + '.tif'
            # participant_image_name = 'mask' + image_name.split('man_seg')[1][:3] + '.tif'
            im_gt = tf.imread(os.path.join(gt_path, image_name))
            im_allmasks = tf.imread(os.path.join(input_path, participant_image_name))
            
            unique_objs = np.unique(im_gt)
            unique_objs = unique_objs[unique_objs != 0]     # Remove 0 (background) if present # TODO not sure about this line

            # 5. Inner Loop: Iterate through unique object labels found within that file/image
            for cur_label in unique_objs:
                # 5.1 Create Input Mask (The rater's segmentation)
                mask_im = np.zeros(im_gt.shape, dtype='float32') # creates empty mask JUST for the given object, firstly full-size
                mask_im[im_allmasks == cur_label] = 1.0 # to get the mask where rater's input matches the current GT object label
                # 5.2. Create Target Mask (The Ground Truth for this specific object)
                gt_im = np.zeros(im_gt.shape, dtype=np.float32)
                gt_im[im_gt == cur_label] = 1.0

                rows, cols = np.where(gt_im == 1.0)  # Find coordinates where mask is 1

                if len(rows) == 0:
                    continue # Skip if empty intersection

                # 6. Calculate the bounding box
                center_y = int(np.mean(rows))
                center_x = int(np.mean(cols))

                # 7. Calculate crop start/end
                start_y = max(0, center_y - crop_h // 2)
                start_x = max(0, center_x - crop_w // 2)
                end_y = start_y + crop_h
                end_x = start_x + crop_w
                
                # 8. 
                if end_y > mask_im.shape[0]:  
                    # If the crop goes below the image...
                    end_y = mask_im.shape[0]  # Snap to the bottom edge
                    start_y = end_y - crop_h  # Shift valid start point up to keep size constant

                # 9. Check Right Edge (X) 
                if end_x > mask_im.shape[1]: 
                    # If the crop goes to the right of the image...
                    end_x = mask_im.shape[1]  # Snap to the right edge
                    start_x = end_x - crop_w  # Shift valid start point left to keep size constant
                        ## mask_im = np.zeros((im_allmasks.shape[0], im_allmasks.shape[1]), dtype='float32')
                        ## mask_im = np.zeros((im_allmasks.shape[0], im_allmasks.shape[1]), dtype='float32')
                        ## cur_label = np.unique(im_gt)[-1]
                
                # Perform the crop
                crop_input = mask_im[start_y:end_y, start_x:end_x]
                crop_gt = gt_im[start_y:end_y, start_x:end_x]

                if crop_input.shape[0] != crop_h or crop_input.shape[1] != crop_w:
                    pad_y = crop_h - crop_input.shape[0]
                    pad_x = crop_w - crop_input.shape[1]
                    crop_input = np.pad(crop_input, ((0, pad_y), (0, pad_x)), mode='constant')

                    crop_gt = np.pad(crop_gt, ((0, pad_y), (0, pad_x)), mode = 'constant')
                
                    ## mask_im[im_allmasks == cur_label] = 1.0	# to produce a binary image for each label since DeepFuse is a 2-class classifier
                    ## mask_imgs[ii, :, :] = mask_im
                    ## ii = ii+1
                objects_subimages.append(crop_input)
                objects_gts.append(crop_gt)

    # Convert list to numpy array
    if len(objects_subimages) == 0 or len(objects_gts) == 0:
        raise ValueError("No objects found to process.")
    
    input_mask_imgs = np.array(objects_subimages, dtype='float32')
    gts_imgs = np.array(objects_gts, dtype = 'float32')

    print(f"Created dataset with {input_mask_imgs.shape[0]} patches of size {crop_h}x{crop_w}")
    input_mask_imgs = input_mask_imgs[..., np.newaxis]
    gts_imgs = gts_imgs[..., np.newaxis]
    
    return input_mask_imgs, gts_imgs

# creates single-mask GT images on-the-fly and returns them in a single array!

# def create_gt_data(gt_path):
#     # AUTHORS's original code
#     # mask_imgs = np.zeros((144, 1024, 1024))  
# 	# first dimension is the number of available GT single-mask images (Gold segmentation annotations)
# 	# if mask images contain multiple masks, they should be split into single-mask images beforehand.
# 	# second&third dim. is the dimension of input images
    
#     valid_files = sorted([f for f in os.listdir(gt_path) if f.endswith(".tif")])
#     num_files = len(valid_files)

#     if num_files == 0:
#         raise ValueError(f"No .tif files found in {gt_path}")
#     # Read the first image to get the TRUE shape
#     first_image_path = os.path.join(gt_path, valid_files[0])
#     sample_img = tf.imread(first_image_path)
#     true_shape = sample_img.shape  # This will likely be (1036, 1070)
#     print(f"Detected image shape: {true_shape}")


#     # NEW CODE dynamically assigning number of golden truth images -----------------
#     # Filter for valid TIF files first
#     valid_files = [f for f in os.listdir(gt_path) if f.endswith(".tif")]
#     num_files = len(valid_files)

#     if num_files == 0:
#         raise ValueError(f"No .tif files found in {gt_path}")

#     # Dynamically allocate memory based on what you actually have
    
#     gt_mask_imgs = np.zeros((num_files, true_shape[0], true_shape[1]), dtype='float32')
#     ii = 0
#     for image_name in os.listdir(gt_path):  # Go through all the .tif files in the folder
#         if image_name.endswith(".tif"):
#             print('Reading image', image_name)
#             im_allmasks = tf.imread(os.path.join(gt_path, image_name))
#             im_allmasks [im_allmasks > 0.5] = 1.0 # to produce a binary image
#             im_allmasks [im_allmasks < 0.5] = 0.0
#             gt_mask_imgs[ii, :, :] = im_allmasks
#             ii = ii+1

#     gt_mask_imgs = gt_mask_imgs[..., np.newaxis]
#     return gt_mask_imgs
