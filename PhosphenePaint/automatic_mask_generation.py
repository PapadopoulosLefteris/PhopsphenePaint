from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator 
import cv2
import numpy as np
import torch
from skimage.feature import CENSURE
from skimage.color import rgb2gray
from skimage.filters import gaussian
# from segment_anything_hq import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
import os
import pickle

def histogram_equalization_color(img):
    yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # Equalize the histogram of the Y channel (luminance)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
        
    # Convert the image back to BGR
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    return equalized_image

def create_grid(image):
    # Calculate the spacing between points
    spacing_x = image.shape[1] // 32
    spacing_y = image.shape[0] // 32

    # Create an array of grid points
    points = []
    for y in range(spacing_y // 2, image.shape[0], spacing_y):
        for x in range(spacing_x // 2, image.shape[1], spacing_x):
            points.append([x, y])

    return points


def full_mask(img,path):
    mask_generator = SAM2AutomaticMaskGenerator(sam)
    mask_generator = mask_generator
    masks = mask_generator.generate(img)

    final = []
    for i in masks:
        final.append(np.uint8(i['segmentation'])*255)
    final = np.array(final)
    print(final.shape)



    np.save(path,final)

def jaccard_similarity(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    res = np.count_nonzero(intersection)/(np.min([np.count_nonzero(mask1),np.count_nonzero(mask2)])+1)
    return res
    # return np.count_nonzero(intersection) / np.count_nonzero(union)

def remove_duplicates(masks, threshold=0.8):
    keep_indices = []

    for i in range(len(masks)):
        check = 0
        if keep_indices==[]:
            keep_indices.append(i)
        else:
            for j in keep_indices:
                sim = jaccard_similarity(masks[i],masks[j])
                if sim>threshold:
                    check = 1
                    break
            if check == 0:
                keep_indices.append(i)
    
    print(keep_indices)


    return masks[keep_indices]

def sampled(img, path):
    predictor = SAM2ImagePredictor(sam)
    predictor.set_image(img)


    detector = CENSURE()
    detector.detect(rgb2gray(img))

    # print(detector.keypoints)
    # points = detector.keypoints
    points = create_grid(img)

    masks = []
    for i in range(len(points)):
        mask, score, _ = predictor.predict(point_coords=np.array([points[i]]),point_labels=[1], multimask_output=True)
        
        
        # masks.append(mask)
        for j in range(len(score)):
            if score[j]>0.92:
                masks.append(mask[j])

    masks = np.array(masks)
    print("Before removal", masks.shape)
    masks = remove_duplicates(masks)
    print("After removal", masks.shape)
    masks = np.uint8(masks)*255
    np.save(path, masks)
    print("Done")




path_to_model = "C:/Users/Administrator/Desktop/SAM2/sam2/checkpoints/sam2.1_hiera_large.pt"
path_to_config = "C:/Users/Administrator/Desktop/SAM2/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"



device = torch.device('cuda:0')
device = 'cpu'

sam = build_sam2(path_to_config, path_to_model, device=device, apply_postprocessing=False)

path = "images/scene_5.png"
m = "masks/mask_5.npy"

img = cv2.imread(path)
img = cv2.resize(img,(512,512))


full_mask(img,m)