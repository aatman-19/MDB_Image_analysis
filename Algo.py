from math import *
import cv2
import numpy as np
import os

# calculate number of images in a folder
num_img = 0
for im in os.listdir("test_images"):
    num_img += 1


def intensity_code_histogram():
    img_folder = "test_images"
    feature_mat = np.zeros(78).reshape(num_img, 26)
    count = 0
    for cur_img in os.listdir(img_folder):
        image = cv2.imread(os.path.join(img_folder, cur_img))
        histogram = np.zeros(26, dtype=np.int32)
        histogram[0] = image.size
        for row in image:
            for pixel in row:
                # open cv follows BGR -> B[0], G[1], R[2]
                intensity = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]
                bin_index = int(intensity / 10)
                histogram[bin_index + 1] += 1
        feature_mat[count] = histogram
        # print(histogram)
        count += 1
    return feature_mat


def manhattan_dist(selected_img, other_img):


class Algo:
    def __init__(self):
        self.cc_feature_matrix = [[0] * 64] * num_img
        self.ic_feature_matrix = [[0] * 26] * num_img


ic_bins = intensity_code_histogram()
np.set_printoptions(suppress=True)
print(ic_bins)
