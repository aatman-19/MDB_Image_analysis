from math import *
import cv2
import numpy as np
import os

# calculate number of images in a folder
num_img = 0
for im in os.listdir("test_images"):
    num_img += 1


class Algo:
    def __init__(self):
        self.cc_feature_matrix = [[0] * 64] * num_img
        self.ic_feature_matrix = [[0] * 26] * num_img
        self.num_images = num_img


def color_code_histogram():
    img_folder = "test_images"
    feature_mat = np.zeros(num_img * 65).reshape(num_img, 65)
    count = 0
    for cur_img in os.listdir(img_folder):
        image = cv2.imread(os.path.join(img_folder, cur_img))
        histogram = np.zeros(65, dtype=np.int32)
        histogram[0] = image.size
        for row in image:
            for pixel in row:
                r, g, b = pixel[2], pixel[1], pixel[0]  # Extract R, G, and B values

                # Extract the two most significant bits from each channel
                r_bits = (r >> 6) & 0b11
                g_bits = (g >> 6) & 0b11
                b_bits = (b >> 6) & 0b11

                # Concatenate the bits into a decimal value (RGB format)
                concat_value = (r_bits << 4) | (g_bits << 2) | b_bits
                histogram[concat_value + 1] += 1
        feature_mat[count] = histogram
        count += 1
    return feature_mat


def intensity_code_histogram():
    img_folder = "test_images"
    feature_mat = np.zeros(num_img * 26).reshape(num_img, 26)
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
    d = 0
    for i in range(1, 26):
        d += abs((selected_img[i] / selected_img[0]) - (other_img[i] / other_img[0]))
    return d


def get_distance_vector(selected_img):
    dist_vector = np.zeros(num_img, dtype=np.float64)
    for i in range(0, num_img):
        dist_vector[i] = (manhattan_dist(selected_img, f_mat[i]))
    return dist_vector


# testing functions for INTENSITY
f_mat = color_code_histogram()
np.set_printoptions(suppress=True)
print(np.array2string(f_mat, separator=', '))

print("\n >now getting distance vector")
dv = get_distance_vector(f_mat[1])
print(np.array2string(dv, separator=', '))
