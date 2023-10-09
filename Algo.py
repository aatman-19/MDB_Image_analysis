from math import *
import cv2
import numpy as np
import os
from multiprocessing import Pool

# calculate number of images in a folder
num_img = 0
for im in os.listdir("images"):
    num_img += 1


class Algo:
    def __init__(self):
        self.cc_feature_matrix = [[0] * 65] * num_img
        self.ic_feature_matrix = [[0] * 26] * num_img
        self.num_images = num_img


# def color_code_histogram():
#     img_folder = "test_images"
#     feature_mat = np.zeros(num_img * 65).reshape(num_img, 65)
#     count = 0
#     for cur_img in os.listdir(img_folder):
#         image = cv2.imread(os.path.join(img_folder, cur_img))
#         histogram = np.zeros(65, dtype=np.int32)
#         histogram[0] = image.size
#         for row in image:
#             for pixel in row:
#                 r, g, b = pixel[2], pixel[1], pixel[0]  # Extract R, G, and B values
#
#                 # Extract the two most significant bits from each channel
#                 r_bits = (r >> 6) & 0b11
#                 g_bits = (g >> 6) & 0b11
#                 b_bits = (b >> 6) & 0b11
#
#                 # Concatenate the bits into a decimal value (RGB format)
#                 concat_value = (r_bits << 4) | (g_bits << 2) | b_bits
#                 histogram[concat_value + 1] += 1
#         feature_mat[count] = histogram
#         count += 1
#     return feature_mat


def process_image_cc(image_path):
    image = cv2.imread(image_path)
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
            if (concat_value == 64):
                histogram[concat_value] += 1
            else:
                histogram[concat_value + 1] += 1
    return histogram


def color_code_feature_map(img_folder_):
    image_paths = [os.path.join(img_folder_, cur_img) for cur_img in os.listdir(img_folder_)]
    histograms = pmap(process_image_cc, image_paths)  # Process images in parallel

    num_imgs = len(image_paths)
    feature_mat = np.zeros(num_imgs * 65).reshape(num_imgs, 65)
    for count, histogram in enumerate(histograms):
        feature_mat[count] = histogram
    return feature_mat


# IC functions
def process_image_ic(image_path):
    image = cv2.imread(image_path)
    histogram = np.zeros(26, dtype=np.int32)
    histogram[0] = image.size
    for row in image:
        for pixel in row:
            intensity = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]
            bin_index = int(intensity / 10)
            if (bin_index == 25):
                histogram[bin_index] += 1
            else:
                histogram[bin_index + 1] += 1
    return histogram


def pmap(func, iterable, num_processes=None):
    if num_processes is None:
        num_processes = os.cpu_count()  # Use all available CPU cores by default

    with Pool(num_processes) as pool:
        return pool.map(func, iterable)


def intensity_code_feature_map(img_folder_):
    image_paths = [os.path.join(img_folder_, cur_img) for cur_img in os.listdir(img_folder_)]
    histograms = pmap(process_image_ic, image_paths)  # Process images in parallel

    num_imgs = len(image_paths)
    feature_mat = np.zeros(num_imgs * 26).reshape(num_imgs, 26)
    for count, histogram in enumerate(histograms):
        feature_mat[count] = histogram
    return feature_mat


# def intensity_code_histogram():
#     img_folder = "test_images"
#     feature_mat = np.zeros(num_img * 26).reshape(num_img, 26)
#     count = 0
#     for cur_img in os.listdir(img_folder):
#         image = cv2.imread(os.path.join(img_folder, cur_img))
#         histogram = np.zeros(26, dtype=np.int32)
#         histogram[0] = image.size
#         for row in image:
#             for pixel in row:
#                 # open cv follows BGR -> B[0], G[1], R[2]
#                 intensity = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]
#                 bin_index = int(intensity / 10)
#                 histogram[bin_index + 1] += 1
#         feature_mat[count] = histogram
#         # print(histogram)
#         count += 1
#     return feature_mat


def manhattan_dist(selected_img, other_img):
    d = 0
    for i in range(1, 26):
        d += abs((selected_img[i] / selected_img[0]) - (other_img[i] / other_img[0]))
    return d


def get_distance_vector(selected_img, f_mat):
    dist_vector = np.zeros(num_img, dtype=np.float64)
    for i in range(0, num_img):
        dist_vector[i] = (manhattan_dist(selected_img, f_mat[i]))
    return dist_vector


# testing parallelized functions for INTENSITY
if __name__ == "__main__":
    img_folder = "images"
    result = color_code_feature_map(img_folder)
    np.set_printoptions(suppress=True)
    print(np.array2string(result, separator=", "))

    print("\n Now calculating the distance array")
    dv = get_distance_vector(result[99], result)
    print(dv)
