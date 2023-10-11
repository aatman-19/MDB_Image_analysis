from functools import partial
import cv2
import numpy as np
import os
from multiprocessing import Pool

# calculate number of images in a folder
num_img = 0
for im in os.listdir("images"):
    num_img += 1


class Algo:

    def __init__(self, algo_code, preview_image_index):
        self.cc_feature_matrix = [[0] * 64] * num_img
        self.ic_feature_matrix = [[0] * 26] * num_img
        self.num_images = num_img
        self.preview_image_index = preview_image_index
        self.algo_code = algo_code
        self.bin_size = [26, 65]

    def color_code_feature_map(self, img_folder):
        global num_img
        num_img = len(os.listdir(img_folder))
        feature_mat = np.zeros(num_img * 65, dtype=np.int32).reshape(num_img, 65)
        count = 0
        for cur_img in os.listdir(img_folder):
            image = cv2.imread(os.path.join(img_folder, cur_img))
            histogram = np.zeros(65, dtype=np.int32)
            histogram[0] = image.size
            # Extract R, G, and B channels and reshape for efficient processing
            r_channel = image[:, :, 2] >> 6
            g_channel = image[:, :, 1] >> 6
            b_channel = image[:, :, 0] >> 6
            # Concatenate the bits into a 6-bit color code value
            color_code = (r_channel << 4) | (g_channel << 2) | b_channel

            # Compute the histogram using NumPy's bincount
            hist = np.bincount(color_code.ravel(), minlength=64).astype(np.int32)
            histogram[1:] = hist

            feature_mat[count] = histogram
            count += 1

        return feature_mat

    def intensity_code_feature_map(self, img_folder):
        img_paths = os.listdir(img_folder)
        feature_mat = np.zeros((len(img_paths), 26), dtype=np.int32)

        for count in range(len(img_paths)):
            cur_img = img_paths[count]
            image = cv2.imread(os.path.join(img_folder, cur_img))

            # Extract the R, G, and B channels
            b_channel = image[:, :, 0]
            g_channel = image[:, :, 1]
            r_channel = image[:, :, 2]
            intensity = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
            hist, bin_edges = np.histogram(intensity, bins=26, range=(0, 260), density=False)
            hist[24] += hist[25]
            histogram = [image.size] + list(hist.astype(np.int64)[:25])
            feature_mat[count] = histogram

        return feature_mat

    # IC functions
    # def process_image_ic(self, image_path):
    #     image = cv2.imread(image_path)
    #     histogram = np.zeros(26, dtype=np.int32)
    #     histogram[0] = image.size
    #     for row in image:
    #         for pixel in row:
    #             intensity = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]
    #             bin_index = int(intensity / 10)
    #             if (bin_index == 25):
    #                 histogram[bin_index] += 1
    #             else:
    #                 histogram[bin_index + 1] += 1
    #     return histogram
    #
    # def pmap(self, func, iterable, num_processes=None):
    #     if num_processes is None:
    #         num_processes = os.cpu_count()  # Use all available CPU cores by default
    #
    #     with Pool(num_processes) as pool:
    #         return pool.map(func, iterable)
    #
    # def intensity_code_feature_map(self, img_folder_):
    #     image_paths = [os.path.join(img_folder_, cur_img) for cur_img in os.listdir(img_folder_)]
    #     histograms = self.pmap(self.process_image_ic, image_paths)  # Process images in parallel
    #
    #     num_imgs = len(image_paths)
    #     feature_mat = np.zeros(num_imgs * 26).reshape(num_imgs, 26)
    #     for count, histogram in enumerate(histograms):
    #         feature_mat[count] = histogram
    #     return feature_mat

    def manhattan_dist(self, selected_img, other_img, bin_size):
        d = 0
        for i in range(1, bin_size):
            d += abs((selected_img[i] / selected_img[0]) - (other_img[i] / other_img[0]))
        return d

    def get_distance_vector(self, selected_img, f_mat, bin_size):
        dist_vector = np.zeros(num_img, dtype=np.float64)
        for i in range(0, num_img):
            dist_vector[i] = (self.manhattan_dist(selected_img, f_mat[i], bin_size))
        return dist_vector


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


if __name__ == "__main__":
    img_folder = "images"
    result = intensity_code_histogram(img_folder)
    np.set_printoptions(suppress=True)
    print(np.array2string(result, separator=", "))

    # print("\n Now calculating the distance array")
    # dv = get_distance_vector(result[99], result, 26)
    # print(dv)
