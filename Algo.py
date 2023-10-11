import cv2
import numpy as np
import os

# calculate number of images in a folder
num_img = 0
for im in os.listdir("images"):
    num_img += 1


class Algo:

    def __init__(self, algo_code, preview_image_index):
        self.num_images = num_img
        self.preview_image_index = preview_image_index
        self.algo_code = algo_code
        self.bin_size = [26, 65]

    def color_code_feature_matrix(self, img_folder):
        global num_img
        img_paths = os.listdir(img_folder)
        feature_mat = np.zeros((len(img_paths), 65), dtype=np.int32)

        for count in range(len(img_paths)):
            cur_img = img_paths[count]
            image = cv2.imread(os.path.join(img_folder, cur_img))

            # Storing 2 most significant bits from r,g,b channels (opencv follows BGR hence B=0,G=1,R=2 index)
            r_channel = image[:, :, 2] >> 6
            g_channel = image[:, :, 1] >> 6
            b_channel = image[:, :, 0] >> 6

            # concatenating the values to get 6-bit color code
            color_code = (r_channel << 4) | (g_channel << 2) | b_channel

            # using np.bincount, to create 64 bins corresponding to each value in 6 bit color code
            hist = np.bincount(color_code.ravel(), minlength=64).astype(np.int32)

            # storing the image size in 0th index (as instructed by Prof. Min Chen), and concat the rest of hist to get final histogram
            histogram = [image.size] + list(hist)
            feature_mat[count] = histogram

        return feature_mat

    def intensity_code_feature_matrix(self, img_folder):
        img_paths = os.listdir(img_folder)
        feature_mat = np.zeros((len(img_paths), 26), dtype=np.int32)

        for count in range(len(img_paths)):
            cur_img = img_paths[count]
            image = cv2.imread(os.path.join(img_folder, cur_img))

            # Extract the R, G, and B channels
            b_channel = image[:, :, 0]
            g_channel = image[:, :, 1]
            r_channel = image[:, :, 2]

            # calculating intensity acc to the provided formula
            intensity = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel

            # initially dividing into bins of edge 10 ranging from 0 to 260
            hist, bin_edges = np.histogram(intensity, bins=26, range=(0, 260), density=False)

            # Since max range is actually 255, adding all the values of 26th bin to the 25th - adjust for uneven bin size (240,255)
            hist[24] += hist[25]

            # 0th index - image size, concat it with rest of the hist to get final histogram
            histogram = [image.size] + list(hist.astype(np.int64)[:25])
            feature_mat[count] = histogram

        return feature_mat

    # calculates manhattan dist between 2 images
    def manhattan_dist(self, selected_img, other_img, bin_size):
        d = 0
        for i in range(1, bin_size):
            d += abs((selected_img[i] / selected_img[0]) - (other_img[i] / other_img[0]))
        return d

    # goes through the feature matrix & returns a distance vector containing the distance of selected image from all other images
    def get_distance_vector(self, selected_img, f_mat, bin_size):
        dist_vector = np.zeros(num_img, dtype=np.float64)
        for i in range(0, num_img):
            dist_vector[i] = (self.manhattan_dist(selected_img, f_mat[i], bin_size))
        return dist_vector

# parallellized functions
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
