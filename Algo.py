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
        self.bin_size = [26, 65, 90]
        self.rf_inputs = []

    def normalized_icc_feature_matrix(self, img_folder):
        img_paths = os.listdir(img_folder)
        histogram_icc = self.icc_feature_matrix(img_folder)
        feature_mat = []

        # generating features from histogram
        print(f"histogram shape: {histogram_icc.shape}")
        for i in range(0, histogram_icc.shape[0]):
            feature_row = []
            for j in range(0, histogram_icc.shape[1]):
                feature_row.append(histogram_icc[i][j] / 98304)
                # print(feature_row)
            feature_mat.append(feature_row)

        feature_mat = np.array(feature_mat)
        avg_arr = []
        std_arr = []
        for j in range(0, feature_mat.shape[1]):
            std = np.std(feature_mat[:, j], axis=0)
            avg = np.mean(feature_mat[:, j], axis=0)
            std_arr.append(std)
            avg_arr.append(avg)
        std_arr = np.array(std_arr)
        avg_arr = np.array(avg_arr)
        print(f"feature_matrix shape = {feature_mat.shape}")
        # generating norm_feature matrix from features
        normalized_feature_mat = []
        print(f"shapes: mean_arr: {avg_arr.shape}, std_arr:{std_arr.shape}")
        print(f"mean_arr[1]: {avg_arr}")
        print(f"std_arr[1]: {std_arr}")

        for i in range(0, feature_mat.shape[0]):
            norm_row = []
            for j in range(0, feature_mat.shape[1]):
                if std_arr[j] == 0:
                    norm_row.append(feature_mat[i][j])
                else:
                    norm_row.append((feature_mat[i][j] - avg_arr[j]) / std_arr[j])
            # print(norm_row)
            normalized_feature_mat.append(norm_row)
        # print(normalized_feature_mat)
        return normalized_feature_mat

    def icc_feature_matrix(self, img_folder):
        img_paths = os.listdir(img_folder)
        ic_mat = self.intensity_code_feature_matrix(img_folder)
        cc_mat = self.color_code_feature_matrix(img_folder)
        feature_mat = np.concatenate((ic_mat[:, 1:], (cc_mat[:, 1:])), axis=1)
        feature_mat = np.array(feature_mat)
        # print(normalized_feature_mat.shape)

        return feature_mat

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

    def manhattan_dist_norm(self, selected_img, other_img, bin_size, weights):
        d = 0
        for i in range(0, bin_size):
            d += abs(selected_img[i] - other_img[i]) * weights[i]
        return d

    def get_norm_distance_vector(self, selected_img, f_mat, bin_size, weights):
        norm_dist_vector = np.zeros(num_img, dtype=np.float64)
        for i in range(0, num_img):
            norm_dist_vector[i] = self.manhattan_dist_norm(selected_img, f_mat[i], bin_size, weights)
        return norm_dist_vector

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
    def fetch_indices(self, selected_img, img_folder, selected_img_paths):

        img_paths = os.listdir(img_folder)
        img_paths_list = [os.path.join(img_folder, filename) for filename in img_paths if
                          filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
        org_indices = []
        for i in range(0, 100):
            for j in range(0, len(selected_img_paths)):
                if img_paths_list[i] == selected_img_paths[j]:
                    org_indices.append(i)
                elif img_paths_list[i] == selected_img:
                    org_indices.append(i)
        ret = np.unique(org_indices)
        return ret

    def get_weights(self, rf_inputs, selected_img, curr_img_list, img_folder):
        selected_img_paths = []
        for i in range(0, 100):
            if rf_inputs[i]:
                selected_img_paths.append(curr_img_list[i])
        norm_f_mat = self.normalized_icc_feature_matrix(img_folder)
        indices = self.fetch_indices(selected_img, img_folder, selected_img_paths)
        # print(indices)
        num_selected = len(selected_img_paths)
        # get features of selected imgs from the norm_f_mat
        features_to_update = np.array([norm_f_mat[i] for i in indices])
        # print(f"shape of features:{features_to_update.shape}")
        # print(features_to_update)
        std_of_features = []
        # zero_std_value = False
        zero_std_cols = []
        zero_std_indices = []
        non_zero_std_indices = []

        for j in range(0, 89):
            std = np.std(features_to_update[:, j], axis=0)
            if std == 0:
                # zero_std_value = True
                zero_std_indices.append(j)
            else:
                non_zero_std_indices.append(j)
            std_of_features.append(std)

        # print(f"std array={std_of_features}")

        updated_weights = [0] * 89
        # print(updated_weights)

        # if not zero_std_value:
        #     for i in range(0, 89):
        #         print("std_not_zero")
        #         updated_weights[i] = (1 / std_of_features[i])
        # # edge cases in std (where std == 0)
        # else:
        #     print("std_zero")
        #     for col in zero_std_cols:
        #         mean_col = np.mean(features_to_update[:, col], axis=0)
        #         if mean_col == 0.00:
        #             updated_weights[col] = 0
        #         elif mean_col != 0:
        #             non_zero_features = [std_of_features[j] for j in non_zero_std_indices]
        #             updated_weights[col] = min(non_zero_features)/2
        # print(f"updated w : {updated_weights}")

        for i in non_zero_std_indices:
            # print("non_zero_std")
            updated_weights[i] = (1 / std_of_features[i])

        for i in zero_std_indices:
            # print("zero_std")
            mean_col = np.mean(features_to_update[:, i], axis=0)
            if mean_col == 0:
                updated_weights[i] = 0
            elif mean_col != 0:
                non_zero_features = [std_of_features[j] for j in non_zero_std_indices]
                updated_weights[i] = min(non_zero_features) / 2

        total_w = sum(updated_weights)
        norm_weights = [(updated_w / total_w) for updated_w in updated_weights]

        return norm_weights
