import cv2

from Algo import Algo
import numpy as np
import os

if __name__ == "__main__":
    img_folder_ = "images"
    obj = Algo(2, 0)
    norm_f_mat = Algo.normalized_icc_feature_matrix(obj, img_folder_)
    norm_f_mat = np.array(norm_f_mat)

    dv = Algo.get_norm_distance_vector(obj, norm_f_mat[obj.preview_image_index], norm_f_mat, 89, None)
    # np.set_printoptions(suppress=False)
    # print(np.array2string(dv, separator=", "))
    print(dv)

    # print("\n Now calculating the distance array")
    # dv = Algo.get_distance_vector(obj, result[obj.preview_image_index], result, obj.bin_size[obj.algo_code])
    # print(dv)
    # image_paths = os.listdir(img_folder_)
    #
    # # Filter the list to include only image files (e.g., jpg, jpeg, png)
    # image_paths_list = [os.path.join(img_folder_, filename) for filename in image_paths if
    #                     filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
    #
    # # Step 1: Get the sorted indices
    # sorted_indices = np.argsort(dv)
    # print(sorted_indices)
    # # Step 2: Rearrange the image_gallery based on sorted indices
    # sorted_images = [image_paths_list[i] for i in sorted_indices]
    #
    # for image in sorted_images:
    #     print(image)
