import cv2

from Algo import Algo
import numpy as np
import os

if __name__ == "__main__":
    img_folder_ = "images"
    image_paths = os.listdir(img_folder_)
    image_paths_list = [os.path.join(img_folder_, filename) for filename in image_paths if
                        filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
    obj = Algo(2, 0)
    norm_f_mat = Algo.normalized_icc_feature_matrix(obj, img_folder_)
    rf_inputs = [False] * 100
    true = [0, 1, 3, 4, 7, 8, 12, 13, 17]
    for i in true:
        rf_inputs[i] = True
    n_weights = Algo.get_weights(obj, rf_inputs, image_paths_list[obj.preview_image_index], image_paths_list,
                                 img_folder_)
    icc_f_mat = Algo.icc_feature_matrix(obj, img_folder_)
    # print(n_weights)
    norm_f_mat = np.array(norm_f_mat)
    print(f"image-1 path: {image_paths_list[obj.preview_image_index]}")
    print(f"image-1 features: {norm_f_mat[0]}")

    print(norm_f_mat.shape)
    # print(icc_f_mat)
    dv = Algo.get_norm_distance_vector(obj, norm_f_mat[obj.preview_image_index], norm_f_mat, 89, n_weights)
    # np.set_printoptions(suppress=False)
    # print(np.array2string(dv, separator=", "))
    print(dv)
