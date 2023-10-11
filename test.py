import cv2

from Algo import Algo
import numpy as np
import os

if __name__ == "__main__":
    img_folder_ = "images"
    obj = Algo(0, 1)
    result = Algo.intensity_code_feature_map(obj, img_folder_)
    np.set_printoptions(suppress=True)
    print(np.array2string(result, separator=", "))

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
