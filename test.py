from Algo import Algo
import numpy as np

if __name__ == "__main__":
    img_folder_ = "images"
    obj = Algo("Intensity code", 99)
    result = Algo.intensity_code_feature_map(obj, img_folder_)
    np.set_printoptions(suppress=True)
    print(np.array2string(result, separator=", "))

    print("\n Now calculating the distance array")
    dv = Algo.get_distance_vector(obj, result[obj.preview_image_index], result, 26)
    print(dv)
