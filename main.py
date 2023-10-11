import gradio as gr
import os
from Algo import Algo
import numpy as np

# Get a list of all the image files in the "images" directory
image_folder = "images"
image_paths = os.listdir(image_folder)

# Filter the list to include only image files (e.g., jpg, jpeg, png)
image_paths_list = [os.path.join(image_folder, filename) for filename in image_paths if
                    filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

# GLOBAL VARS
curr_preview_index = 0
curr_algo_index = 0

with gr.Blocks() as demo:
    def update_preview_image(evt: gr.SelectData):
        global curr_preview_index
        curr_preview_index = evt.index
        # print(curr_preview_index)
        return image_paths_list[evt.index]


    def show_result_gallery():
        result_paths_list = image_paths_list
        obj = Algo(curr_algo_index, curr_preview_index)
        if curr_algo_index == 0:
            feature_matrix = Algo.intensity_code_feature_map(obj, image_folder)
        else:
            feature_matrix = Algo.color_code_feature_map(obj, image_folder)

        distance_vector = Algo.get_distance_vector(obj, feature_matrix[obj.preview_image_index], feature_matrix,
                                                   obj.bin_size[obj.algo_code])

        sorted_indices = np.argsort(distance_vector)
        sorted_images = [result_paths_list[i] for i in sorted_indices]

        return {result_gallery: gr.Gallery(visible=True, value=sorted_images)}


    def dropdown_algo_setter(evt: gr.SelectData):
        # obj = Algo(evt.index, curr_preview_index)
        global curr_algo_index
        curr_algo_index = evt.index
        print(f"algo selected ={evt.value}, preview_index = {curr_preview_index}")


    def result_img_path_setter():
        pass


    with gr.Row():
        with gr.Column():
            gallery = gr.Gallery(
                columns=10,  # Number of columns in the grid
                rows=10,  # Number of rows in the grid
                object_fit="contain",  # How images should fit within each grid cell
                value=image_paths_list,  # List of dictionaries with image data
                visible=True,
                allow_preview=False,
                label="Select an image"
            )

        with gr.Column():
            gr.HTML("<center><h1>Preview</h1></center>")
            preview_image = gr.Image(value="images/1.jpg", show_label=False, label="Preview Image", width=400,
                                     height=270)
            methods = ["Intensity", "Color code"]
            dropdown = gr.Dropdown(choices=methods, visible=True, label="Select Method")
            button = gr.Button("Run")

    gr.HTML("'<hr>'")
    with gr.Row():
        with gr.Column():
            gr.HTML("<center><h1>Result View</h1></center>")
            result_gallery = gr.Gallery(
                columns=10,  # Number of columns in the grid
                rows=10,  # Number of rows in the grid
                object_fit="contain",  # How images should fit within each grid cell
                value=image_paths_list,  # List of dictionaries with image data
                visible=False,
                min_width=600,
                label="Result List",
                interactive="True"
            )

    gallery.select(update_preview_image, None, preview_image)
    dropdown.select(dropdown_algo_setter)

    button.click(show_result_gallery, None, result_gallery)

if __name__ == "__main__":
    demo.launch()
