import gradio as gr
import os
from Algo import Algo
import numpy as np

# GLOBAL VARS
image_folder = "images"
image_paths = os.listdir(image_folder)

image_paths_list = [os.path.join(image_folder, filename) for filename in image_paths if
                    filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

curr_preview_index = 0
curr_algo_index = 0

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # functions for event listeners
    def update_preview_image(evt: gr.SelectData):
        global curr_preview_index
        curr_preview_index = evt.index
        # print(curr_preview_index)
        return image_paths_list[evt.index]


    def show_result_gallery():
        result_paths_list = image_paths_list

        # creating an object pf Algo class
        obj = Algo(curr_algo_index, curr_preview_index)

        # calling appropriate feature_matrix function based on selected algorithm
        if curr_algo_index == 0:
            feature_matrix = Algo.intensity_code_feature_matrix(obj, image_folder)
        else:
            feature_matrix = Algo.color_code_feature_matrix(obj, image_folder)

        distance_vector = Algo.get_distance_vector(obj, feature_matrix[obj.preview_image_index], feature_matrix,
                                                   obj.bin_size[obj.algo_code])

        # using argsort to get the indices of sorted array
        sorted_indices = np.argsort(distance_vector)

        # using the indices to create a result list, without losing the correct image path
        sorted_images = [result_paths_list[i] for i in sorted_indices]
        print(f"algo selected ={curr_algo_index}, preview_index = {curr_preview_index}")
        return {result_gallery: gr.Gallery(visible=True, value=sorted_images)}

    def dropdown_algo_setter(evt: gr.SelectData):
        global curr_algo_index
        curr_algo_index = evt.index


    # GUI
    with gr.Row():
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                columns=10,
                rows=10,
                object_fit="contain",
                value=image_paths_list,
                visible=True,
                allow_preview=False,
                label="Select an image"
            )

        with gr.Column(variant='panel'):
            gr.HTML("<center><h1>Preview</h1></center>")
            preview_image = gr.Image(value="images/1.jpg", show_label=False, label="Preview Image", width=300)
            methods = ["Intensity", "Color code"]
            dropdown = gr.Dropdown(choices=methods, visible=True, label="Select Method")
            button = gr.Button("Run")

    # GUI - Result view
    gr.HTML("<hr>")
    with gr.Row():
        with gr.Column(variant='panel'):
            gr.HTML("<center><h2>Result View</h2></center>")
            result_gallery = gr.Gallery(
                columns=10,
                rows=10,
                object_fit="contain",
                value=image_paths_list,
                visible=False,
                min_width=600,
                label="Result List",
                interactive="True"
            )

    # Event listeners for preview_gallery, run button & dropdown menu
    gallery.select(update_preview_image, None, preview_image)
    dropdown.select(dropdown_algo_setter)
    button.click(show_result_gallery, None, result_gallery)

if __name__ == "__main__":
    demo.launch()
