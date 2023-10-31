import gradio as gr
import os
from Algo import Algo
import numpy as np

# GLOBAL VARS
image_folder = "images"
image_paths = os.listdir(image_folder)

image_paths_list = [os.path.join(image_folder, filename) for filename in image_paths if
                    filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

captions = [i[7:0] for i in image_paths_list]

curr_preview_index = 0
curr_algo_index = None
curr_sorted_list = image_paths_list
curr_cb_index = 0
rf_iteration = 0
box = []

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # functions for event listeners - a1
    def update_preview_image(evt: gr.SelectData):
        global curr_preview_index
        curr_preview_index = evt.index
        rf_reset()
        # print(curr_preview_index)
        return image_paths_list[evt.index]


    def show_result_gallery():
        global curr_sorted_list
        result_paths_list = image_paths_list

        # creating an object pf Algo class
        obj = Algo(curr_algo_index, curr_preview_index)

        # calling appropriate feature_matrix function based on selected algorithm
        if curr_algo_index == 0:
            feature_matrix = Algo.intensity_code_feature_matrix(obj, image_folder)
        elif curr_algo_index == 1:
            feature_matrix = Algo.color_code_feature_matrix(obj, image_folder)
        else:
            feature_matrix = Algo.icc_feature_matrix(obj, image_folder)

        distance_vector = Algo.get_distance_vector(obj, feature_matrix[obj.preview_image_index], feature_matrix,
                                                   obj.bin_size[obj.algo_code])

        # using argsort to get the indices of sorted array
        sorted_indices = np.argsort(distance_vector)

        # using the indices to create a result list, without losing the correct image path
        sorted_images = [result_paths_list[i] for i in sorted_indices]
        curr_sorted_list = sorted_images
        print(f"algo selected ={curr_algo_index}, preview_index = {curr_preview_index}")
        return {result_gallery: gr.Gallery(visible=True, value=sorted_images)}
        # return {rf_grid: gr.Column(visible=True)}


    def dropdown_algo_setter(evt: gr.SelectData):
        global curr_algo_index
        curr_algo_index = evt.index


    # functions for Relevance feedback algo
    def rf_reset():
        global rf_iteration
        rf_iteration = 0


    def rf_itr_increment():
        global rf_iteration
        rf_iteration += 1


    def rf_inputs(*checkboxes):
        global box
        box = checkboxes
        print(f"checkbox: {checkboxes}")


    def rf_cb_reset(*checkboxes):
        reset_cb = [False] * 100
        checkboxes = tuple(reset_cb)
        return checkboxes


    def show_rf_result_grid(*rf_checkboxes):
        global curr_sorted_list
        rf_images_list_ = []
        if rf_iteration == 0:
            weights = [1 / 89] * 89
            rf_obj = Algo(2, curr_preview_index)
            feature_matrix = Algo.normalized_icc_feature_matrix(rf_obj, image_folder)
            distance_vector = Algo.get_norm_distance_vector(rf_obj, feature_matrix[rf_obj.preview_image_index],
                                                            feature_matrix, 89,
                                                            weights)

            # using argsort to get the indices of sorted array
            sorted_indices = np.argsort(distance_vector)
            curr_sorted_list = [image_paths_list[i] for i in sorted_indices]
            for i in curr_sorted_list:
                rf_images_list_.append(gr.Image(value=i, label=i[7:]))
            # print(curr_sorted_list)

        elif rf_iteration >= 1:
            rf_obj = Algo(2, curr_preview_index)
            feature_matrix = Algo.normalized_icc_feature_matrix(rf_obj, image_folder)
            weights = Algo.get_weights(rf_obj, rf_checkboxes, feature_matrix[rf_obj.preview_image_index],
                                       curr_sorted_list, image_folder)
            distance_vector = Algo.get_norm_distance_vector(rf_obj, feature_matrix[rf_obj.preview_image_index],
                                                            feature_matrix, 89, weights)
            sorted_indices = np.argsort(distance_vector)
            curr_sorted_list = [image_paths_list[i] for i in sorted_indices]
            for i in curr_sorted_list:
                rf_images_list_.append(gr.Image(value=i, label=i[7:]))

        rf_itr_increment()
        print(f"rf iteration: {rf_iteration}")
        return rf_images_list_


    def make_rf_result_visible():
        return {rf_result: gr.Column(visible=True)}


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
            methods = ["Intensity", "Color code", "Intensity + Color code"]
            dropdown = gr.Dropdown(choices=methods, visible=True, label="Select Method")
            button = gr.Button("Run")
            rf_button = gr.Button("Run RF")

    # GUI - Result view
    gr.HTML("<hr>")
    with gr.Row():
        with gr.Column(variant='panel'):
            # gr.HTML("<center><h2>Result View</h2></center>")
            result_gallery = gr.Gallery(
                columns=10,
                rows=10,
                object_fit="contain",
                value=image_paths_list,
                visible=False,
                min_width=600,
                label="Result List (without RF)",
                interactive="True",
            )

    # GUI - RF Result view
    gr.HTML("<hr>")
    rf_result = gr.Column(visible=False, variant='panel')
    with rf_result:
        gr.HTML("<center><h2>RF Result View</h2></center>")
        cols = 5
        rows = 20
        count = 0
        rf_checkbox = []
        rf_images_list = []
        for i in range(0, rows):
            with gr.Row(equal_height=True):
                for j in range(0, cols):
                    with gr.Group():
                        rf_images_list.append(
                            gr.Image(value=curr_sorted_list[count], label=image_paths_list[count][7:]))
                        rf_checkbox.append(gr.Checkbox(label=str(count), info="Relevant"))
                        count += 1

    # Event listeners for preview_gallery, run button & dropdown menu
    gallery.select(fn=update_preview_image, inputs=None, outputs=preview_image)
    dropdown.select(dropdown_algo_setter)
    button.click(fn=show_result_gallery, inputs=None, outputs=result_gallery)
    rf_button.click(fn=make_rf_result_visible, inputs=None, outputs=rf_result)
    rf_button.click(fn=rf_inputs, inputs=rf_checkbox, outputs=None)
    rf_button.click(fn=show_rf_result_grid, inputs=rf_checkbox, outputs=rf_images_list)
    rf_button.click(fn=rf_cb_reset, inputs=rf_checkbox, outputs=rf_checkbox)

if __name__ == "__main__":
    demo.launch()
