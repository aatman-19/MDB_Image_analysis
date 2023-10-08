import gradio as gr
import os

# Get a list of all the image files in the "images" directory
image_folder = "images"
image_paths = os.listdir(image_folder)

# Filter the list to include only image files (e.g., jpg, jpeg, png)
image_paths_list = [os.path.join(image_folder, filename) for filename in image_paths if
                    filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

with gr.Blocks() as demo:

    def update_preview_image(evt: gr.SelectData):
        return image_paths_list[evt.index]

    def show_result_gallery():
        return {result_gallery: gr.Gallery(visible=True)}

    with gr.Row():
        with gr.Column():
            gallery = gr.Gallery(
                columns=10,  # Number of columns in the grid
                rows=10,  # Number of rows in the grid
                object_fit="contain",  # How images should fit within each grid cell
                value=image_paths_list,  # List of dictionaries with image data
                visible=True,
                allow_preview=False,
                label = "Select an image"
            )

        with gr.Column():
            gr.HTML("<center><h1>Preview</h1></center>")
            preview_image = gr.Image(value="images/1.jpg",show_label=False, label="Preview Image",width = 400, height = 270)
            Algo = {"Intensity", "Color code"}
            gr.Dropdown(choices=Algo, visible=True, label="Select Method")
            button = gr.Button("Run")

    gr.HTML("'<hr>'")
    with gr.Row():
        with gr.Column():
            gr.HTML("<center><h1>Result View</h1></center>")
            result_gallery = gr.Gallery(
                columns=5,  # Number of columns in the grid
                rows=10,  # Number of rows in the grid
                object_fit="contain",  # How images should fit within each grid cell
                value=image_paths_list,  # List of dictionaries with image data
                visible=False,
                min_width=600,
                label = "Result List",
                interactive="True"
            )

    gallery.select(update_preview_image,None,preview_image)
    button.click(show_result_gallery,None,result_gallery)


if __name__ == "__main__":
    demo.launch()


