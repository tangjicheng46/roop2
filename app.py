import gradio as gr
import cv2
from roop.core import run_with_param

gradio_temp_path = "~/temp"
gradio_face = "~/temp/face.jpg"
gradio_input_video = "~/temp/input.mp4"
gradio_output_video = "~/temp/output.mp4"


def process_video(face_image, input_video):
    run_with_param(face_image, input_video, gradio_output_video)
    return gradio_output_video


inputs = [
    gr.inputs.Image(type="filepath", label="Input face image"),
    gr.inputs.Video(type="mp4", label="Input video")
]

output = gr.outputs.Video(type="mp4", label="Output video")

app = gr.Interface(fn=process_video, inputs=inputs,
                   outputs=output, title="Faceswap App")

app.launch(share=True)
