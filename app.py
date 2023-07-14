import gradio as gr
import roop2

OUTPUT_VIDEO = "/app/output.mp4"

def process_video(input_image_file, input_video_file):
    roop2.main(input_image_file, input_video_file, OUTPUT_VIDEO)
    return OUTPUT_VIDEO


inputs = [
    gr.inputs.Image(type="filepath", label="Input face image"),
    gr.inputs.Video(type="mp4", label="Input video")
]

output = gr.outputs.Video(type="mp4", label="Output video")

app = gr.Interface(fn=process_video, inputs=inputs,
                   outputs=output, title="Faceswap App", server_timeout=6000)

app.launch(share=True)
