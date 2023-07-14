import os
import cv2
import argparse
import insightface
from moviepy.editor import VideoFileClip

from roop2_config import INSWAPPER_ONNX_FILE, TEMP_VIDEO_FILE

PROVIDERS_LIST = ["CUDAExecutionProvider", "CPUExecutionProvider"]

FACE_ANALYSER = insightface.app.FaceAnalysis(
    name='buffalo_l', providers=PROVIDERS_LIST)
FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))

FACE_SWAPPER = insightface.model_zoo.get_model(INSWAPPER_ONNX_FILE, providers=PROVIDERS_LIST)


def get_one_face(frame):
    faces = FACE_ANALYSER.get(frame)
    if len(faces) == 0:
        return None
    return faces[0]


def swap_face(input_face, frame_face, frame):
    return FACE_SWAPPER.get(frame, frame_face, input_face, paste_back=True)


def process_video(input_image_file, input_video_file, output_video_file):
    input_face = get_one_face(cv2.imread(input_image_file))
    if input_face == None:
        print("No face in image")
        return

    video = cv2.VideoCapture(input_video_file)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    output_video = cv2.VideoWriter(
        output_video_file, fourcc, fps, (width, height))

    i_frame = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_face = get_one_face(frame)
        if frame_face == None:
            output_frame = frame
        else:
            output_frame = swap_face(input_face, frame_face, frame)

        output_video.write(output_frame)

        i_frame += 1
        if i_frame % 10 == 0:
            print(f"{i_frame} / {total_frames}")

    video.release()
    output_video.release()


def copy_audio(source_video_file, target_video_file, output_video_file):
    print(f"[copy_audio] source video: {source_video_file}")
    print(f"[copy_audio] target video: {target_video_file}")
    print(f"[copy_audio] output video: {output_video_file}")
    video1 = VideoFileClip(source_video_file)
    video2 = VideoFileClip(target_video_file)

    audio = video1.audio

    video2 = video2.set_audio(audio)

    video2.write_videofile(output_video_file)


def main(input_image_file, input_video_file, output_video_file):
    process_video(input_image_file, input_video_file, TEMP_VIDEO_FILE)
    copy_audio(input_video_file, TEMP_VIDEO_FILE, output_video_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Roop2')

    parser.add_argument('--input-image', help='path to the input image file')
    parser.add_argument('--input-video', help='path to the input video file')
    parser.add_argument('--output-video', help='path to the output video file')

    args = parser.parse_args()

    main(args.input_image, args.input_video, args.output_video)