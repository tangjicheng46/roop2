import cv2
import insightface

PROVIDERS_LIST = ['CUDAExecutionProvider', 'CPUExecutionProvider']

FACE_ANALYSER = insightface.app.FaceAnalysis(
    name='buffalo_l', providers=PROVIDERS_LIST)
FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))

root_path = "/Users/tangjicheng/roop2/"
input_image = root_path + "image/song1.jpg"
input_video = root_path + "video/1.mp4"
output_video = root_path + "test1.mp4"
swap_face_model_path = root_path + "models/inswapper_128.onnx"
FACE_SWAPPER = insightface.model_zoo.get_model(
    swap_face_model_path, providers=PROVIDERS_LIST)


def get_one_face(frame):
    faces = FACE_ANALYSER.get(frame)
    if len(faces) == 0:
        return None
    return faces[0]


def swap_face(input_face, frame_face, frame):
    return FACE_SWAPPER.get(frame, frame_face, input_face, paste_back=True)


def process_video(input_face, input_file, output_file):
    video = cv2.VideoCapture(input_file)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    output = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

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

        output.write(output_frame)

        i_frame += 1
        if i_frame % 100 == 0:
            print(f"{i_frame} / {total_frames}")

    video.release()
    output.release()


def main(image_file, video_file, output_file):
    input_face = get_one_face(cv2.imread(image_file))
    if input_face == None:
        print("No face in image")
        return
    process_video(input_face, video_file, output_file)


if __name__ == "__main__":
    main(input_image,
         input_video, output_video)
