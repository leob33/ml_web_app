from pathlib import Path
from typing import Tuple

import cv2

from src.ml_utils.tflite_object_detection import ObjectDetectionModel


def create_object_detection_model(model_directory: Path) -> ObjectDetectionModel:
    model = ObjectDetectionModel(labels_path=model_directory / Path("ml_utils/model_data/label_map.txt"),
                                 model_path=model_directory / Path("ml_utils/model_data/ssd_mobilenet_v2_5.tflite"))
    return model


def create_video_writer_and_reader(video_file_path: Path, output_directory_path: Path, prefix: str = 'prediction') \
        -> Tuple[cv2.VideoWriter, cv2.VideoCapture]:
    video_reader = cv2.VideoCapture(str(video_file_path))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_reader.get(3))
    frame_height = int(video_reader.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    if not output_directory_path.exists():
        raise NotADirectoryError
    video_writer = cv2.VideoWriter(str(output_directory_path / f'{prefix}_{video_file_path.stem}.MP4'),
                                   fourcc,
                                   fps,
                                   (frame_width, frame_height))
    return video_writer, video_reader


def make_prediction_on_video(video_file_path: Path,
                             model: ObjectDetectionModel,
                             output_directory_path: Path) -> None:
    video_writer, video_reader = create_video_writer_and_reader(video_file_path, output_directory_path)
    while True:
        _, frame = video_reader.read()
        if frame is None:
            break

        frame_with_bounding_boxes_and_score = model.predict(image=frame, filter_threshold=0.3, draw_boxes=True)
        print("prediction done")
        video_writer.write(frame_with_bounding_boxes_and_score)
    video_reader.release()
    video_writer.release()
