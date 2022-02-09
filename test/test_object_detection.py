from pathlib import Path

from src.ml_utils.tflite_object_detection import ObjectDetectionModel
import cv2


def test_object_detection_model_on_video_return_list_of_dic_with_prediction_score_for_each_label():
    video = "data/test/2020-10-28_Sus_scrofa_03.MP4"
    model = ObjectDetectionModel(labels_path=Path("data/label_map.txt"),
                                 model_path=Path("data/ssd_mobilenet_v2_5.tflite"))

    vs = cv2.VideoCapture(str(video))
    while True:
        _, frame = vs.read()
        if frame is None:
            break
        results = model.predict(image=frame, filter_threshold=0.05, draw_boxes=False)
        print(results)


def test_object_detection_model_on_video_return_another_video_with_labels_drawn():
    video = Path("data/video_sample/2020-10-28_Sus_scrofa_03.MP4")
    model = ObjectDetectionModel(labels_path=Path("data/model_data/label_map.txt"),
                                 model_path=Path("data/model_data/ssd_mobilenet_v2_5.tflite"))

    vs = cv2.VideoCapture(str(video))
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))
    out = cv2.VideoWriter(f'data/model_output/prediction_{video.stem}.mp4',
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                          10,
                          (frame_width, frame_height))
    while True:
        _, frame = vs.read()
        if frame is None:
            break
        frame_with_bounding_boxes_and_score = model.predict(image=frame, filter_threshold=0.3, draw_boxes=True)
        out.write(frame_with_bounding_boxes_and_score)
        out.release()
