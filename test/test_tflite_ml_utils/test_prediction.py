from pathlib import Path

from tflite_ml_app.video_prediction import create_object_detection_model, create_video_writer_and_reader


def test_video_is_properly_generated():
    video_file_path = Path('data/video_sample/2020-10-28_Sus_scrofa_03.MP4')
    video_writer, video_reader = create_video_writer_and_reader(video_file_path=video_file_path,
                                                                output_directory_path=Path('test_output'),
                                                                prefix='identical')

    while True:
        _, frame = video_reader.read()
        if frame is None:
            break
        video_writer.write(frame)
    video_writer.release()
    video_reader.release()


def test_prediction_video_is_properly_generated():
    video_file_path = Path('data/video_sample/2020-10-28_Sus_scrofa_03.MP4')
    video_writer, video_reader = create_video_writer_and_reader(video_file_path=video_file_path,
                                                                output_directory_path=Path('test_output'),
                                                                prefix='prediction')
    model = create_object_detection_model(model_directory=Path('/tflite_ml_app'
                                                               '/model_data'))

    while True:
        _, frame = video_reader.read()
        if frame is None:
            break
        frame = model.predict(frame, 0.4, True)
        video_writer.write(frame)
    video_writer.release()
    video_reader.release()

