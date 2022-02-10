import glob

import tensorflow as tf

from tf_object_detection import read_image_np_array, run_inference_on_image
from viz_utils import generate_human_readable_results


def test_validity_of_humain_readable_detection_results():
    # Given
    batch_of_images = glob.glob('minimal_test_sample/*.jpg')
    detect_fn = tf.saved_model.load('/Users/leo.babonnaud/lab/ml_web_app/src/saved_model')
    labels = ['Chevreuil Européen', 'Renard roux', 'Martre des pins', "Sanglier d'Eurasie"]

    # When
    human_readable_detection_results = []
    for image_path in batch_of_images:
        image_np = read_image_np_array(image_path=image_path)
        detections = run_inference_on_image(image=image_np, detection_fn=detect_fn)
        human_readable_detection_results.extend(
            generate_human_readable_results(results_of_prediction=detections, labels=labels))

    # Then
    expected_human_readable_detection_results = [{'Martre des pins': 0.74},
                                                 {'Renard roux': 0.96},
                                                 {'Chevreuil Européen': 0.93},
                                                 {"Sanglier d'Eurasie": 0.84}]

    assert human_readable_detection_results == expected_human_readable_detection_results