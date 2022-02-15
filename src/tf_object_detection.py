from typing import Dict, List

import numpy as np

import tensorflow as tf
from PIL import Image


def read_image_np_array(image_path: str):
    return np.array(Image.open(image_path))


def read_image_as_tensor(image_np: np.ndarray):
    input_tensor = tf.convert_to_tensor(image_np)
    return input_tensor[tf.newaxis, ...]


def convert_detections_into_np_array_with_correct_dtype(detections: Dict[str, tf.Tensor]) \
        -> [Dict[str, np.ndarray]]:
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in
                  detections.items()}  # transform tensors to np
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections


def convert_detections_into_a_list_of_dict(detections: Dict[str, np.ndarray], filter_threshold: float) -> List[
    Dict[str, float]]:
    results = []
    for i in range(detections['num_detections']):
        if detections['detection_scores'][i] >= filter_threshold:
            result = {
                'bounding_box': detections['detection_boxes'][i],
                'class_id': detections['detection_classes'][i],
                'score': detections['detection_scores'][i]
            }
            results.append(result)

    return results


def run_inference_on_image(image, detection_fn, filter_threshold):
    input_tensor = read_image_as_tensor(image_np=image)
    detections = detection_fn(input_tensor)
    detections = convert_detections_into_np_array_with_correct_dtype(detections=detections)
    detections = convert_detections_into_a_list_of_dict(detections=detections, filter_threshold=filter_threshold)
    return detections