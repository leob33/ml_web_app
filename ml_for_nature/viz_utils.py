from typing import List, Dict, Any

import cv2
import numpy as np


def generate_human_readable_results(results_of_prediction: List[Dict[str, Any]],
                                    labels) \
        -> list[dict[Any, float]]:
    info = []
    if len(results_of_prediction) > 0:
        for result in results_of_prediction:
            dic = {labels[int(result["class_id"] - 1)]: round(float(result["score"]), 2)}
            info.append(dic)
    return info


def annotate_raw_image_with_prediction_results(
        results_of_prediction: List[Dict[str, Any]],
        image: np.ndarray,
        labels: list,
        text_size: float = 0.45,
) \
        -> np.ndarray:
    if len(results_of_prediction) > 0:

        for result in results_of_prediction:
            xmin, ymin, xmax, ymax = _get_bounding_boxes_coordinates(image, result)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            _add_class_label_to_image(image, result, xmin, ymin, text_size, labels=labels)

    return image


def _get_bounding_boxes_coordinates(image, result_of_prediction) -> tuple:
    imh, imw = image.shape[0:-1]

    ymin = int(max(1, (result_of_prediction["bounding_box"][0] * imh)))
    xmin = int(max(1, (result_of_prediction["bounding_box"][1] * imw)))
    ymax = int(min(imh, (result_of_prediction["bounding_box"][2] * imh)))
    xmax = int(min(imw, (result_of_prediction["bounding_box"][3] * imw)))

    return xmin, ymin, xmax, ymax


def _add_class_label_to_image(image, results_of_prediction, xmin, ymin, text_size, labels) -> None:
    object_name = labels[int(results_of_prediction["class_id"] - 1)]
    label = f'{object_name}: {int(results_of_prediction["score"] * 100)}%'
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2)
    label_ymin = max(ymin, label_size[1] + 10)
    cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10),
                  (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                  cv2.FILLED)
    cv2.putText(image, label, (xmin, label_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 2)
