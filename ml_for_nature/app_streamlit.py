import glob
import random
from typing import List, Dict

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

from tf_object_detection import read_image_np_array, run_inference_on_image
from viz_utils import generate_human_readable_results, annotate_raw_image_with_prediction_results


def _display_one_prediction(prediction: Dict[str, float]):
    for key, value in prediction.items():
        return f"I see a {key} with {value*100}% confidence"


def display_predictions(predictions: List[Dict[str, float]]) -> None:
    if not predictions:
        st.write('I cannot detect anything here...')
        return
    st.write(" ".join(map(_display_one_prediction, predictions)))


@st.cache(allow_output_mutation=True)
def load_model():
    return tf.saved_model.load(MODEL_PATH)


IMAGE_PATH = 'ml_for_nature/test_images/*.jpg'
MODEL_PATH = 'ml_for_nature/saved_model'
LABELS = ['Chevreuil Européen', 'Renard roux', 'Martre des pins', "Sanglier d'Eurasie"]

batch_of_images = glob.glob(IMAGE_PATH)
detect_fn = load_model()

st.title("Prediction on Nature !")
image_file_buffer = st.file_uploader("Try it with your own image")

if image_file_buffer:
    image_np = np.array(Image.open(image_file_buffer), dtype=np.uint8)

else:
    an_image_path = random.choice(batch_of_images)
    image_np = read_image_np_array(image_path=an_image_path)

detections = run_inference_on_image(image=image_np, detection_fn=detect_fn, filter_threshold=0.6)

clicked = st.sidebar.checkbox('Show me where is the animal')
if clicked:
    image_with_annotations = annotate_raw_image_with_prediction_results(results_of_prediction=detections,
                                                                        image=image_np.copy(),
                                                                        labels=LABELS,
                                                                        text_size=0.9)
    st.image(image_with_annotations, use_column_width=True)

else:
    st.image(image_np, use_column_width=True)
    st.subheader("Predictions 👇")
    prediction_result = generate_human_readable_results(results_of_prediction=detections, labels=LABELS)
    display_predictions(prediction_result)

st.button("👉 Shuffle & predict 👈")
