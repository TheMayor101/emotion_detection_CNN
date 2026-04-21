# This file is responsible for the non-GUI functions inside GUI.py

# TODO: change file name

import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import numpy.typing as npt

# Path to the trained CNN model
path_to_model: str = "model/emotion_recognition_model.h5"

model: tf.keras.Model = tf.keras.models.load_model(path_to_model)


def preprocess_image(
    img_path: str,
) -> npt.NDArray[np.float32]:
    """
    Preprocess the image so the model will be able to work with it.
    Converts the image to an array, expands dimensions to fit the model
    input shape, and normalizes pixel values to the 0-1 range.
    """
    img = image.load_img(
        img_path, color_mode='grayscale', target_size=(48, 48)
    )
    img_array: npt.NDArray[np.float32] = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_emotion(
    img_array: npt.NDArray[np.float32],
) -> int:
    """
    Predicts the emotion from the preprocessed image array.
    """
    predictions: npt.NDArray[np.float32] = model.predict(img_array)
    return int(np.argmax(predictions))
