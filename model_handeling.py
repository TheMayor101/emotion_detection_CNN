# This file is responsible for the non-GUI functions inside GUI.py

import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Path to the trained CNN model
path_to_model = "model/emotion_recognition_model.h5"


model = tf.keras.models.load_model(path_to_model)

def preprocess_image(img_path):
    """
    Preprocess the image so the model will be able to work with it.
    a few operations are done. converted the image to and array, expended it dimenations to fit the model needs and normalized the values.
    """
    img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
    
    img_array = image.img_to_array(img)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array /= 255.0
    
    return img_array

def predict_emotion(img_array):
    """
    Predicts the emotion from the preprocessed image array.
    """
    # Make predictions using the loaded model
    predictions = model.predict(img_array)
    
    # Return the index of the highest probability class
    return np.argmax(predictions)
