# This file is responsible for the non-GUI functions inside GUI.py

import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Path to the trained CNN model
path_to_model = "model/emotion_recognition_model.h5"

# Load the trained CNN model
model = tf.keras.models.load_model(path_to_model)

def preprocess_image(img_path):
    """
    Preprocess the image so the model will be able to work with it.
    
    Parameters:
    img_path (str): Path to the image file.
    
    Returns:
    np.ndarray: Preprocessed image array.
    """
    # Load the image in grayscale mode and resize to 48x48 pixels
    img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
    
    # Convert the image to an array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the model's input shape (1, 48, 48, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to be between 0 and 1
    img_array /= 255.0
    
    # Return the preprocessed image array
    return img_array

def predict_emotion(img_array):
    """
    Predict the emotion from the preprocessed image array.
    
    Parameters:
    img_array (np.ndarray): Preprocessed image array.
    
    Returns:
    int: Predicted emotion class index.
    """
    # Make predictions using the loaded model
    predictions = model.predict(img_array)
    
    # Return the index of the highest probability class
    return np.argmax(predictions)
