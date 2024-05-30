#this file is responsible for the non GUI functions inside app.py

import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Path to the trained CNN model
path_to_model = "model/emotion_recognition_model.h5"

# Load the trained CNN model
model = tf.keras.models.load_model(path_to_model)

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    #returning the new images after we adapted it our model's needs
    return img_array

# Function to predict emotion
def predict_emotion(img_array):
    predictions = model.predict(img_array)
    #we only want the max value of the predictions which is the emotion the model thinks is the most dominent in the image.
    return np.argmax(predictions)
