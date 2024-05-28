import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import random

# Path to the trained CNN model
path_to_model = "model/emotion_recognition_model.h5"
# Path to the test folder containing images
test_folder = 'data/test/sad'

# Load the trained CNN model
model = tf.keras.models.load_model(path_to_model)

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to map emotion index to text
def get_emotion_text(index):
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    return emotions[index]

# Function to randomly select an image from the test folder
def get_random_image_path(folder):
    images = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not images:
        raise FileNotFoundError(f"No images found in the directory: {folder}")
    return random.choice(images)

# Create the main application window
class EmotionRecognitionApp:
    def __init__(self, root):
        # Initialize scores for the user and the model
        self.user_score = 0
        self.model_score = 0

        # Initialize the GUI
        self.root = root
        self.root.title("Emotion Recognition")

        self.label = tk.Label(root, text="Select the emotion you recognize in the picture:")
        self.label.pack(pady=10)

        self.emotion_var = tk.IntVar()
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.radio_buttons = []
        for i, emotion in enumerate(self.emotions):
            rb = tk.Radiobutton(root, text=emotion, variable=self.emotion_var, value=i)
            rb.pack(anchor='w')
            self.radio_buttons.append(rb)

        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_emotion)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

        self.load_random_image()

    def load_random_image(self):
        # Load a random image
        self.img_path = get_random_image_path(test_folder)
        img = Image.open(self.img_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def predict_emotion(self):
        user_guess = self.emotion_var.get()
        img_array = preprocess_image(self.img_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        result_text = f"CNN Model Prediction: {get_emotion_text(predicted_class)}\n"
        result_text += f"Your Guess: {get_emotion_text(user_guess)}\n"

        # Check if the user's guess matches the model's prediction
        if predicted_class == user_guess:
            self.user_score += 1
            self.model_score += 1
            result_text += "You got it right!\n"
        else:
            # If the user's guess is wrong, end the game
            result_text += "Sorry, wrong guess! Game Over.\n"
            result_text += f"Final Scores - You: {self.user_score}, Model: {self.model_score}\n"
            self.predict_button.config(state=tk.DISABLED)  # Disable the predict button
        result_text += f"Model Score for this round: {self.model_score}"
        self.result_label.config(text=result_text)

        if predicted_class == user_guess:
            self.load_random_image()

if __name__ == "__main__":
    if not os.path.exists(test_folder):
        raise FileNotFoundError(f"The directory {test_folder} does not exist.")
    if not os.listdir(test_folder):
        raise FileNotFoundError(f"The directory {test_folder} is empty.")
    
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
