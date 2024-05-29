import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import model
import utils

# Path to the test folder containing images
test_folder = 'data/test/sad'

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
        self.img_path = utils.get_random_image_path(test_folder)
        img = Image.open(self.img_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def predict_emotion(self):
        user_guess = self.emotion_var.get()
        img_array = model.preprocess_image(self.img_path)
        predicted_class = model.predict_emotion(img_array)

        result_text = f"CNN Model Prediction: {utils.get_emotion_text(predicted_class)}\n"
        result_text += f"Your Guess: {utils.get_emotion_text(user_guess)}\n"

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
