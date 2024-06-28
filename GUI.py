# Imports
import os
import random
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import model_handeling


# Constants
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
TEST_FOLDER = 'data/test/'
IMG_SIZE = (400, 400)
USER_INITIAL_LIVES = 3
MODEL_INITIAL_LIVES = 3
FULLSCREEN_MODE = True
FONT_TYPE = "Arial"
BOLD_TYPE = "bold"


class EmotionRecognitionApp:
    def __init__(self, root):
        """
        Initializes all the values for the game
        """
        self.user_lives = USER_INITIAL_LIVES
        self.model_lives = MODEL_INITIAL_LIVES
        self.user_score = 0
        self.model_score = 0
        self.root = root

        self.root.title("Emotion Recognition")
        self.root.attributes("-fullscreen", FULLSCREEN_MODE)

        self.setup_ui()
        self.load_random_image()

    def setup_ui(self):
        self.label = tk.Label(self.root, text="Please Select the emotion you recognize in the image", font=(FONT_TYPE, 16, BOLD_TYPE))
        self.label.pack(pady=10)

        self.emotion_var = tk.IntVar()
        for i, emotion in enumerate(EMOTIONS):
            rb = tk.Radiobutton(self.root, text=emotion, variable=self.emotion_var, value=i, font=(FONT_TYPE, 14))
            rb.pack(anchor='w', padx=20, pady=5)

        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=10)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_emotion, bg="blue", fg="white", font=(FONT_TYPE, 14, BOLD_TYPE))
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=(FONT_TYPE, 14))
        self.result_label.pack(pady=10)

        self.setup_scoreboard()
        self.setup_control_buttons()

    def setup_scoreboard(self):
        """
        Drawing the score on the screen
        """
        tk.Label(self.root, text="The Score", font=(FONT_TYPE, 16, BOLD_TYPE)).pack()

        self.user_score_label = tk.Label(self.root, text=f"User: {self.user_score}", font=(FONT_TYPE, 14))
        self.user_score_label.pack()

        self.model_score_label = tk.Label(self.root, text=f"Model: {self.model_score}", font=(FONT_TYPE, 14))
        self.model_score_label.pack()

        self.user_lives_label = tk.Label(self.root, text=f"User Lives: {self.user_lives}", font=(FONT_TYPE, 14))
        self.user_lives_label.pack()

        self.model_lives_label = tk.Label(self.root, text=f"Model Lives: {self.model_lives}", font=(FONT_TYPE, 14))
        self.model_lives_label.pack()

    def setup_control_buttons(self):
        """
        Drawing the the Restart and Exit buttons
        """
        self.restart_button = tk.Button(self.root, text="Restart", command=self.restart_game, state=tk.DISABLED, bg="green", fg="white", font=(FONT_TYPE, 14, BOLD_TYPE))
        self.restart_button.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=50)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit, bg="red", fg="white", font=(FONT_TYPE, 14, BOLD_TYPE))
        self.exit_button.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)

    def load_random_image(self):
        """
        Loading a random image from the test folder and presenting it on the screen
        """
        choosen_folder = random.choice(os.listdir(TEST_FOLDER))
        random_image_path = os.path.join(TEST_FOLDER, choosen_folder, random.choice(os.listdir(os.path.join(TEST_FOLDER, choosen_folder))))

        img = Image.open(random_image_path).resize(IMG_SIZE)
        img_tk = ImageTk.PhotoImage(img)

        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk
        self.img_path = random_image_path

        correct_answer = os.path.basename(os.path.dirname(self.img_path))
        self.correct_answer_index = EMOTIONS.index(correct_answer)

    def predict_emotion(self):
        """
        Predict the emotion of the displayed image and update scores and lives
        """
        if self.user_lives <= 0 or self.model_lives <= 0:
            messagebox.showinfo("Game Over", "The game is over. You can restart the game or exit it.")
            return

        user_guess = self.emotion_var.get()
        img_array = model_handeling.preprocess_image(self.img_path)
        predicted_class = model_handeling.predict_emotion(img_array)

        result_text = f"CNN Model Prediction: {EMOTIONS[predicted_class]}\nYour Guess: {EMOTIONS[user_guess]}\nCorrect Answer: {EMOTIONS[self.correct_answer_index]}\n"

        if predicted_class == self.correct_answer_index:
            if user_guess == self.correct_answer_index:
                result_text += "You and the model both got it right!\n"
                self.user_score += 1
                self.model_score += 1
            else:
                result_text += "Model got it right!\n"
                self.model_score += 1
                self.user_lives -= 1
        else:
            if user_guess == self.correct_answer_index:
                result_text += "You got it right!\n"
                self.user_score += 1
                self.model_lives -= 1
            else:
                result_text += "Sorry, wrong guess!\n"
                self.user_lives -= 1
                self.model_lives -= 1

        result_text += f"Remaining Lives - User: {self.user_lives}, Model: {self.model_lives}\n"

        if self.user_lives == 0 or self.model_lives == 0:
            winner = "User" if self.user_lives > 0 else "Model" if self.model_lives > 0 else "No one"
            result_text += f"Game Over! {winner} wins!"
            self.restart_button.config(state=tk.NORMAL)

        self.result_label.config(text=result_text)
        self.user_score_label.config(text=f"User: {self.user_score}")
        self.model_score_label.config(text=f"Model: {self.model_score}")
        self.user_lives_label.config(text=f"User Lives: {self.user_lives}")
        self.model_lives_label.config(text=f"Model Lives: {self.model_lives}")

        if self.user_lives > 0 and self.model_lives > 0:
            self.load_random_image()

    def restart_game(self):
        """
        Restart the game by resetting scores and lives
        """
        self.user_lives = USER_INITIAL_LIVES
        self.model_lives = MODEL_INITIAL_LIVES
        self.user_score = 0
        self.model_score = 0
        self.result_label.config(text="")
        self.load_random_image()
        self.restart_button.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
