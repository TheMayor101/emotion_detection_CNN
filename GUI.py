import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import model
import random  


# Function to map emotion index to text
def get_emotion_text(index):
    emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
    return emotions[index]

# Function to randomly select an image from the test folder
def get_random_image_path(folder):
    images = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not images:
        raise FileNotFoundError(f"No images found in the directory: {folder}")
    return random.choice(images)

test_folder = 'data/test/'

class EmotionRecognitionApp:
    def __init__(self, root):
        self.user_lives = 3
        self.model_lives = 3
        self.user_score = 0
        self.model_score = 0
        self.prev_image_path = None  # Initialize prev_image_path attribute

        self.root = root
        self.root.title("Emotion Recognition")
        self.root.attributes("-fullscreen", True)  # Set fullscreen mode

        self.label = tk.Label(root, text="Select the emotion you recognize in the picture:", font=("Helvetica", 14, "bold"), fg="blue")
        self.label.pack(pady=10)

        self.emotion_var = tk.IntVar()
        self.emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
        self.radio_buttons = []
        for i, emotion in enumerate(self.emotions):
            rb = tk.Radiobutton(root, text=emotion, variable=self.emotion_var, value=i, font=("Helvetica", 12, "bold"), fg="green")
            rb.pack(anchor='w', padx=20, pady=5)  
            self.radio_buttons.append(rb)

        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_emotion, bg="orange", fg="white", font=("Helvetica", 12, "bold"))
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

        self.score_label = tk.Label(root, text="Scores:", font=("Helvetica", 14, "bold"))
        self.score_label.pack()
        
        self.user_score_label = tk.Label(root, text=f"User: {self.user_score}", font=("Helvetica", 12))
        self.user_score_label.pack()

        self.model_score_label = tk.Label(root, text=f"Model: {self.model_score}", font=("Helvetica", 12))
        self.model_score_label.pack()

        self.user_lives_label = tk.Label(root, text=f"User Lives: {self.user_lives}", font=("Helvetica", 12))
        self.user_lives_label.pack()

        self.model_lives_label = tk.Label(root, text=f"Model Lives: {self.model_lives}", font=("Helvetica", 12))
        self.model_lives_label.pack()

        self.restart_button = tk.Button(root, text="Restart", command=self.restart_game, state=tk.DISABLED, bg="red", fg="white", font=("Helvetica", 12, "bold"))
        self.restart_button.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=50)  # Position the restart button below the exit button

        self.exit_button = tk.Button(root, text="Exit", command=root.quit, bg="gray", fg="white", font=("Helvetica", 12, "bold"))
        self.exit_button.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)  # Position the exit button at the top right corner

        self.load_random_image()

    def load_random_image(self):
        # Create a label for the previous image if it exists
        if hasattr(self, "prev_img_label"):
            self.prev_img_label.destroy()  # Remove the previous image label if it exists

        # Check if there's a previous image
        if self.prev_image_path:
            # Load and resize the previous image
            prev_img = Image.open(self.prev_image_path)
            prev_img = prev_img.resize((200, 200))
            prev_img_tk = ImageTk.PhotoImage(prev_img)

            # Create a label for the previous image and pack it in the middle of the left side
            self.prev_img_label = tk.Label(self.root, image=prev_img_tk)
            self.prev_img_label.image = prev_img_tk
            self.prev_img_label.pack(side=tk.LEFT, padx=(50, 10), pady=(self.root.winfo_height() // 2 - 100, 10))  # Position it in the middle-left

        # Load a new random image
        random_folder = random.choice(os.listdir(test_folder))
        random_image_path = os.path.join(test_folder, random_folder, random.choice(os.listdir(os.path.join(test_folder, random_folder))))
        img = Image.open(random_image_path)
        img = img.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk
        self.img_path = random_image_path

        # Store the correct answer for the new image
        correct_answer = os.path.basename(os.path.dirname(self.img_path))
        self.correct_answer_index = self.emotions.index(correct_answer)

        # Store the current image path as the previous image path
        self.prev_image_path = random_image_path



    def predict_emotion(self):
        if self.user_lives <= 0 or self.model_lives <= 0:
            messagebox.showinfo("Game Over", "The game is over. You can restart the game or exit it.")
            return

        user_guess = self.emotion_var.get()
        img_array = model.preprocess_image(self.img_path)
        predicted_class = model.predict_emotion(img_array)

        result_text = f"CNN Model Prediction: {self.emotions[predicted_class]}\n"
        result_text += f"Your Guess: {self.emotions[user_guess]}\n"
        result_text += f"Correct Answer: {self.emotions[self.correct_answer_index]}\n"

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
        self.user_lives = 3
        self.model_lives = 3
        self.user_score = 0
        self.model_score = 0
        self.result_label.config(text="")
        self.predict_button.config(state=tk.NORMAL)
        self.restart_button.config(state=tk.DISABLED)
        self.user_score_label.config(text=f"User: {self.user_score}")
        self.model_score_label.config(text=f"Model: {self.model_score}")
        self.user_lives_label.config(text=f"User Lives: {self.user_lives}")
        self.model_lives_label.config(text=f"Model Lives: {self.model_lives}")
        self.load_random_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()