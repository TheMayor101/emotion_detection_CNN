import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import model
import random

# Function to map emotion index to text
def get_emotion_text(index):
    """
    Function to map emotion index to text
    """
    emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
    return emotions[index]

# Function to randomly select an image from the test folder
def get_random_image_path(folder):
    """
    Function to randomly select an image from the test folder
    """
    images = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return random.choice(images)

# Directory containing test images
test_folder = 'data/test/'

class EmotionRecognitionApp:
    def __init__(self, root):
        self.user_lives = 3  # Initialize user lives
        self.model_lives = 3  # Initialize model lives
        self.user_score = 0  # Initialize user score
        self.model_score = 0  # Initialize model score

        self.root = root
        self.root.title("Emotion Recognition")  # Set window title
        self.root.attributes("-fullscreen", True)  # Set fullscreen mode

        self.setup_ui()  # Setup the user interface

        # Load initial random image
        self.load_random_image()

    def setup_ui(self):
        """Setup the UI elements."""
        # Instruction label
        self.label = tk.Label(self.root, text="Select the emotion you recognize in the picture:", font=("Arial", 16, "bold"), fg="black")
        self.label.pack(pady=10)

        # Variable to store the selected emotion index
        self.emotion_var = tk.IntVar()
        self.emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]  # List of emotions
        self.radio_buttons = []

        # Create radio buttons for emotion selection
        for i, emotion in enumerate(self.emotions):
            rb = tk.Radiobutton(self.root, text=emotion, variable=self.emotion_var, value=i, font=("Arial", 14), fg="black")
            rb.pack(anchor='w', padx=20, pady=5)
            self.radio_buttons.append(rb)  # Add radio button to the list

        # Image display label
        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=10)

        # Predict button
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_emotion, bg="blue", fg="white", font=("Arial", 14, "bold"))
        self.predict_button.pack(pady=10)

        # Result label
        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Score and lives display
        self.setup_scoreboard()  # Setup the scoreboard for displaying scores and lives

        # Restart and exit buttons
        self.setup_control_buttons()  # Setup control buttons for restarting and exiting the game

    def setup_scoreboard(self):
        """Setup the scoreboard for displaying scores and lives."""
        self.score_label = tk.Label(self.root, text="Scores:", font=("Arial", 16, "bold"))
        self.score_label.pack()
        
        self.user_score_label = tk.Label(self.root, text=f"User: {self.user_score}", font=("Arial", 14))
        self.user_score_label.pack()

        self.model_score_label = tk.Label(self.root, text=f"Model: {self.model_score}", font=("Arial", 14))
        self.model_score_label.pack()

        self.user_lives_label = tk.Label(self.root, text=f"User Lives: {self.user_lives}", font=("Arial", 14))
        self.user_lives_label.pack()

        self.model_lives_label = tk.Label(self.root, text=f"Model Lives: {self.model_lives}", font=("Arial", 14))
        self.model_lives_label.pack()

    def setup_control_buttons(self):
        """Setup control buttons for restarting and exiting the game."""
        self.restart_button = tk.Button(self.root, text="Restart", command=self.restart_game, state=tk.DISABLED, bg="green", fg="white", font=("Arial", 14, "bold"))
        self.restart_button.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=50)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit, bg="red", fg="white", font=("Arial", 14, "bold"))
        self.exit_button.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)

    def load_random_image(self):
        """Load a random image from the test folder and display it."""
        random_folder = random.choice(os.listdir(test_folder))  # Select a random folder from the test folder
        random_image_path = os.path.join(test_folder, random_folder, random.choice(os.listdir(os.path.join(test_folder, random_folder))))  # Select a random image from the selected folder
        
        img = Image.open(random_image_path)  # Open the selected image
        img = img.resize((400, 400))  # Resize the image
        img_tk = ImageTk.PhotoImage(img)  # Convert image to PhotoImage for Tkinter
        
        self.img_label.config(image=img_tk)  # Update the image label with the new image
        self.img_label.image = img_tk  # Keep a reference to avoid garbage collection
        self.img_path = random_image_path  # Store the path of the new image

        # Store the correct answer for the new image
        correct_answer = os.path.basename(os.path.dirname(self.img_path))
        self.correct_answer_index = self.emotions.index(correct_answer)

    def predict_emotion(self):
        """Predict the emotion of the displayed image and update scores and lives."""
        if self.user_lives <= 0 or self.model_lives <= 0:
            # Show game over message if user or model lives are zero
            messagebox.showinfo("Game Over", "The game is over. You can restart the game or exit it.")
            return

        user_guess = self.emotion_var.get()  # Get the user's selected emotion
        img_array = model.preprocess_image(self.img_path)  # Preprocess the image for model prediction
        predicted_class = model.predict_emotion(img_array)  # Get the model's prediction

        # Prepare the result text
        result_text = (f"CNN Model Prediction: {self.emotions[predicted_class]}\n"
                       f"Your Guess: {self.emotions[user_guess]}\n"
                       f"Correct Answer: {self.emotions[self.correct_answer_index]}\n")

        # Update scores and lives based on predictions
        if predicted_class == self.correct_answer_index:
            if user_guess == self.correct_answer_index:
                # Both user and model got it right
                result_text += "You and the model both got it right!\n"
                self.user_score += 1  # Increment user score
                self.model_score += 1  # Increment model score
            else:
                # Only model got it right
                result_text += "Model got it right!\n"
                self.model_score += 1  # Increment model score
                self.user_lives -= 1  # Decrement user lives
        else:
            if user_guess == self.correct_answer_index:
                # Only user got it right
                result_text += "You got it right!\n"
                self.user_score += 1  # Increment user score
                self.model_lives -= 1  # Decrement model lives
            else:
                # Both user and model got it wrong
                result_text += "Sorry, wrong guess!\n"
                self.user_lives -= 1  # Decrement user lives
                self.model_lives -= 1  # Decrement model lives

        # Update remaining lives text
        result_text += f"Remaining Lives - User: {self.user_lives}, Model: {self.model_lives}\n"

        if self.user_lives == 0 or self.model_lives == 0:
            # If either user or model lives are zero, game over
            winner = "User" if self.user_lives > 0 else "Model" if self.model_lives > 0 else "No one"
            result_text += f"Game Over! {winner} wins!"
            self.restart_button.config(state=tk.NORMAL)  # Enable the restart button

        self.result_label.config(text=result_text)  # Update result label with the result text
        self.user_score_label.config(text=f"User: {self.user_score}")  # Update user score label
        self.model_score_label.config(text=f"Model: {self.model_score}")  # Update model score label
        self.user_lives_label.config(text=f"User Lives: {self.user_lives}")  # Update user lives
        self.model_lives_label.config(text=f"Model Lives: {self.model_lives}")

        if self.user_lives > 0 and self.model_lives > 0:
            self.load_random_image()

    def restart_game(self):
        """Restart the game by resetting scores and lives."""
        self.user_lives = 3
        self.model_lives = 3
        self.user_score = 0
        self.model_score = 0
        self.result_label.config(text="")  # Clear result label
        self.load_random_image()  # Load a new random image
        self.restart_button.config(state=tk.DISABLED)  # Disable the restart button

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
