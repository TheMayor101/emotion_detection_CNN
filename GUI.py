import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import model
import random

def get_emotion_text(index):
    """
    Function to map emotion index to text
    """
    emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
    return emotions[index]

def get_random_image_path(folder):
    """
    Function to randomly select an image from the test folder
    """
    images = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    return random.choice(images) #using the choice function from the random libary to chose a random image

test_folder = 'data/test/'  # Directory containing test images

class EmotionRecognitionApp:
    def __init__(self, root):
        self.user_lives = 3  # Initialize user lives
        self.model_lives = 3  # Initialize model lives
        self.user_score = 0  # Initialize user score
        self.model_score = 0  # Initialize model score
        self.prev_image_path = None  # Initialize prev_image_path attribute

        self.root = root
        self.root.title("Emotion Recognition")  # Set window title
        self.root.attributes("-fullscreen", True)  # Set fullscreen mode

        # Instruction label
        self.label = tk.Label(root, text="Select the emotion you recognize in the picture:", font=("Helvetica", 14, "bold"), fg="blue")
        self.label.pack(pady=10)

        self.emotion_var = tk.IntVar()  # Variable to store the selected emotion index
        self.emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]  # List of emotions
        self.radio_buttons = []  # List to store radio button widgets
        
        # Create radio buttons for emotion selection
        for i, emotion in enumerate(self.emotions):
            rb = tk.Radiobutton(root, text=emotion, variable=self.emotion_var, value=i, font=("Helvetica", 12, "bold"), fg="green")
            rb.pack(anchor='w', padx=20, pady=5)
            self.radio_buttons.append(rb)  # Add radio button to the list

        self.img_label = tk.Label(root)  # Label to display the image
        self.img_label.pack(pady=10)

        # Predict button
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_emotion, bg="orange", fg="white", font=("Helvetica", 12, "bold"))
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 12))  # Label to display prediction results
        self.result_label.pack(pady=10)

        self.score_label = tk.Label(root, text="Scores:", font=("Helvetica", 14, "bold"))  # Label to display score text
        self.score_label.pack()
        
        # Display user score
        self.user_score_label = tk.Label(root, text=f"User: {self.user_score}", font=("Helvetica", 12))
        self.user_score_label.pack()

        # Display model score
        self.model_score_label = tk.Label(root, text=f"Model: {self.model_score}", font=("Helvetica", 12))
        self.model_score_label.pack()

        # Display user lives
        self.user_lives_label = tk.Label(root, text=f"User Lives: {self.user_lives}", font=("Helvetica", 12))
        self.user_lives_label.pack()

        # Display model lives
        self.model_lives_label = tk.Label(root, text=f"Model Lives: {self.model_lives}", font=("Helvetica", 12))
        self.model_lives_label.pack()

        # Restart button
        self.restart_button = tk.Button(root, text="Restart", command=self.restart_game, state=tk.DISABLED, bg="red", fg="white", font=("Helvetica", 12, "bold"))
        self.restart_button.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=50)  # Position the restart button

        # Exit button
        self.exit_button = tk.Button(root, text="Exit", command=root.quit, bg="gray", fg="white", font=("Helvetica", 12, "bold"))
        self.exit_button.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)  # Position the exit button

        # Load initial random image
        self.load_random_image()

    def load_random_image(self):
        """
        Function to load a random image from the test folder and display it
        """
        if hasattr(self, "prev_img_label"):
            self.prev_img_label.destroy()  # Remove the previous image label if it exists

        if self.prev_image_path:
            # Load and resize the previous image
            prev_img = Image.open(self.prev_image_path)
            prev_img = prev_img.resize((200, 200))
            prev_img_tk = ImageTk.PhotoImage(prev_img)

            # Display the previous image
            self.prev_img_label = tk.Label(self.root, image=prev_img_tk)
            self.prev_img_label.image = prev_img_tk
            self.prev_img_label.pack(side=tk.LEFT, padx=(50, 10), pady=(self.root.winfo_height() // 2 - 100, 10))

        # Select a new random image from the test folder
        random_folder = random.choice(os.listdir(test_folder))
        random_image_path = os.path.join(test_folder, random_folder, random.choice(os.listdir(os.path.join(test_folder, random_folder))))
        img = Image.open(random_image_path)
        img = img.resize((400, 400))  # Resize the new image
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)  # Update the image label with the new image
        self.img_label.image = img_tk
        self.img_path = random_image_path  # Store the path of the new image

        # Store the correct answer for the new image
        correct_answer = os.path.basename(os.path.dirname(self.img_path))
        self.correct_answer_index = self.emotions.index(correct_answer)

        # Store the current image path as the previous image path
        self.prev_image_path = random_image_path

    def predict_emotion(self):
        """
        Function to predict the emotion of the displayed image and update the scores and lives
        """
        if self.user_lives <= 0 or self.model_lives <= 0:
            # Show game over message if user or model lives are zero
            messagebox.showinfo("Game Over", "The game is over. You can restart the game or exit it.")
            return

        user_guess = self.emotion_var.get()  # Get the user's selected emotion
        img_array = model.preprocess_image(self.img_path)  # Preprocess the image for model prediction
        predicted_class = model.predict_emotion(img_array)  # Get the model's prediction

        # Prepare the result text
        result_text = f"CNN Model Prediction: {self.emotions[predicted_class]}\n"
        result_text += f"Your Guess: {self.emotions[user_guess]}\n"
        result_text += f"Correct Answer: {self.emotions[self.correct_answer_index]}\n"

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