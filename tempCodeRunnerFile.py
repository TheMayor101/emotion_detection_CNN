    if self.user_lives <= 0 or self.model_lives <= 0:
            messagebox.showinfo("Game Over", "The game is over. You can restart the game.")
            return

        user_guess = self.emotion_var.get()
        img_array = model.preprocess_image(self.img_path)
        predicted_class = model.predict_emotion(img_array)

        # Call the check_guesses function
        self.check_guesses(user_guess, predicted_class)

        result_text = f"CNN Model Prediction: {self.emotions[predicted_class]}\n"
        result_text += f"Your Guess: {self.emotions[user_guess]}\n"
        result_text += f"Correct Answer: {self.emotions[self.correct_answer_index]}\n"

        if predicted_class == self.correct_answer_index and user_guess == self.correct_answer_index:
            result_text += "You and the model both got it right!\n"
        elif predicted_class == self.correct_answer_index:
            result_text += "You got it right!\n"
        elif user_guess == self.correct_answer_index:
            result_text += "Model got it right!\n"
        else:
            self.user_lives -= 1
            result_text += "Sorry, wrong guess!\n"
            result_text += f"Remaining Lives - User: {self.user_lives}, Model: {self.model_lives}\n"
            if self.user_lives == 0 or self.model_lives == 0:
                result_text += "Game Over!"
                self.restart_button.config(state=tk.NORMAL)

        if predicted_class != self.correct_answer_index:
            self.model_lives -= 1

        self.result_label.config(text=result_text)
        self.user_score_label.config(text=f"User: {self.user_score}")
        self.model_score_label.config(text=f"Model: {self.model_score}")
        self.user_lives_label.config(text=f"User Lives: {self.user_lives}")
        self.model_lives_label.config(text=f"Model Lives: {self.model_lives}")

        if self.user_lives > 0 and self.model_lives > 0:
            self.load_random_image()