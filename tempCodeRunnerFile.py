    # Create a label for the previous image if it exists
        if hasattr(self, "prev_img_label"):
            self.prev_img_label.destroy()  # Remove the previous image label if it exists

        # Check if there's a previous image
        if self.prev_image_path:
            # Load and resize the previous image
            prev_img = Image.open(self.prev_image_path)
            prev_img = prev_img.resize((200, 200))
            prev_img_tk = ImageTk.PhotoImage(prev_img)

            # Create a label for the previous image and pack it to the middle-left side
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