import os
import random

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
