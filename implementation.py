import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

path_to_model = "model/emotion_recognition_model.h5"
img_path = "predictions/happy.jpeg"

# Load the trained CNN model
model = tf.keras.models.load_model(path_to_model)

# Define a function to preprocess the image
def preprocess_image(img_path):
    # Load the image and convert it to grayscale
    img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Reshape the array to match the expected input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values to be in the range [0, 1]
    img_array /= 255.0
    
    return img_array

# Function to get user's emotion guess
def get_user_guess():
    print("Which emotion do you recognize in the picture?")
    print("1. Angry\n2. Disgust\n3. Fear\n4. Happy\n5. Sad\n6. Surprise\n7. Neutral")
    guess = int(input("Enter the corresponding number: ")) - 1
    return guess

# Function to map emotion index to text
def get_emotion_text(index):
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    return emotions[index]

# Function to display the image
def display_image(img_path):
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Display the image
display_image(img_path)

# Preprocess the image
img_array = preprocess_image(img_path)

# Make predictions
predictions = model.predict(img_array)

# Assuming the model returns a single class prediction
predicted_class = np.argmax(predictions)

# Get user's guess
user_guess = get_user_guess()

# Compare user's guess with model's prediction
print("\nCNN Model Prediction:", get_emotion_text(predicted_class))
print("Your Guess:", get_emotion_text(user_guess))

# Calculate scores
model_score = 1 if predicted_class == user_guess else 0
print("\nCNN Model Score:", model_score)
