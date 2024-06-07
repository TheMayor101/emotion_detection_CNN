import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk, ImageOps
import model  # Importing the model module

# Function to predict emotion and show results in a new window
def predict_emotion():
    if not file_path:
        result_label.config(text="Please select an image first")
        return

    img = model.preprocess_image(file_path)  # Using the model's preprocess function
    emotion_idx = model.predict_emotion(img)  # Using the model's predict function
    probabilities = model.model.predict(img)[0]
    emotion_text = f"Predicted Emotion: {emotion_labels[emotion_idx]}"
    
    # Create a new window to show the prediction and image
    prediction_window = Toplevel(root)
    prediction_window.title("Prediction Result")
    prediction_window.geometry("400x500")
    prediction_window.configure(bg="white")

    # Display the selected image in the new window
    img_display = Image.open(file_path)
    img_display.thumbnail((300, 300))
    img_display = ImageTk.PhotoImage(img_display)
    img_panel = tk.Label(prediction_window, image=img_display, bg="white")
    img_panel.image = img_display
    img_panel.pack(pady=20)

    # Display the prediction result in the new window
    result_label = tk.Label(prediction_window, text=emotion_text, bg="white", font=("Arial", 16, "bold"))
    result_label.pack(pady=10)

    # Display detailed probabilities in the new window
    prob_text = "Probabilities:\n" + "\n".join([f"{label}: {prob:.2f}" for label, prob in zip(emotion_labels, probabilities)])
    prob_label = tk.Label(prediction_window, text=prob_text, bg="white", font=("Arial", 12))
    prob_label.pack(pady=10)

# Function to display selected image
def show_image():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        result_label.config(text="")
        prob_label.config(text="")

# Set up the GUI
root = tk.Tk()
root.title("Emotion Recognition")
root.geometry("600x700")
root.configure(bg="white")

file_path = ""
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# UI Elements
btn_select = tk.Button(root, text="Select Image", command=show_image, bg="blue", fg="white", font=("Arial", 14, "bold"))
btn_select.grid(row=0, column=0, padx=10, pady=10)

btn_predict = tk.Button(root, text="Predict Emotion", command=predict_emotion, bg="blue", fg="white", font=("Arial", 14, "bold"))
btn_predict.grid(row=0, column=1, padx=10, pady=10)

empty_label_left = tk.Label(root, bg="white")
empty_label_left.grid(row=0, column=2, padx=10)

panel = tk.Label(root, bg="white")
panel.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

result_label = tk.Label(root, text="", bg="white", font=("Arial", 16, "bold"))
result_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

prob_label = tk.Label(root, text="", bg="white", font=("Arial", 12))
prob_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()
