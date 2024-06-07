#This is the main file from which the whole project runs.

#importing the file which in I created the model
from model_training import EmotionRecognitionModel
# importing the file with all the ploting functions for visualzing the model's results
import plot_results 


if __name__ == "__main__":
    """
    This is the main file of the project so everything starts running from here.
    """
    #creating a new model using the class EmotionRecognitionModel and initalize function inside of it
    model = EmotionRecognitionModel()
    #when training the model we would like to save the data of the model training for the visualizing the model's results over time
    history = model.train_model()
    model.save_model('emotion_recognition_model.h5') 
    #here we use the history we saved when training the model for our ploting functions
    plot_results.plot_training_history(history)