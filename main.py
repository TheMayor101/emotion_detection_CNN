#This is the main file from which we run the whole program

#importing the three files which in all of my project is written
import data_augmentation
from model_training import EmotionRecognitionModel
import plot_results


#This way of running the main program is a convention in python
#if __name__ == "__main__":
    #data_augmentation.display_augmented_images(data_augmentation.train_generator, 32) #running the data_augmentation file which is first step before starting the training
    #model_training #running the model_training file which in there the training of the model starts.


if __name__ == "__main__":
    model = EmotionRecognitionModel()
    history = model.train_model()
    model.save_model('emotion_recognition_model.h5') #TODO check if I also need this save in addition to the save_model function
    import plot_results
    
    
    plot_results.plot_training_history(history)