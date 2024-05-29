#This is the main file from which we run the whole program

import data_augmentation
import model_training
import plot_results

if __name__ == "__main__":
    data_augmentation.display_augmented_images(data_augmentation.train_generator, 32)
    model_training
