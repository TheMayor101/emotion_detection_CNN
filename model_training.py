#This file is responsible for building the architecture for the CNN model.

#I chose to use the Sequential libary for building my CNN model
from keras.models import Sequential
#importing all the layers
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
# importing the file that is responsible for the data_augmentation process
import data_augmentation  
import os
# Library for visualizing neural network architectures
import visualkeras  

class EmotionRecognitionModel:
    def __init__(self, img_height=48, img_width=48, batch_size=32, epochs=50, train_path="data/train/", test_path="data/test"):
        """
        Initialize the EmotionRecognitionModel with given parameters.
        """
        self.img_height = img_height  # Height of the images
        self.img_width = img_width  # Width of the images
        self.batch_size = batch_size  # Number of samples per batch
        self.epochs = epochs  # Number of training epochs
        self.train_path = train_path  # The Path to the folder which contains the training data
        self.test_path = test_path  # The Path to the folder which contains the test data

        # Initialize and compile the model
        self.model = self.initialize_layers()

        #I chose to use adam as my optimizer, categorical_crossentropy as my loss function and accuracy and the metric to check how well the model is performing
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Print model summary
        print(self.model.summary())

        # Visualize the model architecture
        self.visualize_model()

    def initialize_layers(self):
        """
        Creating the CNN model and adding the layers one after another.
        """
        model = Sequential()  # Initialize the model as a Sequential model

        # First CNN layer: Convolution, Batch Normalization
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 1)))
        model.add(BatchNormalization())

        # Second CNN layer: Convolution, Batch Normalization, Max Pooling, Dropout
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # Third CNN layer: Convolution, Batch Normalization, Max Pooling, Dropout
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # Fourth CNN layer: Convolution, Batch Normalization, Max Pooling, Dropout
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        # Flatten layer to convert 2D matrix data to a vector
        model.add(Flatten())

        # Dense layer for classification
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))  # Dropout to prevent overfitting

        # Output layer: 7 neurons for 7 classes with softmax activation
        model.add(Dense(7, activation='softmax'))

        #returning the newly created CNN model!
        return model

    def visualize_model(self):
        """
        Graphyical Visualize the model architecture using visualkeras.
        """
        img = visualkeras.layered_view(self.model, to_file='model_visual.png')  # Create the image
        img.show()  # Display the image

    def train_model(self):
        """
        Train the model using the data generators from data_augmentation.
        """
        # Count the number of training and testing images
        num_train_imgs = sum([len(files) for r, d, files in os.walk(self.train_path)])
        num_test_imgs = sum([len(files) for r, d, files in os.walk(self.test_path)])

        # Train the model using data augmentation
        history = self.model.fit(data_augmentation.train_generator,
                                 steps_per_epoch=num_train_imgs // self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=data_augmentation.validation_generator,
                                 validation_steps=num_test_imgs // self.batch_size)

        return history

    def save_model(self, filename):
        """
        Save the trained model to a file.
        """
        self.model.save(filename)

