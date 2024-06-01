from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
import data_augmentation
import os
import visualkeras  
from PIL import ImageFont



class EmotionRecognitionModel:
    def __init__(self, img_height=48, img_width=48, batch_size=32, epochs=50, train_path="data/train/", test_path="data/test"):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_path = train_path
        self.test_path = test_path
        self.model = self.initialize_layers()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())
        self.visualize_model()  # Call visualize_model method


    def initialize_layers(self):
        """
        Creating the CNN model and adding the layers one after another
        """
        #first CNN layer
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 1)))
        model.add(BatchNormalization())

        #second CNN layer
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        #third CNN layer
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        #forth CNN layer
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        #flatten layer
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        #Dense layer
        model.add(Dense(7, activation='softmax'))
    
        return model



    def visualize_model(self):
        """
        Visualize the model architecture using visualkeras
        """
        img = visualkeras.layered_view(self.model, to_file='model_visual.png')  # Create the image
        img.show()  # Display the image

    def train_model(self):

        num_train_imgs = sum([len(files) for r, d, files in os.walk(self.train_path)])
        num_test_imgs = sum([len(files) for r, d, files in os.walk(self.test_path)])

        history = self.model.fit(data_augmentation.train_generator,
                                 steps_per_epoch=num_train_imgs // self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=data_augmentation.validation_generator,
                                 validation_steps=num_test_imgs // self.batch_size)

        return history


    #saving the model I trained. What we are actually saving here is the values of the parameters and biases the model perfected during training
    def save_model(self, filename):
        self.model.save(filename)



