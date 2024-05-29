from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
import data_augmentation
import os

class EmotionRecognitionModel:
    def __init__(self, img_height=48, img_width=48, batch_size=32, epochs=50, train_path="data/train/", test_path="data/test"):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_path = train_path
        self.test_path = test_path
        self.model = self.define_model()

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 1)))
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(7, activation='softmax'))

        return model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

    def train_model(self):
        num_train_imgs = sum([len(files) for r, d, files in os.walk(self.train_path)])
        num_test_imgs = sum([len(files) for r, d, files in os.walk(self.test_path)])

        history = self.model.fit(data_augmentation.train_generator,
                                 steps_per_epoch=num_train_imgs // self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=data_augmentation.validation_generator,
                                 validation_steps=num_test_imgs // self.batch_size)

        return history

    def save_model(self, filename):
        self.model.save(filename)

if __name__ == "__main__":
    model = EmotionRecognitionModel()
    model.compile_model()
    history = model.train_model()
    model.save_model('emotion_recognition_model.h5')
    import plot_results
    plot_results.plot_training_history(history)
