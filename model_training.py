#This is the file which in we do the model training


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
#from keras.utils import plot_model
import data_augmentation as da
import os

IMG_HEIGHT = 48
IMG_WIDTH = 48
batch_size = 32
epochs = 50

# Define the CNN architecture
#I chose a model that uses the Sequential framwork in order to organize the layers in my model
#each layer off the code is added using model.add and than I specificed the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
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

#I used the Adam optimizer a loss function called categorical_crossentropy and I check the model's accuracy in order to check it's sucess.
model.compile(optimizer='ADAM', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train the model
train_path = "data/train/"
test_path = "data/test"

num_train_imgs = sum([len(files) for r, d, files in os.walk(train_path)])
num_test_imgs = sum([len(files) for r, d, files in os.walk(test_path)])

history = model.fit(da.train_generator,
                    steps_per_epoch=num_train_imgs // batch_size,
                    epochs=epochs,
                    validation_data=da.validation_generator,
                    validation_steps=num_test_imgs // batch_size)

#I saved the model to a .h5 file in ordre to be able to use it later and not need to run it each time individually.
model.save('emotion_recognition_model.h5')

if __name__ == "__main__":
    import plot_results
    plot_results.plot_training_history(history)
