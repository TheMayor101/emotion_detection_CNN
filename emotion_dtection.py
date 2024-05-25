from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os
from matplotlib import pyplot as plt
import tensorflow as tf
#from kerastuner.tuners import RandomSearch #helps tune the hyperparameters
import numpy as np

#Dataset from: https://www.kaggle.com/msambare/fer2013


#the images we got are 48X48 pixels
IMG_HEIGHT = 48
IMG_WIDTH = 48
batch_size = 32
epochs = 50


train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

# write why we use IMageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255, #we prefer to use numbers between 0-1 than 0-256
    rotation_range=10, # Random rotation by 10 degrees
    shear_range=0.1,
    zoom_range=0.1, # Randomly zoom by 10%
    height_shift_range = 0.1, # Randomly shift height by 10%
    width_shift_range = 0.1, # Randomly shift width by 10%
    brightness_range=[0.9, 1.1], #Randomaly change the brightness of the photo
    horizontal_flip=True, # Randomly flip images horizontally
    fill_mode='nearest') #we fill the new created pixles that may be created after some rotation with the value of the nearest pixels


validation_datagen = ImageDataGenerator(rescale=1. / 255) # for validation we don't want to change the pixels to much because we want to check the model on the images we got.

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# Define the CNN architecture
# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2)) #started with 0.1 decided on 0.2

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

model.compile(optimizer='ADAM', loss='categorical_crossentropy', metrics=['accuracy']) #check wether we should use a learning rate(lr) or check the diffult leraning rate
print(model.summary())




# Train the model
train_path = "data/train/"
test_path = "data/test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)


history = model.fit(train_generator,
                    steps_per_epoch=num_train_imgs // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs // batch_size)

# Generate predictions on the validation dataset
predictions = model.predict(validation_generator)



# Get class labels from the generator
class_labels = list(validation_generator.class_indices.keys())

# Get true labels
true_labels = validation_generator.classes

# Print the probabilities and true label for each image
for i in range(len(predictions)):
    print("Image:", i + 1)
    print("True Label:", class_labels[true_labels[i]])
    print("Probabilities:")
    for j, prob in enumerate(predictions[i]):
        print(f"{class_labels[j]}: {prob}")
    print()


model.save('emotion_recognition_model.h5')


# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
# acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
# val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

